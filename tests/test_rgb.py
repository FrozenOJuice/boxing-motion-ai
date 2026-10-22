"""Synthetic RGB policy checks; no pretrained download or dataset needed."""

from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from boxing_motion_ai.training.rgb import (  # noqa: E402
    GRAY,
    VisualClips,
    crop_box,
    sampling_plan,
    views,
)
from boxing_motion_ai.training.rgb_study import (  # noqa: E402
    augment_visual,
    fit,
    load_settings,
    visual_collate,
)


def clip(v="V4", label=0, row=1):
    n = 8
    return dict(
        x=torch.arange(n * 65).reshape(n, 65).float() / 100,
        mask_x=torch.ones(n, 16),
        label=label,
        recording_id=v,
        source_row=row,
        raw=dict(
            frame_index=np.arange(n) + 12,
            pts=np.arange(n) * 1001 + 42042,
            time_base_num=np.ones(n, int),
            time_base_den=np.full(n, 30000),
            scene_id=np.zeros(n, int),
            segment_id=np.zeros(n, int),
        ),
    )


def test_sampling_exact_pts_is_causal_and_resets_at_boundaries():
    c = clip()
    selected, mapping = sampling_plan(c)
    assert selected == [0, 3, 6]  # 29.97 FPS, never rounded to nominal 30.
    assert mapping == [12, 12, 12, 15, 15, 15, 18, 18]
    c["raw"]["scene_id"][2:] = 1
    c["raw"]["frame_index"][4:] += 1
    _, mapping = sampling_plan(c)
    assert mapping[2] == 14 and mapping[4] == 17
    c["raw"]["pts"][1] = c["raw"]["pts"][0]
    with pytest.raises(ValueError, match="PTS"):
        sampling_plan(c)


def test_crop_requires_all_anchors_and_views_redact_hud_without_source_mutation():
    xy, valid = np.zeros((33, 3)), np.ones(33, bool)
    xy[[11, 12], :2] = [[0.4, 0.35], [0.6, 0.35]]
    xy[[23, 24], :2] = [[0.45, 0.6], [0.55, 0.6]]
    box, fallback = crop_box(xy, valid, (400, 200))
    assert not fallback and box[2] - box[0] == 225
    source = Image.new("RGB", (400, 200), "red")
    original = source.tobytes()
    crop, flip, background, bgflip = views(source, (0, 0, 400, 400), "V4")
    assert source.tobytes() == original
    assert crop.getpixel((20, 10)) == GRAY
    assert np.array_equal(np.asarray(flip), np.asarray(crop)[:, ::-1])
    assert np.array_equal(np.asarray(bgflip), np.asarray(background)[:, ::-1])
    valid[11] = False
    assert crop_box(xy, valid, (400, 200))[1]


def test_visual_features_mirror_pose_and_cache_indices_without_metadata_inputs(tmp_path):
    c = clip()
    values = np.arange(3 * 4 * 576, dtype=np.float32).reshape(3, 4, 576)
    np.savez(tmp_path / "features.npz", frame_index=[12, 15, 18], features=values)
    ds = SimpleNamespace(clips=[c], provenance=dict(recording_id="V4"))
    rgb = VisualClips(ds, tmp_path, "rgb")
    hybrid = VisualClips(ds, tmp_path, "hybrid")
    background = VisualClips(ds, tmp_path, "background")
    assert torch.equal(background[0]["x"][0], background[0]["x"][-1])
    assert rgb[0]["x"].shape == (8, 576)
    assert hybrid[0]["x"].shape == (8, 641)
    from boxing_motion_ai.training.augmentation import mirror_features

    torch.testing.assert_close(hybrid[0]["reflected_x"][:, :65], mirror_features(c["x"]))
    assert torch.equal(rgb[0]["x"][0], rgb[0]["x"][1])
    torch.testing.assert_close(rgb[0]["reflected_x"][0], torch.from_numpy(values[0, 1]))
    changed = deepcopy(c)
    changed.update(source_row=999, label=5, timestamp_seconds="not an input")
    other = VisualClips(SimpleNamespace(clips=[changed], provenance=ds.provenance), tmp_path, "rgb")
    assert torch.equal(other[0]["x"], rgb[0]["x"])
    ds.provenance = dict(recording_id="V9")
    with pytest.raises(ValueError, match="non-V9"):
        VisualClips(ds, tmp_path / "does-not-exist", "rgb")


def test_mirror_uses_one_coin_for_complete_rgb_pose_pair_and_keeps_inputs():
    c = {**clip(), "reflected_x": -clip()["x"]}
    batch = visual_collate([c] * 12)
    before = batch["x"].clone()
    transformed, coins = augment_visual(batch, torch.Generator().manual_seed(1))
    assert coins.any() and (~coins).any()
    assert torch.equal(transformed["x"][coins], batch["reflected_x"][coins])
    assert torch.equal(transformed["x"][~coins], batch["x"][~coins])
    assert torch.equal(batch["x"], before)


def test_review_protocol_and_training_are_reproducible(tmp_path, monkeypatch):
    load_settings("configs/level1_rgb.yaml")
    monkeypatch.setattr("boxing_motion_ai.training.rgb_study.plot", lambda *a: None)
    train = [clip(label=i, row=i) for i in range(6)]
    dev = [clip(v="V5", label=i, row=i) for i in range(6)]
    for c in [*train, *dev]:
        c["reflected_x"] = -c["x"]
    base = dict(
        epochs=1,
        batch_size=4,
        threads=1,
        hidden=32,
        dropout=0.1,
        learning_rate=0.001,
        weight_decay=0.001,
    )

    def ds(rows):
        return SimpleNamespace(clips=rows)

    from boxing_motion_ai.training.multi_recording import Pool

    a, audit_a = fit(Pool([ds(train)]), Pool([ds(dev)]), base, tmp_path / "a")
    b, audit_b = fit(Pool([ds(train)]), Pool([ds(dev)]), base, tmp_path / "b")
    assert audit_a == audit_b
    for x, y in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(x, y)


def test_decoder_matches_exact_synthetic_pts_and_rejects_nonexistent_frame(tmp_path):
    av = pytest.importorskip("av")
    from boxing_motion_ai.training.rgb_review import decode_requested

    path = tmp_path / "test.mkv"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("ffv1", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "bgr0"
        for i in range(10):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), i * 20, np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    with av.open(str(path)) as source:
        frames = list(source.decode(video=0))
        times = [f.pts * f.time_base for f in frames]
    result = list(decode_requested(path, {i: times[i] for i in (0, 5, 9)}))
    assert [r[0] for r in result] == [0, 5, 9]
    assert [r[2] for r in result] == [times[i] for i in (0, 5, 9)]
    with pytest.raises(ValueError, match="exact source"):
        list(decode_requested(path, {5: times[5] + Fraction(1, 100000)}))


def test_cache_rejects_policy_source_or_byte_changes(tmp_path):
    from boxing_motion_ai.training.rgb import validate_cache
    from boxing_motion_ai.vision.pipeline import sha256

    feature = tmp_path / "features.npz"
    feature.write_bytes(b"synthetic embeddings")
    inputs = dict(policy="v1", video_sha256="original", source_rows=[1, 3])
    saved = dict(inputs=inputs, feature_sha256=sha256(feature))
    validate_cache(saved, inputs, feature)
    for change in (dict(policy="v2"), dict(video_sha256="other"), dict(source_rows=[1, 2])):
        with pytest.raises(ValueError, match="Stale"):
            validate_cache(saved, {**inputs, **change}, feature)
    feature.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        validate_cache(saved, inputs, feature)


def test_encoder_is_frozen_and_eval_without_downloading_weights(tmp_path, monkeypatch):
    models = pytest.importorskip("torchvision.models")
    from boxing_motion_ai.training.rgb import encoder

    fake = torch.nn.Sequential(torch.nn.Linear(3, 576), torch.nn.Dropout(0.5))
    fake.classifier = torch.nn.Linear(576, 6)
    monkeypatch.setattr(models, "mobilenet_v3_small", lambda **kw: fake)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    (directory / "mobilenet_v3_small-047dcff4.pth").write_bytes(b"synthetic weights")
    model, _, _ = encoder()
    assert model is fake and not model.training
    assert all(not p.requires_grad for p in model.parameters())
    assert isinstance(model.classifier, torch.nn.Identity)


def test_sampling_cannot_change_past_when_future_frames_are_appended():
    full = clip()
    prefix = deepcopy(full)
    prefix["raw"] = {k: v[:4] for k, v in prefix["raw"].items()}
    assert sampling_plan(prefix)[1] == sampling_plan(full)[1][:4]


def test_readable_report_includes_absence_controls_and_confusions():
    from boxing_motion_ai.training.experiment import metrics
    from boxing_motion_ai.training.rgb_report import render
    from boxing_motion_ai.training.velocity_transfer import supported_metrics

    m = supported_metrics(metrics([0, 0, 1], [0, 1, 1]))
    case = dict(
        fold=dict(heldout=["V6"]),
        runs={
            mode: {v: m for v in ("V5", "V6", "heldout_group")}
            for mode in ("pose", "rgb", "hybrid", "mask", "background")
        },
        duration_control={v: m for v in ("V5", "V6", "heldout_group")},
    )
    report = render(dict(cases={"hold-V6": case}))
    assert "absent" in report and "Rows=true" in report
    assert "supported" in report.lower() and "duration" in report
    assert "background" in report and "precision" in report
