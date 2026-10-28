"""Synthetic causal windows, missing data, loss masks, streaming and exact event metrics."""

from copy import deepcopy
from fractions import Fraction as Q

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
from boxing_motion_ai.detection.data import collate, encode_source, make_windows  # noqa: E402
from boxing_motion_ai.detection.metrics import Event, Exposure, event_metrics  # noqa: E402
from boxing_motion_ai.detection.model import BinaryDetector  # noqa: E402
from boxing_motion_ai.detection.smoke import Config, masked_loss  # noqa: E402


def source(n=50):
    coordinates = np.full((n, 33, 3), 0.5)
    coordinates[:, 11:13, 1] = 0.2
    coordinates[:, 15, 0] = np.arange(n) * 0.001
    return dict(
        frame_index=np.arange(n),
        pts=np.arange(n) * 1001,
        time_base_num=np.ones(n, dtype=int),
        time_base_den=np.full(n, 30000),
        coordinates=coordinates,
        confidence=np.ones((n, 33, 2)),
        valid_mask=np.ones((n, 33), bool),
        pose_status=np.zeros(n, int),
        scene_id=np.zeros(n, int),
        segment_id=np.zeros(n, int),
    )


def interval(start=20, stop=45):
    return dict(
        id="synthetic",
        provenance=dict(recording="V4", start=start, stop=stop),
        supervision_runs=[
            dict(start=start, stop=start + 5, activity=1),
            dict(start=start + 5, stop=stop, activity=0),
        ],
    )


def test_windows_preserve_indices_unknown_warmup_and_every_observable_target_once():
    raw = source()
    before = deepcopy(raw)
    x, observed, epoch, _ = encode_source(raw, 16 / 9)
    windows, labels = make_windows(x, observed, epoch, interval(), 8)
    supervised = [
        int(f) for w in windows for f, m in zip(w["frames"], w["loss_mask"], strict=True) if m
    ]
    assert supervised == list(range(20, 45))
    assert windows[0]["frames"][0] == 6
    assert (windows[0]["labels"][:14] == -1).all()
    assert labels.tolist() == [1] * 5 + [0] * 20
    assert x.shape == (50, 65)
    for k in raw:
        np.testing.assert_array_equal(raw[k], before[k])


@pytest.mark.parametrize("kind", ["missing", "anchor", "cut", "scene", "segment", "timegap"])
def test_no_model_history_or_derivative_crosses_discontinuities(kind):
    raw = source()
    cuts = []
    if kind in ("missing", "anchor"):
        raw["coordinates"][28, 11] = np.nan
        raw["valid_mask"][28, 11] = False
        if kind == "missing":
            raw["pose_status"][28] = 2
    elif kind == "cut":
        cuts = [28]
    elif kind in ("scene", "segment"):
        raw[kind + "_id"][28:] = 1
    else:
        raw["pts"][28:] += 10000
    x, observed, epoch, _ = encode_source(raw, 1.0, cuts)
    windows, _ = make_windows(x, observed, epoch, interval(), 32)
    assert not x[28, 41:].any()
    for w in windows:
        if (w["frames"] >= 29).any() and w["frames"][-1] >= 29:
            assert w["frames"][0] >= (29 if kind in ("missing", "anchor") else 28)
    if kind in ("missing", "anchor"):
        assert 28 not in [
            int(f) for w in windows for f, m in zip(w["frames"], w["loss_mask"], strict=True) if m
        ]


def test_partial_missing_arm_is_masked_not_discarded_or_filled():
    raw = source()
    raw["coordinates"][25, 13] = np.nan
    raw["valid_mask"][25, 13] = False
    x, observed, epoch, _ = encode_source(raw, 1.0)
    assert observed[25]
    assert not x[25, :40].reshape(8, 5)[2].any()
    assert not x[25, 41:].reshape(8, 3)[2].any()
    windows, _ = make_windows(x, observed, epoch, interval())
    assert any(25 in w["frames"][w["loss_mask"].numpy()] for w in windows)


def test_future_motion_does_not_change_past_features_or_logits_and_streaming_matches():
    raw = source()
    x, _, _, _ = encode_source(raw, 1.0)
    changed = deepcopy(raw)
    changed["coordinates"][30:, 15, 0] += 0.1
    y, _, _, _ = encode_source(changed, 1.0)
    torch.testing.assert_close(x[:30], y[:30], rtol=0, atol=0)
    torch.manual_seed(4)
    model = BinaryDetector()
    model.eval()
    with torch.no_grad():
        full = model(x[None])[0]
        altered = model(y[None])[0]
        torch.testing.assert_close(full[:30], altered[:30], rtol=0, atol=0)
        state = None
        stream = []
        for frame in x:
            p, state = model.step(frame, state)
            stream.append(p)
        torch.testing.assert_close(torch.stack(stream), full.sigmoid(), rtol=1e-5, atol=1e-6)
        p, _ = model.step(x[35], state, reset=True)
        torch.testing.assert_close(p, model(x[35:36][None])[0, 0].sigmoid())


def test_warmup_chunk_predictions_match_full_past_and_padding_never_changes_the_past():
    raw = source()
    x, observed, epoch, _ = encode_source(raw, 1.0)
    windows, _ = make_windows(x, observed, epoch, interval(), 8)
    model = BinaryDetector()
    model.eval()
    with torch.no_grad():
        full = model(x[None])[0]
        for w in windows:
            logits = model(w["x"][None])[0]
            torch.testing.assert_close(
                logits[w["loss_mask"]],
                full[w["frames"][w["loss_mask"].numpy()]],
                rtol=1e-5,
                atol=1e-6,
            )
        padded = torch.cat((x, torch.randn(10, 65)), dim=0)
        torch.testing.assert_close(model(padded[None])[0, : len(x)], full, rtol=0, atol=0)


def test_loss_ignores_unknown_padding_and_warmup_and_backpropagates_only_supervised_logits():
    logits = torch.zeros(1, 5, requires_grad=True)
    labels = torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0]])
    mask = torch.tensor([[False, False, True, True, False]])
    loss = masked_loss(logits, labels, mask, 2.0)
    altered = labels.clone()
    altered[~mask] = 1 - labels[~mask]
    assert loss == masked_loss(logits, altered, mask, 2.0)
    loss.backward()
    assert (logits.grad[~mask] == 0).all()
    assert (logits.grad[mask] != 0).all()
    with pytest.raises(ValueError, match="Empty"):
        masked_loss(logits, labels, torch.zeros_like(mask), 1.0)


def event(name, start, stop, emit=None, recording="V4"):
    return Event(name, recording, Q(start), Q(stop), None if emit is None else Q(emit))


def test_event_metrics_duplicate_alarms_misses_latency_and_reviewed_minutes():
    refs = [event("a", 1, 2), event("b", 4, 5)]
    preds = [event("x", 1, 2, Q(6, 5)), event("duplicate", 1, 2, Q(13, 10))]
    result = event_metrics(refs, preds, [Exposure("V4", Q(0), Q(60))])
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["precision"] == result["recall"] == 0.5
    assert result["false_positives_per_minute"] == 1
    assert result["latency_seconds"]["count"] == 1
    assert result["latency_seconds"]["median"] == pytest.approx(0.3)


def test_matcher_maximizes_cardinality_before_iou_and_never_crosses_recordings():
    refs = [event("a", 0, 3), event("b", 2, 5)]
    preds = [event("flexible", 0, 5, 1), event("only_a", 0, 2, 1)]
    result = event_metrics(refs, preds, [Exposure("V4", Q(0), Q(6))])
    assert result["tp"] == 2
    other = event_metrics(
        refs,
        [event("wrong_recording", 0, 3, 1, "V6")],
        [Exposure("V4", Q(0), Q(6)), Exposure("V6", Q(0), Q(6))],
    )
    assert (other["tp"], other["fp"], other["fn"]) == (0, 1, 2)


def test_maximum_iou_tie_break_empty_events_and_undefined_metrics():
    result = event_metrics(
        [event("r", 1, 3)],
        [event("partial", 1, 2, 1), event("exact", 1, 3, 1)],
        [Exposure("V4", Q(0), Q(10))],
    )
    assert result["matches"][0]["prediction"] == "exact"
    empty = event_metrics([], [], [Exposure("V4", Q(0), Q(10))])
    assert empty["precision"] is empty["recall"] is empty["f1"] is None
    assert empty["latency_seconds"]["median"] is None
    assert empty["false_positives_per_minute"] == 0


@pytest.mark.parametrize("kind", ["unknown", "overlap", "emission", "zero"])
def test_metrics_reject_unverified_exposure_and_missing_emission_times(kind):
    coverage = [Exposure("V4", Q(0), Q(5))]
    preds = [event("p", 1, 2, 1)]
    if kind == "unknown":
        preds = [event("p", 4, 6, 4)]
    elif kind == "overlap":
        coverage.append(Exposure("V4", Q(4), Q(8)))
    elif kind == "emission":
        preds = [event("p", 1, 2)]
    else:
        coverage = []
    with pytest.raises(ValueError):
        event_metrics([], preds, coverage)


def test_collation_only_right_pads_and_preserves_loss_accounting():
    raw = source()
    x, observed, epoch, _ = encode_source(raw, 1.0)
    windows, _ = make_windows(x, observed, epoch, interval(), 8)
    batch, labels, mask = collate(windows)
    assert batch.shape[0] == len(windows)
    assert int(mask.sum()) == 25
    for i, w in enumerate(windows):
        n = len(w["x"])
        torch.testing.assert_close(batch[i, :n], w["x"])
        assert not mask[i, n:].any()
        assert not mask[i, : int(w["loss_mask"].nonzero()[0])].any()
        assert (labels[i][mask[i]] >= 0).all()


@pytest.mark.parametrize("recording", ["V5", "V6", "V7", "V8", "V10", "V9"])
def test_smoke_configuration_cannot_expand_training_sources(recording):
    import json
    from pathlib import Path

    config = json.loads((Path(__file__).parents[1] / "configs/detector_smoke_v1.json").read_text())
    config["training_recording"] = recording
    with pytest.raises(ValueError):
        Config.model_validate(config)


def test_checkpoint_inference_roundtrip_preserves_source_pts_and_reports_abstention(
    tmp_path, monkeypatch
):
    import hashlib
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from boxing_motion_ai.detection import smoke
    from boxing_motion_ai.detection.data import VERSION as FEATURE_VERSION
    from boxing_motion_ai.detection.model import VERSION

    raw = source()
    raw["coordinates"][28, 11] = np.nan
    raw["valid_mask"][28, 11] = False
    x, observed, epoch, _ = encode_source(raw, 1.0)
    region = interval()
    windows, labels = make_windows(x, observed, epoch, region, 8)
    ds = SimpleNamespace(raw=raw, windows=windows, intervals=[region], labels={"synthetic": labels})
    monkeypatch.setattr(smoke, "ReviewedWindows", lambda *_: ds)
    config = json.loads((Path(__file__).parents[1] / "configs/detector_smoke_v1.json").read_text())
    manifest = tmp_path / "reports/synthetic/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    config["dataset"] = {
        "path": "reports/synthetic/manifest.json",
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/continuous_dataset_v1.json").write_text(json.dumps(config["dataset"]))
    run = tmp_path / "reports/run"
    run.mkdir()
    model = BinaryDetector()
    torch.save(
        dict(
            version=VERSION,
            feature_version=FEATURE_VERSION,
            config=config,
            state_dict=model.state_dict(),
            dataset_sha256=config["dataset"]["sha256"],
        ),
        run / "checkpoint.pt",
    )
    (run / "checkpoint.sha256").write_text(
        hashlib.sha256((run / "checkpoint.pt").read_bytes()).hexdigest()
    )
    result = smoke.predict(tmp_path, "reports/run", "reports/predictions")
    rows = [
        json.loads(line)
        for line in (tmp_path / "reports/predictions/frames.jsonl").read_text().splitlines()
    ]
    assert [r["frame_index"] for r in rows] == list(range(20, 45))
    assert rows[0]["timestamp_exact"] == str(Q(20 * 1001, 30000))
    missing = next(r for r in rows if r["frame_index"] == 28)
    assert missing["probability_punch"] is None and missing["observable"] is False
    assert result["unobservable_rows"] == 1
    assert result["streaming_window_max_error"] < 1e-5
    with pytest.raises(ValueError, match="new ignored"):
        smoke.predict(tmp_path, "reports/run", "reports/predictions")
    (run / "checkpoint.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checkpoint hash mismatch"):
        smoke.predict(tmp_path, "reports/run", "reports/other")
