"""Synthetic positive clips only; no real dataset or network required."""

import json
from fractions import Fraction

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from boxing_motion_ai.training.data import (  # noqa: E402
    CLASSES,
    PoseClips,
    collate,
    exact_times,
    features,
)
from boxing_motion_ai.training.experiment import (  # noqa: E402
    load_config,
    metrics,
    run_experiment,
    seed_all,
)
from boxing_motion_ai.training.models import CausalTCN, StaticBaseline  # noqa: E402
from boxing_motion_ai.vision.contracts import JOINTS  # noqa: E402
from boxing_motion_ai.vision.pipeline import sha256  # noqa: E402


def dump(path, value):
    path.write_text(json.dumps(value))


def synthetic(tmp_path, video="V4", scene_cut=False):
    from boxing_motion_ai.data.pose_quality import ANCHORS, ARMS, CORE, POLICIES

    root = tmp_path / video
    root.mkdir()
    pose_run = root / "source"
    pose_run.mkdir()
    dump(pose_run / "manifest.json", {"synthetic": True, "video_id": video})
    dump(pose_run / "poses.jsonl", {"video_id": video, "source_width": 160, "source_height": 90})
    n = 36
    xyz = np.full((n, 33, 3), 0.5)
    xyz[:, 11:13, 1] = 0.2
    xyz[:, 23:25, 1] = 0.7
    # Six synthetic classes have distinct, finite wrist positions (not real results).
    xyz[:, 15, 0] = np.repeat(np.linspace(0.2, 0.7, 6), 6)
    xyz[:, 16, 0] = 0.45 if video == "V4" else 0.46
    confidence = np.ones((n, 33, 2))
    valid = np.ones((n, 33), dtype=bool)
    xyz[0, 13] = np.nan
    confidence[0, 13] = np.nan
    valid[0, 13] = False
    scene = np.zeros(n, dtype=np.int64)
    if scene_cut:
        scene[3:] = 1
    arrays = dict(
        frame_index=np.arange(n),
        pts=np.arange(n) * 1001,
        time_base_num=np.ones(n, dtype=np.int64),
        time_base_den=np.full(n, 30000, dtype=np.int64),
        coordinates=xyz,
        confidence=confidence,
        valid_mask=valid,
        scene_id=scene,
        segment_id=np.zeros(n, dtype=np.int64),
        pose_status=np.zeros(n, dtype=np.uint8),
    )
    np.savez_compressed(root / "poses.npz", **arrays)
    manifest = {
        "schema_version": "recording-preparation-1.1" if video == "V4" else "v5-preparation-1.0",
        "scope": "full_video",
        "video_id": video,
        "source_frame_count": n,
        "pose_contract": {
            "joint_order": list(JOINTS),
            "confidence_order": ["visibility", "presence"],
        },
        "inputs": {
            "pose_manifest_sha256": sha256(pose_run / "manifest.json"),
            "poses_sha256": sha256(pose_run / "poses.jsonl"),
        },
        "npz_sha256": sha256(root / "poses.npz"),
        "mapping_uncertainty": {"provisional": True},
    }
    dump(root / "preparation.json", manifest)
    candidates = [
        dict(
            source_row=i + 2,
            annotation_eligible=True,
            exclusion_reasons=[],
            array_slice_half_open=[i * 6, i * 6 + 6],
            punch_type_provisional=name,
            flags=["class_unverified"],
        )
        for i, name in enumerate(CLASSES)
    ]
    candidates.append(
        dict(source_row=99, annotation_eligible=False, exclusion_reasons=["questionable_row"])
    )
    (root / "candidates.jsonl").write_text("\n".join(map(json.dumps, candidates)))
    index = {
        "schema_version": "recording-mask-policy-index-1.1",
        "video_id": video,
        "policy": "mask_aware70",
        "definition": POLICIES["mask_aware70"],
        "max_missing_frame_fraction": 0.1,
        "source_mask_threshold": 0.5,
        "joint_groups": {
            "core": list(CORE),
            "anchors": list(ANCHORS),
            "arms": [list(x) for x in ARMS],
        },
        "inputs": {
            k: sha256(root / k) for k in ("preparation.json", "poses.npz", "candidates.jsonl")
        },
        "selected_source_rows": list(range(2, 8)),
    }
    dump(root / "policy.json", index)
    return {
        "recording": video,
        "dataset": str(root),
        "pose_run": str(pose_run),
        "policy": str(root / "policy.json"),
    }


def test_loader_preserves_exact_contract_and_exclusions(tmp_path):
    ds = PoseClips(synthetic(tmp_path), "V4")
    assert len(ds) == 6
    c = ds[0]
    assert c["raw"]["frame_index"].tolist() == list(range(6))
    assert exact_times(c)[1] == Fraction(1001, 30000)
    assert np.isnan(c["raw"]["coordinates"][0, 13]).all()
    assert not c["raw"]["valid_mask"][0, 13]
    assert torch.isfinite(c["x"]).all()
    assert c["mask_x"][0, 2] == 0
    assert c["annotation"]["flags"] == ["class_unverified"]
    assert ds.provenance["boxer_id"] is None
    assert 99 not in ds.provenance["source_rows"]
    b = collate([ds[0], {**ds[1], "x": ds[1]["x"][:3], "mask_x": ds[1]["mask_x"][:3]}])
    assert b["lengths"].tolist() == [6, 3]
    assert b["time_mask"].sum().item() == 9
    assert not b["x"][1, 3:].any()


def test_hash_and_split_checks(tmp_path):
    spec = synthetic(tmp_path)
    with pytest.raises(ValueError, match="split mismatch"):
        PoseClips(spec, "V5")
    with pytest.raises(ValueError, match="Only V4"):
        PoseClips(spec, "V9")
    with open(tmp_path / "V4/candidates.jsonl", "a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="hashes"):
        PoseClips(spec, "V4")


def test_policy_change_and_scene_crossing_rejected(tmp_path):
    spec = synthetic(tmp_path, scene_cut=True)
    with pytest.raises(ValueError, match="scene/segment"):
        PoseClips(spec, "V4")
    path = tmp_path / "V4/policy.json"
    index = json.loads(path.read_text())
    index["definition"]["core_entries"] = 0.1
    dump(path, index)
    with pytest.raises(ValueError, match="Unapproved quality"):
        PoseClips(spec, "V4")


def test_normalization_is_frame_local_and_invalid_values_cannot_leak():
    xyz = np.ones((3, 33, 3)) * 0.4
    xyz[:, 11:13, 1] = 0.2
    xyz[:, 23:25, 1] = 0.7
    conf = np.ones((3, 33, 2))
    mask = np.ones((3, 33), dtype=bool)
    mask[:, 13] = False
    x, masks, _ = features(xyz, conf, mask, 16 / 9)
    xyz[:, 13] = np.nan
    conf[:, 13] = -999
    changed, _, _ = features(xyz, conf, mask, 16 / 9)
    np.testing.assert_array_equal(x, changed)
    assert np.all(x[:, 10:15] == 0)  # left elbow's five gated channels
    # Translation + uniform scaling in image coordinates preserves normalized XY.
    moved = xyz * 0.6 + 0.1
    equivalent, _, _ = features(moved, conf, mask, 16 / 9)
    np.testing.assert_allclose(x, equivalent, atol=1e-6)
    xyz[2, 11:13, 1] = 0.1
    future, _, _ = features(xyz, conf, mask, 16 / 9)
    np.testing.assert_array_equal(x[:2], future[:2])
    mask[0, 11] = False
    failed, mask_only, normal = features(xyz, conf, mask, 16 / 9)
    assert not normal[0] and not failed[0].any()
    assert masks[0, 0] == 1 and mask_only[0, 0] == 0


@pytest.mark.parametrize("cls", [StaticBaseline, CausalTCN])
def test_padding_invariance_finite_gradients_and_seed(cls):
    seed_all(7, 1)
    model = cls(8).eval()
    x = torch.randn(1, 5, 8)
    original = model(x, torch.ones(1, 5, dtype=torch.bool))
    padded = torch.cat((x, torch.randn(1, 7, 8)), dim=1)
    mask = torch.arange(12)[None] < 5
    torch.testing.assert_close(original, model(padded, mask))
    original.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    seed_all(7, 1)
    repeat = cls(8).eval()
    torch.testing.assert_close(original, repeat(x, torch.ones(1, 5, dtype=torch.bool)))


def test_tcn_causal_prefix_and_static_order_invariance():
    seed_all(9, 1)
    x = torch.randn(2, 20, 8)
    model = CausalTCN(8).eval()
    changed = x.clone()
    changed[:, 7:] += 30
    torch.testing.assert_close(model.encode(x)[:, :7], model.encode(changed)[:, :7])
    static = StaticBaseline(8)
    mask = torch.ones(2, 20, dtype=torch.bool)
    torch.testing.assert_close(static(x, mask), static(x.flip(1), mask))


def test_metrics_support_and_undefined_precision():
    result = metrics(list(range(6)), [0] * 6)
    assert result["accuracy"] == pytest.approx(1 / 6)
    assert result["macro_f1"] == pytest.approx((2 / 7) / 6)
    assert result["per_class"]["cross"]["precision"] == 0
    assert result["confusion_matrix"][5][0] == 1


def test_synthetic_training_report_checkpoints_and_split_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "schema_version": "level1-experiment-1.0",
        "seed": 1,
        "epochs": 2,
        "batch_size": 6,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "hidden": 8,
        "dropout": 0.1,
        "threads": 1,
        "limit_per_class": None,
        "train": synthetic(tmp_path, "V4"),
        "development": synthetic(tmp_path, "V5"),
        "runs": ["static", "tcn", "static_mask_only", "tcn_mask_only"],
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "reports/test"
    result = run_experiment(path, output, progress=lambda _: None)
    assert result["static"]["development"]["per_class"]["jab"]["support"] == 1
    assert len((output / "tcn/curves.csv").read_text().splitlines()) == 3
    checkpoint = torch.load(output / "tcn/best.pt", weights_only=True)
    assert checkpoint["config"]["train"]["recording"] == "V4"
    assert (output / "tcn/confusion.png").is_file()
    assert (output / "results.md").is_file()
    assert json.loads((output / "provenance.json").read_text())["seed"] == 1
    with pytest.raises(ValueError, match="already exists"):
        run_experiment(path, output)
    config["development"]["recording"] = "V9"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="V9 reserved"):
        load_config(path)
