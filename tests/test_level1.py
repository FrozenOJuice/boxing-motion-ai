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
        "confirmed_scene_cuts": [],
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


def test_mirrored_training_uses_original_evaluation_and_records_decisions(tmp_path, monkeypatch):
    from boxing_motion_ai.training import experiment

    monkeypatch.chdir(tmp_path)
    config = {
        "schema_version": "level1-experiment-1.1",
        "seed": 8,
        "epochs": 2,
        "batch_size": 6,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "hidden": 8,
        "dropout": 0.1,
        "threads": 1,
        "limit_per_class": None,
        "mirror_probability": 0.5,
        "train": synthetic(tmp_path, "V4"),
        "development": synthetic(tmp_path, "V5"),
        "runs": ["tcn", "tcn_mask_only"],
    }
    path = tmp_path / "mirror.yaml"
    path.write_text(yaml.safe_dump(config))
    evaluate_original = experiment.evaluate
    evaluations = []

    def checked_evaluate(model, loader, feature_key, weights):
        # Evaluation uses the source dataset itself, never an augmented wrapper.
        clip = loader.dataset[0]
        assert clip["raw"]["coordinates"][1, 15, 0] == 0.2
        evaluations.append(loader.dataset.provenance["recording_id"])
        return evaluate_original(model, loader, feature_key, weights)

    monkeypatch.setattr(experiment, "evaluate", checked_evaluate)
    output = tmp_path / "reports/mirror"
    experiment.run_experiment(path, output, progress=lambda _: None)
    assert set(evaluations) == {"V4", "V5"}
    logs = [
        json.loads((output / name / "augmentation.json").read_text()) for name in config["runs"]
    ]
    assert logs[0] == logs[1]
    assert all(e["seen"] == 6 for e in logs[0])
    assert sum(len(e["mirrored_source_rows"]) for e in logs[0]) > 0
    state = torch.load(output / "tcn/last.pt", weights_only=True)
    assert "augmentation_rng_state" in state
    for probability in (0.1, True):
        config["mirror_probability"] = probability
        path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match="Mirror probability"):
            load_config(path)


def test_visibility_config_training_and_unaugmented_evaluation(tmp_path, monkeypatch):
    from boxing_motion_ai.training import experiment

    monkeypatch.chdir(tmp_path)
    config = {
        "schema_version": "level1-experiment-1.2",
        "seed": 8,
        "epochs": 2,
        "batch_size": 6,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "hidden": 8,
        "dropout": 0.1,
        "threads": 1,
        "limit_per_class": None,
        "mirror_probability": 0.5,
        "visibility_probability": 0.5,
        "train": synthetic(tmp_path, "V4"),
        "development": synthetic(tmp_path, "V5"),
        "runs": ["tcn", "tcn_mask_only"],
    }
    path = tmp_path / "visibility.yaml"
    path.write_text(yaml.safe_dump(config))
    evaluate = experiment.evaluate
    seen = []

    def checked(model, loader, feature_key, weights):
        assert loader.dataset[0]["raw"]["valid_mask"][1].all()
        assert loader.dataset[0]["x"][1, 2 * 5 + 4] == 1
        seen.append(loader.dataset.provenance["recording_id"])
        return evaluate(model, loader, feature_key, weights)

    monkeypatch.setattr(experiment, "evaluate", checked)
    output = tmp_path / "reports/visibility"
    experiment.run_experiment(path, output, progress=lambda _: None)
    assert set(seen) == {"V4", "V5"}
    logs = [
        json.loads((output / name / "augmentation.json").read_text()) for name in config["runs"]
    ]
    assert logs[0] == logs[1]
    assert sum(len(e["visibility_events"]) for e in logs[0]) > 0
    state = torch.load(output / "tcn/last.pt", weights_only=True)
    assert "visibility_rng_state" in state
    for probability in (0.1, True):
        config["visibility_probability"] = probability
        path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match="Visibility probability"):
            load_config(path)


def test_context_wrapper_preserves_original_examples_features_raw_missingness(tmp_path):
    from boxing_motion_ai.training.context import ContextClips

    base = PoseClips(synthetic(tmp_path), "V4")
    ds = ContextClips(base, 0.65)
    assert len(ds) == len(base) == 6
    assert ds.provenance["source_rows"] == base.provenance["source_rows"]
    assert ds[0]["context_length"] == 0  # Do not exclude recording-start examples.
    assert ds[1]["context_length"] == 6
    assert ds[5]["context_length"] == 19
    for c, original in zip(ds.clips, base.clips, strict=True):
        offset = c["context_length"]
        torch.testing.assert_close(c["x"][offset:], original["x"], rtol=0, atol=0)
        torch.testing.assert_close(c["mask_x"][offset:], original["mask_x"], rtol=0, atol=0)
        assert c["pool_mask"].sum() == len(original["x"])
        assert not c["pool_mask"][:offset].any()
        for key in original["raw"]:
            np.testing.assert_array_equal(c["raw"][key][offset:], original["raw"][key])
    assert np.isnan(ds[1]["raw"]["coordinates"][0, 13]).all()
    assert not ds[1]["raw"]["valid_mask"][0, 13]
    assert exact_times(ds[1])[1] == Fraction(1001, 30000)


def test_context_training_keeps_punch_pooling_and_visibility_disabled(tmp_path, monkeypatch):
    from boxing_motion_ai.training import experiment

    monkeypatch.chdir(tmp_path)
    config = {
        "schema_version": "level1-experiment-1.3",
        "seed": 8,
        "epochs": 2,
        "batch_size": 6,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "hidden": 8,
        "dropout": 0.1,
        "threads": 1,
        "limit_per_class": None,
        "mirror_probability": 0.5,
        "visibility_probability": 0.0,
        "context_seconds": 0.65,
        "train": synthetic(tmp_path, "V4"),
        "development": synthetic(tmp_path, "V5"),
        "runs": ["tcn", "tcn_mask_only"],
    }
    path = tmp_path / "context.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "reports/context"
    experiment.run_experiment(path, output, progress=lambda _: None)
    p = json.loads((output / "provenance.json").read_text())
    assert p["context_policy"]["seconds"] == 0.65
    assert p["train"]["class_counts"] == p["development"]["class_counts"]
    for model in config["runs"]:
        logs = json.loads((output / model / "augmentation.json").read_text())
        assert not any(e["visibility_events"] for e in logs)
    config["visibility_probability"] = 0.5
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="visibility disabled"):
        load_config(path)


def test_real_context_truncates_at_cuts_and_segments_without_dropping_examples(tmp_path):
    from boxing_motion_ai.training.context import ContextClips

    spec = synthetic(tmp_path)
    root = tmp_path / "V4"
    with np.load(root / "poses.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["scene_id"][6:] = 1
    arrays["segment_id"][18:] = 1
    np.savez_compressed(root / "poses.npz", **arrays)
    manifest = json.loads((root / "preparation.json").read_text())
    manifest.update(confirmed_scene_cuts=[6], npz_sha256=sha256(root / "poses.npz"))
    dump(root / "preparation.json", manifest)
    policy = json.loads((root / "policy.json").read_text())
    policy["inputs"] = {k: sha256(root / k) for k in policy["inputs"]}
    dump(root / "policy.json", policy)
    base = PoseClips(spec, "V4")
    ds = ContextClips(base, 0.65)
    assert len(ds) == 6
    assert ds[1]["context_length"] == 0
    assert ds[2]["context_length"] == 6
    assert ds[3]["context_length"] == 0
    assert ds[4]["raw"]["frame_index"][0] == 18
    # A confirmed but otherwise uncatalogued cut inside the target must fail closed.
    manifest["confirmed_scene_cuts"] = [4, 6]
    dump(root / "preparation.json", manifest)
    policy["inputs"] = {k: sha256(root / k) for k in policy["inputs"]}
    dump(root / "policy.json", policy)
    with pytest.raises(ValueError, match="Punch crosses"):
        ContextClips(PoseClips(spec, "V4"), 0.65)


def test_incomplete_extraction_is_rejected_before_context_loading(tmp_path):
    spec = synthetic(tmp_path)
    root = tmp_path / "V4"
    with np.load(root / "poses.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["frame_index"][6:] += 1
    np.savez_compressed(root / "poses.npz", **arrays)
    manifest = json.loads((root / "preparation.json").read_text())
    manifest["npz_sha256"] = sha256(root / "poses.npz")
    dump(root / "preparation.json", manifest)
    policy = json.loads((root / "policy.json").read_text())
    policy["inputs"] = {k: sha256(root / k) for k in policy["inputs"]}
    dump(root / "policy.json", policy)
    with pytest.raises(ValueError, match="complete, contiguous"):
        PoseClips(spec, "V4")
