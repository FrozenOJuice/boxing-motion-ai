"""Synthetic, read-only geometry checks; no source data or model training needed."""

from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from boxing_motion_ai.training.augmentation import mirror_features  # noqa: E402
from boxing_motion_ai.training.limb_investigation import (  # noqa: E402
    clip_measurements,
    describe_clips,
    load_settings,
    write_report,
)
from boxing_motion_ai.training.limbs import (  # noqa: E402
    MASK_COLUMNS,
    REFLECT_COLUMNS,
    limb_features,
)


def pose(n=4):
    x = torch.zeros(n, 65)
    joint = x[:, :40].reshape(n, 8, 5)
    joint[..., 2:] = 1
    joint[..., :2] = torch.tensor(
        [
            [-0.3, 0],
            [0.3, 0],
            [-0.3, -0.5],
            [0.3, 0.5],
            [-0.8, -0.5],
            [0.3, 1.0],
            [-0.2, 1],
            [0.2, 1],
        ]
    )
    x[:, 40] = 1
    raw = dict(
        frame_index=np.arange(n),
        pts=np.arange(n) * 1001,
        time_base_num=np.ones(n, int),
        time_base_den=np.full(n, 30000, int),
        scene_id=np.zeros(n, int),
        segment_id=np.zeros(n, int),
        pose_status=np.zeros(n, int),
    )
    return x, raw


def test_known_angles_and_vectors_and_inputs_unchanged():
    x, _ = pose()
    before = x.clone()
    result = limb_features(x).reshape(4, 2, 17)
    torch.testing.assert_close(result[:, 0, 9], torch.zeros(4))  # 90 degrees
    torch.testing.assert_close(result[:, 1, 9], -torch.ones(4))  # 180 degrees
    torch.testing.assert_close(result[:, 0, :2], torch.tensor([-0.5, -0.5]).expand(4, 2))
    assert result[..., list(MASK_COLUMNS)].all()
    torch.testing.assert_close(x, before, rtol=0, atol=0)


def test_invalid_elbow_does_not_destroy_available_wrist_relative_geometry():
    x, _ = pose()
    joint = x[:, :40].reshape(4, 8, 5)
    joint[:, 2, 4] = 0
    joint[:, 2, :2] = 99999  # Finite invalid payload must never be trusted.
    a = limb_features(x).reshape(4, 2, 17)
    assert a[:, 0, 2].all()
    assert not a[:, 0, 3:].any()
    assert a[:, 1, list(MASK_COLUMNS)].all()
    joint[:, 2, :2] = -99999
    torch.testing.assert_close(limb_features(x).reshape_as(a), a, rtol=0, atol=0)


@pytest.mark.parametrize("joint_index", [0, 1, 6, 7])
def test_unusable_torso_anchor_masks_orientation(joint_index):
    x, _ = pose()
    x[:, :40].reshape(4, 8, 5)[:, joint_index, 4] = 0
    a = limb_features(x).reshape(4, 2, 17)
    assert not a[..., 11:].any()
    x[:, 40] = 0  # Existing normalization validity gates every candidate feature.
    assert not limb_features(x).any()


def test_degenerate_bone_masks_angles_without_filling_joints():
    x, _ = pose()
    joint = x[:, :40].reshape(4, 8, 5)
    joint[:, 2, :2] = joint[:, 0, :2]
    a = limb_features(x).reshape(4, 2, 17)
    assert not a[:, 0, 3:6].any()
    assert not a[:, 0, 9:14].any()
    assert a[:, 0, 2].all() and a[:, 0, 8].all()


def test_mirror_covariance_with_anatomical_swap_and_signed_orientation():
    x, _ = pose()
    a = limb_features(x).reshape(4, 2, 17)
    reflected = a[:, [1, 0]].clone()
    reflected[..., list(REFLECT_COLUMNS)] *= -1
    actual = limb_features(mirror_features(x)).reshape_as(a)
    torch.testing.assert_close(actual, reflected, rtol=0, atol=0)
    twice = reflected[:, [1, 0]].clone()
    twice[..., list(REFLECT_COLUMNS)] *= -1
    torch.testing.assert_close(twice, a, rtol=0, atol=0)


def test_torso_relative_angles_remove_in_plane_rotation_but_vectors_rotate():
    x, _ = pose()
    a = limb_features(x).reshape(4, 2, 17)
    rotated = x.clone()
    xy = rotated[:, :40].reshape(4, 8, 5)[..., :2]
    xy.copy_(torch.stack((-xy[..., 1], xy[..., 0]), -1))
    b = limb_features(rotated).reshape_as(a)
    torch.testing.assert_close(b[..., 9:], a[..., 9:], atol=1e-6, rtol=0)
    torch.testing.assert_close(b[..., :2], torch.stack((-a[..., 1], a[..., 0]), -1))


def test_frame_local_causality_and_batch_support():
    x, _ = pose()
    before = limb_features(x)
    x[2:, :40].reshape(2, 8, 5)[..., :2] *= 2
    torch.testing.assert_close(limb_features(x)[:2], before[:2], rtol=0, atol=0)
    torch.testing.assert_close(limb_features(x[None])[0], limb_features(x))


def settings():
    return load_settings(Path("configs/level1_limb_investigation.yaml"))


@pytest.mark.parametrize("break_kind", ["cut", "missing", "frame_gap", "time_gap"])
def test_angle_jump_diagnostics_never_cross_discontinuities(break_kind):
    x, raw = pose()
    x[2:, :40].reshape(2, 8, 5)[:, 4, :2] = torch.tensor([-0.3, -1.0])
    c = dict(x=x, raw=raw, source_row=17, label=3)
    control = clip_measurements(c, settings(), Fraction(1, 10))
    assert control["sides"][0]["large_jumps"] == 1
    if break_kind == "cut":
        raw["scene_id"][2:] = 1
    elif break_kind == "missing":
        raw["pose_status"][2] = 2
    elif break_kind == "frame_gap":
        raw["frame_index"][2:] += 3
    else:
        raw["pts"][2:] += 30000
    c = dict(x=x, raw=raw, source_row=17, label=3)
    a = clip_measurements(c, settings(), Fraction(1, 10))
    assert a["sides"][0]["large_jumps"] == 0
    assert a["source_row"] == 17 and a["class"] == "rear_hook"


def test_report_retains_absence_and_sparse_side_counts(tmp_path):
    x, raw = pose()
    x[:, :40].reshape(4, 8, 5)[:, 3, 4] = 0
    c = clip_measurements(dict(x=x, raw=raw, source_row=1, label=3), settings(), Fraction(1, 10))
    a = describe_clips([c])
    assert a["by_side"][0]["clips_majority_elbow_angle"] == 1
    assert a["by_side"][1]["clips_any_elbow_angle"] == 0
    assert a["coverage"][1][0] == 1
    assert a["coverage"][1][3] == 0
    empty = describe_clips([])
    result = dict(
        decision="Investigation only",
        recordings={"V4": dict(all=a, by_class={"rear_hook": a, "jab": empty})},
    )
    write_report(tmp_path, result)
    text = (tmp_path / "summary.md").read_text()
    assert "no eligible examples" in text and "100.00% / 0.00%" in text


def test_scope_rejects_reserved_or_changed_inputs_before_dataset_access(tmp_path):
    a = settings()
    for field, value in [
        ("recordings", ["V9"]),
        ("base_config", "unapproved"),
        ("short_projected_segment_torso_units", 0.25),
    ]:
        b = deepcopy(a)
        b[field] = value
        path = tmp_path / "settings.yaml"
        path.write_text(yaml.safe_dump(b))
        with pytest.raises(ValueError, match="V9 reserved"):
            load_settings(path)


def test_optional_wrapper_preserves_base_inputs_masks_labels_and_source_records():
    from types import SimpleNamespace

    from boxing_motion_ai.training.limbs import LimbClips

    x, raw = pose()
    x[0, :40].reshape(8, 5)[2, 4] = 0  # Asymmetric geometry validity must swap too.
    clip = dict(x=x, raw=raw, mask_x=torch.ones(4, 16), label=4, source_row=42)
    ds = SimpleNamespace(clips=[clip], provenance={"recording_id": "V4", "source_rows": [42]})
    augmented = LimbClips(ds)
    result = augmented[0]
    assert len(augmented) == 1 and result["x"].shape == (4, 99)
    torch.testing.assert_close(result["x"][:, :65], x, rtol=0, atol=0)
    torch.testing.assert_close(result["mask_x"][:, :16], clip["mask_x"], rtol=0, atol=0)
    assert result["label"] == 4 and result["raw"] is raw
    assert ds.provenance == {"recording_id": "V4", "source_rows": [42]}
    reflected = mirror_features(result["x"])
    recomputed = torch.cat((mirror_features(x), limb_features(mirror_features(x))), -1)
    torch.testing.assert_close(reflected, recomputed, atol=1e-6, rtol=0)
    torch.testing.assert_close(mirror_features(reflected), result["x"], atol=0, rtol=0)
    masks = result["mask_x"].clone()
    masks[0, 16:] = torch.tensor([0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
    reflected_masks = mirror_features(masks, mask_only=True)
    torch.testing.assert_close(reflected_masks[0, 16:22], masks[0, 22:28], rtol=0, atol=0)
    torch.testing.assert_close(reflected_masks[0, 22:28], masks[0, 16:22], rtol=0, atol=0)
    torch.testing.assert_close(
        mirror_features(mirror_features(masks, mask_only=True), mask_only=True),
        masks,
        rtol=0,
        atol=0,
    )
    ds.provenance["recording_id"] = "V9"
    with pytest.raises(ValueError, match="reserved"):
        LimbClips(ds)


def test_zero_column_initialization_preserves_reference_predictions_and_rng():
    from boxing_motion_ai.training.experiment import seed_all
    from boxing_motion_ai.training.models import CausalTCN
    from boxing_motion_ai.training.velocity import expand_input

    seed_all(123, 1)
    original = CausalTCN(41)
    reference = expand_input(deepcopy(original), 65).eval()
    rng = torch.get_rng_state().clone()
    geometry = expand_input(deepcopy(original), 99).eval()
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(
        reference.input.weight, geometry.input.weight[:, :65], rtol=0, atol=0
    )
    assert not geometry.input.weight[:, 65:].any()
    x, _ = pose()
    appended = torch.cat((x, limb_features(x)), -1)
    mask = torch.ones(1, 4, dtype=torch.bool)
    torch.testing.assert_close(
        reference(x[None], mask), geometry(appended[None], mask), rtol=0, atol=1e-6
    )


def test_geometry_config_keeps_v4_v5_split_and_interventions_frozen(tmp_path):
    from boxing_motion_ai.training.experiment import load_config
    from boxing_motion_ai.training.limb_study import load_study

    base = yaml.safe_load(Path("configs/level1_initial.yaml").read_text())
    base.update(
        schema_version="level1-experiment-1.6",
        feature_representation="xy_velocity_limb",
        mirror_probability=0.5,
        runs=["tcn", "tcn_mask_only"],
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(base))
    assert load_config(path)["feature_representation"] == "xy_velocity_limb"
    for field, value in [
        ("mirror_probability", 0),
        ("visibility_probability", 0.5),
        ("context_seconds", 0.3),
        ("feature_representation", "angles_only"),
    ]:
        bad = {**base, field: value}
        path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError):
            load_config(path)
    bad = deepcopy(base)
    bad["train"]["recording"] = "V6"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="V4 train"):
        load_config(path)
    study = load_study(Path("configs/level1_limb_study.yaml"))
    study["seeds"] = [1]
    path.write_text(yaml.safe_dump(study))
    with pytest.raises(ValueError, match="V9 reserved"):
        load_study(path)


def test_study_aggregation_and_report_preserve_unsupported_classes(tmp_path):
    from boxing_motion_ai.training.experiment import metrics
    from boxing_motion_ai.training.limb_study import ARMS, aggregate, write_report
    from boxing_motion_ai.training.velocity_transfer import SEEDS, supported_metrics

    reference = supported_metrics(metrics([2, 2, 4], [2, 0, 2]))
    added = supported_metrics(metrics([2, 2, 4], [2, 2, 4]))
    runs = {
        str(s): {
            arm: {m: deepcopy(r) for m in ("tcn", "tcn_mask_only")}
            for arm, r in zip(ARMS, (reference, added), strict=True)
        }
        for s in SEEDS
    }
    a = aggregate(runs)
    assert a["tcn"][ARMS[1]]["per_class_recall"]["jab"] is None
    assert a["tcn"][ARMS[1]]["supported_macro_f1"]["mean"] == 1
    assert a["tcn"][ARMS[1]]["macro_f1"]["mean"] == pytest.approx(1 / 3)
    assert a["tcn"]["paired_supported_macro_f1_gain"]["mean"] > 0
    support = {
        c: {
            "retained": r["support"],
            "status": "present" if r["support"] else "absent_from_source_annotations",
        }
        for c, r in reference["per_class"].items()
    }
    result = dict(
        warning="Provisional", recordings={"V6": dict(source_support=support, aggregate=a)}
    )
    write_report(tmp_path, result, {})
    report = (tmp_path / "summary.md").read_text()
    assert "absent_from_source_annotations | unsupported" in report
    assert "Rows true / columns predicted" in report
    assert "supported-class F1" in report
