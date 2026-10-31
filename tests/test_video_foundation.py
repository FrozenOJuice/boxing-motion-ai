"""Full-body, native video timing and task-isolation contracts. No training or network calls."""

from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from boxing_motion_ai.data.video_foundation import admitted, safe_weak_rows
from boxing_motion_ai.vision.full_body import CHANNELS, encode
from boxing_motion_ai.vision.video_foundation import encoder, full_body_crop, pixels, windows


def raw(n=40, rate=25):
    xy = np.zeros((n, 33, 3), np.float32)
    xy[:, :, :2] = [0.5, 0.5]
    xy[:, 11, :2] = [0.4, 0.3]
    xy[:, 12, :2] = [0.6, 0.3]
    xy[:, 23, :2] = [0.45, 0.6]
    xy[:, 24, :2] = [0.55, 0.6]
    xy[:, 15, 0] = 0.35 + np.arange(n) * 0.001
    xy[:, 31, :2] = [0.4, 0.9]
    xy[:, 32, :2] = [0.6, 0.9]
    return dict(
        coordinates=xy,
        confidence=np.ones((n, 33, 2), np.float32),
        valid_mask=np.ones((n, 33), bool),
        pose_status=np.zeros(n, int),
        frame_index=np.arange(n),
        pts=np.arange(n),
        time_base_num=np.ones(n, int),
        time_base_den=np.full(n, rate),
        scene_id=np.zeros(n, int),
        segment_id=np.zeros(n, int),
    )


def test_fullbody_schema_contains_feet_and_exact_time_velocity():
    r = raw()
    x, normal, _ = encode(r, 1)
    assert x.shape == (40, 209) and len(set(CHANNELS)) == 209 and normal.all()
    vx = CHANNELS.index("left_wrist.vx")
    valid = CHANNELS.index("left_wrist.velocity_valid")
    assert x[0, valid] == 0 and x[1, valid] == 1
    assert x[1, vx] == pytest.approx(0.001 / 0.3 * 25, rel=1e-4)
    assert x[:, CHANNELS.index("left_foot_index.valid")].all()
    r["time_base_den"][:] = 30
    y, _, _ = encode(r, 1)
    assert y[1, vx] / x[1, vx] == pytest.approx(30 / 25, rel=1e-4)


def test_missing_and_offscreen_joints_are_masked_not_filled_and_originals_stay_intact():
    r = raw()
    r["valid_mask"][10, 15] = False
    r["coordinates"][10, 15] = np.nan
    r["coordinates"][12, 31, 0] = 1.2
    original = deepcopy(r)
    x, _, _ = encode(r, 1)
    offset = CHANNELS.index("left_wrist.x")
    assert (x[10, offset : offset + 8] == 0).all()
    assert x[11, offset + 7] == 0 and x[12, offset + 7] == 1
    assert x[12, CHANNELS.index("left_foot_index.valid")] == 0
    for k in r:
        assert np.array_equal(r[k], original[k], equal_nan=True)


def test_missing_torso_invalidates_normalization_without_creating_zero_motion_evidence():
    r = raw()
    r["valid_mask"][10, 23] = False
    x, normal, _ = encode(r, 1)
    assert not normal[10] and not x[10].any()


def test_native_discontinuities_reset_derivatives_and_video_windows():
    for kind in ("scene", "missing_frame", "gap"):
        r = raw()
        if kind == "scene":
            r["scene_id"][20:] = 1
        elif kind == "missing_frame":
            r["frame_index"][20:] += 1
        else:
            r["pts"][20:] += 20
        x, _, reset = encode(r, 1)
        assert reset[20] and x[20, CHANNELS.index("left_wrist.velocity_valid")] == 0
        assert all(not (a < 20 < z) for _, a, z in windows(40, reset))


def test_explicit_source_cut_resets_pose_derivatives_as_well_as_video_context():
    r = raw()
    cuts = np.zeros(40, bool)
    cuts[20] = True
    x, _, reset = encode(r, 1, cuts)
    assert reset[20] and x[20, CHANNELS.index("left_wrist.velocity_valid")] == 0
    with pytest.raises(ValueError):
        encode(r, 1, np.zeros(39, bool))


def test_root_translation_is_retained_but_estimator_z_is_not_treated_as_3d_rotation():
    r = raw()
    x, _, _ = encode(r, 1)
    r["coordinates"][:, :, 0] += np.arange(40)[:, None] * 0.0001
    r["coordinates"][:, :, 2] = 1000
    y, _, _ = encode(r, 1)
    assert y[2, CHANNELS.index("root.vx")] > 0
    assert np.allclose(
        x[:, CHANNELS.index("axis_alignment.cos")], y[:, CHANNELS.index("axis_alignment.cos")]
    )
    r2 = deepcopy(r)
    r2["coordinates"][:, :, 2] = -1000
    assert np.array_equal(encode(r, 1)[0], encode(r2, 1)[0])


def test_invalid_clock_and_aspect_fail_closed():
    r = raw()
    r["pts"][10] = r["pts"][9]
    with pytest.raises(ValueError):
        encode(r, 1)
    with pytest.raises(ValueError):
        encode(raw(), 0)


def test_native_windows_keep_every_center_without_fabricating_edge_context():
    for rate in (24, 30):
        r = raw(rate=rate)
        _, _, reset = encode(r, 1)
        p = windows(40, reset)
        assert len(p) == 25 and p[0] == (7, 0, 16) and p[-1] == (31, 24, 40)
        # Same genuine native inventory, differing physical context duration, never 10 Hz hold.
        assert Q(r["pts"][p[0][2] - 1] - r["pts"][0], rate) == Q(15, rate)
    assert windows(15, np.zeros(15, bool)) == []


def test_crop_contains_all_observed_body_points_and_empty_fallback_is_explicit():
    r = raw()
    box, fallback = full_body_crop(r, (640, 480))
    assert not fallback
    for p in r["coordinates"][:, 11:33, :2].reshape(-1, 2) * [640, 480]:
        assert box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]
    r["valid_mask"][:] = False
    assert full_body_crop(r, (640, 480)) == ((0, 0, 640, 480), True)


def test_letterbox_retains_frame_corners_through_official_center_crop(monkeypatch):
    from torchvision.transforms import functional as F

    from boxing_motion_ai.vision.video_foundation import MOTION_POLICY, RGB_POLICY

    monkeypatch.setitem(RGB_POLICY["hud_rectangles_normalized"], "V4", [])
    monkeypatch.setitem(MOTION_POLICY["source_identifier_rectangles"], "V4", [])
    a = np.zeros((224, 224, 3), np.uint8)
    a[:20, :20] = [255, 0, 0]
    a[:20, -20:] = [0, 255, 0]
    a[-20:, :20] = [0, 0, 255]
    a[-20:, -20:] = 255
    im = F.center_crop(pixels(Image.fromarray(a), (0, 0, 224, 224), "V4"), [112, 112])
    b = np.asarray(im)
    assert np.array_equal(b[0, 0], [255, 0, 0]) and np.array_equal(b[-1, -1], [255, 255, 255])
    assert np.array_equal(b[0, -1], [0, 255, 0]) and np.array_equal(b[-1, 0], [0, 0, 255])


def test_identifier_union_is_static_and_missing_anchors_do_not_expose_unredacted_identity():
    from boxing_motion_ai.vision.video_foundation import shirt_identifiers

    r = raw()
    rectangles = shirt_identifiers(r, (640, 480))
    assert len(rectangles) == 1 and rectangles[0][0] < rectangles[0][2]
    r["valid_mask"][:, [11, 12, 23, 24]] = False
    assert shirt_identifiers(r, (640, 480)) == [(0, 0, 640, 480)]


def test_reserved_or_held_sources_cannot_enter_weak_or_localization_tensors():
    for v in ("V1", "V2", "V3", "V5", "V7", "V8", "V9", "V10", "V11", "BoxingWeb"):
        assert not admitted(v, "weak_classification") and not admitted(v, "existing_localization")
        with pytest.raises(ValueError):
            pixels(Image.new("RGB", (20, 20)), (0, 0, 20, 20), v)


def test_curation_removal_is_not_restored_by_raw_policy_eligibility():
    clips = [
        dict(recording_id="V4", source_row=i, annotation=dict(exclusion_reasons=[])) for i in (1, 2)
    ]
    assert [c["source_row"] for c in safe_weak_rows(clips, [2])] == [2]
    with pytest.raises(ValueError):
        safe_weak_rows(clips, [3])
    clips[1]["annotation"]["exclusion_reasons"] = ["hold"]
    with pytest.raises(ValueError):
        safe_weak_rows(clips, [2])


def test_weak_overlay_excludes_manual_overlap_and_known_label_conflicts_without_relabeling():
    from boxing_motion_ai.data.video_foundation_validation import eligibility

    clips = [
        dict(id=f"V6:{i}", recording="V6", source_frames=span, clip_class="jab")
        for i, span in enumerate(([10, 20], [30, 40], [50, 60]))
    ]
    snapshot = dict(
        intervals=[
            dict(
                provenance=dict(recording="V6"),
                excluded_events=[dict(start=19, stop=22)],
                events=[dict(start=30, stop=40, punch_type="cross")],
            )
        ]
    )
    result = eligibility(clips, snapshot)
    assert [c["weak_training_eligible"] for c in result] == [False, False, True]
    assert all(c["clip_class"] == "jab" for c in clips)
    assert not any(c["localization_loss"] or c["background_loss"] for c in result)


def test_encoder_missing_or_altered_pin_never_downloads(tmp_path):
    p = tmp_path / "missing.pth"
    with pytest.raises(ValueError, match="missing/changed"):
        encoder(dict(path=str(p), sha256="0" * 64), "cpu")
    p.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="missing/changed"):
        encoder(dict(path=str(p), sha256="0" * 64), "cpu")


def test_original_weak_foundation_roles_and_truth_levels_if_private_artifacts_exist():
    import json

    p = Path("reports/boxingvi/video-foundation-dataset-v1-1/manifest.json")
    if not p.exists():
        pytest.skip("private feature foundation not generated")
    d = json.loads(p.read_text())
    assert (
        len(d["weak_clips"]) == 815 and d["new_recordings"] == 0 and not d["localization_changed"]
    )
    assert set(d["recordings"]) == {"V4", "V6"}
    for c in d["weak_clips"]:
        assert not c["localization_loss"] and not c["background_loss"]
        assert not c["class_verified"] and not c["instance_boundaries_verified"]


def test_frozen_encoder_is_eval_only_when_local_weights_exist():
    from boxing_motion_ai.vision.observation_study import read

    cfg = read("configs/video_foundation_v1.json")["pretrained_encoder"]
    if not Path(cfg["path"]).exists():
        pytest.skip("optional official video checkpoint unavailable")
    model, _ = encoder(cfg, "cpu")
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert isinstance(model.fc, torch.nn.Identity)


def test_private_foundation_validation_rejects_a_changed_native_join_without_touching_originals(
    tmp_path,
):
    import json
    import shutil

    from boxing_motion_ai.data.video_foundation_validation import verify

    dataset = Path("reports/boxingvi/video-foundation-dataset-v1-1")
    video = Path("reports/boxingvi/video-foundation-features-v1-2")
    if not (dataset / "manifest.json").exists() or not (video / "manifest.json").exists():
        pytest.skip("private final cache unavailable")
    copy = tmp_path / "dataset"
    shutil.copytree(dataset, copy)
    p = copy / "manifest.json"
    d = json.loads(p.read_text())
    c = d["weak_clips"][0]
    c["feature_rows"][0], c["feature_rows"][1] = c["feature_rows"][1], c["feature_rows"][0]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="frame_join_changed"):
        verify(copy, video, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_snapshot_pin_keeps_provisional_evidence_separate_from_new_instance_truth():
    from boxing_motion_ai.analysis.common import sha
    from boxing_motion_ai.vision.observation_study import read

    p = Path("configs/video_foundation_snapshot_v1.json")
    if not p.exists():
        pytest.skip("snapshot not pinned yet")
    cfg = read(p)
    assert cfg["status"] == "blocked_for_expanded_training_experiment"
    assert cfg["new_verified_instance_events"] == cfg["new_recordings"] == 0
    assert cfg["eligible_weak_clips"] < cfg["weak_inventory"] and not cfg["models_trained"]
    for key in ("dataset", "eligibility", "pretrained_features"):
        path = Path(cfg[key]["path"])
        assert path.parts[:2] == ("reports", "boxingvi") and "V9" not in path.parts
        if path.exists():
            assert sha(path) == cfg[key]["sha256"]
