"""Diagnostic-only pose transforms: no weights, inference or landmark recovery claims."""

from fractions import Fraction as Q

import numpy as np
import pytest

from boxing_motion_ai.vision.observation_study import (
    CASES,
    VARIANTS,
    coverage,
    fresh,
    map_landmarks,
    mask_matched_rtmpose,
    reset_before,
)


def test_explicit_inventory_excludes_held_and_evaluation_sources():
    assert len(CASES) == len(set(CASES)) == 19
    assert all(c.startswith(("V4-", "V6-")) or c == "no-person" for c in CASES)
    assert "V4-bg-4" not in CASES and "V4-bg-5" not in CASES
    assert len(VARIANTS) == 5


def test_crop_inverse_mapping_preserves_anatomy_scores_and_depth_scale():
    points = [(0.25, 0.75, -0.1, 0.9, 0.8)] * 33
    mapped = map_landmarks(points, (100, 200, 500, 800), (1000, 1000))
    np.testing.assert_allclose(mapped[15], (0.2, 0.65, -0.04, 0.9, 0.8))
    assert points[15] == (0.25, 0.75, -0.1, 0.9, 0.8)
    assert map_landmarks(None, (0, 0, 10, 10), (100, 100)) is None


def test_outside_crop_cannot_become_valid_by_mapping_inside_source():
    p = map_landmarks([(1.1, 0.2, 0.0, 0.9, 0.9)] * 33, (100, 100, 500, 500), (1000, 1000))
    assert p[0][0] == 0.54  # Unclipped estimate, not a recovered joint.
    assert p[0][-2:] == (0.0, 0.0)


def test_tracking_reset_uses_source_pts_masks_cuts_and_segments():
    raw = dict(scene_id=np.array([0, 0, 0, 1, 1, 1]), segment_id=np.array([0, 0, 0, 0, 0, 1]))
    video = dict(
        frames=[
            dict(timestamp_exact=str(t))
            for t in (Q(0), Q(1, 25), Q(1, 5), Q(6, 25), Q(7, 25), Q(8, 25))
        ],
        confirmed_scene_cuts=[4],
    )
    activity = dict(
        effective_reset_gap_seconds_exact="1/10", frames=[dict(discontinuity_before=False)] * 6
    )
    assert reset_before(raw, video, activity) == [True, False, True, True, True, True]
    activity["frames"] = [dict(discontinuity_before=True)] * 6
    assert all(reset_before(raw, video, activity))


def test_rtmpose_coordinate_sensitivity_never_creates_valid_joint_or_scores():
    raw = dict(
        coordinates=np.ones((2, 33, 3)) * 0.5,
        confidence=np.ones((2, 33, 2)) * 0.9,
        valid_mask=np.ones((2, 33), bool),
    )
    raw["valid_mask"][0, 16] = False
    xy = np.ones((2, 8, 2)) * 25
    xy[1, 4] = [200, 25]  # Outside source: retain original, do not clip.
    result = mask_matched_rtmpose(raw, xy, (100, 100))
    np.testing.assert_array_equal(result["valid_mask"], raw["valid_mask"])
    np.testing.assert_array_equal(result["confidence"], raw["confidence"])
    np.testing.assert_array_equal(raw["coordinates"], np.ones((2, 33, 3)) * 0.5)
    np.testing.assert_allclose(result["coordinates"][0, 15, :2], 0.25)
    np.testing.assert_allclose(result["coordinates"][0, 16, :2], 0.5)
    np.testing.assert_allclose(result["coordinates"][1, 15, :2], 0.5)


def test_coverage_breaks_usable_runs_at_missing_observations_and_resets():
    data = dict(
        times=list(map(Q, ("0", ".04", ".08", ".12", ".16"))),
        stop_time=Q(".20"),
        reset=np.array([True, False, True, False, False]),
        arms={
            "left": dict(
                wrist_valid=np.array([True, True, True, False, True]), elbow_valid=np.ones(5, bool)
            )
        },
    )
    r = coverage(data)["left"]
    assert r["wrist"]["frames"] == 4
    assert r["wrist"]["longest_seconds"] == 0.08
    assert r["complete"]["fraction"] == 0.8
    split = coverage(data, {"left": {"tracking_jump_resets": [1]}})["left"]
    assert split["wrist"]["frames"] == 4
    assert split["wrist"]["longest_seconds"] == 0.04


def test_fresh_refuses_existing_or_outside_report_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "reports/existing"
    existing.mkdir(parents=True)
    with pytest.raises(ValueError):
        fresh(tmp_path / "outside")
    with pytest.raises(ValueError):
        fresh(existing)


def test_study_manifest_binds_source_inventory_and_frozen_policy(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from boxing_motion_ai.analysis.common import sha
    from boxing_motion_ai.detection import instances_v2
    from boxing_motion_ai.vision.observation_study import VERSION, study_manifest

    policy = Path("configs/punch_instances_v2.json").read_bytes()
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    Path("configs/punch_instances_v2.json").write_bytes(policy)
    out = Path("reports/study")
    out.mkdir(parents=True)
    value = dict(
        schema_version=VERSION,
        cases=dict.fromkeys(CASES, "unused"),
        variants=VARIANTS,
        segmentation_code_sha256=sha(Path(instances_v2.__file__)),
        segmentation_policy_sha256=sha(Path("configs/punch_instances_v2.json")),
    )
    path = out / "manifest.json"
    path.write_text(json.dumps(value))
    assert study_manifest(out)["schema_version"] == VERSION
    value["cases"]["V4-bg-4"] = "forbidden"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="inventory"):
        study_manifest(out)
    del value["cases"]["V4-bg-4"]
    value["segmentation_policy_sha256"] = "changed"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="segmentation"):
        study_manifest(out)
