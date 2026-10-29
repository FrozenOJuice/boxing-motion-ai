"""V2 phase proposals: real permitted pose traces + synthetic failure boundaries."""

import json
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import pytest
from test_automatic_punches import sequence
from test_uploaded_activity import prepared

np = pytest.importorskip("numpy")
from boxing_motion_ai.analysis.automatic import classify_candidates  # noqa: E402
from boxing_motion_ai.detection import instances  # noqa: E402
from boxing_motion_ai.detection import instances_v2 as v2  # noqa: E402
from boxing_motion_ai.detection.instance_diagnostics import compare  # noqa: E402

ROOT = Path(__file__).parents[1]
POLICY = v2.Policy.model_validate(
    json.loads((ROOT / "configs/punch_instances_v2.json").read_text())
)
V1 = instances.Policy.model_validate(
    json.loads((ROOT / "configs/punch_instances_v1.json").read_text())
)


def test_isolated_and_consecutive_cycles_do_not_require_one_punch_per_activity_bout():
    for cycles, count in [
        ((("left", 10, 30),), 1),
        ((("left", 10, 30), ("right", 40, 60)), 2),
        ((("left", 10, 30), ("left", 40, 60)), 2),
    ]:
        raw, video, activity = sequence(cycles)
        a = v2.segment(raw, video, activity, POLICY)
        assert len(a["candidates"]) == count
        assert a == v2.segment(raw, video, activity, POLICY)
        for p in a["candidates"]:
            assert p["start_frame"] < p["peak_frame"] < p["stop_frame_exclusive"]
            assert p["evidence_through_frame"] >= p["stop_frame_exclusive"]
            assert p["classification_eligible"] and "first_emission" not in p
        assert "offline" in a["processing_mode"]


def test_changed_guard_position_is_not_a_required_return_coordinate():
    raw, video, activity = sequence()
    drift = np.clip((np.arange(100) - 10) / 20, 0, 1) * 0.3 * 0.3 / (4 / 3)
    raw["coordinates"][:, 15, 0] += drift
    assert not instances.segment(raw, video, activity, V1)["candidates"]
    assert len(v2.segment(raw, video, activity, POLICY)["candidates"]) == 1


def test_missing_elbow_preserves_wrist_evidence_but_blocks_unsupported_classification():
    raw, video, activity = sequence()
    raw["coordinates"][10:31, 13] = np.nan
    raw["valid_mask"][10:31, 13] = False
    result = v2.segment(raw, video, activity, POLICY)
    assert len(result["candidates"]) == 1
    assert not result["candidates"][0]["classification_eligible"]

    class Never:
        def classify(self, *args):
            raise AssertionError("unsupported primary-arm classification")

    row = classify_candidates(result, Never(), raw, video)["candidates"][0]
    assert (
        row["classification"]["status"] == "unavailable" and "scores" not in row["classification"]
    )
    assert result["unresolved_activity"]


@pytest.mark.parametrize(
    "kind", ["wrist", "torso", "gap", "scene", "segment", "cut", "reset", "jump"]
)
def test_no_smoothing_or_candidate_crosses_missing_observations_or_discontinuities(kind):
    raw, video, activity = sequence()
    if kind in ["wrist", "torso"]:
        j = 15 if kind == "wrist" else 11
        raw["coordinates"][20, j] = np.nan
        raw["valid_mask"][20, j] = False
    elif kind == "gap":
        raw["pts"][20:] += 1000
        video = prepared(raw)
        for i, row in enumerate(video["frames"]):
            activity["frames"][i]["timestamp_exact"] = row["timestamp_exact"]
    elif kind in ["scene", "segment"]:
        raw[kind + "_id"][20:] = 1
    elif kind == "cut":
        video["confirmed_scene_cuts"] = [20]
    elif kind == "jump":
        raw["coordinates"][20, 15, 0] = 0.99
    else:
        activity["frames"][20]["discontinuity_before"] = True
    result = v2.segment(raw, video, activity, POLICY)
    assert not any(p["start_frame"] < 20 < p["stop_frame_exclusive"] for p in result["candidates"])
    if kind == "jump":
        assert 20 in result["arm_motion"]["left"]["tracking_jump_resets"]


def test_overlapping_arm_phases_remain_unresolved_and_are_not_arbitrarily_trimmed():
    raw, video, activity = sequence((("left", 10, 40), ("right", 20, 50)))
    result = v2.segment(raw, video, activity, POLICY)
    assert len(result["candidates"]) == 2
    assert all(not p["classification_eligible"] for p in result["candidates"])
    assert result["candidates"][0]["stop_frame_exclusive"] > result["candidates"][1]["start_frame"]
    assert result["unresolved_activity"]


def test_activity_overlap_can_support_recovery_after_the_motion_peak():
    raw, video, activity = sequence()
    activity["bouts"] = [dict(bout_id="late-activity", start_frame=21, stop_frame_exclusive=40)]
    result = v2.segment(raw, video, activity, POLICY)
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["peak_frame"] < 21
    activity["bouts"] = [dict(bout_id="one-frame", start_frame=20, stop_frame_exclusive=21)]
    assert not v2.segment(raw, video, activity, POLICY)["candidates"]


def test_stationary_jitter_and_rigid_body_translation_are_not_instance_evidence():
    raw, video, activity = sequence(())
    raw["coordinates"][:, :, 0] += np.sin(np.arange(100) / 5)[:, None] * 0.12
    raw["coordinates"][:, 15, 0] += np.sin(np.arange(100)) * 0.002
    assert not v2.segment(raw, video, activity, POLICY)["candidates"]


def fixture_motion(case):
    arms = {}
    for arm, t in case["arms"].items():
        arms[arm] = {}
        for joint in ["wrist", "elbow"]:
            arms[arm][joint] = np.array(
                [point or [0.0, 0.0] for point in t[joint]], dtype=np.float32
            )
            arms[arm][joint + "_valid"] = np.array([point is not None for point in t[joint]])
    reset = np.zeros(len(case["times_exact"]), bool)
    reset[case["resets"]] = True
    return dict(
        times=[Q(t) for t in case["times_exact"]],
        stop_time=Q(case["stop_time_exact"]),
        reset=reset,
        arms=arms,
        bouts=case["bouts"],
    )


def test_permitted_real_motion_fixtures_preserve_v2_bounds_and_do_not_read_reference_labels():
    f = json.loads((ROOT / "tests/fixtures/punch_motion_v2.json").read_text())
    from boxing_motion_ai.analysis.common import sha

    assert f["policy_sha256"] == sha(ROOT / "configs/punch_instances_v2.json")
    for case in f["cases"]:
        data = fixture_motion(case)
        result = v2.segment_motion(data, POLICY)
        actual = [
            {k: p[k] for k in ["arm", "start_frame", "peak_frame", "stop_frame_exclusive"]}
            for p in result["candidates"]
        ]
        assert actual == case["expected_candidates"], case["name"]
        changed = deepcopy(case)
        changed["references_for_comparison_only"] = [
            {"type": "invented", "start_frame": 0, "stop_frame_exclusive": 9999}
        ]
        assert result == v2.segment_motion(fixture_motion(changed), POLICY)
        if case["name"] == "V6-motion-6":
            assert not result["candidates"]
            assert any(
                p["reason"] == "no_sustained_activity_overlap" for p in result["rejected_proposals"]
            )


def test_real_review_comparison_reports_misses_and_background_false_proposal():
    f = json.loads((ROOT / "tests/fixtures/punch_motion_v2.json").read_text())
    total_refs = total_matches = 0
    for case in f["cases"]:
        if not case["name"].startswith("V4-bg"):
            continue
        data = fixture_motion(case)
        result = v2.segment_motion(data, POLICY)
        prov = case["provenance"]
        core = (
            prov["review_start"] - prov["source_start"],
            prov["review_stop"] - prov["source_start"],
        )
        comparison = compare(
            result["candidates"],
            case["references_for_comparison_only"],
            data["times"],
            data["stop_time"],
            core,
        )
        total_refs += comparison["reviewed_reference_count"]
        total_matches += len(comparison["matches_iou_half"])
        if case["name"] == "V4-bg-3":
            assert comparison["candidate_without_reference_overlap"] == [0]
    assert (
        total_refs == 11 and total_matches == 2
    )  # Development diagnostic, not a held-out accuracy claim.


def test_exact_source_clock_is_preserved_with_variable_frame_spacing():
    raw, video, activity = sequence()
    raw["pts"][1::2] += 3
    video = prepared(raw)
    for i, row in enumerate(video["frames"]):
        activity["frames"][i]["timestamp_exact"] = row["timestamp_exact"]
    result = v2.segment(raw, video, activity, POLICY)
    assert result["candidates"]

    class Unavailable:
        def classify(self, *args):
            return dict(status="unavailable", reason="test only")

    p = classify_candidates(result, Unavailable(), raw, video)["candidates"][0]
    assert Q(p["start_timestamp_exact"]) == Q(int(raw["pts"][p["start_frame"]]), 1000)
    assert p["source_start"]["frame_index"] == p["start_frame"]
    bad = deepcopy(activity)
    bad["frames"][20]["timestamp_exact"] = "999"
    with pytest.raises(ValueError, match="timing"):
        v2.segment(raw, video, bad, POLICY)


def test_overlap_comparison_maximizes_cardinality_without_fabricating_emission_times():
    times = [Q(i) for i in range(6)]
    refs = [
        dict(event_id="a", type="jab", start_frame=0, stop_frame_exclusive=3),
        dict(event_id="b", type="cross", start_frame=2, stop_frame_exclusive=5),
    ]
    preds = [
        dict(start_frame=0, stop_frame_exclusive=5, peak_frame=2),
        dict(start_frame=0, stop_frame_exclusive=2, peak_frame=1),
    ]
    report = compare(preds, refs, times, Q(6), (0, 6))
    assert {(p["candidate_index"], p["reference_id"]) for p in report["matches_iou_half"]} == {
        (0, "b"),
        (1, "a"),
    }
    assert not report["missed_references"]
    with pytest.raises(ValueError, match="outside source"):
        compare(
            [dict(start_frame=0, stop_frame_exclusive=7, peak_frame=2)], refs, times, Q(6), (0, 6)
        )


def test_v1_diagnostic_census_counts_independent_failures_without_changing_v1_results():
    from boxing_motion_ai.detection.instance_diagnostics import v1_audit

    raw, video, activity = sequence()
    drift = np.clip((np.arange(100) - 10) / 20, 0, 1) * 0.3 * 0.3 / (4 / 3)
    raw["coordinates"][:, 15, 0] += drift
    before = instances.segment(raw, video, activity, V1)
    report = v1_audit(v2.motion(raw, video, activity), before, V1)
    assert report["activity_gated_pairs"] > 0
    assert report["independent_failed_conditions"]["return_distance"] > 0
    assert before == instances.segment(raw, video, activity, V1)
