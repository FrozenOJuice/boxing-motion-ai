"""Policy development uses synthetic evidence only. No V5 fixture or score access."""

from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import pytest

from boxing_motion_ai.detection.localization_decode import decode as v1
from boxing_motion_ai.detection.localization_decode_v2 import Policy, decode
from boxing_motion_ai.vision.observation_study import read

POLICY = read(Path(__file__).parents[1] / "configs/localization_decoder_v2.json")


def fixture(n=100, rate=25):
    video = dict(
        frames=[
            dict(frame_index=i, timestamp_exact=str(Q(i, rate)), media_time=float(Q(i, rate)))
            for i in range(n)
        ],
        stop_timestamp_exact=str(Q(n, rate)),
        duration_seconds=float(Q(n, rate)),
    )
    scores = np.zeros((n, 3), np.float32)
    scores[:, 0] = 0.9
    obs = np.ones(n, bool)
    reset = np.zeros(n, bool)
    reset[0] = True
    return scores, obs, reset, video


def spans(result):
    return [(r["start_frame"], r["stop_frame_exclusive"]) for r in result["candidates"]]


def test_isolated_and_continuously_active_consecutive_cycles_use_only_real_peaks():
    p, o, r, v = fixture()
    p[[10, 30], 1] = 0.9
    p[[20, 40], 2] = 0.9
    result = decode(p, o, r, v, POLICY)
    assert spans(result) == [(10, 21), (30, 41)]
    assert result["boundary_peaks"] == v1(p, o, r, v, POLICY)["boundary_peaks"]
    assert result["candidates"][0]["start_timestamp_exact"] == "2/5"
    assert result["candidates"][0]["stop_timestamp_exact"] == "21/25"
    assert decode(p, o, r, v, POLICY) == result


def test_stale_onset_expires_at_existing_maximum_span_without_fabricating_an_end():
    p, o, r, v = fixture(150)
    p[[5, 90], 1] = 0.9
    p[100, 2] = 0.9
    assert spans(v1(p, o, r, v, POLICY)) == []
    result = decode(p, o, r, v, POLICY)
    assert spans(result) == [(90, 101)]
    assert any(x["reason"] == "onset_expired_at_maximum_span" for x in result["unresolved"])


def test_quiet_reset_prevents_pending_onset_from_poisoning_a_later_cycle():
    p, o, r, v = fixture()
    p[[10, 50], 1] = 0.9
    p[60, 2] = 0.9
    p[25:40, 0] = 0.1
    assert spans(v1(p, o, r, v, POLICY)) == []
    result = decode(p, o, r, v, POLICY)
    assert spans(result) == [(50, 61)]
    assert result["predicted_inactivity_resets"][0]["reason"].endswith("not_verified_background")
    assert any(
        x["reason"] == "onset_interrupted_by_predicted_inactivity" for x in result["unresolved"]
    )


def test_short_invalid_first_end_does_not_consume_a_usable_later_end():
    p, o, r, v = fixture()
    p[10, 1] = 0.9
    p[[11, 20], 2] = 0.9
    assert spans(v1(p, o, r, v, POLICY)) == []
    assert spans(decode(p, o, r, v, POLICY)) == [(10, 21)]


def test_incompatible_late_onset_does_not_destroy_a_supported_earlier_pair():
    p, o, r, v = fixture()
    p[[10, 29], 1] = 0.9
    p[30, 2] = 0.9
    assert spans(v1(p, o, r, v, POLICY)) == []
    assert spans(decode(p, o, r, v, POLICY)) == [(10, 31)]


def test_competing_onsets_and_competing_ends_remain_unresolved():
    for starts, ends in [([10, 15], [25]), ([10], [20, 25]), ([10, 15], [25, 30])]:
        p, o, r, v = fixture()
        p[starts, 1] = 0.9
        p[ends, 2] = 0.9
        result = decode(p, o, r, v, POLICY)
        assert not result["candidates"]
        assert any(x["reason"] == "competing_boundary_assignments" for x in result["unresolved"])


def test_coincident_boundaries_are_not_given_an_invented_order():
    p, o, r, v = fixture()
    p[20, 1:] = 0.9
    result = decode(p, o, r, v, POLICY)
    assert not result["candidates"]
    assert any(x["reason"] == "coincident_onset_end_ambiguous" for x in result["unresolved"])


def test_missing_pose_resets_and_censored_boundaries_never_bridge_evidence():
    for missing in [True, False]:
        p, o, r, v = fixture()
        p[10, 1] = 0.9
        p[30, 2] = 0.9
        if missing:
            o[20] = False
        else:
            r[20] = True
        result = decode(p, o, r, v, POLICY)
        assert not result["candidates"]
        assert bool(result["unobservable"]) == missing
    p, o, r, v = fixture()
    p[0, 1] = 0.9
    p[10, 2] = 0.9
    assert not decode(p, o, r, v, POLICY)["candidates"]


def test_no_boundary_evidence_and_weak_activity_do_not_generate_instances():
    p, o, r, v = fixture()
    assert not decode(p, o, r, v, POLICY)["candidates"]
    p[10, 1] = 0.9
    p[20, 2] = 0.9
    p[:, 0] = 0.1
    result = decode(p, o, r, v, POLICY)
    assert not result["candidates"]
    assert result["possible_activity"] == []


def test_different_native_rate_and_source_origin_keep_exact_timing():
    p, o, r, v = fixture(rate=30)
    p[10, 1] = 0.9
    p[20, 2] = 0.9
    for f in v["frames"]:
        f["timestamp_exact"] = str(Q(f["timestamp_exact"]) + Q(1001, 7))
    v["stop_timestamp_exact"] = str(Q(v["stop_timestamp_exact"]) + Q(1001, 7))
    result = decode(p, o, r, v, POLICY)
    assert spans(result) == [(10, 21)]
    assert Q(result["candidates"][0]["start_timestamp_exact"]) == Q(1001, 7) + Q(1, 3)
    assert Q(result["candidates"][0]["stop_timestamp_exact"]) == Q(1001, 7) + Q(7, 10)


def test_policy_and_invalid_clock_fail_closed():
    with pytest.raises(ValueError):
        Policy.model_validate(POLICY | dict(boundary_threshold=0.4))
    p, o, r, v = fixture()
    v["stop_timestamp_exact"] = "0"
    with pytest.raises(ValueError):
        decode(p, o, r, v, POLICY)


def test_unique_but_overlapping_pairs_remain_unresolved_without_arm_evidence():
    p, o, r, v = fixture()
    p[[10, 29], 1] = 0.9
    p[[30, 78], 2] = 0.9
    result = decode(p, o, r, v, POLICY)
    assert not result["candidates"]
    assert any(
        x["reason"] == "overlapping_pairs_without_arm_assignment" for x in result["unresolved"]
    )


def test_coincident_boundaries_cannot_be_ignored_to_create_a_long_pair():
    p, o, r, v = fixture()
    p[[10, 20], 1] = 0.9
    p[[20, 30], 2] = 0.9
    assert not decode(p, o, r, v, POLICY)["candidates"]


def test_development_replay_requires_a_frozen_training_binding_before_loading_cases(
    tmp_path, monkeypatch
):
    import boxing_motion_ai.detection.localization_decoder_study as study

    called = []
    monkeypatch.setattr(study, "cases", lambda phase: called.append(phase))
    with pytest.raises(ValueError):
        study.run("development", tmp_path / "result")
    assert called == [] and not (tmp_path / "result").exists()
    freeze = tmp_path / "freeze.json"
    freeze.write_text('{"code_and_policy": {}}')
    with pytest.raises(ValueError):
        study.run("development", tmp_path / "result", freeze)
    assert called == []
