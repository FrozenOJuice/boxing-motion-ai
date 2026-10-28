"""Synthetic discovery scores never become action labels or human confirmations."""

import numpy as np

from boxing_motion_ai.data.motion_candidates import motion_metrics, select_diverse


def arrays(n=150):
    xyz = np.zeros((n, 33, 3)) + 0.5
    xyz[:, [11, 12], 1] = 0.3
    xyz[:, [23, 24], 1] = 0.7
    xyz[:, [27, 28], 1] = 0.9
    return dict(
        coordinates=xyz,
        valid_mask=np.ones((n, 33), bool),
        pts=np.arange(n) * 40,
        time_base_num=np.ones(n, int),
        time_base_den=np.full(n, 1000),
    )


def test_stationary_valid_pose_is_only_a_motion_measurement():
    a = arrays()
    r = motion_metrics(a, 0, 150, 16 / 9)
    assert r["arm_p90"] == 0 and r["feet_p50"] == 0 and r["anchor_coverage"] == 1
    assert not {"label", "no_punch", "decision", "verified"}.intersection(r)
    assert a["valid_mask"].all()


def test_elapsed_time_changes_speed_without_imputing_invalid_joints():
    a = arrays()
    a["coordinates"][:, 15, 0] += np.arange(150) * 0.001
    original = a["coordinates"].copy()
    fast = motion_metrics(a, 0, 150, 1)
    a["pts"] *= 2
    slow = motion_metrics(a, 0, 150, 1)
    assert abs(fast["arm_p90"] - 2 * slow["arm_p90"]) < 1e-8
    np.testing.assert_equal(a["coordinates"], original)
    a["valid_mask"][10:100, 15] = False
    assert motion_metrics(a, 0, 150, 1) is None


def test_unreliable_feet_anchors_gaps_and_tracking_spikes_are_rejected():
    for joint in [11, 15, 27]:
        a = arrays()
        a["valid_mask"][30:80, joint] = False
        assert motion_metrics(a, 0, 150, 1) is None
    a = arrays()
    a["pts"][50:] += 1000
    assert motion_metrics(a, 0, 150, 1) is None
    a = arrays()
    a["coordinates"][50, 15, 0] = 100
    assert motion_metrics(a, 0, 150, 1) is None


def test_activity_strata_are_deterministic_separated_and_label_free():
    windows = []
    for i in range(30):
        windows.append(
            dict(
                start=i * 1000,
                start_seconds=i * 40,
                metrics=dict(
                    arm_p90=0.1 + i * 0.07,
                    arm_p50=0.05 + i * 0.01,
                    feet_p50=0.1 + (i % 4) * 0.2,
                    feet_coverage=1.0,
                    hip_p50=0.05 + (i % 3) * 0.2,
                ),
            )
        )
    first = select_diverse(windows, 1200)
    assert first == select_diverse(windows, 1200) and len(first) == 6
    assert [x["stratum"] for x in first] == [
        "quiet_motion",
        "quiet_motion",
        "lower_body_motion",
        "lower_body_motion",
        "moderate_arm_motion",
        "mixed_motion",
    ]
    assert all(
        abs(x["start_seconds"] - y["start_seconds"]) >= 30
        for i, x in enumerate(first)
        for y in first[i + 1 :]
    )
    assert all("decision" not in w and "label" not in w for w in first)
