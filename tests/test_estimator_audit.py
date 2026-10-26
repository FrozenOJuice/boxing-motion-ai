"""Synthetic exact-frame and common-joint comparison checks; no estimator downloads."""

from fractions import Fraction

import numpy as np
import pytest

from boxing_motion_ai.vision.estimator_audit import (
    COCO,
    MP,
    bounded_indices,
    case_box,
    common_mp,
    motion_stats,
)


def arrays():
    return dict(
        frame_index=np.arange(10),
        scene_id=np.zeros(10, int),
        segment_id=np.zeros(10, int),
        coordinates=np.ones((10, 33, 3)) * 0.5,
        confidence=np.ones((10, 33, 2)),
        valid_mask=np.ones((10, 33), bool),
    )


def test_common_mapping_preserves_masks_and_scores():
    a = arrays()
    a["coordinates"][4, 13, :2] = [0.2, 0.3]
    a["valid_mask"][4, 13] = False
    xy, scores, mask = common_mp(a, [4], (100, 200))
    np.testing.assert_allclose(xy[0, 2], [20, 60])
    assert not mask[0, 2]
    assert MP == [11, 12, 13, 14, 15, 16, 23, 24]
    assert COCO == [5, 6, 7, 8, 9, 10, 11, 12]
    assert scores.shape == (1, 8, 2)
    assert not a["valid_mask"][4, 13]


def test_context_stops_at_cut_or_segment_boundary():
    a = arrays()
    a["scene_id"][:3] = 1
    a["segment_id"][7:] = 1
    assert bounded_indices(a, 4, 5) == [3, 4, 5, 6]
    with pytest.raises(ValueError, match="discontinuity"):
        bounded_indices(a, 2, 4)


def test_missing_frames_reversed_and_long_intervals_rejected():
    a = arrays()
    a["frame_index"][5:] += 1
    with pytest.raises(ValueError, match="Missing"):
        bounded_indices(a, 4, 6)
    for start, end, max_n in [(6, 4, 100), (0, 10, 100), (0, 9, 2)]:
        with pytest.raises(ValueError):
            bounded_indices(arrays(), start, end, maximum=max_n)


def test_motion_uses_exact_time_and_never_crosses_invalid_or_gap():
    xy = np.zeros((4, 1, 2))
    xy[:, 0, 0] = [0, 1, 3, 100]
    valid = np.ones((4, 1), bool)
    idx = np.array([0, 1, 2, 4])
    zeros = np.zeros(4, int)
    times = list(map(Fraction, ["0", ".04", ".12", ".16"]))
    r = motion_stats(xy, valid, idx, times, zeros, zeros, 1)
    assert r["pairs"] == 2
    assert r["speed_p50"] == 25
    valid[1] = False
    assert motion_stats(xy, valid, idx, times, zeros, zeros, 1)["pairs"] == 0
    valid[:] = True
    scene = np.array([0, 1, 2, 3])
    assert motion_stats(xy, valid, idx, times, scene, zeros, 1)["pairs"] == 0


def test_motion_rejects_nonpositive_or_large_time_delta():
    xy = np.zeros((3, 1, 2))
    valid = np.ones((3, 1), bool)
    ids = np.arange(3)
    zero = np.zeros(3, int)
    r = motion_stats(xy, valid, ids, list(map(Fraction, ["0", "0", "1"])), zero, zero, 1)
    assert r["pairs"] == 0 and r["speed_p95"] is None


def test_crop_uses_valid_points_only_and_falls_back_without_them():
    a = arrays()
    a["valid_mask"][:] = False
    assert case_box(a, [0], (100, 200)) == [0.0, 0.0, 100.0, 200.0]
    a["valid_mask"][0, :10] = True
    a["coordinates"][0, :5, :2] = [0.2, 0.2]
    a["coordinates"][0, 5:10, :2] = [0.8, 0.8]
    a["coordinates"][0, 10, :2] = [100, 100]
    box = case_box(a, [0], (100, 200))
    assert 0 <= box[0] < box[2] <= 100 and 0 <= box[1] < box[3] <= 200
