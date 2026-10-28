"""Timestamp-based source proposals are not reference labels or detector predictions."""

from fractions import Fraction

import numpy as np
import pytest

from boxing_motion_ai.services.activity_suggestions import propose


def test_exact_quarter_second_gap_overlaps_and_censoring_preserve_source():
    arrays = dict(pts=np.arange(40), time_base_num=np.ones(40, int), time_base_den=np.full(40, 100))
    entry = dict(id="V10-eval-1", recording="V10", start=2, stop=35, source_inputs={"x": "hash"})
    rows = [
        dict(
            source_row=i,
            source_interval_half_open=b,
            raw_start=b[0],
            raw_end=b[1] - 1,
            annotation_end_assumption="inclusive; unverified",
        )
        for i, b in enumerate(([0, 4], [3, 5], [29, 31], [30, 39]), 1)
    ]
    result = propose(entry, rows, arrays)
    assert [(s["start"], s["stop"]) for s in result["suggestions"]] == [(2, 35)]
    # First group ends at .05; .29-.05=.24 merges, including chained overlaps.
    assert result["suggestions"][0]["members"][0]["source_annotation"] == rows[0]
    assert result["suggestions"][0]["timing"] == {"start": "1/50", "stop_exclusive": "7/20"}
    assert not result["suggestions"][0]["verified"]
    assert result["suggestions"][0]["source"] == "automated"
    assert propose(entry, rows, arrays) == result
    rows[2]["source_interval_half_open"] = [30, 31]
    assert len(propose(entry, rows, arrays)["suggestions"]) == 2  # Exactly .25 stays separate.


@pytest.mark.parametrize("fps", [25, Fraction(30000, 1001), 30])
def test_actual_pts_drive_merging_not_nominal_fps(fps):
    base = Fraction(1, 1) / fps
    arrays = dict(
        pts=np.arange(60) * base.numerator,
        time_base_num=np.ones(60, int),
        time_base_den=np.full(60, base.denominator),
    )
    entry = dict(id="V7-eval-1", recording="V7", start=0, stop=59, source_inputs={})
    rows = [
        dict(source_row=1, source_interval_half_open=[1, 3]),
        dict(source_row=2, source_interval_half_open=[10, 12]),
    ]
    result = propose(entry, rows, arrays)
    assert len(result["suggestions"]) == (2 if fps == 25 else 1)
    assert result["gaps"] == "unknown_until_complete_human_review"
    assert result["detector_predictions_used"] is False


def test_invalid_rows_are_reported_not_repaired_and_empty_is_unknown():
    arrays = dict(pts=np.arange(10), time_base_num=np.ones(10, int), time_base_den=np.full(10, 25))
    entry = dict(id="V7-eval-1", recording="V7", start=1, stop=5, source_inputs={})
    rows = [
        dict(source_row=1, source_interval_half_open=[4, 2]),
        dict(source_row=2, source_interval_half_open=[True, 3]),
        dict(source_row=3, source_interval_half_open=None),
        dict(source_row=4, source_interval_half_open=[6, 8]),
    ]
    result = propose(entry, rows, arrays)
    assert not result["suggestions"] and len(result["issues"]) == 3
    assert "unknown" in result["gaps"]
