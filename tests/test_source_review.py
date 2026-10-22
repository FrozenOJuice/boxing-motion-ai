"""Synthetic review diagnostics never restore source labels or touch reserved inputs."""

from fractions import Fraction

import pytest

from boxing_motion_ai.data.source_review import (
    GROUPS,
    SOURCES,
    hypotheses,
    overlap_relations,
    planning_scenario,
    write_player,
)


def row(n, a, b, label="Rear Uppercut"):
    return {"source_row": n, "raw_start": a, "raw_end": b, "source_class": label}


def test_overlap_geometry_does_not_classify_motion_or_restore_labels():
    target = row(5, 20, 30)
    rows = [target, row(4, 10, 20), row(6, 25, 35), row(7, 22, 24), row(8, 20, 30)]
    result = overlap_relations(target, rows)
    assert [r["kind"] for r in result] == [
        "shared_endpoint",
        "partial_overlap",
        "nested_interval",
        "identical_interval",
    ]
    assert not any(r["training_restored"] for r in result)
    assert all("action_verified" not in r for r in result)
    assert target == row(5, 20, 30)


def test_reversed_intervals_do_not_create_overlap_approvals():
    assert overlap_relations(row(2, 30, 20), [row(3, 15, 35)]) == []
    assert overlap_relations(row(2, 20, 30), [row(3, 35, 15)]) == []


def test_v6_helper_is_separate_hypothesis_not_applied_to_raw_mapping():
    times = [Fraction(i * 1001, 24000) for i in range(600)]
    source = row(2, 200, 210)
    h = hypotheses(source, times, "V6")
    assert h["raw"] == [200, 210]
    assert h["helper_minus_157"] == [43, 53]
    assert h["clock_24"] == [200, 210]
    assert h["clock_30"] != h["raw"]
    assert source == row(2, 200, 210)


def test_reversed_source_bounds_remain_diagnostic_only():
    times = [Fraction(i, 25) for i in range(100)]
    assert hypotheses(row(2, 20, 10), times, "V10") == {"reversed_bounds_diagnostic_only": [10, 20]}


def test_capacity_estimate_is_explicit_assumption_and_zero_stays_zero():
    result = planning_scenario(32)
    assert result["capacity_scenario"] == 25
    assert result["structural_ceiling"] == 32
    assert "including zero" in result["note"]
    assert planning_scenario(0)["capacity_scenario"] == 0
    with pytest.raises(ValueError):
        planning_scenario(-1)


def test_review_scope_and_source_groups_preserve_reserved_split():
    assert set(SOURCES) == {"V7", "V6", "V10"}
    assert "V5" not in SOURCES and "V9" not in SOURCES
    assert GROUPS["V7"] == GROUPS["V8"]
    assert GROUPS["V6"] != GROUPS["V10"] != GROUPS["V7"]


def test_player_uses_portable_paths_and_exact_source_pts(tmp_path):
    case = {
        "id": "synthetic",
        "source_class": "Rear Hook",
        "hypothesis": "raw",
        "indices": [0, 1],
        "training_restored": False,
    }
    write_player([case], [Fraction(0), Fraction(1001, 24000)], tmp_path)
    text = (tmp_path / "review.html").read_text()
    assert "1001/24000" in text
    assert "frames/${f.i}.jpg" in text
    assert "No exclusions restored" in text
    assert "https://" not in text and "PAYLOAD" not in text
