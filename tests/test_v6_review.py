"""Synthetic helpers, overlap and planning evidence; no source footage needed."""

from copy import deepcopy

from boxing_motion_ai.data.v6_review import (
    RECOMMENDED_HOLDS,
    conditional_capacity,
    display_intervals,
    helper_audit,
    pair_inventory,
)


def event(n, a, b, label="Rear Hook"):
    return {
        "source_row": n,
        "raw_start": a,
        "raw_end": b,
        "source_class": label,
        "source_cells": {
            "D": f"=B{n}-A{n}",
            "E": f"=A{n}-157",
            "F": f"=B{n}-157",
            "G": None,
            "H": n - 2,
        },
    }


def test_helper_audit_reports_literals_without_inventing_time_or_identity():
    rows = [event(2, 157, 171), event(3, 311, 320)]
    rows[1]["source_cells"].update(E=13, F=22)
    auxiliary = [{"source_row": 4, "source_cells": {"H": 2}}]
    before = deepcopy(rows)
    result = helper_audit(rows, auxiliary)
    assert result["matching_formula_counts"] == {"D": 2, "E": 1, "F": 1}
    assert [(a["source_row"], a["column"], a["value"]) for a in result["anomalies"]] == [
        (3, "E", 13),
        (3, "F", 22),
    ]
    assert not result["H_row_minus_2_mismatches"]
    assert result["auxiliary_only_rows"] == 1
    assert "not evidence" in result["interpretation"]
    assert rows == before


def test_geometry_does_not_restore_rows_or_claim_two_actions():
    rows = [event(2, 10, 20), event(3, 20, 30), event(4, 21, 30), event(5, 27, 35)]
    before = deepcopy(rows)
    result = pair_inventory(rows)
    assert result["geometry_counts"] == {
        "shared_endpoint": 1,
        "nested_interval": 1,
        "partial_overlap": 2,
    }
    assert not any(p["training_restored"] for p in result["pairs"])
    assert rows == before


def test_alternative_sheet_marks_hypothesis_not_unshifted_source_interval():
    row = event(685, 44325, 44338)
    assert display_intervals(row, "raw", [44325, 44338], [row]) == [[685, 44325, 44338]]
    assert display_intervals(row, "clock_24", [44281, 44294], [row]) == [[685, 44281, 44294]]
    assert row["raw_start"] == 44325 and row["raw_end"] == 44338


def test_capacity_excludes_flagged_and_review_held_without_promising_yield():
    inventory = {
        "by_class": {"rear_uppercut": {"source": 4, "structurally_eligible": 3}},
        "rows": [
            {
                "source_row": n,
                "punch_type_provisional": "rear_uppercut",
                "structurally_eligible": n != 4,
            }
            for n in [2, 3, 4, 685]
        ],
    }
    result = conditional_capacity(inventory, [685])
    c = result["by_class"]["rear_uppercut"]
    assert c["review_constrained_ceiling"] == 2
    assert c["candidate_source_rows"] == [2, 3]
    assert c["capacity_scenario_80_percent"] == 1
    assert c["approved_training_examples"] == 0
    assert "including zero" in result["limitation"]
    assert 685 in RECOMMENDED_HOLDS and 686 in RECOMMENDED_HOLDS
