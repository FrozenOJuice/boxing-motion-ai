"""Synthetic approved-subset safety and paired evaluation checks; no dataset downloads."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from boxing_motion_ai.data.curated_subset import conservative_selection


def plans():
    old = dict(
        records={"V4": {"selected_source_rows": [1, 2, 3]}, "V5": {"selected_source_rows": [1, 2]}}
    )
    current = dict(
        blockers=["targeted_human_review_pending"],
        pending_human_queue=["V4:2"],
        records={"V4": {"selected_source_rows": [1, 3, 4]}, "V5": {"selected_source_rows": [1, 2]}},
    )
    revision = SimpleNamespace(actor_type="human", disposition="hold", verification="partial")
    ledger = SimpleNamespace(cases={"V4:2": SimpleNamespace(revisions=[revision])})
    return old, current, ledger


def test_approved_subset_keeps_partial_hold_and_historical_exclusion():
    old, current, ledger = plans()
    before = deepcopy(ledger)
    assert conservative_selection(old, current, ledger, ["V4:2"]) == {"V4": [1, 3]}
    assert ledger == before
    assert current["pending_human_queue"] == ["V4:2"]
    assert old["records"]["V4"]["selected_source_rows"] == [1, 2, 3]


@pytest.mark.parametrize(
    "blocker",
    [
        "human_verified_anchor_class_coverage_incomplete",
        "confirmed_changes_need_separate_preparation",
        "training_class_absent",
    ],
)
def test_approval_cannot_bypass_other_readiness_failures(blocker):
    old, current, ledger = plans()
    current["blockers"].append(blocker)
    with pytest.raises(ValueError, match="Other curation"):
        conservative_selection(old, current, ledger, ["V4:2"])


def test_unapproved_pending_case_and_automated_hold_rejected():
    old, current, ledger = plans()
    with pytest.raises(ValueError, match="explicit approval"):
        conservative_selection(old, current, ledger, [])
    ledger.cases["V4:2"].revisions[-1].actor_type = "assistant"
    with pytest.raises(ValueError, match="human hold"):
        conservative_selection(old, current, ledger, ["V4:2"])


def test_pending_case_cannot_be_selected_or_claimed_accepted():
    old, current, ledger = plans()
    current["records"]["V4"]["selected_source_rows"].append(2)
    with pytest.raises(ValueError, match="entered training"):
        conservative_selection(old, current, ledger, ["V4:2"])
    ledger.cases["V4:2"].revisions[-1].disposition = "accept"
    with pytest.raises(ValueError, match="human hold"):
        conservative_selection(old, current, ledger, ["V4:2"])


def test_development_membership_cannot_change():
    old, current, ledger = plans()
    current["records"]["V5"]["selected_source_rows"].pop()
    with pytest.raises(ValueError, match="Frozen V5"):
        conservative_selection(old, current, ledger, ["V4:2"])


def test_filter_preserves_exact_tensors_and_provenance():
    pytest.importorskip("torch")
    from boxing_motion_ai.training.curation_study import subset

    clips = [dict(source_row=i, raw=object(), x=object(), mask_x=object()) for i in (1, 2, 3)]
    original = SimpleNamespace(clips=clips, provenance={"hashes": "unchanged"})
    result = subset(original, [1, 3])
    assert result.clips[0] is original.clips[0]
    assert result.clips[1] is original.clips[2]
    assert len(original.clips) == 3
    assert original.provenance == {"hashes": "unchanged"}
    for rows in ([], [1, 1], [99]):
        with pytest.raises(ValueError):
            subset(original, rows)


def test_paired_evaluation_rejects_different_labels_or_membership():
    pytest.importorskip("torch")
    from boxing_motion_ai.training.curation_study import paired_metrics

    predictions = [dict(recording_id="V5", source_row=1, target="jab", prediction="jab")]
    a, b = paired_metrics(predictions, deepcopy(predictions))
    assert a == b
    for key, value in (("recording_id", "V4"), ("source_row", 2), ("target", "cross")):
        changed = deepcopy(predictions)
        changed[0][key] = value
        with pytest.raises(ValueError, match="membership or labels"):
            paired_metrics(predictions, changed)
