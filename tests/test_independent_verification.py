"""Independent human checks, historical hashes and source-bound provenance."""

import itertools
import json

import pytest
from test_dashboard_services import workspace as workspace
from test_review_workspace import Frames, decision

from boxing_motion_ai.data.annotation_review import AXES, LABELS, ReviewLedger, digest
from boxing_motion_ai.services import DashboardService
from boxing_motion_ai.services.artifacts import ServiceError
from boxing_motion_ai.services.review_workspace import ReviewWorkspace


def verify(store, axes, **changes):
    fields = dict(
        action="verify",
        proposed_label="jab",
        proposed_bounds=dict(start=100, end=103, convention="zero_based_decoded_inclusive"),
        inspected_frames=[100, 103],
        lead_rear_evidence="",
        uncertainty="",
        notes="Human selected the explicitly checked aspects.",
        verification=dict(
            schema_version="annotation-verification-1.0", **{axis: axis in axes for axis in AXES}
        ),
    )
    fields.update(changes)
    return decision(store, **fields)


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize(
    "axes", [set(c) for n in range(1, 4) for c in itertools.combinations(AXES, n)]
)
def test_every_combination_and_class_remains_honest(workspace, axes, label):
    store = ReviewWorkspace(DashboardService(workspace[0]))
    item = verify(store, axes, proposed_label=label)
    store.save("V4", 1, item, Frames())
    reopened = ReviewWorkspace(DashboardService(workspace[0])).snapshot()
    review = reopened.review("V4", 1).data
    revision = review["latest"]
    full = axes == set(AXES)
    assert revision["disposition"] == ("accept" if full else "hold")
    assert review["human_accepted"] == full
    assert review["verification_status"] == ("fully_verified" if full else "partially_verified")
    assert revision["confirmed_label"] == (label if {"punch_type", "hand"} <= axes else None)
    assert (revision["confirmed_bounds"] is not None) == ("timing" in axes)
    assert {k for k, v in revision["verification"].items() if k in AXES and v} == axes
    assert revision["verification"]["origins"] == {axis: item.request_id for axis in axes}
    assert reopened.review("V4", 1).data["source"]["punch_type_provisional"] == "jab"
    plan = reopened.curation_status().data["proposal"]
    if not full:
        assert "V4:1" in plan["pending_human_queue"]
        assert "V4:1" not in plan["human_accepted_cases"]


def test_origins_survive_other_reviewer_revision_and_reopening(workspace):
    store = ReviewWorkspace(DashboardService(workspace[0]))
    first = verify(store, {"punch_type"}, reviewer="Alice")
    receipt = store.save("V4", 1, first, Frames())
    before = (workspace[0] / receipt["ledger_path"]).read_bytes()
    second = verify(store, {"punch_type", "hand"}, reviewer="Bob")
    store.save("V4", 1, second, Frames())
    third = verify(store, set(AXES), reviewer="Carol")
    store.save("V4", 1, third, Frames())
    reopened = ReviewWorkspace(DashboardService(workspace[0]))
    state = reopened.snapshot().edit_state("V4", 1).data
    assert len(state["history"]) == 4
    assert state["history"][-1]["verification"]["origins"] == dict(
        punch_type=first.request_id, hand=second.request_id, timing=third.request_id
    )
    assert (workspace[0] / receipt["ledger_path"]).read_bytes() == before
    # Saving inherited checks needs no invented fresh frame inspection or new attribution.
    carried = verify(reopened, set(AXES), reviewer="Dana", inspected_frames=[])
    reopened.save("V4", 1, carried, Frames())
    last = reopened.snapshot().edit_state("V4", 1).data["history"][-1]
    assert last["verification"]["origins"] == state["history"][-1]["verification"]["origins"]
    # Human explicitly unchecks timing; the past timing confirmation remains in history.
    revoked = verify(reopened, {"punch_type", "hand"}, reviewer="Dana", inspected_frames=[])
    reopened.save("V4", 1, revoked, Frames())
    assert reopened.snapshot().review("V4", 1).data["verification_status"] == "partially_verified"
    assert reopened.snapshot().edit_state("V4", 1).data["history"][-1]["confirmed_bounds"] is None
    reopened.save("V4", 1, verify(reopened, set(), inspected_frames=[]), Frames())
    assert reopened.snapshot().review("V4", 1).data["verification_status"] == "unverified"


def test_checks_cannot_migrate_to_an_edited_target_without_fresh_evidence(workspace):
    store = ReviewWorkspace(DashboardService(workspace[0]))
    store.save("V4", 1, verify(store, set(AXES)), Frames())
    for fields in [
        dict(proposed_label="cross"),
        dict(proposed_bounds=dict(start=101, end=104, convention="zero_based_decoded_inclusive")),
    ]:
        item = verify(store, set(AXES), inspected_frames=[], **fields)
        with pytest.raises(ServiceError):
            store.save("V4", 1, item, Frames())
    assert len(store.snapshot().edit_state("V4", 1).data["history"]) == 2


def test_legacy_human_checks_reuse_evidence_without_rewriting_history(workspace):
    store = ReviewWorkspace(DashboardService(workspace[0]))
    old = decision(
        store,
        action="confirm",
        proposed_label="jab",
        proposed_bounds=dict(start=100, end=103, convention="zero_based_decoded_inclusive"),
        inspected_frames=[100, 103],
        lead_rear_evidence="Human verified the arm against stance.",
    )
    receipt = store.save("V4", 1, old, Frames())
    raw = (workspace[0] / receipt["ledger_path"]).read_bytes()
    legacy = json.loads(raw)
    parsed = ReviewLedger.model_validate(legacy)
    assert parsed.model_dump(mode="json") == legacy
    assert digest(parsed.cases["V4:1"].revisions[-1].model_dump(mode="json")) == digest(
        legacy["cases"]["V4:1"]["revisions"][-1]
    )
    store.save("V4", 1, verify(store, set(AXES), inspected_frames=[]), Frames())
    last = store.snapshot().edit_state("V4", 1).data["history"][-1]
    assert set(last["verification"]["origins"].values()) == {old.request_id}
    assert (workspace[0] / receipt["ledger_path"]).read_bytes() == raw


@pytest.mark.parametrize(
    "axes,indices", [({"punch_type"}, []), ({"hand"}, [99]), ({"timing"}, [100]), (set(), [])]
)
def test_new_checks_require_real_native_evidence(workspace, axes, indices):
    store = ReviewWorkspace(DashboardService(workspace[0]))
    with pytest.raises(ServiceError):
        store.save("V4", 1, verify(store, axes, inspected_frames=indices), Frames())


def test_timing_does_not_certify_type_or_hand_and_rejects_scene_change(workspace):
    class CutFrames(Frames):
        def frame(self, recording, index):
            frame, pixels = super().frame(recording, index)
            frame["scene_id"] = int(index >= 102)
            return frame, pixels

    store = ReviewWorkspace(DashboardService(workspace[0]))
    with pytest.raises(ServiceError, match="confirmation_crosses_scene_or_segment"):
        store.save("V4", 1, verify(store, {"timing"}), CutFrames())
    # A movement/hand review can remain partial while timing is unresolved.
    store.save("V4", 1, verify(store, {"hand"}), CutFrames())
    latest = store.snapshot().review("V4", 1).data["latest"]
    assert latest["confirmed_label"] is None
    assert latest["confirmed_bounds"] is None


def test_false_or_assistant_provenance_is_rejected(workspace):
    store = ReviewWorkspace(DashboardService(workspace[0]))
    store.save("V4", 1, verify(store, {"punch_type"}), Frames())
    ledger = store.snapshot().ledger.model_dump(mode="json")
    item = ledger["cases"]["V4:1"]["revisions"][-1]
    item["verification"]["origins"]["punch_type"] = ledger["cases"]["V4:1"]["revisions"][0][
        "revision_id"
    ]
    with pytest.raises(ValueError, match="verified human origin"):
        ReviewLedger.model_validate(ledger)
    item["verification"]["origins"]["punch_type"] = item["revision_id"]
    item["label_status"] = "confirmed"
    item["confirmed_label"] = "jab"
    with pytest.raises(ValueError, match="both punch type AND lead/rear hand"):
        ReviewLedger.model_validate(ledger)
