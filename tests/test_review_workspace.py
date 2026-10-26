"""Synthetic review transactions; never attest real dataset cases."""

import json
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_dashboard_services import workspace as workspace

from boxing_motion_ai.services import DashboardService
from boxing_motion_ai.services.artifacts import ServiceError
from boxing_motion_ai.services.review_workspace import DIRECTORY, Decision, ReviewWorkspace


class Frames:
    def frame(self, recording, index):
        return dict(
            frame_index=index,
            pts=str(index * 1001),
            time_base="1/30000",
            exact_timestamp=f"{index * 1001}/30000",
            scene_id=0,
            segment_id=0,
        ), None


def decision(store, **changes):
    state = store.snapshot().edit_state("V4", 1).data
    fields = dict(
        request_id=str(uuid.uuid4()),
        expected_ledger_sha256=state["ledger_sha256"],
        expected_revision_sha256=state["revision_sha256"],
        source_candidate_sha256=state["source_candidate_sha256"],
        action="hold",
        reviewer="Synthetic tester",
        notes="Synthetic boundary review",
        uncertainty="Boundary unresolved",
        lead_rear_evidence="",
        inspected_frames=[],
        attest_human=True,
    )
    fields.update(changes)
    return Decision(**fields)


def test_restart_revision_idempotency_and_original_preservation(workspace):
    root = workspace[0]
    originals = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    store = ReviewWorkspace(DashboardService(root))
    first = decision(store)
    receipt = store.save("V4", 1, first, Frames())
    assert store.save("V4", 1, first, Frames()) == receipt
    version = root / receipt["ledger_path"]
    before = version.read_bytes()
    restarted = ReviewWorkspace(DashboardService(root))
    assert restarted.snapshot().edit_state("V4", 1).data["history"][-1]["actor_type"] == "human"
    second = decision(restarted, action="exclude", uncertainty="")
    restarted.save("V4", 1, second, Frames())
    state = restarted.snapshot().edit_state("V4", 1).data
    assert len(state["history"]) == 3
    assert state["history"][-1]["supersedes_sha256"] == second.expected_revision_sha256
    assert version.read_bytes() == before
    assert all(p.read_bytes() == content for p, content in originals.items())
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from boxing_motion_ai.services import DashboardService; "
            "from boxing_motion_ai.services.review_workspace import ReviewWorkspace; "
            "import sys; store = ReviewWorkspace(DashboardService(sys.argv[1])); "
            "print(store.snapshot().ledger.ledger_version)",
            str(root),
        ],
        text=True,
    )
    assert int(output) == state["ledger_version"]
    assert restarted.snapshot().curation_status().status == "available"


def test_confirm_and_separate_correction(workspace):
    store = ReviewWorkspace(DashboardService(workspace[0]))
    item = decision(
        store,
        action="confirm",
        proposed_label="rear_hook",
        proposed_bounds=dict(start=101, end=104, convention="zero_based_decoded_inclusive"),
        inspected_frames=[101, 104],
        lead_rear_evidence="Synthetic stance evidence",
    )
    receipt = store.save("V4", 1, item, Frames())
    assert receipt["corrections_applied"] is False
    assert receipt["verified_frames"][0]["pts"] == "101101"
    live = store.snapshot()
    assert live.review("V4", 1).data["source"]["punch_type_provisional"] == "jab"
    assert live.edit_state("V4", 1).data["history"][-1]["confirmed_label"] == "rear_hook"


@pytest.mark.parametrize(
    "changes",
    [
        dict(attest_human=False),
        dict(attest_human=1),
        dict(reviewer=" "),
        dict(inspected_frames=[-1]),
        dict(inspected_frames=[1, 1]),
        dict(unsafe_path="/tmp/file"),
    ],
)
def test_invalid_payload(workspace, changes):
    with pytest.raises(ValueError):
        decision(ReviewWorkspace(DashboardService(workspace[0])), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        dict(uncertainty=""),
        dict(inspected_frames=[200]),
        dict(action="confirm", proposed_label="jab"),
        dict(
            action="confirm",
            proposed_label="jab",
            proposed_bounds=dict(start=1, end=2, convention="zero_based_decoded_inclusive"),
        ),
    ],
)
def test_validation_no_publication(workspace, changes):
    store = ReviewWorkspace(DashboardService(workspace[0]))
    item = decision(store, **changes)
    with pytest.raises(ServiceError):
        store.save("V4", 1, item, Frames())
    assert not list((workspace[0] / DIRECTORY).glob("v*"))


def test_concurrent_stale_writers(workspace):
    root = workspace[0]
    store = ReviewWorkspace(DashboardService(root))
    decisions = [decision(store), decision(store)]

    def write(item):
        try:
            ReviewWorkspace(DashboardService(root)).save("V4", 1, item, Frames())
            return "saved"
        except ServiceError as exc:
            return exc.status

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(write, decisions)) == ["saved", "stale"]
    assert len(store.snapshot().edit_state("V4", 1).data["history"]) == 2


def test_audit_tampering_detected(workspace):
    root = workspace[0]
    store = ReviewWorkspace(DashboardService(root))
    receipt = store.save("V4", 1, decision(store), Frames())
    path = (root / receipt["ledger_path"]).with_name("receipt.json")
    value = json.loads(path.read_text())
    value["decision"]["notes"] = "tampered"
    path.write_text(json.dumps(value))
    with pytest.raises(ServiceError):
        store.snapshot()


def test_symlink_and_reserved_recording(workspace):
    root = workspace[0]
    store = ReviewWorkspace(DashboardService(root))
    item = decision(store)
    with pytest.raises(ServiceError):
        store.save("V9", 1, item, Frames())
    directory = root / DIRECTORY
    (directory / ".lock").unlink()
    (directory / ".lock").symlink_to(root / "configs/dashboard_catalogue.json")
    with pytest.raises(ServiceError):
        store.snapshot()


def test_interrupted_transaction_never_published(workspace, monkeypatch):
    store = ReviewWorkspace(DashboardService(workspace[0]))
    item = decision(store)

    def fail(*args, **kwargs):
        raise OSError("synthetic interruption")

    monkeypatch.setattr("boxing_motion_ai.services.review_workspace.os.rename", fail)
    with pytest.raises(ServiceError):
        store.save("V4", 1, item, Frames())
    assert store.snapshot().ledger.ledger_version == 1


def test_structured_human_attestation_and_uncertainty_preserve_correction_history(workspace):
    """No freeform prose required; explicit human assertions still bind exact evidence."""
    root = workspace[0]
    originals = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    store = ReviewWorkspace(DashboardService(root))
    assertion = (
        "I personally verified the punch label, the lead/rear arm from the boxer’s stance, "
        "and the start/end frames."
    )
    first = decision(
        store,
        action="confirm",
        reviewer="Alice",
        notes=(
            f"Reviewer selected Yes for the original annotation. Explicit attestation: {assertion}"
        ),
        lead_rear_evidence=f"Reviewer explicitly attested: {assertion}",
        uncertainty="",
        proposed_label="jab",
        proposed_bounds=dict(start=100, end=103, convention="zero_based_decoded_inclusive"),
        inspected_frames=[100, 103],
    )
    receipt = store.save("V4", 1, first, Frames())
    first_bytes = (root / receipt["ledger_path"]).read_bytes()
    corrected = decision(
        store,
        action="confirm",
        reviewer="Bob",
        notes="Reviewer correction: the opening frame belongs to the preceding movement.",
        lead_rear_evidence=f"Reviewer explicitly attested: {assertion}",
        uncertainty="",
        proposed_label="cross",
        proposed_bounds=dict(start=101, end=104, convention="zero_based_decoded_inclusive"),
        inspected_frames=[101, 104],
    )
    store.save("V4", 1, corrected, Frames())
    unsure = decision(
        store,
        reviewer="Bob",
        action="ambiguous",
        notes="Reviewer selected: Lead or rear arm is unclear.",
        uncertainty="Lead or rear arm is unclear",
        proposed_label=None,
        proposed_bounds=None,
    )
    store.save("V4", 1, unsure, Frames())
    reopened = ReviewWorkspace(DashboardService(root)).snapshot()
    history = reopened.edit_state("V4", 1).data["history"]
    assert [r["reviewer"] for r in history[-3:]] == ["Alice", "Bob", "Bob"]
    assert history[-3]["confirmed_label"] == "jab"
    assert history[-2]["confirmed_label"] == "cross"
    assert history[-2]["confirmed_bounds"]["start"] == 101
    assert history[-1]["disposition"] == "hold"
    assert history[-1]["confirmed_label"] is None
    assert history[-1]["confirmed_bounds"] is None
    assert history[-1]["lead_rear_evidence"] is None
    assert reopened.review("V4", 1).data["human_accepted"] is False
    assert (root / receipt["ledger_path"]).read_bytes() == first_bytes
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    # The backend still rejects empty evidence/notes and missing native boundaries.
    with pytest.raises(ValueError):
        decision(store, notes="")
    invalid = decision(
        store,
        action="confirm",
        proposed_label="jab",
        proposed_bounds=dict(start=100, end=103, convention="zero_based_decoded_inclusive"),
    )
    with pytest.raises(ServiceError):
        store.save("V4", 1, invalid, Frames())
