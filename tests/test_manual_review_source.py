"""Anonymous manual review, automated separation, and legacy-name preservation."""

from datetime import datetime

import pytest
from test_dashboard_services import workspace as workspace
from test_independent_verification import verify
from test_review_workspace import Frames, decision

from boxing_motion_ai.data.annotation_review import Revision, review_source, verified_axes
from boxing_motion_ai.services import DashboardService
from boxing_motion_ai.services.review_workspace import ReviewWorkspace, encode, sha


def test_anonymous_manual_checks_keep_timestamps_and_origins(workspace):
    root = workspace[0]
    store = ReviewWorkspace(DashboardService(root))
    selected = {"punch_type", "hand", "timing"}
    first = verify(store, selected, reviewer=None)
    assert "reviewer" not in first.model_dump(mode="json")
    receipt = store.save("V4", 1, first, Frames())
    assert receipt["source"] == "manual"
    assert "reviewer" not in receipt["decision"]
    before = (root / receipt["ledger_path"]).read_bytes()
    second = verify(store, {"punch_type"}, reviewer=None, inspected_frames=[])
    store.save("V4", 1, second, Frames())
    reopened = ReviewWorkspace(DashboardService(root)).snapshot()
    history = reopened.edit_state("V4", 1).data["history"]
    assert history[-1]["source"] == "manual"
    assert "reviewer" not in history[-1]
    assert datetime.fromisoformat(history[-1]["reviewed_at"]).utcoffset() is not None
    assert history[-1]["verification"]["origins"] == {"punch_type": first.request_id}
    assert history[-1]["confirmed_label"] is None
    assert history[-1]["disposition"] == "hold"
    assert history[-2]["source"] == "manual"
    assert history[-2]["verification"]["timing"] is True
    assert (root / receipt["ledger_path"]).read_bytes() == before
    assert reopened.review("V4", 1).data["review_source"] == "manual"


def test_realistic_legacy_receipt_and_named_history_are_not_rewritten(workspace):
    root = workspace[0]
    store = ReviewWorkspace(DashboardService(root))
    receipt = store.save("V4", 1, verify(store, {"punch_type"}, reviewer="Alice"), Frames())
    # Construct an old-format fixture, only in this synthetic temporary workspace.
    ledger = store.snapshot().ledger.model_dump(mode="json")
    ledger["cases"]["V4:1"]["revisions"][-1].pop("source")
    ledger_raw = encode(ledger)
    ledger_path = root / receipt["ledger_path"]
    ledger_path.write_bytes(ledger_raw)
    config = store.base._config().model_dump(mode="json")
    config["review_ledger"] = dict(path=receipt["ledger_path"], sha256=sha(ledger_raw))
    config_raw = encode(config)
    (root / receipt["config_path"]).write_bytes(config_raw)
    receipt.pop("source")
    receipt.update(ledger_sha256=sha(ledger_raw), config_sha256=sha(config_raw))
    receipt_path = ledger_path.with_name("receipt.json")
    receipt_path.write_bytes(encode(receipt))
    frozen = {p: p.read_bytes() for p in (ledger_path, receipt_path, root / receipt["config_path"])}
    reopened = ReviewWorkspace(DashboardService(root))
    previous = reopened.snapshot().edit_state("V4", 1).data["history"][-1]
    assert previous["reviewer"] == "Alice"
    assert "source" not in previous
    reopened.save("V4", 1, verify(reopened, {"punch_type", "hand"}, reviewer=None), Frames())
    history = reopened.snapshot().edit_state("V4", 1).data["history"]
    assert history[-2] == previous
    assert history[-1]["source"] == "manual" and "reviewer" not in history[-1]
    assert all(p.read_bytes() == raw for p, raw in frozen.items())


def test_automated_origin_never_becomes_manual_verification(workspace):
    store = ReviewWorkspace(DashboardService(workspace[0]))
    old = store.snapshot().ledger.cases["V4:1"].revisions[-1]
    assert review_source(old) == "automated"
    assert not verified_axes(old)
    automated = old.model_dump(mode="json")
    automated.pop("reviewer")
    automated["source"] = "automated"
    parsed = Revision.model_validate(automated)
    assert review_source(parsed) == "automated" and not verified_axes(parsed)
    automated["source"] = "manual"
    with pytest.raises(ValueError, match="match the review actor"):
        Revision.model_validate(automated)
    manual = store.save("V4", 1, verify(store, {"punch_type"}, reviewer=None), Frames())
    updated = store.snapshot().ledger.cases["V4:1"].revisions[-1].model_dump(mode="json")
    updated["source"] = "automated"
    with pytest.raises(ValueError, match="match the review actor"):
        Revision.model_validate(updated)
    updated["actor_type"] = "assistant"
    with pytest.raises(ValueError, match="Independent confirmations"):
        Revision.model_validate(updated)
    assert manual["source"] == "manual"


def test_clients_cannot_choose_source_or_actor(workspace):
    store = ReviewWorkspace(DashboardService(workspace[0]))
    for values in [dict(source="manual"), dict(source="automated"), dict(actor_type="human")]:
        with pytest.raises(ValueError):
            decision(store, reviewer=None, **values)
