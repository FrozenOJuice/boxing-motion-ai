"""Synthetic review/export fixtures; never label real footage or train a detector."""

import pytest
from test_dashboard_api import media as media  # noqa: F401
from test_dashboard_services import workspace as workspace  # noqa: F401
from test_interval_events import events as events  # noqa: F401
from test_interval_events import mutate, punch
from test_interval_review import pilot as pilot  # noqa: F401

from boxing_motion_ai.data import continuous_dataset as dataset
from boxing_motion_ai.data.verified_background import workspace_at
from boxing_motion_ai.services.artifacts import ServiceError
from boxing_motion_ai.services.interval_review import interval_timing
from boxing_motion_ai.services.review_workspace import encode, sha


@pytest.fixture
def data(events, monkeypatch):
    root, client = events
    ws = workspace_at(root, "original")
    original = ws.detail("V4-bg-1")["entry"]
    spec = ws.base._recording("V4")
    arrays = ws.frames._load("V4", spec, ws.base._preparation("V4", spec))
    negative = original | dict(
        id="V4-motion-1",
        start=5,
        stop=6,
        source_group_guard=["V4"],
        timing=interval_timing(arrays, 5, 6),
    )
    # The absence validator itself has real integration tests; isolate composition here.
    path = "reports/synthetic-background/manifest.json"
    (root / path).parent.mkdir(parents=True)
    review_path = "reports/synthetic-background/review.json"
    (root / review_path).write_bytes(encode(dict(source="manual", decision="no_punch")))
    background = dict(
        review_head="a" * 64,
        intervals=[
            dict(
                provenance=negative,
                id="V4-motion-1",
                review=dict(path=review_path, sha256=sha((root / review_path).read_bytes())),
            )
        ],
    )
    (root / path).write_bytes(encode(background))
    receipt = root / "reports/synthetic-background/review-backup-sha256.json"
    receipt.write_bytes(encode({}))
    (root / dataset.BACKGROUND_PIN).write_bytes(
        encode(
            dict(
                path=path,
                sha256=sha((root / path).read_bytes()),
                backup_receipt_sha256=sha(receipt.read_bytes()),
            )
        )
    )
    monkeypatch.setattr(dataset, "validate_background", lambda _: background)
    monkeypatch.setattr(dataset, "SCOPE", ("V4-bg-1",))
    return root, client, background


def complete(client, event=None):
    p = event or punch(1, 3)
    assert (
        mutate(client, event=p, inspected_frames=sorted({p["start"], p["stop"] - 1}))[0].status_code
        == 200
    )
    response, _ = mutate(
        client,
        "complete",
        watched_entire=True,
        all_punches_labeled=True,
        inspected_frames=[1, 2, 3, 4],
    )
    assert response.status_code == 200, response.text
    return p


def test_only_exhaustively_reviewed_gaps_get_background_and_unknown_stays_unknown(data):
    root, client, _ = data
    complete(client)
    result = dataset.prepare(root)
    assert result["summary"]["positive_events"] == 1
    assert result["summary"]["activity_1_frames"] == 2
    assert result["summary"]["activity_0_frames"] == 3
    assert result["summary"]["activity_1_seconds"] == "2/25"
    assert result["default_unreviewed_state"] == "unknown"
    assert result["detector_training_authorized"] is False
    assert result["development_intervals"] == result["evaluation_intervals"] == []
    assert result["readiness"]["detector_experiment_ready"] is False
    assert all(
        1 <= r["start"] < r["stop"] <= 5
        for i in result["intervals"]
        if i["kind"] == "exhaustive_punch_interval"
        for r in i["supervision_runs"]
    )
    assert dataset.validate(root) == result
    with pytest.raises(ServiceError, match="continuous_version_already_exists"):
        dataset.prepare(root)


def test_incomplete_positive_records_are_excluded_and_never_generate_negative_gaps(data):
    root, client, _ = data
    assert mutate(client, event=punch(), inspected_frames=[1, 2])[0].status_code == 200
    result = dataset.collect(root)
    assert len(result["intervals"]) == 1
    assert result["excluded_intervals"][0]["id"] == "V4-bg-1"
    assert result["summary"]["positive_events"] == 0
    with pytest.raises(ServiceError, match="continuous_no_verified_positive_events"):
        dataset.prepare(root)


def test_explicit_fifth_hold_wins_even_if_a_later_review_is_completed(data, monkeypatch):
    root, client, _ = data
    complete(client)
    original = dataset.IntervalEvents.detail

    def detail(self, key):
        if key == "V4-bg-5":
            return dict(head="b" * 64, history=[], events=[dict()], complete=True)
        return original(self, key)

    monkeypatch.setattr(dataset.IntervalEvents, "detail", detail)
    monkeypatch.setattr(dataset, "SCOPE", ("V4-bg-1", "V4-bg-5"))
    result = dataset.collect(root)
    assert result["summary"]["positive_events"] == 1
    assert result["excluded_intervals"][0]["reason"] == "user_held_incomplete"


def test_overlaps_and_uncertain_types_preserve_binary_activity_but_mask_type_loss(data):
    root, client, _ = data
    p = punch(1, 4)
    assert mutate(client, event=p, inspected_frames=[1, 3])[0].status_code == 200
    complete(client, punch(2, 4) | dict(punch_type="unknown", type_uncertain=True))
    result = dataset.collect(root)
    assert result["summary"]["overlap_pairs"] == 1
    assert result["summary"]["uncertain_types"] == 1
    assert result["summary"]["activity_1_frames"] == 3
    mixed = [
        r for i in result["intervals"] for r in i["supervision_runs"] if len(r["event_ids"]) == 2
    ]
    assert mixed and mixed[0]["activity"] == 1 and mixed[0]["type_loss_enabled"] is False


def test_source_group_and_conflicting_supervision_are_blockers(data):
    root, client, bg = data
    complete(client)
    p = bg["intervals"][0]["provenance"]
    p["start"], p["stop"] = 2, 3
    with pytest.raises(ServiceError, match="continuous_supervision_overlap"):
        dataset.collect(root)
    p["recording"] = "V5"
    with pytest.raises(ServiceError, match="continuous_split_leakage"):
        dataset.collect(root)


def test_new_review_invalidates_snapshot_and_backup_changes_are_detected(data):
    root, client, _ = data
    p = complete(client)
    dataset.prepare(root)
    target = root / dataset.OUTPUT / "manifest.json"
    before = target.read_bytes()
    assert mutate(client, event=p | dict(punch_type="cross"))[0].status_code == 200
    with pytest.raises(ServiceError, match="continuous_snapshot_stale"):
        dataset.validate(root)
    assert target.read_bytes() == before


def test_backup_hash_is_pinned_by_the_dataset_manifest(data):
    root, client, _ = data
    complete(client)
    result = dataset.prepare(root)
    path = next(iter(result["source_receipts"]))
    target = root / dataset.OUTPUT / "review-backup" / path
    target.write_bytes(b"{}")
    with pytest.raises(ServiceError, match="artifact_hash_mismatch"):
        dataset.validate(root)
