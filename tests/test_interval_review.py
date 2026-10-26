"""Interval review tests use synthetic footage and temporary immutable ledgers only."""

import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_dashboard_api import media as media  # noqa: F401
from test_dashboard_services import workspace as workspace  # noqa: F401

from boxing_motion_ai.api.app import create_app
from boxing_motion_ai.services.artifacts import ServiceError
from boxing_motion_ai.services.interval_review import IntervalDecision, interval_timing
from boxing_motion_ai.services.review_workspace import encode, sha

HEADERS = {"origin": "http://127.0.0.1:5174", "x-review-client": "dashboard-v0.2"}


@pytest.fixture
def pilot(media):
    root, app, client, _, arrays = media
    service = app.state.service
    spec = service._recording("V4")
    prep = service._preparation("V4", spec)
    entry = dict(
        id="V4-bg-1",
        recording="V4",
        role="train_candidate",
        purpose="background_candidate",
        start=0,
        stop=1,
        frame_count=8,
        context_start=0,
        context_end=1,
        source_inputs=spec.inputs,
        video_sha256=prep["inputs"]["video_sha256"],
        timing=interval_timing(arrays, 0, 1),
    )
    manifest = dict(
        schema_version="interval-pilot-1.0",
        roles={
            "V4": "train_candidate",
            "V5": "development",
            "V7": "evaluation_reserved",
            "V8": "evaluation_reserved",
        },
        entries=[entry],
    )
    path = root / "reports/boxingvi/interval-review-pilot-v1/manifest.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(encode(manifest))
    (root / "configs/interval_review_pilot_v1.json").write_bytes(
        encode(dict(path=str(path.relative_to(root)), sha256=sha(path.read_bytes())))
    )
    return root, client, manifest


def payload(client, decision="no_punch"):
    return dict(
        request_id=str(uuid.uuid4()),
        expected_head=client.get("/api/intervals").json()["data"]["head"],
        decision=decision,
        watched_entire=True,
        visible_continuous=True,
        inspected_frames=[0],
        unannotated_punches="no",
        notes="Synthetic fixture",
    )


def test_persistence_revision_history_and_idempotent_save(pilot):
    root, client, _ = pilot
    assert client.get("/api/intervals/V4-bg-1").json()["data"]["status"] == "unknown"
    first = payload(client)
    r = client.post("/api/intervals/V4-bg-1", json=first, headers=HEADERS)
    assert r.status_code == 200, r.text
    assert r.json()["data"]["status"] == "verified_background"
    before = (root / "reports/boxingvi/interval-reviews-v1/v00000001/review.json").read_bytes()
    assert client.post("/api/intervals/V4-bg-1", json=first, headers=HEADERS).status_code == 200
    restarted = TestClient(create_app(root), base_url="http://127.0.0.1:8000")
    assert len(restarted.get("/api/intervals").json()["data"]["history"]) == 1
    second = payload(restarted, "punch_present") | {"unannotated_punches": "yes"}
    saved = restarted.post("/api/intervals/V4-bg-1", json=second, headers=HEADERS)
    assert saved.json()["data"]["status"] == "needs_event_labeling"
    assert saved.json()["data"]["exhaustive_event_reference"] is False
    assert (
        before == (root / "reports/boxingvi/interval-reviews-v1/v00000001/review.json").read_bytes()
    )
    third = payload(restarted, "not_sure")
    assert (
        restarted.post("/api/intervals/V4-bg-1", json=third, headers=HEADERS).json()["data"][
            "status"
        ]
        == "unresolved"
    )
    assert len(restarted.get("/api/intervals").json()["data"]["history"]) == 3


@pytest.mark.parametrize(
    "change",
    [
        {"watched_entire": False},
        {"visible_continuous": False},
        {"watched_entire": "true"},
        {"source": "manual"},
        {"inspected_frames": []},
        {"inspected_frames": [7]},
        {"unannotated_punches": "yes"},
    ],
)
def test_no_automatic_absence_certification(pilot, change):
    _, client, _ = pilot
    r = client.post("/api/intervals/V4-bg-1", json=payload(client) | change, headers=HEADERS)
    assert r.status_code == 422, r.text
    assert client.get("/api/intervals").json()["data"]["history"] == []


def test_stale_revision_reserved_recording_and_unsafe_paths(pilot):
    _, client, _ = pilot
    stale = payload(client)
    assert (
        client.post("/api/intervals/V4-bg-1", json=payload(client), headers=HEADERS).status_code
        == 200
    )
    assert client.post("/api/intervals/V4-bg-1", json=stale, headers=HEADERS).status_code == 409
    assert client.get("/api/intervals/V9-bg-1").status_code == 404
    assert (
        client.post("/api/intervals/V9-bg-1", json=payload(client), headers=HEADERS).status_code
        == 404
    )
    assert client.post("/api/intervals/V4-bg-1", json=payload(client)).status_code == 403


def test_manifest_tampering_and_storage_symlink_rejected(pilot, tmp_path):
    root, client, _ = pilot
    storage = root / "reports/boxingvi/interval-reviews-v1"
    storage.symlink_to(tmp_path)
    assert client.get("/api/intervals").status_code == 422
    storage.unlink()
    path = root / "reports/boxingvi/interval-review-pilot-v1/manifest.json"
    path.write_text(path.read_text() + " ")
    assert client.get("/api/intervals").status_code == 409


def test_timing_cuts_gaps_and_exact_endpoints():
    a = dict(
        frame_index=np.arange(5),
        pts=np.arange(5) * 1001,
        time_base_num=np.ones(5, int),
        time_base_den=np.full(5, 30000),
        scene_id=np.zeros(5, int),
        segment_id=np.zeros(5, int),
    )
    assert interval_timing(a, 0, 4)["stop_exclusive"] == "1001/7500"
    with pytest.raises(ServiceError):
        interval_timing(a, 0, 4, [2])
    a["segment_id"][2:] = 1
    with pytest.raises(ServiceError):
        interval_timing(a, 0, 4)
    a["segment_id"][:] = 0
    a["frame_index"][2] = 9
    with pytest.raises(ServiceError):
        interval_timing(a, 0, 4)
    a["frame_index"][2] = 2
    a["pts"][2] = a["pts"][1]
    with pytest.raises(ServiceError):
        interval_timing(a, 0, 4)


def test_strict_decision_rejects_automated_source():
    with pytest.raises(ValidationError):
        IntervalDecision(
            request_id=str(uuid.uuid4()),
            expected_head="a" * 64,
            decision="no_punch",
            source="automated",
        )


def test_conflicting_source_annotation_cannot_become_negative(pilot, monkeypatch):
    from boxing_motion_ai.services.interval_review import IntervalWorkspace

    _, client, _ = pilot
    original = IntervalWorkspace.detail

    def conflicting(self, key):
        detail = original(self, key)
        detail["annotations"] = [{"source_row": 1}]
        return detail

    monkeypatch.setattr(IntervalWorkspace, "detail", conflicting)
    r = client.post("/api/intervals/V4-bg-1", json=payload(client), headers=HEADERS)
    assert r.status_code == 422
    assert "source_punch_conflict" in r.text
    assert client.get("/api/intervals").json()["data"]["history"] == []


def test_failed_publish_is_not_a_review_and_old_history_is_retained(pilot):
    root, client, _ = pilot
    client.get("/api/intervals")
    pending = root / "reports/boxingvi/interval-reviews-v1/pending-interrupted"
    pending.mkdir()
    (pending / "review.json").write_text("incomplete")
    assert client.get("/api/intervals").json()["data"]["history"] == []
    assert (
        client.post("/api/intervals/V4-bg-1", json=payload(client), headers=HEADERS).status_code
        == 200
    )


def test_manifest_split_overlap_and_reserved_recording_guards(pilot):
    root, client, manifest = pilot
    from copy import deepcopy

    path = root / "reports/boxingvi/interval-review-pilot-v1/manifest.json"
    bad = []
    v = deepcopy(manifest)
    v["entries"][0]["recording"] = "V9"
    bad.append(v)
    v = deepcopy(manifest)
    v["entries"][0]["purpose"] = "evaluation_block"
    bad.append(v)
    v = deepcopy(manifest)
    v["roles"]["V8"] = "train_candidate"
    bad.append(v)
    v = deepcopy(manifest)
    v["entries"].append(v["entries"][0] | {"id": "V4-bg-2"})
    bad.append(v)
    for value in bad:
        path.write_bytes(encode(value))
        (root / "configs/interval_review_pilot_v1.json").write_bytes(
            encode(dict(path=str(path.relative_to(root)), sha256=sha(path.read_bytes())))
        )
        assert client.get("/api/intervals").status_code == 422
