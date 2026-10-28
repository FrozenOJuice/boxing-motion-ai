"""Only explicit latest human absence decisions can enter the frozen background manifest."""

import json
import uuid

import pytest
from test_dashboard_api import media as media  # noqa: F401
from test_dashboard_services import workspace as workspace  # noqa: F401
from test_interval_pilots import install_motion
from test_interval_review import HEADERS, payload
from test_interval_review import pilot as pilot  # noqa: F401

from boxing_motion_ai.data.interval_pilot import ROLES
from boxing_motion_ai.data.verified_background import (
    OUTPUT,
    collect,
    prepare,
    validate,
    workspace_at,
)
from boxing_motion_ai.services.artifacts import ServiceError
from boxing_motion_ai.services.review_workspace import encode, sha


def reviewed(pilot, decision="no_punch"):
    root, client = install_motion(pilot)
    path = root / "reports/boxingvi/interval-review-motion-v2/manifest.json"
    m = json.loads(path.read_text())
    ws = workspace_at(root)
    spec = ws.base._recording("V4")
    a = ws.frames._load("V4", spec, ws.base._preparation("V4", spec))
    m["roles"] = ROLES
    m["entries"][0].update(
        source_group_guard=["V4"],
        selection_stratum="quiet_motion",
        boxer_id=None,
        session_id=None,
        endpoint_pts={
            name: {k: int(a[k][i]) for k in ("pts", "time_base_num", "time_base_den")}
            for name, i in (("start", 0), ("stop_exclusive", 1))
        },
    )
    path.write_bytes(encode(m))
    (root / "configs/interval_review_motion_v2.json").write_bytes(
        encode(dict(path=str(path.relative_to(root)), sha256=sha(path.read_bytes())))
    )
    clip = client.get("/api/intervals/V4-motion-1/playback?pilot=motion-v2").json()["data"]
    body = payload(client, decision) | {
        "expected_head": client.get("/api/intervals?pilot=motion-v2").json()["data"]["head"],
        "inspected_frames": [],
        "playback_coverage": dict(
            schema_version="continuous-viewing-1.0",
            policy="visible-realtime-1.0",
            interval_id="V4-motion-1",
            cache_key=clip["cache_key"],
            segments=[dict(start=0, end=0.04, wall_seconds=0.04, rate=1.0, samples=2)],
        ),
    }
    result = client.post("/api/intervals/V4-motion-1?pilot=motion-v2", json=body, headers=HEADERS)
    assert result.status_code == 200, result.text
    return root, client, body


def test_frozen_export_roundtrip_exact_pts_backup_and_no_overwrite(pilot):
    root, _, _ = reviewed(pilot)
    review = root / "reports/boxingvi/interval-reviews-motion-v2/v00000001/review.json"
    before = review.read_bytes()
    result = prepare(root, expected_intervals=1)
    assert result["summary"]["frames"] == 1
    assert result["summary"]["duration_exact_seconds"] == "1/25"
    assert result["default_unreviewed_state"] == "unknown"
    assert result["context_is_verified"] is False
    assert result["detector_training_authorized"] is False
    item = result["intervals"][0]
    assert item["review"]["validation"]["completion_mode"] == "continuous_playback"
    assert item["review"]["inspected_frames"] == []
    assert item["review"]["movement_tags"] == []
    assert item["endpoint_pts"]["stop_exclusive"]["pts"] == 40
    assert validate(root) == result
    assert (root / OUTPUT / "review-backup" / review.relative_to(root)).read_bytes() == before
    assert review.read_bytes() == before
    with pytest.raises(ServiceError, match="background_version_already_exists"):
        prepare(root, expected_intervals=1)


@pytest.mark.parametrize("decision", ["punch_present", "not_sure"])
def test_unresolved_or_positive_never_export_as_background(pilot, decision):
    root, _, _ = reviewed(pilot, decision)
    with pytest.raises(ServiceError, match="background_not_verified"):
        collect(workspace_at(root))
    assert not (root / OUTPUT).exists()


def test_latest_revision_invalidates_frozen_export_without_rewriting_it(pilot):
    root, client, body = reviewed(pilot)
    prepare(root, expected_intervals=1)
    target = root / OUTPUT / "manifest.json"
    before = target.read_bytes()
    body.update(
        request_id=str(uuid.uuid4()),
        expected_head=client.get("/api/intervals?pilot=motion-v2").json()["data"]["head"],
        decision="not_sure",
    )
    assert (
        client.post(
            "/api/intervals/V4-motion-1?pilot=motion-v2", json=body, headers=HEADERS
        ).status_code
        == 200
    )
    with pytest.raises(ServiceError, match="background_not_verified"):
        validate(root)
    assert target.read_bytes() == before


def test_tampered_backup_is_detected(pilot):
    root, _, _ = reviewed(pilot)
    prepare(root, expected_intervals=1)
    target = (
        root
        / OUTPUT
        / "review-backup/reports/boxingvi/interval-reviews-motion-v2/v00000001/review.json"
    )
    target.write_bytes(b"{}")
    with pytest.raises(ServiceError, match="artifact_hash_mismatch"):
        validate(root)


def test_incomplete_pilot_cannot_publish_default_twelve_interval_version(pilot):
    root, _, _ = reviewed(pilot)
    with pytest.raises(ServiceError, match="motion_pilot_not_complete"):
        prepare(root)
    assert not (root / OUTPUT).exists()


def test_even_a_new_confirmed_revision_requires_a_new_export_version(pilot):
    root, client, body = reviewed(pilot)
    prepare(root, expected_intervals=1)
    body.update(
        request_id=str(uuid.uuid4()),
        expected_head=client.get("/api/intervals?pilot=motion-v2").json()["data"]["head"],
        notes="Updated human note; same absence decision",
    )
    r = client.post("/api/intervals/V4-motion-1?pilot=motion-v2", json=body, headers=HEADERS)
    assert r.status_code == 200, r.text
    with pytest.raises(ServiceError, match="background_snapshot_stale"):
        validate(root)
