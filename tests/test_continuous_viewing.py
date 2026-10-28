"""Continuous viewing is versioned evidence, never a fabricated list of inspected frames."""

import pytest
from fastapi.testclient import TestClient
from test_dashboard_api import media as media  # noqa: F401
from test_dashboard_services import workspace as workspace  # noqa: F401
from test_interval_review import HEADERS, payload
from test_interval_review import pilot as pilot  # noqa: F401

from boxing_motion_ai.api.app import create_app
from boxing_motion_ai.services.review_workspace import encode, sha


def setup_view(pilot):
    root, client, manifest = pilot
    manifest["entries"][0]["context_end"] = 0
    path = root / "reports/boxingvi/interval-review-pilot-v1/manifest.json"
    path.write_bytes(encode(manifest))
    (root / "configs/interval_review_pilot_v1.json").write_bytes(
        encode(dict(path=str(path.relative_to(root)), sha256=sha(path.read_bytes())))
    )
    clip = client.get("/api/intervals/V4-bg-1/playback").json()["data"]
    body = payload(client) | {
        "inspected_frames": [],
        "playback_coverage": dict(
            schema_version="continuous-viewing-1.0",
            policy="visible-realtime-1.0",
            cache_key=clip["cache_key"],
            interval_id="V4-bg-1",
            segments=[dict(start=0.0, end=0.04, rate=1.0, wall_seconds=0.04, samples=2)],
        ),
    }
    return root, client, body


def test_continuous_watch_can_save_without_claiming_frame_inspection_and_preserves_legacy(pilot):
    root, client, body = setup_view(pilot)
    legacy = client.post("/api/intervals/V4-bg-1", json=payload(client), headers=HEADERS)
    assert legacy.status_code == 200, legacy.text
    old = root / "reports/boxingvi/interval-reviews-v1/v00000001/review.json"
    raw = old.read_bytes()
    assert "playback_coverage" not in legacy.json()["data"]["payload"]
    body["expected_head"] = client.get("/api/intervals").json()["data"]["head"]
    result = client.post("/api/intervals/V4-bg-1", json=body, headers=HEADERS)
    assert result.status_code == 200, result.text
    r = result.json()["data"]
    assert r["payload"]["inspected_frames"] == []
    assert r["viewing_validation"]["completion_mode"] == "continuous_playback"
    assert r["viewing_validation"]["exact_complete"] is False
    restarted = TestClient(create_app(root), base_url="http://127.0.0.1:8000")
    history = restarted.get("/api/intervals").json()["data"]["history"]
    assert len(history) == 2 and raw == old.read_bytes()
    assert restarted.post("/api/intervals/V4-bg-1", json=body, headers=HEADERS).status_code == 200


@pytest.mark.parametrize(
    "segments",
    [
        [dict(start=0.0, end=0.02, rate=1.0, wall_seconds=0.02, samples=2)],
        [dict(start=0.02, end=0.04, rate=1.0, wall_seconds=0.02, samples=2)],
        [
            dict(start=0.0, end=0.01, rate=1.0, wall_seconds=0.01, samples=2),
            dict(start=0.02, end=0.04, rate=1.0, wall_seconds=0.02, samples=2),
        ],
        [dict(start=0.0, end=0.04, rate=1.0, wall_seconds=0.001, samples=2)],
        [dict(start=0.0, end=0.04, rate=2.0, wall_seconds=0.02, samples=2)],
        [dict(start=0.0, end=0.05, rate=1.0, wall_seconds=0.05, samples=2)],
    ],
)
def test_incomplete_skipped_overspeed_or_outside_clip_is_not_absence(pilot, segments):
    _, client, body = setup_view(pilot)
    body["playback_coverage"]["segments"] = segments
    r = client.post("/api/intervals/V4-bg-1", json=body, headers=HEADERS)
    assert r.status_code == 422, r.text
    assert client.get("/api/intervals").json()["data"]["history"] == []


def test_source_binding_and_explicit_attestation_cannot_be_bypassed(pilot):
    _, client, body = setup_view(pilot)
    body["playback_coverage"]["cache_key"] = "0" * 64
    assert client.post("/api/intervals/V4-bg-1", json=body, headers=HEADERS).status_code == 409
    body["playback_coverage"]["cache_key"] = client.get("/api/intervals/V4-bg-1/playback").json()[
        "data"
    ]["cache_key"]
    body["watched_entire"] = False
    assert client.post("/api/intervals/V4-bg-1", json=body, headers=HEADERS).status_code == 422


def test_partial_evidence_is_valid_for_positive_findings_but_does_not_certify_absence(pilot):
    _, client, body = setup_view(pilot)
    body["decision"] = "punch_present"
    body["playback_coverage"]["segments"][0].update(end=0.02, wall_seconds=0.02)
    r = client.post("/api/intervals/V4-bg-1", json=body, headers=HEADERS)
    assert r.status_code == 200, r.text
    assert r.json()["data"]["status"] == "needs_event_labeling"
    assert r.json()["data"]["viewing_validation"]["complete"] is False


def test_rate_changes_can_complete_the_viewing_interval(pilot):
    _, client, body = setup_view(pilot)
    body["playback_coverage"]["segments"] = [
        dict(start=0.0, end=0.02, rate=1.0, wall_seconds=0.02, samples=2),
        dict(start=0.02, end=0.04, rate=0.25, wall_seconds=0.08, samples=2),
    ]
    assert client.post("/api/intervals/V4-bg-1", json=body, headers=HEADERS).status_code == 200
