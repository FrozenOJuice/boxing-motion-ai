"""Pilot isolation preserves old decisions; all writes here use temporary synthetic footage."""

import json
import uuid

from fastapi.testclient import TestClient
from test_dashboard_api import media as media  # noqa: F401
from test_dashboard_services import workspace as workspace  # noqa: F401
from test_interval_review import HEADERS, payload
from test_interval_review import pilot as pilot  # noqa: F401

from boxing_motion_ai.api.app import create_app
from boxing_motion_ai.services.review_workspace import encode, sha


def install_motion(pilot):
    root, client, original = pilot
    raw = encode(
        original | {"entries": [original["entries"][0] | {"id": "V4-motion-1", "context_end": 0}]}
    )
    path = root / "reports/boxingvi/interval-review-motion-v2/manifest.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    (root / "configs/interval_review_motion_v2.json").write_bytes(
        encode(dict(path=str(path.relative_to(root)), sha256=sha(raw)))
    )
    return root, client


def test_independent_heads_histories_and_restart(pilot):
    root, client = install_motion(pilot)
    first = client.post(
        "/api/intervals/V4-bg-1", json=payload(client, "punch_present"), headers=HEADERS
    )
    assert first.status_code == 200
    original = (root / "reports/boxingvi/interval-reviews-v1/v00000001/review.json").read_bytes()
    motion = client.get("/api/intervals?pilot=motion-v2").json()["data"]
    assert motion["history"] == []
    body = payload(client) | {"request_id": str(uuid.uuid4()), "expected_head": motion["head"]}
    result = client.post("/api/intervals/V4-motion-1?pilot=motion-v2", json=body, headers=HEADERS)
    assert result.status_code == 200, result.text
    assert (
        client.post(
            "/api/intervals/V4-motion-1?pilot=motion-v2", json=body, headers=HEADERS
        ).status_code
        == 200
    )
    restarted = TestClient(create_app(root), base_url="http://127.0.0.1:8000")
    assert len(restarted.get("/api/intervals?pilot=motion-v2").json()["data"]["history"]) == 1
    assert len(restarted.get("/api/intervals").json()["data"]["history"]) == 1
    assert (
        original
        == (root / "reports/boxingvi/interval-reviews-v1/v00000001/review.json").read_bytes()
    )
    assert restarted.get("/api/intervals/V4-motion-1").status_code == 404
    assert restarted.get("/api/intervals/V4-bg-1?pilot=motion-v2").status_code == 404
    assert restarted.get("/api/intervals?pilot=../../other").status_code == 422
    assert restarted.get("/api/intervals/V9-motion-1?pilot=motion-v2").status_code == 404


def test_motion_clip_urls_keep_the_pilot_binding(pilot):
    _, client = install_motion(pilot)
    r = client.get("/api/intervals/V4-motion-1/playback?pilot=motion-v2")
    assert r.status_code == 200, r.text
    url = r.json()["data"]["video_url"]
    assert url.endswith("?pilot=motion-v2")
    assert client.get(url).status_code == 200
    assert client.get("/api/intervals?pilot=motion-v2").json()["data"]["history"] == []


def test_motion_pilot_cannot_register_development_or_evaluation_entries(pilot):
    root, client = install_motion(pilot)
    path = root / "reports/boxingvi/interval-review-motion-v2/manifest.json"
    m = json.loads(path.read_text())
    m["entries"][0].update(id="V5-motion-1", recording="V5", role="development")
    path.write_bytes(encode(m))
    (root / "configs/interval_review_motion_v2.json").write_bytes(
        encode(dict(path=str(path.relative_to(root)), sha256=sha(path.read_bytes())))
    )
    assert client.get("/api/intervals?pilot=motion-v2").status_code == 422
