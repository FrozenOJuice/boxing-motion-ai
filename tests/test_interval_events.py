"""Real tiny synthetic video, immutable human-event transactions; no actual reviews written."""

import json
import uuid
from fractions import Fraction

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_dashboard_api import media as media  # noqa: F401
from test_dashboard_services import pin, publish
from test_dashboard_services import workspace as workspace  # noqa: F401
from test_interval_review import HEADERS, payload
from test_interval_review import pilot as pilot  # noqa: F401

from boxing_motion_ai.api.app import create_app
from boxing_motion_ai.services.interval_review import interval_timing
from boxing_motion_ai.services.review_workspace import encode, sha

URL = "/api/intervals/V4-bg-1/events"


@pytest.fixture
def events(pilot, workspace):
    root, _, manifest = pilot
    video = root / "data/raw/boxingvi/RGB_videos/V4.mkv"
    with av.open(str(video), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for i in range(8):
            f = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), i * 20, dtype=np.uint8), format="rgb24"
            )
            f.pts, f.time_base = i * 40, Fraction(1, 1000)
            for packet in stream.encode(f):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    spec = workspace[1]["sources"]["V4"]
    directory = root / spec["dataset"]
    with np.load(directory / "poses.npz") as z:
        arrays = dict(z)
    arrays["pts"] = np.arange(8) * 40
    arrays["time_base_num"][:] = 1
    arrays["time_base_den"][:] = 1000
    np.savez_compressed(directory / "poses.npz", **arrays)
    spec["inputs"]["poses.npz"] = pin(root, f"{spec['dataset']}/poses.npz")["sha256"]
    prep = json.loads((directory / "preparation.json").read_text())
    prep.update(npz_sha256=spec["inputs"]["poses.npz"], confirmed_scene_cuts=[])
    prep["inputs"]["video_sha256"] = sha(video.read_bytes())
    (directory / "preparation.json").write_bytes(encode(prep))
    spec["inputs"]["preparation.json"] = sha((directory / "preparation.json").read_bytes())
    publish(workspace)
    e = manifest["entries"][0]
    e.update(
        start=1,
        stop=5,
        context_start=0,
        context_end=4,
        source_inputs=spec["inputs"],
        video_sha256=prep["inputs"]["video_sha256"],
        timing=interval_timing(arrays, 1, 5),
    )
    path = root / "reports/boxingvi/interval-review-pilot-v1/manifest.json"
    path.write_bytes(encode(manifest))
    (root / "configs/interval_review_pilot_v1.json").write_bytes(
        encode(dict(path=str(path.relative_to(root)), sha256=sha(path.read_bytes())))
    )
    client = TestClient(create_app(root), base_url="http://127.0.0.1:8000")
    r = client.post(
        "/api/intervals/V4-bg-1",
        json=payload(client, "punch_present") | {"inspected_frames": [1]},
        headers=HEADERS,
    )
    assert r.status_code == 200, r.text
    return root, client


def mutate(client, operation="upsert", **extra):
    body = dict(
        request_id=str(uuid.uuid4()),
        expected_head=client.get(URL).json()["data"]["head"],
        operation=operation,
        **extra,
    )
    return client.post(URL, json=body, headers=HEADERS), body


def punch(start=1, stop=3, **extra):
    return dict(
        event_id=str(uuid.uuid4()),
        start=start,
        stop=stop,
        punch_type="jab",
        type_uncertain=False,
        **extra,
    )


def test_add_overlap_consecutive_edit_remove_restart_and_original_unchanged(events):
    root, client = events
    old = root / "reports/boxingvi/interval-reviews-v1/v00000001/review.json"
    before = old.read_bytes()
    a, b, c = punch(1, 2), punch(2, 4), punch(1, 4)
    for p in (a, b, c):
        r, body = mutate(
            client,
            event=p,
            inspected_frames=[p["start"], p["stop"] - 1]
            if p["stop"] - p["start"] > 1
            else [p["start"]],
        )
        assert r.status_code == 200, r.text
        assert client.post(URL, json=body, headers=HEADERS).status_code == 200
    first = root / "reports/boxingvi/interval-events-v1/V4-bg-1/v00000001/review.json"
    first_bytes = first.read_bytes()
    a.update(punch_type="unknown", type_uncertain=True)
    r, _ = mutate(
        client, event=a
    )  # Unchanged, previously inspected boundaries need not be reinspected.
    assert r.status_code == 200, r.text
    assert r.json()["data"]["events"][0]["endpoint_pts"]["start"]["pts"] == 40
    r, _ = mutate(client, "remove", event_id=b["event_id"])
    assert r.status_code == 200, r.text
    restarted = TestClient(create_app(root), base_url="http://127.0.0.1:8000")
    state = restarted.get(URL).json()["data"]
    assert len(state["events"]) == 2 and len(state["history"]) == 5
    assert state["uncertain_types"] == 1 and state["complete"] is False
    assert first.read_bytes() == first_bytes and old.read_bytes() == before
    assert all(not r["background_inferred"] and not r["training_enabled"] for r in state["history"])


def test_completion_needs_full_coverage_explicit_attestation_and_edits_reset(events):
    _, client = events
    p = punch()
    assert mutate(client, event=p, inspected_frames=[1, 2])[0].status_code == 200
    assert mutate(client, "complete")[0].status_code == 422
    assert (
        mutate(
            client, "complete", watched_entire=True, all_punches_labeled=True, inspected_frames=[1]
        )[0].status_code
        == 422
    )
    clip = client.get("/api/intervals/V4-bg-1/playback").json()["data"]
    evidence = dict(
        schema_version="continuous-viewing-1.0",
        policy="visible-realtime-1.0",
        cache_key=clip["cache_key"],
        interval_id="V4-bg-1",
        segments=[dict(start=0, end=0.2, rate=1, wall_seconds=0.2, samples=4)],
    )
    r, _ = mutate(
        client,
        "complete",
        watched_entire=True,
        all_punches_labeled=True,
        playback_coverage=evidence,
    )
    assert r.status_code == 200, r.text
    assert client.get(URL).json()["data"]["complete"] is True
    assert mutate(client, "remove", event_id=p["event_id"])[0].status_code == 200
    state = client.get(URL).json()["data"]
    assert state["complete"] is False and state["history"][-2]["complete"] is True
    assert (
        mutate(
            client,
            "complete",
            watched_entire=True,
            all_punches_labeled=True,
            playback_coverage=evidence,
        )[0].status_code
        == 422
    )


@pytest.mark.parametrize(
    "event,seen",
    [
        (punch(), []),
        (punch(), [1]),
        (punch(0, 6), [0, 5]),
        (punch(5, 6), [5]),
        (punch() | {"start": True}, [1, 2]),
        (punch() | {"punch_type": "unknown"}, [1, 2]),
    ],
)
def test_invalid_or_uninspected_boundaries_never_persist(events, event, seen):
    _, client = events
    r, _ = mutate(client, event=event, inspected_frames=seen)
    assert r.status_code == 422, r.text
    assert client.get(URL).json()["data"]["history"] == []


def test_stale_writes_and_scope_guards(events):
    _, client = events
    p = punch()
    r, body = mutate(client, event=p, inspected_frames=[1, 2])
    assert r.status_code == 200
    body.update(request_id=str(uuid.uuid4()), event=p | {"punch_type": "cross"})
    assert client.post(URL, json=body, headers=HEADERS).status_code == 409
    for key in ("V9-bg-1", "V5-bg-1", "V6-bg-1", "V4-motion-1", "V4-bg-6", "..%2Fsecret"):
        assert client.get(f"/api/intervals/{key}/events").status_code == 404
    assert client.post(URL, json=body).status_code == 403


def test_edge_crossing_punch_keeps_full_context_bounds_without_clipping(events):
    _, client = events
    r, _ = mutate(client, event=punch(0, 2), inspected_frames=[0, 1])
    assert r.status_code == 200, r.text
    event = r.json()["data"]["events"][0]
    assert event["start"] == 0 and event["stop"] == 2
    assert event["partial_overlap"] is True
    assert event["endpoint_pts"]["start"]["pts"] == 0
    assert event["endpoint_pts"]["stop_exclusive"]["pts"] == 80
