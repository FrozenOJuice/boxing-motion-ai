"""Synthetic-only activity references; real review history is never written by tests."""

import copy
import json
import shutil
import uuid

import pytest
from fastapi.testclient import TestClient
from test_dashboard_api import media as media  # noqa: F401
from test_dashboard_services import pin, publish
from test_dashboard_services import workspace as workspace  # noqa: F401
from test_interval_events import events as events  # noqa: F401
from test_interval_review import HEADERS
from test_interval_review import pilot as pilot  # noqa: F401

from boxing_motion_ai.api.app import create_app
from boxing_motion_ai.services import activity_evaluation as module
from boxing_motion_ai.services.review_workspace import encode

URL = "/api/activity-evaluation/V7-eval-1"


@pytest.fixture
def activity(events, workspace, monkeypatch):
    root, _ = events
    # Preserve the synthetic previous ledger; repin a separate synthetic evaluation workspace.
    (root / "reports/boxingvi/interval-reviews-v1").rename(root / "synthetic-old-ledger")
    spec = workspace[1]["sources"]["V7"]
    old = workspace[1]["sources"]["V4"]
    for name in ("poses.npz", "preparation.json"):
        shutil.copyfile(root / old["dataset"] / name, root / spec["dataset"] / name)
    prep_path = root / spec["dataset"] / "preparation.json"
    prep = json.loads(prep_path.read_text())
    prep["video_id"] = "V7"
    prep_path.write_bytes(encode(prep))
    for name in ("poses.npz", "preparation.json"):
        spec["inputs"][name] = pin(root, f"{spec['dataset']}/{name}")["sha256"]
    shutil.copyfile(
        root / "data/raw/boxingvi/RGB_videos/V4.mkv",
        root / "data/raw/boxingvi/RGB_videos/V7.mkv",
    )
    publish(workspace)
    path = root / "reports/boxingvi/interval-review-pilot-v1/manifest.json"
    manifest = json.loads(path.read_text())
    entry = copy.deepcopy(manifest["entries"][0])
    entry.update(
        id="V7-eval-1",
        recording="V7",
        role="evaluation_reserved",
        purpose="evaluation_block",
        source_group_guard=["V7", "V8"],
        source_inputs=spec["inputs"],
    )
    manifest["entries"] = [entry]
    manifest["roles"]["V7"] = "evaluation_reserved"
    path.write_bytes(encode(manifest))
    ref = pin(root, str(path.relative_to(root)))
    (root / "configs/interval_review_pilot_v1.json").write_bytes(encode(ref))
    protocol = json.loads(open("configs/activity_eval_protocol_v1.json").read())
    protocol.update(interval_ids=["V7-eval-1"], interval_manifest=ref)
    monkeypatch.setattr(module, "SCOPE", ("V7-eval-1",))
    for name in ("checkpoint", "matching_code", "decoder_config"):
        target = root / "configs" / f"synthetic-{name}"
        target.write_text("synthetic fixture, never executed")
        protocol[name] = pin(root, str(target.relative_to(root)))
    (root / module.PROTOCOL).write_bytes(encode(protocol))
    client = TestClient(create_app(root), base_url="http://127.0.0.1:8000")
    return root, client


def change(client, operation="upsert", **extra):
    body = dict(
        request_id=str(uuid.uuid4()),
        expected_head=client.get(URL).json()["data"]["head"],
        operation=operation,
        **extra,
    )
    return client.post(URL, json=body, headers=HEADERS), body


def region(kind="activity", start=1, stop=3):
    return dict(event_id=str(uuid.uuid4()), kind=kind, start=start, stop=stop)


def finish(client, **extra):
    return change(
        client,
        "complete",
        watched_entire=True,
        all_activity_labeled=True,
        inspected_frames=[1, 2, 3, 4],
        **extra,
    )[0]


def test_manual_regions_persist_revise_remove_and_invalidate_completion(activity):
    root, client = activity
    before = (root / "synthetic-old-ledger/v00000001/review.json").read_bytes()
    p = region()
    r, body = change(client, event=p, inspected_frames=[1, 2])
    assert r.status_code == 200, r.text
    assert client.post(URL, json=body, headers=HEADERS).status_code == 200
    assert finish(client).status_code == 200
    restarted = TestClient(create_app(root), base_url="http://127.0.0.1:8000")
    state = restarted.get(URL).json()["data"]
    assert state["complete"] and state["primary_eligible"] and state["predictions_hidden"]
    assert state["events"][0]["left_censored"]
    assert state["events"][0]["endpoint_pts"]["start"]["pts"] == 40
    r, _ = change(client, event=p | {"kind": "uncertain"})
    assert r.status_code == 200, r.text
    assert finish(client).status_code == 200
    state = client.get(URL).json()["data"]
    assert state["complete"] and not state["primary_eligible"] and state["uncertain_sections"] == 1
    assert change(client, "remove", event_id=p["event_id"])[0].status_code == 200
    assert not client.get(URL).json()["data"]["complete"]
    assert (root / "synthetic-old-ledger/v00000001/review.json").read_bytes() == before
    assert all(r["source"] == "manual" and not r["training_enabled"] for r in state["history"])


def test_empty_requires_full_viewing_and_explicit_no_activity(activity):
    _, client = activity
    assert finish(client).status_code == 422
    assert (
        change(
            client,
            "complete",
            watched_entire=True,
            all_activity_labeled=True,
            no_activity_observed=True,
            inspected_frames=[1],
        )[0].status_code
        == 422
    )
    assert finish(client, no_activity_observed=True).status_code == 200
    assert client.get(URL).json()["data"]["primary_eligible"]


def test_invalid_boundaries_coverage_scope_protocol_and_stale_writes(activity):
    root, client = activity
    for p, seen in (
        (region(), []),
        (region(start=0), [0, 2]),
        (region(stop=6), [1, 5]),
        (region() | {"start": True}, [1, 2]),
    ):
        assert change(client, event=p, inspected_frames=seen)[0].status_code == 422
    r, body = change(client, event=region(), inspected_frames=[1, 2])
    assert r.status_code == 200
    body["request_id"] = str(uuid.uuid4())
    assert client.post(URL, json=body, headers=HEADERS).status_code == 409
    for key in ("V9-eval-1", "V5-eval-1", "V4-bg-1", "V7-eval-3", "..%2Fsecret"):
        assert client.get(f"/api/activity-evaluation/{key}").status_code == 404
    assert client.post(URL, json=body).status_code == 403
    (root / "configs/synthetic-decoder_config").write_text("changed")
    assert client.get(URL).status_code == 409


def test_close_activity_regions_require_combining_unless_explicitly_uncertain(activity):
    _, client = activity
    for p in (region(start=1, stop=2), region(start=3, stop=5)):
        r, _ = change(
            client,
            event=p,
            inspected_frames=[p["start"], p["stop"] - 1]
            if p["stop"] - p["start"] > 1
            else [p["start"]],
        )
        assert r.status_code == 200, r.text
    assert finish(client).json()["issues"] == ["combine_activity_bouts_under_quarter_second"]
    r, _ = change(client, event=region("uncertain", 2, 4), inspected_frames=[2, 3])
    assert r.status_code == 200
    assert finish(client).status_code == 200
    assert not client.get(URL).json()["data"]["primary_eligible"]


def test_freeze_requires_completion_backs_up_history_and_detects_later_edits(activity):
    from boxing_motion_ai.data.activity_reference import OUTPUT, collect, freeze, validate
    from boxing_motion_ai.services.artifacts import ServiceError

    root, client = activity
    assert not collect(root)["summary"]["ready_to_freeze"]
    with pytest.raises(ServiceError, match="activity_reviews_incomplete"):
        freeze(root)
    assert not (root / OUTPUT).exists()
    p = region()
    assert change(client, event=p, inspected_frames=[1, 2])[0].status_code == 200
    assert finish(client).status_code == 200
    frozen = freeze(root)
    assert frozen == validate(root)
    original = root / module.DIRECTORY / "V7-eval-1/v00000001/review.json"
    backup = root / OUTPUT / "review-backup" / module.DIRECTORY / "V7-eval-1/v00000001/review.json"
    assert backup.read_bytes() == original.read_bytes()
    with pytest.raises(ServiceError, match="activity_reference_already_exists"):
        freeze(root)
    assert change(client, event=p | {"kind": "uncertain"})[0].status_code == 200
    with pytest.raises(ServiceError, match="activity_reference_stale"):
        validate(root)


def test_playback_completion_reuses_visible_coverage_contract(activity):
    _, client = activity
    clip = client.get("/api/intervals/V7-eval-1/playback").json()["data"]
    evidence = dict(
        schema_version="continuous-viewing-1.0",
        policy="visible-realtime-1.0",
        cache_key=clip["cache_key"],
        interval_id="V7-eval-1",
        segments=[dict(start=0, end=0.2, rate=1, wall_seconds=0.2, samples=4)],
    )
    flags = dict(watched_entire=True, all_activity_labeled=True, no_activity_observed=True)
    r, _ = change(client, "complete", playback_coverage=evidence, **flags)
    assert r.status_code == 200, r.text
    evidence["cache_key"] = "0" * 64
    assert change(client, "complete", playback_coverage=evidence, **flags)[0].status_code == 409
