"""Temporary synthetic reviews only; no real human ledger is written by these tests."""

import json
import shutil
import uuid
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_dashboard_api import media as media  # noqa: F401
from test_dashboard_services import publish
from test_dashboard_services import workspace as workspace  # noqa: F401
from test_interval_events import events as events  # noqa: F401
from test_interval_review import HEADERS
from test_interval_review import pilot as pilot  # noqa: F401

from boxing_motion_ai.api.app import create_app
from boxing_motion_ai.data.interval_pilot import GROUPS, ROLES
from boxing_motion_ai.data.localization_evidence import canonical_events, snapshot, supervision
from boxing_motion_ai.services.artifacts import ServiceError
from boxing_motion_ai.services.localization_review import LocalizationDecision
from boxing_motion_ai.services.review_workspace import encode, sha


@pytest.fixture(params=["V6", "V5"])
def localization(events, workspace, request):
    root, _ = events
    v = request.param
    source = workspace[1]["sources"][v]
    v4 = workspace[1]["sources"]["V4"]
    directory = root / source["dataset"]
    shutil.copyfile(root / v4["dataset"] / "poses.npz", directory / "poses.npz")
    video = root / f"data/raw/boxingvi/RGB_videos/{v}.mkv"
    shutil.copyfile(root / "data/raw/boxingvi/RGB_videos/V4.mkv", video)
    prep = json.loads((root / v4["dataset"] / "preparation.json").read_text())
    prep.update(
        video_id=v,
        schema_version="v5-preparation-1.0" if v == "V5" else "recording-preparation-1.1",
    )
    (directory / "preparation.json").write_bytes(encode(prep))
    for name in ("preparation.json", "poses.npz"):
        source["inputs"][name] = sha((directory / name).read_bytes())
    if v == "V5":
        rows = [json.loads(x) for x in (directory / "candidates.jsonl").read_text().splitlines()]
        rows[0]["array_slice_half_open"] = [2, 3]
        (directory / "candidates.jsonl").write_text("\n".join(map(json.dumps, rows)))
        source["inputs"]["candidates.jsonl"] = sha((directory / "candidates.jsonl").read_bytes())
    publish(workspace)
    entry = json.loads(
        (root / "reports/boxingvi/interval-review-pilot-v1/manifest.json").read_text()
    )["entries"][0]
    entry.update(
        id=f"{v}-motion-101",
        recording=v,
        role=ROLES[v],
        purpose="localization_review",
        source_group_guard=GROUPS[v],
        source_inputs=source["inputs"],
    )
    manifest = dict(schema_version="interval-pilot-1.0", roles=ROLES, entries=[entry])
    path = root / "reports/boxingvi/localization-review-v1/manifest.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(encode(manifest))
    (root / "configs/localization_review_v1.json").write_bytes(
        encode(dict(path=str(path.relative_to(root)), sha256=sha(path.read_bytes())))
    )
    client = TestClient(create_app(root), base_url="http://127.0.0.1:8000")
    return root, client, entry


def mutate(client, entry, **payload):
    url = f"/api/localization-events/{entry['id']}"
    payload = dict(
        request_id=str(uuid.uuid4()),
        expected_head=client.get(url).json()["data"]["head"],
        **payload,
    )
    return client.post(url, json=payload, headers=HEADERS), payload


def test_empty_completion_explicit_visibility_and_full_view(localization):
    root, client, entry = localization
    old = root / "reports/boxingvi/interval-events-v1"
    before = {str(p): p.read_bytes() for p in old.rglob("*.json")}
    completion = dict(
        operation="complete",
        watched_entire=True,
        all_punches_labeled=True,
        no_punches_observed=True,
        inspected_frames=[1, 2, 3, 4],
    )
    assert mutate(client, entry, **completion)[0].status_code == 422
    assert (
        mutate(client, entry, **(completion | dict(visible_continuous=True, inspected_frames=[1])))[
            0
        ].status_code
        == 422
    )
    response, payload = mutate(client, entry, **completion, visible_continuous=True)
    assert response.status_code == 200, response.text
    assert response.json()["data"]["complete"]
    assert response.json()["data"]["background_inferred"] is False
    url = f"/api/localization-events/{entry['id']}"
    assert client.post(url, json=payload, headers=HEADERS).status_code == 200
    restarted = TestClient(create_app(root), base_url="http://127.0.0.1:8000")
    assert restarted.get(url).json()["data"]["complete"]
    assert before == {str(p): p.read_bytes() for p in old.rglob("*.json")}
    assert client.get(f"/api/intervals/{entry['id']}/events").status_code == 404


def test_unknown_type_uncertain_ranges_exact_frames_and_replay(localization):
    _, client, entry = localization
    event = dict(
        event_id=str(uuid.uuid4()),
        start=1,
        stop=3,
        kind="uncertain",
        punch_type="unknown",
        type_uncertain=True,
    )
    assert (
        mutate(client, entry, operation="upsert", event=event, inspected_frames=[1])[0].status_code
        == 422
    )
    response, stale = mutate(
        client, entry, operation="upsert", event=event, inspected_frames=[1, 2]
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"]["events"][0]["endpoint_pts"]["stop_exclusive"]["pts"] == 120
    response, _ = mutate(
        client,
        entry,
        operation="complete",
        watched_entire=True,
        all_punches_labeled=True,
        inspected_frames=[1, 2, 3, 4],
    )
    assert response.status_code == 200
    state = client.get(f"/api/localization-events/{entry['id']}").json()["data"]
    contract = supervision(entry, state["events"], state["complete"])
    assert contract["supervision_runs"][0]["activity"] is None
    assert contract["supervision_runs"][-1]["activity"] == 0
    stale.update(request_id=str(uuid.uuid4()))
    assert (
        client.post(
            f"/api/localization-events/{entry['id']}", json=stale, headers=HEADERS
        ).status_code
        == 409
    )
    response, _ = mutate(client, entry, operation="remove", event_id=event["event_id"])
    assert response.status_code == 200 and not response.json()["data"]["complete"]
    assert client.get("/api/localization-events/V9-motion-101").status_code == 404
    playback = client.get(f"/api/intervals/{entry['id']}/playback?pilot=localization-v1")
    assert playback.status_code == 200, playback.text
    assert playback.json()["data"]["video_url"].endswith("?pilot=localization-v1")
    if entry["recording"] == "V5":
        detail = client.get(f"/api/intervals/{entry['id']}?pilot=localization-v1").json()["data"]
        assert len(detail["annotations"]) == 1


def test_unknown_censoring_and_no_pose_background_assumption():
    entry = dict(start=10, stop=20)
    punch = dict(event_id="a", start=9, stop=14, kind="punch")
    uncertain = dict(event_id="b", start=13, stop=17, kind="uncertain")
    result = supervision(entry, [punch, uncertain], False)
    assert [r["activity"] for r in result["supervision_runs"]] == [1, None, None, None]
    assert not result["pose_masks_are_label_masks"]
    assert result["event_boundaries"][0]["left_censored"]
    assert not result["event_boundaries"][0]["onset_supervised"]
    assert not result["event_boundaries"][0]["end_supervised"]
    assert result["boundary_absence_domain"] == []
    assert supervision(entry, [], False)["supervision_runs"][0]["activity"] is None


def test_dedup_conflicts_and_source_roles():
    anchor = dict(recording="V6", start=10, stop=15, source_id="V6:7", event_id="a")
    p = dict(recording="V6", role=ROLES["V6"], source_group_guard=GROUPS["V6"])
    interval = dict(
        id="queue", provenance=p, events=[anchor | dict(event_id="b")], supervision_runs=[]
    )
    result = canonical_events([], [anchor], [interval])
    assert len(result) == 1 and len(result[0]["origins"]) == 2
    wrong = deepcopy(interval)
    wrong["events"][0]["stop"] = 16
    with pytest.raises(ServiceError, match="conflicting_anchor"):
        canonical_events([], [anchor], [wrong])
    wrong = deepcopy(interval)
    wrong["supervision_runs"] = [dict(start=11, stop=13, activity=0)]
    with pytest.raises(ServiceError, match="human_positive_background_conflict"):
        canonical_events([], [anchor], [wrong])
    for v in ["V5", "V7", "V8", "V9", "V10"]:
        wrong = deepcopy(interval)
        wrong["provenance"]["recording"] = v
        with pytest.raises(ServiceError):
            canonical_events([], [], [wrong])


def test_snapshot_exclusive_backup_and_no_outside_reports(tmp_path):
    p = tmp_path / "reports/original.json"
    p.parent.mkdir()
    p.write_bytes(b"original human evidence")
    result = dict(source_receipts={"reports/original.json": sha(p.read_bytes())})
    out = tmp_path / "reports/new-snapshot"
    snapshot(tmp_path, out, result)
    assert (out / "review-backup/reports/original.json").read_bytes() == p.read_bytes()
    with pytest.raises(ServiceError):
        snapshot(tmp_path, out, result)
    with pytest.raises(ServiceError):
        snapshot(tmp_path, tmp_path / "outside", result)


def test_no_human_claims_from_bad_payload():
    with pytest.raises(ValidationError):
        LocalizationDecision.model_validate(
            dict(
                request_id=str(uuid.uuid4()),
                expected_head="a" * 64,
                operation="upsert",
                no_punches_observed=True,
                event=dict(
                    event_id=str(uuid.uuid4()),
                    start=1,
                    stop=3,
                    kind="uncertain",
                    punch_type="jab",
                    type_uncertain=False,
                ),
            )
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(recording="V7", id="V7-motion-101"),
        dict(recording="V9", id="V9-motion-101"),
        dict(role="evaluation_reserved"),
        dict(source_group_guard=["V6"]),
    ],
)
def test_localization_registry_rejects_reserved_roles_and_groups(localization, change):
    root, client, entry = localization
    if change == dict(source_group_guard=["V6"]) and entry["recording"] == "V5":
        change = dict(source_group_guard=["V1", "V6"])
    path = root / "reports/boxingvi/localization-review-v1/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["entries"][0].update(change)
    path.write_bytes(encode(manifest))
    (root / "configs/localization_review_v1.json").write_bytes(
        encode(dict(path=str(path.relative_to(root)), sha256=sha(path.read_bytes())))
    )
    assert client.get("/api/intervals?pilot=localization-v1").status_code == 422
