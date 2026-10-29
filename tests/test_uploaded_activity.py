"""Synthetic upload/stream contracts; never opens a registered video or review."""

import json
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import pytest
from test_dashboard_services import workspace as workspace
from test_decoder import frame
from test_detector import source

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
from boxing_motion_ai.analysis.common import sha, write_json  # noqa: E402
from boxing_motion_ai.analysis.jobs import AnalysisError, AnalysisJobs  # noqa: E402
from boxing_motion_ai.detection.activity_bouts import ActivityDecoder  # noqa: E402
from boxing_motion_ai.detection.decoder import DecoderConfig  # noqa: E402
from boxing_motion_ai.detection.model import BinaryDetector  # noqa: E402
from boxing_motion_ai.detection.upload import predict_arrays, validate_timing  # noqa: E402


def group(values):
    d = ActivityDecoder(DecoderConfig())
    for i, p in enumerate(values):
        d.push(frame(i, p))
    return d.finish(len(values), Q(len(values), 25))


@pytest.mark.parametrize(
    "values,events,bouts",
    [
        ([0.1] * 50, 0, 0),
        ([0.1] * 3 + [0.9] * 5 + [0.1] * 20, 1, 1),
        ([0.1] + [0.9] * 4 + [0.1] * 3 + [0.9] * 4 + [0.1] * 20, 2, 1),
        ([0.1] + [0.9] * 4 + [0.1] * 9 + [0.9] * 4 + [0.1] * 20, 2, 2),
    ],
)
def test_activity_not_individual_punches(values, events, bouts):
    result = group(values)
    assert len(result["decoder_events"]) == events
    assert len(result["bouts"]) == bouts
    if bouts:
        first = result["bouts"][0]
        assert first["first_emission"]["frame_index"] == first["estimated_start"]["frame_index"] + 2
        assert not first["right_censored"]
        assert first["verified"] is False


def test_exact_quarter_second_separates_and_candidate_before_deadline_merges():
    # 100 Hz ensures exact 1/4 boundary; candidate activation takes 80 ms.
    for pause, expected in [(24, 1), (25, 2)]:
        d = ActivityDecoder(DecoderConfig())
        values = [0.1] * 3 + [0.9] * 10 + [0.1] * pause + [0.9] * 10 + [0.1] * 40
        for i, p in enumerate(values):
            d.push(frame(i, p, fps=100))
        assert len(d.finish(len(values), Q(len(values), 100))["bouts"]) == expected


@pytest.mark.parametrize("kind", ["missing", "gap", "scene", "segment", "cut"])
def test_grouping_never_bridges_interruptions_and_emissions_are_immutable(kind):
    d = ActivityDecoder(DecoderConfig())
    for i in range(8):
        d.push(frame(i, 0.9 if i < 4 else 0.1))
    prefix = deepcopy(d.updates)
    for i in range(8, 20):
        kwargs = {}
        if kind == "missing" and i == 8:
            kwargs.update(observable=False)
        if kind == "gap":
            kwargs.update(pts=i + 20)
        if kind in ("scene", "segment"):
            kwargs[kind + "_id"] = 1
        if kind == "cut":
            kwargs["discontinuity_before"] = i == 8
        d.push(frame(i, 0.9, **kwargs))
    end = Q(40 if kind == "gap" else 20, 25)
    result = d.finish(20, end)
    assert len(result["bouts"]) == 2
    assert result["bouts"][0]["right_censored"]
    assert result["bouts"][1]["left_censored"]
    assert d.updates[: len(prefix)] == prefix


def prepared(raw):
    frames = [
        dict(
            frame_index=int(i),
            pts=int(p),
            time_base_num=int(n),
            time_base_den=int(d),
            timestamp_exact=str(Q(int(p * n), int(d))),
        )
        for i, p, n, d in zip(
            raw["frame_index"], raw["pts"], raw["time_base_num"], raw["time_base_den"], strict=True
        )
    ]
    for row in frames:
        row["media_time"] = float(Q(row["timestamp_exact"]) - Q(frames[0]["timestamp_exact"]))
    duration = Q(frames[-1]["timestamp_exact"]) + Q(1001, 30000) - Q(frames[0]["timestamp_exact"])
    return dict(
        duration_seconds=float(duration),
        frame_count=len(frames),
        frames=frames,
        width=640,
        height=480,
        stop_timestamp_exact=str(Q(frames[-1]["timestamp_exact"]) + Q(1001, 30000)),
    )


def test_full_uploaded_stream_native_pts_chunk_equivalence_and_resets():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    raw = source(70)
    raw["pts"] += 60000  # Nonzero native source origin.
    raw["pts"][40:] += 9000
    raw["coordinates"][20, 11] = np.nan
    raw["valid_mask"][20, 11] = False
    video = prepared(raw)
    video["confirmed_scene_cuts"] = [55]
    model = BinaryDetector().eval()
    a = predict_arrays(model, raw, video, "synthetic", DecoderConfig(), chunk_size=7)
    b = predict_arrays(model, raw, video, "synthetic", DecoderConfig(), chunk_size=64)
    assert a == b
    assert len(a["frames"]) == 70
    assert a["frames"][20]["probability_punch"] is None
    assert a["frames"][0]["timestamp_exact"] == "2"
    assert a["observable_frames"] == 69
    assert {r["frame_index"] for r in a["resets"]} == {0, 20, 21, 40, 55}
    assert {r["reason"] for r in a["unobservable_intervals"]} == {
        "timestamp_gap",
        "missing_or_unreliable_pose",
    }
    assert a["model_streaming_batch_max_error"] < 1e-5
    assert a["decoder_streaming_batch_equal"]
    assert all("reviewed" not in r for r in a["frames"])
    # Future movement cannot alter previously emitted rows/events (fixed source clock).
    changed = deepcopy(raw)
    changed["coordinates"][60:, 15, 0] += 0.2
    c = predict_arrays(model, changed, video, "synthetic", DecoderConfig())
    assert a["frames"][:60] == c["frames"][:60]
    broken = deepcopy(video)
    broken["frames"][2]["pts"] += 1
    with pytest.raises(ValueError, match="timing"):
        validate_timing(raw, broken)


def test_all_unobservable_is_not_background_and_does_not_invoke_model():
    raw = source(15)
    raw["coordinates"][:] = np.nan
    raw["valid_mask"][:] = False
    raw["pose_status"][:] = 2
    a = predict_arrays(None, raw, prepared(raw), "synthetic", DecoderConfig())
    assert not a["bouts"] and a["observable_frames"] == 0
    assert all(r["probability_punch"] is None for r in a["frames"])
    assert a["unobservable_intervals"][0]["stop_frame_exclusive"] == 15


def ready(jobs):
    key, directory = jobs.reserve("synthetic.mp4")
    write_json(directory / "job.json", dict(id=key, status="ready"))
    write_json(directory / "video.json", dict(frame_count=3))
    write_json(directory / "pose.json", {})
    jobs.active.remove(key)
    return key, directory


def test_scan_cache_idempotence_failure_recovery_stale_cross_video_and_busy(tmp_path, monkeypatch):
    jobs = AnalysisJobs(tmp_path)
    key, directory = ready(jobs)
    fingerprint = ["1" * 64]
    monkeypatch.setattr(jobs, "_activity_identity", lambda k: fingerprint[0])
    calls = []
    monkeypatch.setattr(jobs.executor, "submit", lambda *a: calls.append(a))
    state = jobs.scan(key)
    assert jobs.scan(key) == state and len(calls) == 1
    with pytest.raises(AnalysisError, match="Wait"):
        jobs.remove(key)
    with pytest.raises(AnalysisError, match="busy"):
        jobs.classify(key, 0, 2)
    monkeypatch.setattr(
        jobs, "run", lambda *a, **k: (_ for _ in ()).throw(AnalysisError("Failure"))
    )
    jobs.process_scan(key, fingerprint[0], state["request_id"])
    assert jobs.activity(key)["status"] == "failed"
    assert "Failure" in jobs.activity(key)["message"]
    state = jobs.scan(key)

    def worker(*args, **kwargs):
        write_json(
            directory / f"activity-{fingerprint[0]}.json",
            dict(
                schema_version="uploaded-activity-detection-1.0",
                video_id=key,
                input_fingerprint=fingerprint[0],
                bouts=[],
            ),
        )

    monkeypatch.setattr(jobs, "run", worker)
    jobs.process_scan(key, fingerprint[0], state["request_id"])
    assert jobs.scan(key)["status"] == "ready" and len(calls) == 2
    fingerprint[0] = "2" * 64
    assert jobs.activity(key)["status"] == "failed"
    jobs.scan(key)
    jobs.process_scan(key, fingerprint[0], "retry")
    output = directory / f"activity-{fingerprint[0]}.json"
    result = json.loads(output.read_text())
    result["video_id"] = "b" * 32
    write_json(output, result)
    state = jobs.read(key, "activity-state.json")
    state["result_sha256"] = sha(output)
    write_json(directory / "activity-state.json", state)
    assert jobs.activity(key)["status"] == "failed"
    state.update(status="processing")
    write_json(directory / "activity-state.json", state)
    assert "interrupted" in jobs.activity(key)["message"]


def test_real_pin_integrity_without_accessing_any_training_or_review_recordings():
    from boxing_motion_ai.analysis.activity_contract import activity_pin

    root = Path(__file__).parents[1]
    checkpoint = root / "reports/boxingvi/detector-smoke-v1-verified/checkpoint.pt"
    if not checkpoint.exists():
        pytest.skip("Optional local smoke checkpoint is not distributed in Git")
    assert activity_pin(root)["checkpoint"]["sha256"] == sha(checkpoint)


def test_missing_checkpoint_and_corrupt_binding_fail_closed(tmp_path):
    from boxing_motion_ai.analysis.activity_contract import activity_pin

    (tmp_path / "configs").mkdir()
    binding = dict(
        schema_version="uploaded-activity-pin-1.0",
        checkpoint=dict(path="missing.pt", sha256="0" * 64),
        source_sha256={},
    )
    write_json(tmp_path / "configs/uploaded_activity_v1.json", binding)
    with pytest.raises((OSError, ValueError)):
        activity_pin(tmp_path)
    (tmp_path / "missing.pt").write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="changed"):
        activity_pin(tmp_path)


def test_upload_scan_api_is_local_only_and_upload_scoped(workspace, monkeypatch):
    from fastapi.testclient import TestClient

    from boxing_motion_ai.api.app import create_app

    app = create_app(workspace[0])
    jobs = app.state.analysis
    key, _ = ready(jobs)
    monkeypatch.setattr(jobs, "_activity_identity", lambda k: "f" * 64)
    submissions = []
    monkeypatch.setattr(jobs.executor, "submit", lambda *a: submissions.append(a))
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    assert client.get(f"/api/analyze/videos/{key}/activity").json()["status"] == "not_started"
    assert client.post(f"/api/analyze/videos/{key}/scan", json={}).status_code == 403
    headers = {"origin": "http://127.0.0.1:8000", "x-review-client": "dashboard-v0.2"}
    for _ in range(2):
        response = client.post(f"/api/analyze/videos/{key}/scan", json={}, headers=headers)
        assert response.status_code == 202 and response.json()["video_id"] == key
    assert len(submissions) == 1
    assert client.get("/api/analyze/videos/V9/activity").status_code == 404
    assert client.post("/api/analyze/videos/V9/scan", json={}, headers=headers).status_code == 404


def test_fingerprint_changes_with_pose_metadata_source_and_video_identity(tmp_path):
    from boxing_motion_ai.analysis.activity_contract import inputs

    root = Path(__file__).parents[1]
    pin = json.loads((root / "configs/uploaded_activity_v1.json").read_text())
    for name in pin["source_sha256"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    checkpoint = tmp_path / pin["checkpoint"]["path"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"synthetic checkpoint: identity tests only, never inference")
    pin["checkpoint"]["sha256"] = sha(checkpoint)
    write_json(tmp_path / "configs/uploaded_activity_v1.json", pin)
    directory = tmp_path / "upload"
    directory.mkdir()
    (directory / "source.video").write_bytes(b"synthetic video")
    (directory / "poses.npz").write_bytes(b"synthetic poses")
    source_hash = sha(directory / "source.video")
    video = dict(schema_version="assisted-video-1.0", source_sha256=source_hash)
    poses = dict(
        schema_version="assisted-poses-1.0",
        source_sha256=source_hash,
        poses_sha256=sha(directory / "poses.npz"),
        joint_order=pin["joint_order"],
        pose_model_sha256=pin["pose_model_sha256"],
        threshold=0.5,
        mode="image",
    )
    write_json(directory / "video.json", video)
    write_json(directory / "pose.json", poses)
    a = inputs(tmp_path, directory, "a")[0]
    assert inputs(tmp_path, directory, "a")[0] == a
    assert inputs(tmp_path, directory, "b")[0] != a
    poses["pose_seconds"] = 2.0
    write_json(directory / "pose.json", poses)
    assert inputs(tmp_path, directory, "a")[0] != a
    (directory / "poses.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance"):
        inputs(tmp_path, directory, "a")
    (directory / "source.video").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="provenance"):
        inputs(tmp_path, directory, "a")


def test_corrupt_state_and_worker_disk_failure_do_not_permanently_lock_scans(tmp_path, monkeypatch):
    from boxing_motion_ai.analysis import jobs as module

    jobs = AnalysisJobs(tmp_path)
    key, directory = ready(jobs)
    monkeypatch.setattr(jobs, "_activity_identity", lambda k: "a" * 64)
    monkeypatch.setattr(jobs.executor, "submit", lambda *a: None)
    (directory / "activity-state.json").write_text("broken json")
    assert jobs.activity(key)["status"] == "failed"
    state = jobs.scan(key)
    monkeypatch.setattr(jobs, "run", lambda *a, **k: (_ for _ in ()).throw(ValueError("fail")))
    monkeypatch.setattr(module, "write_json", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        jobs.process_scan(key, state["input_fingerprint"], state["request_id"])
    assert key not in jobs.scanning
    assert "interrupted" in jobs.activity(key)["message"]
