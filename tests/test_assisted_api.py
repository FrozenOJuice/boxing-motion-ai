"""Local upload boundary and scoped job lifecycle; synthetic files only."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from test_dashboard_services import workspace as workspace  # noqa: E402

from boxing_motion_ai.analysis.common import MAX_BYTES, write_json  # noqa: E402
from boxing_motion_ai.analysis.jobs import AnalysisError, AnalysisJobs  # noqa: E402
from boxing_motion_ai.api.app import create_app  # noqa: E402

HEADERS = {
    "origin": "http://127.0.0.1:8000",
    "x-review-client": "dashboard-v0.2",
    "content-type": "application/octet-stream",
    "x-upload-name": "test.mp4",
}


def test_upload_middleware_limits_and_no_v9_resolution(workspace, monkeypatch):
    root = workspace[0]
    app = create_app(root)
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    monkeypatch.setattr(app.state.analysis.executor, "submit", lambda *a: None)
    assert client.post("/api/analyze/uploads", content=b"video").status_code == 403
    assert (
        client.post(
            "/api/analyze/uploads",
            content=b"video",
            headers={**HEADERS, "origin": "https://evil.example"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/analyze/uploads",
            content=b"v",
            headers={**HEADERS, "content-length": str(MAX_BYTES + 1)},
        ).status_code
        == 413
    )
    assert (
        client.post(
            "/api/analyze/uploads",
            content=b"video",
            headers={**HEADERS, "x-upload-name": "bad.html"},
        ).status_code
        == 422
    )
    assert client.get("/api/analyze/videos/V9").status_code == 404
    assert client.get("/api/analyze/videos").json() == {"videos": []}
    response = client.post("/api/analyze/uploads", content=b"synthetic", headers=HEADERS)
    assert response.status_code == 202
    key = response.json()["id"]
    assert (root / ".analysis-local" / key / "source.video").read_bytes() == b"synthetic"
    assert client.get(f"/api/analyze/videos/{key}").json()["status"] == "queued"
    assert (
        client.post(
            f"/api/analyze/videos/{key}/classify",
            json=dict(start_frame=0, stop_frame_exclusive=2),
            headers={**HEADERS, "content-type": "application/json"},
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/api/analyze/videos/{key}/remove",
            json={},
            headers={**HEADERS, "content-type": "application/json"},
        ).status_code
        == 409
    )
    app.state.analysis.active.remove(key)
    assert "restart" in client.get(f"/api/analyze/videos/{key}").json()["message"]
    assert (
        client.post(
            f"/api/analyze/videos/{key}/remove",
            json={},
            headers={**HEADERS, "content-type": "application/json"},
        ).status_code
        == 200
    )
    assert not (root / ".analysis-local" / key).exists()


def test_incomplete_upload_cleaned_and_symlinks_rejected(workspace, monkeypatch, tmp_path):
    app = create_app(workspace[0])
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    response = client.post(
        "/api/analyze/uploads", content=b"short", headers={**HEADERS, "content-length": "100"}
    )
    assert response.status_code == 422
    assert client.get("/api/analyze/videos").json()["videos"] == []
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    jobs = AnalysisJobs(isolated)
    outside = tmp_path / "outside"
    outside.mkdir()
    jobs.storage.symlink_to(outside, target_is_directory=True)
    with pytest.raises(AnalysisError, match="Unsafe"):
        jobs.reserve("test.mp4")
    assert not list(outside.iterdir())


def test_bounded_jobs_failure_and_scoped_cleanup(tmp_path, monkeypatch):
    jobs = AnalysisJobs(tmp_path)
    a, _ = jobs.reserve("a.mp4")
    b, _ = jobs.reserve("b.mov")
    with pytest.raises(AnalysisError, match="Two"):
        jobs.reserve("c.mp4")
    jobs.abandon(b)

    def fail(*args, **kwargs):
        raise AnalysisError("Video decode failed clearly")

    monkeypatch.setattr(jobs, "run", fail)
    jobs.process(a)
    assert jobs.status(a)["status"] == "failed"
    assert "clearly" in jobs.status(a)["message"]
    unrelated = tmp_path / "protected-review.json"
    unrelated.write_text("protected")
    jobs.remove(a)
    assert unrelated.read_text() == "protected"
    assert not jobs.listing()
    with pytest.raises(AnalysisError):
        jobs.directory("../../protected-review.json")


def test_ready_assets_and_bounds_are_source_scoped(tmp_path):
    jobs = AnalysisJobs(tmp_path)
    key, directory = jobs.reserve("test.mp4")
    write_json(directory / "job.json", dict(id=key, status="ready"))
    write_json(directory / "video.json", dict(frame_count=3))
    write_json(directory / "pose.json", {})
    (directory / "preview.mp4").write_bytes(b"synthetic")
    (directory / "frames").mkdir()
    (directory / "frames/0000.png").write_bytes(b"synthetic")
    jobs.active.remove(key)
    assert jobs.asset(key).name == "preview.mp4"
    assert jobs.asset(key, 0).name == "0000.png"
    with pytest.raises(AnalysisError, match="outside"):
        jobs.asset(key, 3)
    with pytest.raises(AnalysisError, match="start frame"):
        jobs.classify(key, 2, 1)
    with pytest.raises(AnalysisError, match="start frame"):
        jobs.classify(key, 0, 4)


def test_storage_error_is_readable_and_partial_upload_removed(workspace, monkeypatch):
    app = create_app(workspace[0])
    client = TestClient(app, base_url="http://127.0.0.1:8000")

    def full_disk(*args):
        raise OSError("Synthetic full disk")

    monkeypatch.setattr(app.state.analysis, "submit", full_disk)
    response = client.post("/api/analyze/uploads", content=b"synthetic", headers=HEADERS)
    assert response.status_code == 422
    assert "disk space" in response.json()["detail"]
    assert client.get("/api/analyze/videos").json()["videos"] == []
