"""HTTP and exact-PTS integration tests use tiny generated media, never dataset files."""

import hashlib
import io
from fractions import Fraction

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
av = pytest.importorskip("av")
np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402
from test_dashboard_services import pin, publish, save  # noqa: E402
from test_dashboard_services import workspace as workspace  # noqa: E402

from boxing_motion_ai.api.app import create_app  # noqa: E402
from boxing_motion_ai.api.media import decode_exact  # noqa: E402
from boxing_motion_ai.services.artifacts import ServiceError  # noqa: E402


@pytest.fixture
def media(workspace):
    root, config, _, _ = workspace
    v = "V4"
    path = root / f"data/raw/boxingvi/RGB_videos/{v}.mkv"
    path.parent.mkdir(parents=True)
    with av.open(str(path), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        stream.time_base = Fraction(1, 1000)
        for i in range(8):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), i * 25, dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = i * i * 40, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        decoded = list(container.decode(video=0))
        times = [f.pts * Fraction(f.time_base) for f in decoded]
    n = len(times)
    source = config["sources"][v]
    directory = root / source["dataset"]
    xyz = np.zeros((n, 33, 3)) + 0.4
    xyz[:, 13] = np.nan
    valid = np.ones((n, 33), dtype=bool)
    valid[:, 13] = False
    arrays = dict(
        frame_index=np.arange(n),
        pts=np.array([f.pts for f in decoded]),
        time_base_num=np.array([f.time_base.numerator for f in decoded]),
        time_base_den=np.array([f.time_base.denominator for f in decoded]),
        coordinates=xyz,
        confidence=np.ones((n, 33, 2)),
        valid_mask=valid,
        scene_id=np.zeros(n, dtype=np.int64),
        segment_id=np.zeros(n, dtype=np.int64),
    )
    np.savez_compressed(directory / "poses.npz", **arrays)
    source["inputs"]["poses.npz"] = pin(root, f"{source['dataset']}/poses.npz")["sha256"]
    prep = dict(
        video_id=v,
        schema_version="recording-preparation-1.1",
        source_frame_count=n,
        npz_sha256=source["inputs"]["poses.npz"],
        confirmed_scene_cuts=[3],
        inputs=dict(video_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    )
    source["inputs"]["preparation.json"] = save(
        root, f"{source['dataset']}/preparation.json", prep
    )["sha256"]
    publish(workspace)
    app = create_app(root)
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    return root, app, client, times, arrays


def test_api_reuses_services_and_limits_write_routes(workspace):
    client = TestClient(create_app(workspace[0]), base_url="http://127.0.0.1:8000")
    assert client.get("/api/health").json()["mode"] == "local_interactive_review"
    assert client.get("/api/inventory").json()["data"]["available_recordings"] == 6
    assert client.get("/api/reviews").json()["data"]["entries"][0]["id"] == "V4:1"
    case = client.get("/api/cases/V4/1").json()["data"]
    assert case["annotation"]["raw_start"] == 100
    assert case["review"]["data"]["human_accepted"] is False
    assert client.get("/api/recordings/V4/annotations?limit=101").status_code == 422
    assert client.get("/api/cases/V4/99").status_code == 404
    assert client.get("/api/experiments/reference").status_code == 200
    assert client.get("/api/experiments/reference/metrics/seed-1").status_code == 200
    assert client.get("/api/statistics").json()["V4"]["data"]["approved_provisional_examples"] == 5
    for method in ("POST", "PUT", "DELETE", "PATCH"):
        assert (
            client.request(
                method,
                "/api/reviews",
                json={},
                headers={"Origin": "http://127.0.0.1:8000", "X-Review-Client": "dashboard-v0.2"},
            ).status_code
            == 405
        )
    assert client.get("/data/raw/example.mkv").status_code == 404
    assert client.get("/api/health", headers={"Origin": "http://127.0.0.1:5174"}).status_code == 200
    assert client.get("/api/health", headers={"Origin": "http://127.0.0.1:5173"}).status_code == 403
    assert (
        client.get("/api/reviews", headers={"Origin": "https://external.example"}).status_code
        == 403
    )
    assert client.get("/api/reviews", headers={"Host": "external.example"}).status_code == 400


def test_exact_frame_cache_and_pose_masks_preserve_source(media, monkeypatch):
    root, app, client, times, _ = media
    video = root / "data/raw/boxingvi/RGB_videos/V4.mkv"
    before = hashlib.sha256(video.read_bytes()).hexdigest()
    result = client.get("/api/recordings/V4/frames/3")
    assert result.status_code == 200, result.text
    value = result.json()["data"]
    assert Fraction(value["exact_timestamp"]) == times[3]
    assert value["frame_index"] == 3 and value["scene_cut_before"] is True
    assert value["landmarks"][13]["valid"] is False
    assert value["landmarks"][13]["xyz"] == [None, None, None]
    assert value["landmarks"][12]["valid"] is True

    def forbidden(*args):
        pytest.fail("Repeated request must use exact cached pixels")

    monkeypatch.setattr("boxing_motion_ai.api.media.decode_exact", forbidden)
    image = client.get(value["image_url"])
    assert image.headers["x-exact-timestamp"] == str(times[3])
    assert image.headers["x-native-frame"] == "3"
    assert abs(np.asarray(Image.open(io.BytesIO(image.content))).mean() - 75) < 3
    assert hashlib.sha256(video.read_bytes()).hexdigest() == before
    assert len(list((root / ".dashboard-cache").glob("*.jpg"))) == 1


def test_missing_exact_pts_is_not_approximate(media):
    root, _, _, times, _ = media
    with (root / "data/raw/boxingvi/RGB_videos/V4.mkv").open("rb") as video:
        with pytest.raises(ServiceError, match="exact_frame_not_found"):
            decode_exact(video, times[3] + Fraction(1, 1000))


@pytest.mark.parametrize("path", ["V9/frames/0", "V1/frames/0", "V4/frames/-1", "V4/frames/999"])
def test_reserved_or_invalid_frame_never_decodes(media, monkeypatch, path):
    _, _, client, _, _ = media

    def forbidden(*args):
        pytest.fail("Must reject before decode")

    monkeypatch.setattr("boxing_motion_ai.api.media.decode_exact", forbidden)
    assert client.get(f"/api/recordings/{path}").status_code == 422


def test_source_change_refuses_even_cached_frame(media):
    root, _, client, _, _ = media
    assert client.get("/api/recordings/V4/frames/2").status_code == 200
    video = root / "data/raw/boxingvi/RGB_videos/V4.mkv"
    with video.open("ab") as stream:
        stream.write(b"changed")
    response = client.get("/api/recordings/V4/frames/2")
    assert response.status_code == 409
    assert response.json()["issues"] == ["source_hash_mismatch"]


def test_missing_media_and_symlink_do_not_leak_paths(media, tmp_path):
    root, _, client, _, _ = media
    path = root / "data/raw/boxingvi/RGB_videos/V4.mkv"
    path.unlink()
    assert client.get("/api/recordings/V4/frames/1").status_code == 404
    elsewhere = root / "external.mkv"
    elsewhere.write_bytes(b"no footage")
    path.symlink_to(elsewhere)
    response = client.get("/api/recordings/V4/frames/1")
    assert response.status_code == 422 and str(root) not in response.text


def test_cache_directory_symlink_rejected(media):
    root, _, client, _, _ = media
    (root / ".dashboard-cache").symlink_to(root / "data")
    assert client.get("/api/recordings/V4/frames/1").status_code == 422


def test_corrupt_cache_does_not_silently_serve(media):
    root, _, client, _, _ = media
    assert client.get("/api/recordings/V4/frames/1").status_code == 200
    next((root / ".dashboard-cache").glob("*.jpg")).write_bytes(b"corrupt")
    assert client.get("/api/recordings/V4/frames/1").status_code == 409


def test_review_api_persistence_csrf_and_exact_boundaries(media):
    from test_review_workspace import decision

    root, app, client, _, _ = media
    headers = {"Origin": "http://127.0.0.1:8000", "X-Review-Client": "dashboard-v0.2"}
    item = decision(
        app.state.workspace,
        action="confirm",
        proposed_label="jab",
        proposed_bounds=dict(start=4, end=5, convention="zero_based_decoded_inclusive"),
        inspected_frames=[4, 5],
        lead_rear_evidence="Synthetic review only",
    )
    payload = item.model_dump(mode="json")
    assert client.post("/api/reviews/V4/1", json=payload).status_code == 403
    response = client.post("/api/reviews/V4/1", json=payload, headers=headers)
    assert response.status_code == 200, response.text
    restarted = TestClient(create_app(root), base_url="http://127.0.0.1:8000")
    case = restarted.get("/api/cases/V4/1").json()["data"]
    assert case["edit"]["data"]["history"][-1]["confirmed_bounds"]["start"] == 4
    assert case["annotation"]["raw_start"] == 100
    assert restarted.post("/api/reviews/V9/1", json=payload, headers=headers).status_code == 422
    crossing = decision(
        app.state.workspace,
        action="confirm",
        proposed_label="jab",
        proposed_bounds=dict(start=2, end=4, convention="zero_based_decoded_inclusive"),
        inspected_frames=[2, 4],
        lead_rear_evidence="Synthetic review only",
    )
    response = restarted.post(
        "/api/reviews/V4/1", json=crossing.model_dump(mode="json"), headers=headers
    )
    assert response.status_code == 422
    assert "confirmation_crosses_known_cut" in response.text
