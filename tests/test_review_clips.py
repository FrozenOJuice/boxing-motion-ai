"""Local MP4 derivatives preserve native VFR timestamps; sources/reviews stay read-only."""

import io
from fractions import Fraction

import av
import numpy as np
import pytest
from test_dashboard_api import media as media  # noqa: F401
from test_dashboard_services import workspace as workspace  # noqa: F401
from test_interval_review import pilot as pilot  # noqa: F401

from boxing_motion_ai.api.clips import encode_clip, verify_clip
from boxing_motion_ai.services.artifacts import ServiceError


@pytest.mark.parametrize("rate", [Fraction(24), Fraction(25), Fraction(30000, 1001)])
def test_native_rate_roundtrip_and_duration(rate):
    source = io.BytesIO()
    with av.open(source, "w", format="matroska") as c:
        s = c.add_stream("ffv1", rate=rate)
        s.width = s.height = 32
        s.pix_fmt = "yuv420p"
        for i in range(14):
            f = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 10, np.uint8), format="rgb24")
            f.pts, f.time_base = i, 1 / rate
            for packet in s.encode(f):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)
    raw = source.getvalue()
    with av.open(io.BytesIO(raw)) as c:
        times = [f.pts * f.time_base for f in c.decode(video=0)]
    target = io.BytesIO()
    encode_clip(io.BytesIO(raw), target, times[2:12])
    assert verify_clip(io.BytesIO(target.getvalue()), [t - times[2] for t in times[2:12]]) == 9
    # MP4 initialization precedes media fragments; no complete-file download before playback.
    content = target.getvalue()
    assert content.index(b"moov") < content.index(b"mdat")


def test_variable_timing_and_wrong_source_correspondence(media):
    root, _, _, times, _ = media
    path = root / "data/raw/boxingvi/RGB_videos/V4.mkv"
    before = path.read_bytes()
    target = io.BytesIO()
    with path.open("rb") as source:
        encode_clip(source, target, times)
    assert verify_clip(io.BytesIO(target.getvalue()), [t - times[0] for t in times]) == 7
    assert path.read_bytes() == before
    wrong = list(times)
    wrong[1] += Fraction(1, 10000)
    with path.open("rb") as source, pytest.raises(ServiceError, match="pts_mismatch"):
        encode_clip(source, io.BytesIO(), wrong)


def test_cache_range_and_reserved_routes(pilot):
    root, client, manifest = pilot
    # The synthetic VFR fixture has large later gaps; use one exact reviewed frame.
    from boxing_motion_ai.services.review_workspace import encode, sha

    manifest["entries"][0]["context_end"] = 0
    path = root / "reports/boxingvi/interval-review-pilot-v1/manifest.json"
    path.write_bytes(encode(manifest))
    (root / "configs/interval_review_pilot_v1.json").write_bytes(
        encode(dict(path=str(path.relative_to(root)), sha256=sha(path.read_bytes())))
    )
    result = client.get("/api/intervals/V4-bg-1/playback")
    assert result.status_code == 200, result.text
    first = result.json()["data"]
    assert first["cache_hit"] is False and first["frames"] == [0]
    mp4 = root / f".dashboard-cache/clips/{first['cache_key']}.mp4"
    before = mp4.stat().st_mtime_ns
    assert client.get("/api/intervals/V4-bg-1/playback").json()["data"]["cache_hit"] is True
    assert mp4.stat().st_mtime_ns == before
    full = client.get(first["video_url"])
    assert full.status_code == 200 and full.headers["content-type"] == "video/mp4"
    part = client.get(first["video_url"], headers={"Range": "bytes=4-31"})
    assert part.status_code == 206 and part.content == full.content[4:32]
    assert part.headers["content-range"].startswith("bytes 4-31/")
    suffix = client.get(first["video_url"], headers={"Range": "bytes=-16"})
    assert suffix.status_code == 206 and suffix.content == full.content[-16:]
    invalid_range = client.get(first["video_url"], headers={"Range": "bytes=999999-"})
    assert invalid_range.status_code == 416
    assert invalid_range.headers["content-range"].startswith("bytes */")
    assert client.get("/api/intervals/V9-bg-1/playback").status_code == 404
    assert client.get("/api/intervals/V9-bg-1/clip").status_code == 404
    assert client.get("/api/intervals").json()["data"]["history"] == []
    mp4.write_bytes(b"corrupt")
    assert client.get("/api/intervals/V4-bg-1/playback").status_code == 409
