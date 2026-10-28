"""Bounded decode, exact VFR preview mapping and rotation using tiny generated media."""

import json
from fractions import Fraction

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from boxing_motion_ai.analysis import media  # noqa: E402


def source(directory, times=(2000, 2040, 2100, 2140)):
    with av.open(str(directory / "source.video"), "w", format="mp4") as c:
        s = c.add_stream("libx264", rate=25)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        s.codec_context.time_base = Fraction(1, 1000)
        for i, time in enumerate(times):
            rgb = np.full((48, 64, 3), i * 30, dtype=np.uint8)
            rgb[:20, :20, 0] = 240
            f = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            f.pts, f.time_base = time, Fraction(1, 1000)
            for packet in s.encode(f):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_vfr_nonzero_pts_rotation_and_verified_preview(tmp_path, monkeypatch, rotation):
    source(tmp_path)
    monkeypatch.setattr(media, "probe", lambda _: dict(width=64, height=48, rotation=rotation))
    media.prepare(tmp_path)
    value = json.loads((tmp_path / "video.json").read_text())
    assert value["frame_count"] == 4
    assert value["frames"][0]["timestamp_exact"] == "2"
    assert [row["media_time"] for row in value["frames"]] == [0, 0.04, 0.1, 0.14]
    shape = (48, 64) if rotation in (90, 270) else (64, 48)
    assert (value["width"], value["height"]) == shape
    with Image.open(tmp_path / "frames/0000.png") as image:
        assert image.size == shape
        with av.open(str(tmp_path / "source.video")) as original:
            rgb = next(original.decode(video=0)).to_ndarray(format="rgb24")
        np.testing.assert_array_equal(np.asarray(image), np.rot90(rgb, k=rotation // 90))
    with av.open(str(tmp_path / "preview.mp4")) as c:
        times = [f.pts * f.time_base for f in c.decode(video=0)]
    assert times == [Fraction(0), Fraction(1, 25), Fraction(1, 10), Fraction(7, 50)]
    assert value["duration_seconds"] == 0.20


def test_corrupt_decode_and_frame_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "probe", lambda _: dict(width=64, height=48, rotation=0))
    (tmp_path / "source.video").write_bytes(b"not video")
    with pytest.raises(av.error.InvalidDataError):
        media.prepare(tmp_path)
    limited = tmp_path / "limited"
    limited.mkdir()
    source(limited)
    monkeypatch.setattr(media, "MAX_FRAMES", 2)
    with pytest.raises(ValueError, match="1800"):
        media.prepare(limited)


def test_artifact_storage_budget(tmp_path, monkeypatch):
    source(tmp_path)
    monkeypatch.setattr(media, "probe", lambda _: dict(width=64, height=48, rotation=0))
    monkeypatch.setattr(media, "MAX_ARTIFACT_BYTES", 1)
    with pytest.raises(ValueError, match="storage limit"):
        media.prepare(tmp_path)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"duration": "31"}, "30 seconds"),
        ({"duration": "nan"}, "30 seconds"),
        ({"width": 8192}, "dimensions"),
        ({"avg_frame_rate": "120/1"}, "60 FPS"),
        ({"rotation": 45}, "rotation"),
        ({"format_name": "hls"}, "MP4"),
    ],
)
def test_probe_rejects_unsupported_inputs(monkeypatch, change, match):
    value = dict(
        format=dict(format_name="mov,mp4", duration="1"),
        streams=[dict(width=64, height=48, avg_frame_rate="25/1", tags={}, side_data_list=[])],
    )
    for key, item in change.items():
        if key in ("duration", "format_name"):
            value["format"][key] = item
        elif key == "rotation":
            value["streams"][0]["side_data_list"] = [dict(rotation=item)]
        else:
            value["streams"][0][key] = item

    class Result:
        stdout = json.dumps(value).encode()

    monkeypatch.setattr(media.subprocess, "run", lambda *a, **kw: Result())
    with pytest.raises(ValueError, match=match):
        media.probe("synthetic.mp4")
