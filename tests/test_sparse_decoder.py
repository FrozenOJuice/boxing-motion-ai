"""Selected pixels match full decoding exactly, including VFR timestamps."""

import json
from fractions import Fraction

import pytest

from boxing_motion_ai.vision.decoder import decode_video
from boxing_motion_ai.vision.sparse_decoder import choose_decoder, decode_selected


def test_sparse_vfr_pixels_and_metadata_match_full_decode(tmp_path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    path = tmp_path / "synthetic.mkv"
    pts = [0, 33, 70, 110, 180]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=30)
        stream.width, stream.height, stream.pix_fmt = 128, 96, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for i, timestamp in enumerate(pts):
            frame = av.VideoFrame.from_ndarray(
                np.full((96, 128, 3), i * 35, np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    full = list(decode_video(path, 64))
    sparse = list(decode_selected(path, 64, [(1, 3)]))
    assert [f.index for f in sparse] == [0, 1, 2, 3]
    assert sparse[0].rgb is None and sparse[3].rgb is None
    for i in [1, 2]:
        assert sparse[i].time == full[i].time and sparse[i].pts == pts[i]
        assert np.array_equal(sparse[i].rgb, full[i].rgb)
        assert (sparse[i].source_width, sparse[i].source_height) == (128, 96)


def test_legacy_decoder_choice_and_sparse_version_pin(tmp_path):
    assert choose_decoder(tmp_path, []) is decode_video
    assert choose_decoder(tmp_path, [(1, 3)]).func is decode_selected
    (tmp_path / "manifest.json").write_text(json.dumps({"code": {"sha256": {}}}))
    assert choose_decoder(tmp_path, [(1, 3)]) is decode_video
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"code": {"sha256": {"src/boxing_motion_ai/vision/sparse_decoder.py": "changed"}}}
        )
    )
    with pytest.raises(ValueError, match="sparse decoder code changed"):
        choose_decoder(tmp_path, [(1, 3)])
