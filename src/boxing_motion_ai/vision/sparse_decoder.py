"""Sample-only sequential decode: skip pixel conversion outside requested windows.

Every source frame is decoded and its PTS checked. Only selected RGB pixels cross
process boundaries. Full extraction keeps its existing decoder implementation.
"""

import json
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

from .contracts import DecodedFrame
from .decoder import _HEADER, _read_exact, decode_video


def choose_decoder(output, ranges):
    if not ranges:
        return decode_video
    # Old runs keep their original decoder. New sparse runs pin this implementation.
    manifest = Path(output) / "manifest.json"
    if manifest.exists():
        from .pipeline import sha256

        prior = json.loads(manifest.read_text())
        pinned = prior["code"]["sha256"].get("src/boxing_motion_ai/vision/sparse_decoder.py")
        if pinned is None:
            return decode_video
        if pinned != sha256(__file__):
            raise ValueError("Resume refused: sparse decoder code changed")
    return partial(decode_selected, ranges=ranges)


def decode_selected(path, max_width=960, ranges=()):
    from fractions import Fraction

    import numpy as np

    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            [sys.executable, "-m", __spec__.name, str(path), str(max_width), json.dumps(ranges)],
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        try:
            while True:
                header = process.stdout.read(_HEADER.size)
                if not header:
                    break
                if len(header) != _HEADER.size:
                    raise ValueError("Truncated sparse decoder metadata")
                size = _HEADER.unpack(header)[0]
                if size > 16384:
                    raise ValueError("Invalid sparse decoder metadata size")
                meta = json.loads(_read_exact(process.stdout, size))
                rgb = None
                if meta["shape"] is not None:
                    h, w, c = meta["shape"]
                    if c != 3 or not 0 < w <= max_width or not 0 < h <= 32768:
                        raise ValueError("Invalid sparse RGB dimensions")
                    rgb = np.frombuffer(
                        _read_exact(process.stdout, h * w * 3), dtype=np.uint8
                    ).reshape(h, w, 3)
                yield DecodedFrame(
                    meta["index"],
                    meta["pts"],
                    Fraction(meta["time_base"]),
                    meta["source_width"],
                    meta["source_height"],
                    rgb,
                )
            if process.wait():
                errors.seek(0)
                raise ValueError(
                    "Sparse decoder worker failed: "
                    + errors.read().decode(errors="replace")[-4000:]
                )
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _worker():
    from fractions import Fraction

    import av

    path, width, ranges = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 4
        previous = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                raise ValueError(f"Frame {index}: missing timing or corrupt frame")
            time = frame.pts * Fraction(frame.time_base)
            if previous is not None and time <= previous:
                raise ValueError("Non-increasing presentation timestamps")
            previous = time
            selected = any(a <= index < b for a, b in ranges)
            rgb = None
            if selected:
                resized = frame
                if frame.width > width:
                    resized = frame.reformat(
                        width=width, height=round(frame.height * width / frame.width)
                    )
                rgb = resized.to_ndarray(format="rgb24")
            meta = json.dumps(
                {
                    "index": index,
                    "pts": frame.pts,
                    "time_base": str(frame.time_base),
                    "source_width": frame.width,
                    "source_height": frame.height,
                    "shape": None if rgb is None else rgb.shape,
                }
            ).encode()
            sys.stdout.buffer.write(_HEADER.pack(len(meta)))
            sys.stdout.buffer.write(meta)
            if rgb is not None:
                sys.stdout.buffer.write(rgb.tobytes())
            if ranges and index >= ranges[-1][1]:
                break


if __name__ == "__main__":
    _worker()
