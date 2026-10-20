"""Sequential presentation-order decode. No seeking, frame resampling or FPS-derived times."""

import json
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction

from .contracts import DecodedFrame


def _decode_local(path, max_width=960):
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # Bounded software decoding; AV1 must be available in the installed FFmpeg/PyAV build.
        stream.codec_context.thread_count = 4
        previous = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                raise ValueError(f"Frame {index}: missing timing or corrupt decoded frame")
            time_base = Fraction(frame.time_base)
            time = frame.pts * time_base
            if previous is not None and time <= previous:
                raise ValueError(f"Frame {index}: non-increasing presentation timestamp")
            previous = time
            width, height = frame.width, frame.height
            resized = frame
            if width > max_width:
                resized = frame.reformat(width=max_width, height=round(height * max_width / width))
            yield DecodedFrame(
                index, frame.pts, time_base, width, height, resized.to_ndarray(format="rgb24")
            )


_HEADER = struct.Struct("!I")


def _read_exact(stream, size):
    parts = bytearray()
    while len(parts) < size:
        block = stream.read(size - len(parts))
        if not block:
            raise ValueError("Decoder worker ended during a frame")
        parts.extend(block)
    return bytes(parts)


def decode_video(path, max_width=960):
    """Isolate PyAV FFmpeg dylibs from MediaPipe's OpenCV dylibs on macOS.

    Only resized RGB pixels and exact timing metadata cross the local binary pipe.
    This avoids duplicate Objective-C FFmpeg classes in the inference process.
    """
    import numpy as np

    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            [sys.executable, "-m", "boxing_motion_ai.vision.decoder", str(path), str(max_width)],
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        try:
            while True:
                header = process.stdout.read(_HEADER.size)
                if not header:
                    break
                if len(header) != _HEADER.size:
                    raise ValueError("Truncated decoder metadata header")
                size = _HEADER.unpack(header)[0]
                if size > 16384:
                    raise ValueError("Invalid decoder metadata length")
                meta = json.loads(_read_exact(process.stdout, size))
                height, width, channels = meta["shape"]
                if channels != 3 or not 0 < width <= max_width or not 0 < height <= 32768:
                    raise ValueError("Invalid decoded RGB dimensions")
                rgb = np.frombuffer(
                    _read_exact(process.stdout, height * width * 3), dtype=np.uint8
                ).reshape((height, width, 3))
                yield DecodedFrame(
                    meta["index"],
                    meta["pts"],
                    Fraction(meta["time_base"]),
                    meta["source_width"],
                    meta["source_height"],
                    rgb,
                )
            code = process.wait()
            if code:
                errors.seek(0)
                raise RuntimeError(
                    f"Decoder worker failed ({code}): "
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
    for frame in _decode_local(sys.argv[1], int(sys.argv[2])):
        meta = json.dumps(
            {
                "index": frame.index,
                "pts": frame.pts,
                "time_base": str(frame.time_base),
                "source_width": frame.source_width,
                "source_height": frame.source_height,
                "shape": frame.rgb.shape,
            }
        ).encode()
        sys.stdout.buffer.write(_HEADER.pack(len(meta)))
        sys.stdout.buffer.write(meta)
        sys.stdout.buffer.write(frame.rgb.tobytes())


if __name__ == "__main__":
    _worker()
