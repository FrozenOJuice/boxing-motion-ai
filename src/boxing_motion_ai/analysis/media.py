"""Isolated, bounded upload decode and verified buffered preview; no pose imports."""

import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from time import perf_counter

from .common import MAX_ARTIFACT_BYTES, MAX_FRAMES, MAX_SECONDS, sha, write_json


def probe(path):
    try:
        return _probe(path)
    except (
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        KeyError,
        ZeroDivisionError,
    ) as exc:
        raise ValueError(
            "Could not read this video. Re-export it as a standard MP4 or MOV."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Video inspection timed out. Try a shorter MP4 or MOV export.") from exc


def _probe(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        timeout=15,
        check=True,
    )
    value = json.loads(result.stdout)
    if not value.get("streams"):
        raise ValueError("No decodable video stream was found.")
    stream = value["streams"][0]
    if not set(value["format"]["format_name"].split(",")) & {"mov", "mp4", "matroska", "webm"}:
        raise ValueError("Use an MP4, MOV, MKV or WebM video.")
    width, height = stream["width"], stream["height"]
    if not (32 <= width <= 4096 and 32 <= height <= 4096):
        raise ValueError("Use video dimensions between 32 and 4096 pixels per side.")
    duration = value["format"].get("duration")
    if duration is None or not math.isfinite(float(duration)) or float(duration) > MAX_SECONDS:
        raise ValueError("Use a video no longer than 30 seconds with readable duration metadata.")
    rate = Fraction(stream.get("avg_frame_rate", "0/1"))
    if rate <= 0 or rate > 60:
        raise ValueError("Use a video with a frame rate of 60 FPS or lower.")
    rotation = float(stream.get("tags", {}).get("rotate", 0))
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            rotation = float(side["rotation"])
        if "displaymatrix" in side:
            rows = side["displaymatrix"].strip().splitlines()
            matrix = [[int(v) for v in r.split(":", 1)[1].split()] for r in rows]
            if matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0] <= 0:
                raise ValueError(
                    "Reflected display metadata is unsupported. Export a normal video."
                )
    if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > 0.001:
        raise ValueError("Only 0, 90, 180 and 270 degree video rotation is supported.")
    return dict(width=width, height=height, rotation=int(round(rotation)) % 360)


def prepare(directory):
    # PyAV stays in this subprocess, separate from MediaPipe/OpenCV's FFmpeg libraries.
    import av
    import numpy as np
    from PIL import Image

    directory = Path(directory)
    began = perf_counter()
    source = directory / "source.video"
    info = probe(source)
    frames_dir = directory / "frames"
    frames_dir.mkdir()
    times, metadata, durations = [], [], {}
    bytes_used = 0
    target = directory / "preview.mp4"
    with av.open(str(source), options={"protocol_whitelist": "file"}) as reader:
        track = reader.streams.video[0]
        track.codec_context.thread_count = 2
        with av.open(str(target), "w", format="mp4", options={"movflags": "+faststart"}) as writer:
            stream = None
            first = None

            def mux(packet):
                timestamp = packet.pts * packet.time_base
                if timestamp not in durations:
                    raise ValueError("Preview timestamps could not be preserved.")
                duration = durations[timestamp] / packet.time_base
                if duration.denominator != 1:
                    raise ValueError("Preview duration could not be represented exactly.")
                packet.duration = int(duration)
                writer.mux(packet)

            for index, frame in enumerate(reader.decode(track)):
                if index >= MAX_FRAMES:
                    raise ValueError("Use at most 1800 frames (30 seconds at 60 FPS).")
                if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                    raise ValueError("The video has corrupt frames or missing timestamps.")
                if (frame.width, frame.height) != (info["width"], info["height"]):
                    raise ValueError("Video dimensions change during the recording.")
                time = frame.pts * Fraction(frame.time_base)
                first = time if first is None else first
                if times and time <= times[-1]:
                    raise ValueError("Video timestamps must increase continuously.")
                if time - first > MAX_SECONDS:
                    raise ValueError("The decoded video exceeds 30 seconds.")
                if times:
                    durations[times[-1] - first] = time - times[-1]
                times.append(time)
                scale = min(1, 960 / max(frame.width, frame.height))
                w = max(2, int(frame.width * scale) // 2 * 2)
                h = max(2, int(frame.height * scale) // 2 * 2)
                rgb = frame.reformat(width=w, height=h).to_ndarray(format="rgb24")
                rgb = np.ascontiguousarray(np.rot90(rgb, k=info["rotation"] // 90))
                image_path = frames_dir / f"{index:04d}.png"
                Image.fromarray(rgb).save(image_path)
                bytes_used += image_path.stat().st_size
                if bytes_used > MAX_ARTIFACT_BYTES:
                    raise ValueError(
                        "Decoded frames exceed the local storage limit. Use a shorter video."
                    )
                if stream is None:
                    stream = writer.add_stream("libx264", rate=track.average_rate)
                    stream.width, stream.height = rgb.shape[1], rgb.shape[0]
                    stream.pix_fmt = "yuv420p"
                    stream.time_base = stream.codec_context.time_base = Fraction(frame.time_base)
                    stream.codec_context.thread_count = 2
                    stream.options = {"preset": "veryfast", "crf": "20", "bf": "0", "g": "30"}
                preview = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                preview.pts = int((time - first) / stream.codec_context.time_base)
                preview.time_base = stream.codec_context.time_base
                # libx264 has bounded encoder delay, so the next source PTS is known by mux time.
                for packet in stream.encode(preview):
                    mux(packet)
                metadata.append(
                    dict(
                        frame_index=index,
                        pts=frame.pts,
                        time_base_num=frame.time_base.numerator,
                        time_base_den=frame.time_base.denominator,
                        timestamp_exact=str(time),
                        media_time=float(time - first),
                    )
                )
                final_duration = Fraction(frame.duration or 0) * frame.time_base
                if index % 15 == 0:
                    write_json(
                        directory / "progress.json", dict(stage="decoding", frames=index + 1)
                    )
            if len(times) < 2 or final_duration <= 0:
                raise ValueError(
                    "The video needs at least two frames with readable frame durations."
                )
            end = times[-1] + final_duration
            if end - first > MAX_SECONDS:
                raise ValueError("The decoded video exceeds 30 seconds.")
            durations[times[-1] - first] = final_duration
            for packet in stream.encode():
                mux(packet)
    with av.open(str(target)) as verification:
        actual = []
        for decoded in verification.decode(video=0):
            actual.append(decoded.pts * decoded.time_base)
            last_duration = decoded.duration * decoded.time_base
        if actual != [t - first for t in times]:
            raise ValueError("The buffered preview did not preserve source frame timestamps.")
        if last_duration != final_duration:
            raise ValueError("The buffered preview did not preserve the final frame duration.")
    if info["rotation"] in (90, 270):
        info["width"], info["height"] = info["height"], info["width"]
    value = dict(
        schema_version="assisted-video-1.0",
        **info,
        frame_count=len(times),
        frames=metadata,
        duration_seconds=float(end - first),
        stop_timestamp_exact=str(end),
        source_sha256=sha(source),
        preview_sha256=sha(target),
        preview_width=stream.width,
        preview_height=stream.height,
        decoding_seconds=perf_counter() - began,
        frame_policy="presentation-order decoded frames; lossless resized oriented RGB",
    )
    write_json(directory / "video.json", value)


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
