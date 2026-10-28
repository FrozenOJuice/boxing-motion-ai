"""Cached H.264 review derivatives with verified source-to-preview PTS correspondence."""

import math
import os
import re
import threading
import uuid
from fractions import Fraction
from time import perf_counter

from fastapi.responses import Response, StreamingResponse

from ..data.annotation_review import digest
from ..services.artifacts import ServiceError
from ..services.dashboard import require
from ..services.review_workspace import ReviewWorkspace, encode, sha
from .media import signature

VERSION = "interval-mp4-1.0"


def times_for(arrays, indices):
    return [
        Fraction(
            int(arrays["pts"][i]) * int(arrays["time_base_num"][i]), int(arrays["time_base_den"][i])
        )
        for i in indices
    ]


def encode_clip(source, target, times):
    """Sequential exact-PTS decode and VFR encode; never use nominal FPS for selection."""
    import av

    start, stop = times[0], times[-1]
    relative = [t - start for t in times]
    tb = Fraction(1, math.lcm(*(t.denominator for t in relative)))
    require(tb.denominator <= 1_000_000, "unsupported_preview_timebase")
    with av.open(source) as reader:
        track = reader.streams.video[0]
        track.codec_context.thread_count = 2
        reader.seek(int(start / track.time_base), stream=track, backward=True)
        # Fragmented MP4 puts initialization first, allowing buffered browser playback.
        with av.open(
            target,
            "w",
            format="mp4",
            options={
                "movflags": "frag_keyframe+empty_moov+default_base_moof",
                "video_track_timescale": str(tb.denominator),
            },
        ) as writer:
            stream = writer.add_stream("libx264", rate=track.average_rate or 30)
            factor = min(1, 1280 / track.width, 720 / track.height)
            stream.width = max(2, int(track.width * factor) // 2 * 2)
            stream.height = max(2, int(track.height * factor) // 2 * 2)
            stream.pix_fmt = "yuv420p"
            stream.time_base = stream.codec_context.time_base = tb
            stream.codec_context.thread_count = 2
            stream.options = {"preset": "veryfast", "crf": "20", "bf": "0", "g": "30"}
            durations = {relative[i]: times[i + 1] - t for i, t in enumerate(times[:-1])}

            def mux(packet):
                timestamp = packet.pts * packet.time_base
                require(timestamp in durations, "unexpected_encoded_timestamp")
                duration = durations[timestamp] / packet.time_base
                require(duration.denominator == 1, "inexact_encoded_duration")
                packet.duration = int(duration)
                writer.mux(packet)

            index = 0
            for decoded in reader.decode(track):
                require(
                    decoded.pts is not None and not decoded.is_corrupt,
                    "invalid_preview_source_frame",
                )
                actual = decoded.pts * decoded.time_base
                if actual < start:
                    continue
                if actual >= stop:
                    break
                require(
                    index < len(times) - 1 and actual == times[index], "preview_source_pts_mismatch"
                )
                frame = decoded.reformat(stream.width, stream.height, format="yuv420p")
                frame.pts = int(relative[index] / tb)
                frame.time_base = tb
                for packet in stream.encode(frame):
                    mux(packet)
                index += 1
            require(index == len(times) - 1, "incomplete_preview_source")
            for packet in stream.encode():
                mux(packet)
    return stream.width, stream.height


def verify_clip(stream, expected):
    """Decode derivative independently; reject drops, duplication or shifted timing."""
    import av

    with av.open(stream) as container:
        frames = list(container.decode(video=0))
        actual = [f.pts * f.time_base for f in frames]
        require(actual == expected[:-1], "preview_roundtrip_pts_mismatch")
        require(
            frames[-1].duration * frames[-1].time_base == expected[-1] - expected[-2],
            "preview_end_duration_mismatch",
        )
    return len(frames)


class ClipService:
    def __init__(self, intervals, frames):
        self.intervals, self.frames = intervals, frames
        self.store = frames.store
        self.lock = threading.Lock()
        self.verified = {}

    def prepare(self, key):
        detail = self.intervals.detail(key)
        e = detail["entry"]
        source = self.frames.service._recording(e["recording"])
        prep = self.frames.service._preparation(e["recording"], source)
        with self.frames.lock:
            arrays = self.frames._load(e["recording"], source, prep)
        lo = e["context_start"]
        stop = min(e["context_end"] + 1, e["frame_count"] - 1)
        require(
            0 <= lo <= e["start"] < e["stop"] <= stop < e["frame_count"], "invalid_preview_context"
        )
        require(stop - lo <= 1200, "preview_frame_budget_exceeded")
        for k in ("scene_id", "segment_id"):
            require(len(set(arrays[k][lo:stop])) == 1, "preview_context_discontinuity")
        require(
            not any(lo < cut < stop for cut in prep.get("confirmed_scene_cuts", [])),
            "preview_context_crosses_cut",
        )
        indices = list(range(lo, stop))
        times = times_for(arrays, range(lo, stop + 1))
        require(
            times[-1] - times[0] <= 35
            and all(0 < b - a <= Fraction(1, 10) for a, b in zip(times, times[1:], strict=False)),
            "preview_timestamp_discontinuity",
        )
        binding = dict(
            version=VERSION,
            interval_id=key,
            video=e["video_sha256"],
            poses=source.inputs["poses.npz"],
            bounds=[lo, stop],
            pts=list(map(str, times)),
            encoder="libx264-crf20-720p-bf0-vfr",
        )
        cache_key = digest(binding)
        with self.lock:
            with self.store.open_regular(
                f"data/raw/boxingvi/RGB_videos/{e['recording']}.mkv"
            ) as video:
                self.frames._verify(video, e["video_sha256"])
            # Reuse the existing no-symlink directory traversal and cross-process lock.
            with ReviewWorkspace(self.frames.service)._locked(".dashboard-cache/clips") as fd:
                try:
                    receipt, _ = self.store.json(f".dashboard-cache/clips/{cache_key}.json")
                    require(receipt["binding"] == binding, "preview_cache_binding_changed", "stale")
                    with self.store.open_regular(
                        f".dashboard-cache/clips/{cache_key}.mp4"
                    ) as media:
                        sig = signature(media)
                        if self.verified.get(cache_key) != sig:
                            require(
                                sha(media.read()) == receipt["sha256"],
                                "preview_cache_corrupt",
                                "stale",
                            )
                            self.verified[cache_key] = sig
                    return receipt["metadata"] | {"cache_hit": True}
                except ServiceError as exc:
                    if exc.status != "missing":
                        raise
                temp = uuid.uuid4().hex + ".mp4"
                target = os.open(
                    temp, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd
                )
                started = perf_counter()
                try:
                    with os.fdopen(target, "w+b", closefd=True) as output:
                        with self.store.open_regular(
                            f"data/raw/boxingvi/RGB_videos/{e['recording']}.mkv"
                        ) as video:
                            sig = self.frames._verify(video, e["video_sha256"])
                            width, height = encode_clip(video, output, times)
                            require(
                                signature(video) == sig, "source_changed_during_preview", "stale"
                            )
                        output.seek(0)
                        count = verify_clip(output, [t - times[0] for t in times])
                        output.seek(0)
                        raw = output.read()
                        os.fsync(output.fileno())
                    metadata = dict(
                        schema_version=VERSION,
                        recording=e["recording"],
                        interval_id=key,
                        cache_key=cache_key,
                        video_url=f"/api/intervals/{key}/clip",
                        width=width,
                        height=height,
                        frames=indices,
                        media_times=[float(t - times[0]) for t in times[:-1]],
                        source_times=list(map(str, times[:-1])),
                        source_start=str(times[0]),
                        duration=float(times[-1] - times[0]),
                        verified_frame_count=count,
                        encoding_seconds=perf_counter() - started,
                        bytes=len(raw),
                        time_mapping="decoded_pts_roundtrip_verified",
                    )
                    receipt = dict(binding=binding, sha256=sha(raw), metadata=metadata)
                    os.rename(temp, cache_key + ".mp4", src_dir_fd=fd, dst_dir_fd=fd)
                    handle = os.open(
                        temp + ".json",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=fd,
                    )
                    with os.fdopen(handle, "wb") as out:
                        out.write(encode(receipt))
                        out.flush()
                        os.fsync(out.fileno())
                    os.rename(temp + ".json", cache_key + ".json", src_dir_fd=fd, dst_dir_fd=fd)
                    os.fsync(fd)
                    return metadata | {"cache_hit": False}
                finally:
                    for name in (temp, temp + ".json"):
                        try:
                            os.unlink(name, dir_fd=fd)
                        except FileNotFoundError:
                            pass

    def response(self, key, range_header):
        # Metadata/source bindings are rechecked; filenames never come from request paths.
        metadata = self.prepare(key)
        path = f".dashboard-cache/clips/{metadata['cache_key']}.mp4"
        size = metadata["bytes"]
        start, end, status = 0, size - 1, 200
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
            if not match or not any(match.groups()):
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
            first, last = match.groups()
            if first:
                start, end = int(first), min(int(last) if last else size - 1, size - 1)
            else:
                start = max(0, size - int(last))
            if not 0 <= start <= end < size:
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
            status = 206

        def body():
            with self.store.open_regular(path) as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = stream.read(min(remaining, 256 * 1024))
                    require(bool(block), "preview_truncated")
                    remaining -= len(block)
                    yield block

        headers = {"Accept-Ranges": "bytes", "Content-Length": str(end - start + 1)}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return StreamingResponse(
            body(), status_code=status, media_type="video/mp4", headers=headers
        )
