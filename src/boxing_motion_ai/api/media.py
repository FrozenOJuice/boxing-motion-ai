"""Hash-bound exact-PTS review frames; cache writes never touch source artifacts."""

import hashlib
import io
import json
import math
import os
import threading
import uuid
from collections import OrderedDict
from fractions import Fraction

from boxing_motion_ai.data.annotation_review import RECORDINGS, digest
from boxing_motion_ai.services.artifacts import ServiceError
from boxing_motion_ai.services.dashboard import require
from boxing_motion_ai.vision.contracts import EDGES, JOINTS


def signature(stream):
    s = os.fstat(stream.fileno())
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def decode_exact(stream, target):
    """FFmpeg through PyAV: seek to keyframe, then decode until exact rational PTS.

    Unlike browser seeking, a nearby frame is an error. The native index comes from
    the hash-verified full extraction timeline, not nominal FPS or seek position.
    """
    import av

    with av.open(stream) as container:
        track = container.streams.video[0]
        track.codec_context.thread_count = 2
        container.seek(int(target / track.time_base), stream=track, backward=True)
        for count, frame in enumerate(container.decode(track)):
            require(count < 2000, "decode_window_exceeded")
            require(
                frame.pts is not None and frame.time_base is not None and not frame.is_corrupt,
                "invalid_decoded_timing",
            )
            actual = frame.pts * Fraction(frame.time_base)
            if actual < target:
                continue
            require(actual == target, "exact_frame_not_found")
            image = frame.to_image()
            image.thumbnail((1280, 1280))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=90)
            return output.getvalue(), image.width, image.height
    raise ServiceError("missing", "exact_frame_not_found")


class FrameService:
    def __init__(self, service):
        self.service = service
        self.store = service.store
        self.lock = threading.Lock()  # One bounded decoder at a time, including duplicate requests.
        self.verified = {}
        self.arrays = OrderedDict()  # At most two complete recording timelines in memory.

    def _verify(self, stream, expected):
        before = signature(stream)
        key = (before, expected)
        if key not in self.verified:
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
            require(h.hexdigest() == expected, "source_hash_mismatch", "stale")
            require(signature(stream) == before, "source_changed_during_read", "stale")
            self.verified[key] = True
        stream.seek(0)
        return before

    def _load(self, v, source, prep):
        import numpy as np

        path = f"{source.dataset}/poses.npz"
        expected = source.inputs["poses.npz"]
        with self.store.open_regular(path) as stream:
            sig = self._verify(stream, expected)
            key = (v, sig, expected)
            if key not in self.arrays:
                with np.load(stream, allow_pickle=False) as z:
                    arrays = {
                        k: z[k]
                        for k in (
                            "frame_index",
                            "pts",
                            "time_base_num",
                            "time_base_den",
                            "coordinates",
                            "confidence",
                            "valid_mask",
                            "scene_id",
                            "segment_id",
                        )
                    }
                n = len(arrays["frame_index"])
                require(n == prep["source_frame_count"] and n > 0, "incomplete_pose_timeline")
                require(
                    np.array_equal(arrays["frame_index"], np.arange(n)),
                    "missing_native_frame_indices",
                )
                require(
                    arrays["coordinates"].shape == (n, 33, 3)
                    and arrays["confidence"].shape == (n, 33, 2)
                    and arrays["valid_mask"].shape == (n, 33),
                    "invalid_pose_shapes",
                )
                for name in ("pts", "time_base_num", "time_base_den", "scene_id", "segment_id"):
                    require(
                        arrays[name].shape == (n,)
                        and np.issubdtype(arrays[name].dtype, np.integer),
                        "invalid_timeline",
                    )
                require(
                    np.all(arrays["time_base_num"] > 0) and np.all(arrays["time_base_den"] > 0),
                    "invalid_time_base",
                )
                require(arrays["valid_mask"].dtype == np.bool_, "invalid_validity_mask")
                times = [
                    Fraction(int(p) * int(a), int(b))
                    for p, a, b in zip(
                        arrays["pts"], arrays["time_base_num"], arrays["time_base_den"], strict=True
                    )
                ]
                require(
                    all(b > a for a, b in zip(times, times[1:], strict=False)),
                    "nonmonotonic_timeline",
                )
                require(signature(stream) == sig, "source_changed_during_read", "stale")
                for a in arrays.values():
                    a.setflags(write=False)
                self.arrays[key] = arrays
                while len(self.arrays) > 2:
                    self.arrays.popitem(last=False)
            self.arrays.move_to_end(key)
            return self.arrays[key]

    def _cache(self, key, generate):
        # Single dedicated ignored directory; refuse symlinks. Atomic writes via directory FD.
        root = self.store.root
        directory = root / ".dashboard-cache"
        directory.mkdir(exist_ok=True)
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            name = key + ".json"
            try:
                handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
                with os.fdopen(handle, "r") as stream:
                    receipt = json.load(stream)
                handle = os.open(key + ".jpg", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
                with os.fdopen(handle, "rb") as stream:
                    pixels = stream.read(4 * 1024 * 1024)
                require(
                    hashlib.sha256(pixels).hexdigest() == receipt["sha256"],
                    "preview_cache_corrupt",
                    "stale",
                )
                return pixels, receipt["width"], receipt["height"]
            except FileNotFoundError:
                pixels, width, height = generate()
                receipt = json.dumps(
                    dict(sha256=hashlib.sha256(pixels).hexdigest(), width=width, height=height)
                ).encode()
                for suffix, content in ((".jpg", pixels), (".json", receipt)):
                    temporary = uuid.uuid4().hex + ".tmp"
                    handle = os.open(
                        temporary,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=fd,
                    )
                    try:
                        with os.fdopen(handle, "wb") as stream:
                            stream.write(content)
                        os.replace(temporary, key + suffix, src_dir_fd=fd, dst_dir_fd=fd)
                    finally:
                        try:
                            os.unlink(temporary, dir_fd=fd)
                        except FileNotFoundError:
                            pass
                return pixels, width, height
        finally:
            os.close(fd)

    def frame(self, recording, index):
        require(recording in RECORDINGS, "unregistered_recording")
        require(type(index) is int and index >= 0, "invalid_frame_index")
        with self.lock:
            source = self.service._recording(recording)
            prep = self.service._preparation(recording, source)
            require(index < prep["source_frame_count"], "frame_out_of_range")
            arrays = self._load(recording, source, prep)
            target = Fraction(
                int(arrays["pts"][index]) * int(arrays["time_base_num"][index]),
                int(arrays["time_base_den"][index]),
            )
            expected = prep["inputs"]["video_sha256"]
            with self.store.open_regular(f"data/raw/boxingvi/RGB_videos/{recording}.mkv") as video:
                sig = self._verify(video, expected)
                key = digest(
                    dict(
                        version="dashboard-exact-frame-1",
                        video=expected,
                        poses=source.inputs["poses.npz"],
                        frame=index,
                        pts=str(target),
                    )
                )
                pixels, width, height = self._cache(key, lambda: decode_exact(video, target))
                require(signature(video) == sig, "source_changed_during_decode", "stale")
            points = []
            for i in range(33):
                xyz = [
                    float(v) if math.isfinite(v) else None for v in arrays["coordinates"][index, i]
                ]
                confidence = [
                    float(v) if math.isfinite(v) else None for v in arrays["confidence"][index, i]
                ]
                points.append(
                    dict(
                        name=JOINTS[i],
                        xyz=xyz,
                        confidence=confidence,
                        valid=bool(arrays["valid_mask"][index, i]),
                    )
                )
            metadata = dict(
                recording_id=recording,
                frame_index=index,
                pts=str(int(arrays["pts"][index])),
                time_base=str(
                    Fraction(
                        int(arrays["time_base_num"][index]), int(arrays["time_base_den"][index])
                    )
                ),
                exact_timestamp=str(target),
                scene_id=int(arrays["scene_id"][index]),
                segment_id=int(arrays["segment_id"][index]),
                width=width,
                height=height,
                landmarks=points,
                edges=[list(edge) for edge in EDGES],
                image_url=f"/api/recordings/{recording}/frames/{index}/image",
                correspondence="hash_verified_native_index_and_exact_decoded_pts",
                scene_cut_before=index in prep.get("confirmed_scene_cuts", []),
            )
            return metadata, pixels
