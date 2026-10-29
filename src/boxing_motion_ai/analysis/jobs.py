"""Bounded local jobs and scoped cleanup, isolated from research data and ledgers."""

import json
import re
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from .activity_contract import VERSION as ACTIVITY_VERSION
from .activity_contract import inputs as activity_inputs
from .common import EXTENSIONS, MAX_BYTES, pin, regular, sha, write_json


class AnalysisError(ValueError):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


class AnalysisJobs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.storage = self.root / ".analysis-local"
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="assisted-analysis")
        self.active = set()
        self.classifying = set()
        self.scanning = set()

    def directory(self, key):
        if not re.fullmatch(r"[0-9a-f]{32}", key):
            raise AnalysisError("Unknown local video.", 404)
        directory = self.storage / key
        if any(p.is_symlink() for p in (directory, *directory.parents)):
            raise AnalysisError("Unsafe local analysis storage.")
        return directory

    def read(self, key, filename):
        try:
            return json.loads(regular(self.directory(key) / filename).read_text())
        except (OSError, ValueError) as exc:
            raise AnalysisError(
                "Local analysis files are unavailable. Remove the video and upload again.", 404
            ) from exc

    def listing(self):
        if not self.storage.exists():
            return []
        if self.storage.is_symlink():
            raise AnalysisError("Unsafe local analysis storage.")
        return [
            self.status(p.name)
            for p in sorted(self.storage.iterdir())
            if re.fullmatch(r"[0-9a-f]{32}", p.name)
        ]

    def reserve(self, filename):
        with self.lock:
            if len(self.active) >= 2:
                raise AnalysisError(
                    "Two videos are already processing. Wait for one to finish.", 429
                )
            if Path(filename).suffix.lower() not in EXTENSIONS:
                raise AnalysisError("Choose an MP4, MOV, MKV, WebM or M4V video.")
            if len(self.listing()) >= 4:
                raise AnalysisError(
                    "Four local videos are saved. Remove one before uploading another.", 409
                )
            key = uuid.uuid4().hex
            directory = self.directory(key)
            directory.mkdir(parents=True, mode=0o700)
            self.active.add(key)
            write_json(
                directory / "job.json",
                dict(
                    id=key,
                    filename=Path(filename).name[:160],
                    status="uploading",
                    message="Receiving local video",
                    uploaded_bytes=0,
                ),
            )
            return key, directory

    def abandon(self, key):
        with self.lock:
            self.active.discard(key)
            self.remove(key)

    def submit(self, key, size):
        if not 0 < size <= MAX_BYTES:
            raise AnalysisError("Use a nonempty video of at most 150 MiB.")
        with self.lock:
            job = self.read(key, "job.json")
            job.update(status="queued", message="Waiting to process", uploaded_bytes=size)
            write_json(self.directory(key) / "job.json", job)
            self.executor.submit(self.process, key)

    def run(self, key, module, *args, timeout=300):
        directory = self.directory(key)
        with (directory / "worker.log").open("ab") as log:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", module, *map(str, args)],
                    cwd=self.root,
                    stdout=log,
                    stderr=log,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise AnalysisError(
                    "Processing took too long. Try a shorter, lower-resolution video."
                ) from exc
        if result.returncode:
            tail = (directory / "worker.log").read_text(errors="replace")[-6000:]
            errors = [
                line.split("ValueError: ", 1)[1]
                for line in tail.splitlines()
                if "ValueError: " in line
            ]
            if errors:
                raise AnalysisError(errors[-1])
            raise AnalysisError(
                "Video processing failed. Use a standard MP4/MOV export and check that "
                "pose/train dependencies and FFmpeg are installed."
            )

    def process(self, key):
        began = perf_counter()
        directory = self.directory(key)
        job = self.read(key, "job.json")
        try:
            job.update(status="processing", message="Preparing video and extracting poses")
            write_json(directory / "job.json", job)
            self.run(key, "boxing_motion_ai.analysis.media", directory, timeout=120)
            self.run(key, "boxing_motion_ai.analysis.worker", "pose", self.root, directory)
            job.update(status="ready", message="Ready to select a punch")
        except Exception as exc:
            job.update(
                status="failed",
                message=str(exc)
                if isinstance(exc, AnalysisError)
                else "Processing failed. Remove this video and try again.",
            )
        finally:
            job["processing_wall_seconds"] = perf_counter() - began
            write_json(directory / "job.json", job)
            with self.lock:
                self.active.discard(key)

    def status(self, key):
        with self.lock:
            job = self.read(key, "job.json")
            if job["status"] in ("uploading", "queued", "processing") and key not in self.active:
                job.update(
                    status="failed",
                    message="Processing was interrupted by a server restart. "
                    "Remove this video and upload again.",
                )
            if job["status"] == "processing" and (self.directory(key) / "progress.json").is_file():
                job["progress"] = self.read(key, "progress.json")
            if job["status"] == "ready":
                video = self.read(key, "video.json")
                poses = self.read(key, "pose.json")
                job["video"] = video
                job["poses"] = poses
                job["preview_url"] = f"/api/analyze/videos/{key}/preview"
                job["predictions"] = [
                    self.read(key, p.name)
                    for p in sorted(self.directory(key).glob("prediction-*.json"))
                ]
            return job

    def _activity_identity(self, key):
        if self.status(key)["status"] != "ready":
            raise AnalysisError("Wait for video preparation before scanning for activity.", 409)
        try:
            return activity_inputs(self.root, self.directory(key), key)[0]
        except (OSError, ValueError, KeyError) as exc:
            raise AnalysisError(f"Activity detector unavailable: {exc}") from exc

    def _activity_result(self, key, state, fingerprint):
        if state.get("input_fingerprint") != fingerprint:
            raise ValueError("Activity scan inputs have changed. Run Scan for activity again.")
        name = f"activity-{fingerprint}.json"
        if sha(regular(self.directory(key) / name)) != state["result_sha256"]:
            raise ValueError("Cached activity output has changed. Run Scan for activity again.")
        result = self.read(key, name)
        if not (
            result.get("schema_version") == ACTIVITY_VERSION
            and result.get("video_id") == key
            and result.get("input_fingerprint") == fingerprint
        ):
            raise ValueError("Activity result does not match this video and input version.")
        return result

    def activity(self, key):
        self.status(key)  # Validate this upload ID before inspecting its scoped artifacts.
        path = self.directory(key) / "activity-state.json"
        if not path.exists():
            return dict(video_id=key, status="not_started")
        with self.lock:
            try:
                state = self.read(key, path.name)
                if state.get("status") not in {"processing", "ready", "failed"}:
                    raise ValueError("Invalid activity state")
            except (AnalysisError, ValueError, AttributeError):
                return dict(
                    video_id=key,
                    status="failed",
                    message="Activity state is unavailable. Retry the scan.",
                )
            if state.get("video_id") != key:
                return dict(
                    video_id=key,
                    status="failed",
                    message="Activity state belongs to another video.",
                )
            if state["status"] == "processing" and key not in self.scanning:
                return dict(
                    video_id=key,
                    status="failed",
                    message="Activity scanning was interrupted. Retry the scan.",
                )
            if state["status"] == "ready":
                try:
                    result = self._activity_result(key, state, self._activity_identity(key))
                    return {**state, "result": result}
                except (OSError, ValueError, KeyError, AnalysisError) as exc:
                    return dict(video_id=key, status="failed", message=str(exc))
            return state

    def scan(self, key):
        with self.lock:
            fingerprint = self._activity_identity(key)
            if key in self.scanning:
                state = self.read(key, "activity-state.json")
                if state["input_fingerprint"] != fingerprint:
                    raise AnalysisError(
                        "Scan inputs changed while processing. Wait and retry.", 409
                    )
                return state
            old = self.activity(key)
            if old["status"] == "ready":
                return old
            if self.active or self.classifying or self.scanning:
                raise AnalysisError(
                    "Analysis is busy. Wait for processing or classification to finish.", 409
                )
            request_id = uuid.uuid4().hex
            state = dict(
                video_id=key,
                status="processing",
                request_id=request_id,
                input_fingerprint=fingerprint,
                message="Scanning cached poses for possible activity",
            )
            write_json(self.directory(key) / "activity-state.json", state)
            self.scanning.add(key)
            try:
                self.executor.submit(self.process_scan, key, fingerprint, request_id)
            except Exception as exc:
                self.scanning.discard(key)
                state.update(status="failed", message="Could not start activity scan. Retry.")
                write_json(self.directory(key) / "activity-state.json", state)
                raise AnalysisError(state["message"]) from exc
            return state

    def process_scan(self, key, fingerprint, request_id):
        began = perf_counter()
        state = dict(video_id=key, request_id=request_id, input_fingerprint=fingerprint)
        try:
            self.run(
                key,
                "boxing_motion_ai.detection.upload",
                self.root,
                self.directory(key),
                key,
                fingerprint,
                timeout=120,
            )
            if self._activity_identity(key) != fingerprint:
                raise AnalysisError("Activity inputs changed during scanning. Retry.")
            state["result_sha256"] = sha(
                regular(self.directory(key) / f"activity-{fingerprint}.json")
            )
            self._activity_result(key, state, fingerprint)
            state.update(
                status="ready", message="Experimental possible punching activity available"
            )
        except Exception as exc:
            state.update(
                status="failed",
                message=str(exc)
                if isinstance(exc, (AnalysisError, ValueError))
                else "Activity inference failed. Check local artifacts and retry.",
            )
        finally:
            state["scan_wall_seconds"] = perf_counter() - began
            with self.lock:
                try:
                    write_json(self.directory(key) / "activity-state.json", state)
                finally:
                    self.scanning.discard(key)

    def classify(self, key, start, stop):
        with self.lock:
            job = self.status(key)
            if job["status"] != "ready":
                raise AnalysisError("Wait for video processing to finish.", 409)
            count = job["video"]["frame_count"]
            if not (0 <= start < stop <= count):
                raise AnalysisError("Choose a start frame before the end frame, within the video.")
            if key in self.classifying or self.classifying or self.active or self.scanning:
                raise AnalysisError(
                    "Analysis is busy. Wait for processing to finish and try again.", 409
                )
            self.classifying.add(key)
        try:
            name = f"prediction-{start}-{stop}.json"
            try:
                pin(self.root)
                poses = self.read(key, "pose.json")
                if sha(regular(self.directory(key) / "poses.npz")) != poses["poses_sha256"]:
                    raise AnalysisError(
                        "Cached poses have changed. Remove the video and upload again."
                    )
            except (OSError, ValueError) as exc:
                raise AnalysisError(str(exc)) from exc
            if not (self.directory(key) / name).exists():
                self.run(
                    key,
                    "boxing_motion_ai.analysis.worker",
                    "classify",
                    self.root,
                    self.directory(key),
                    start,
                    stop,
                    timeout=45,
                )
            return self.read(key, name)
        finally:
            with self.lock:
                self.classifying.discard(key)

    def remove(self, key):
        with self.lock:
            if key in self.active or key in self.classifying or key in self.scanning:
                raise AnalysisError(
                    "Wait for processing to finish before removing this video.", 409
                )
            directory = self.directory(key)
            if directory.is_dir():
                shutil.rmtree(directory)

    def asset(self, key, frame=None):
        job = self.status(key)
        if job["status"] != "ready":
            raise AnalysisError("The video is not ready for playback.", 409)
        if frame is not None and not 0 <= frame < job["video"]["frame_count"]:
            raise AnalysisError("That frame is outside the video.")
        path = self.directory(key) / ("preview.mp4" if frame is None else f"frames/{frame:04d}.png")
        try:
            return regular(path)
        except ValueError as exc:
            raise AnalysisError(str(exc), 404) from exc
