"""Single-server automatic analysis jobs, using the existing executor and scoped storage."""

import uuid
from time import perf_counter

from .automatic_contract import VERSION, activity_record, inputs
from .common import regular, sha, write_json


class AutomaticJobs:
    def __init__(self, jobs):
        self.jobs = jobs
        self.running = set()

    def identity(self, key):
        from .jobs import AnalysisError

        jobs = self.jobs
        if jobs.status(key)["status"] != "ready":
            raise AnalysisError("Wait for video preparation before automatic analysis.", 409)
        try:
            return inputs(jobs.root, jobs.directory(key), key)
        except (OSError, ValueError, KeyError) as exc:
            raise AnalysisError(f"Automatic analysis unavailable: {exc}") from exc

    def result(self, key, state, fingerprint):
        jobs = self.jobs
        if state.get("input_fingerprint") != fingerprint:
            raise ValueError("Automatic analysis inputs changed. Run analysis again.")
        path = regular(jobs.directory(key) / f"automatic-{fingerprint}.json")
        if sha(path) != state["result_sha256"]:
            raise ValueError("Automatic analysis output changed. Run analysis again.")
        result = jobs.read(key, path.name)
        if not (
            result.get("schema_version") == VERSION
            and result.get("video_id") == key
            and result.get("input_fingerprint") == fingerprint
        ):
            raise ValueError("Automatic result does not match this upload.")
        activity_record(
            jobs.directory(key),
            key,
            result["activity_input_fingerprint"],
            result["activity_result_sha256"],
        )
        return result

    def status(self, key):
        jobs = self.jobs
        jobs.status(key)
        path = jobs.directory(key) / "automatic-state.json"
        if not path.exists():
            return dict(video_id=key, status="not_started")
        with jobs.lock:
            try:
                state = jobs.read(key, path.name)
                if state.get("video_id") != key or state.get("status") not in {
                    "ready",
                    "failed",
                    "processing",
                }:
                    raise ValueError("Automatic analysis state is invalid. Retry analysis.")
                if state["status"] == "processing" and key not in self.running:
                    raise ValueError("Automatic analysis was interrupted. Retry analysis.")
                if state["status"] == "ready":
                    state["result"] = self.result(key, state, self.identity(key)[0])
                return state
            except (OSError, ValueError, KeyError, AttributeError) as exc:
                return dict(video_id=key, status="failed", message=str(exc))

    def start(self, key):
        from .jobs import AnalysisError

        jobs = self.jobs
        with jobs.lock:
            fingerprint = self.identity(key)[0]
            if key in self.running:
                state = jobs.read(key, "automatic-state.json")
                if state["input_fingerprint"] != fingerprint:
                    raise AnalysisError("Automatic inputs changed while busy. Wait and retry.", 409)
                return state
            state = self.status(key)
            if state["status"] == "ready" and not any(
                p.get("classification", {}).get("retryable", False)
                for p in state["result"]["candidates"]
            ):
                return state
            if jobs.active or jobs.classifying or jobs.scanning or self.running:
                raise AnalysisError("Analysis is busy. Wait for the current task to finish.", 409)
            state = dict(
                video_id=key,
                status="processing",
                input_fingerprint=fingerprint,
                request_id=uuid.uuid4().hex,
                message="Preparing activity and punch candidates",
            )
            write_json(jobs.directory(key) / "automatic-state.json", state)
            self.running.add(key)
            try:
                jobs.executor.submit(self.process, key, fingerprint, state["request_id"])
            except Exception as exc:
                self.running.discard(key)
                state.update(status="failed", message="Could not start automatic analysis. Retry.")
                write_json(jobs.directory(key) / "automatic-state.json", state)
                raise AnalysisError(state["message"]) from exc
            return state

    def process(self, key, fingerprint, request_id):
        jobs = self.jobs
        began = perf_counter()
        state = dict(video_id=key, input_fingerprint=fingerprint, request_id=request_id)
        try:
            identity = self.identity(key)
            if identity[0] != fingerprint:
                raise ValueError("Automatic inputs changed while queued. Retry analysis.")
            if jobs.activity(key)["status"] != "ready":
                # Run inline on this same executor. Never enqueue and block waiting on ourselves.
                with jobs.lock:
                    jobs.scanning.add(key)
                jobs.process_scan(key, identity[1]["activity_input_fingerprint"], request_id)
            activity = jobs.activity(key)
            if activity["status"] != "ready":
                raise ValueError(
                    activity.get("message", "Activity scanning failed. Retry analysis.")
                )
            jobs.run(
                key,
                "boxing_motion_ai.analysis.automatic",
                jobs.root,
                jobs.directory(key),
                key,
                fingerprint,
                activity["result_sha256"],
                timeout=120,
            )
            if self.identity(key)[0] != fingerprint:
                raise ValueError("Automatic inputs changed during analysis. Retry.")
            state["result_sha256"] = sha(
                regular(jobs.directory(key) / f"automatic-{fingerprint}.json")
            )
            self.result(key, state, fingerprint)
            state.update(status="ready", message="Experimental punch candidates available")
        except Exception as exc:
            state.update(
                status="failed",
                message=str(exc)
                if isinstance(exc, ValueError)
                else "Automatic analysis failed. Check local artifacts and retry.",
            )
        finally:
            state["analysis_wall_seconds"] = perf_counter() - began
            with jobs.lock:
                try:
                    write_json(jobs.directory(key) / "automatic-state.json", state)
                finally:
                    self.running.discard(key)
