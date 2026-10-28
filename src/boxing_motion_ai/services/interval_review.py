"""Bounded interval review, separate from original punch decisions and training data."""

import os
import re
import uuid
from datetime import UTC, datetime
from fractions import Fraction
from typing import Literal

import numpy as np
from pydantic import Field, StrictBool, StrictInt, model_validator

from ..data.annotation_review import RECORDINGS, StrictModel
from .artifacts import strict_json
from .contracts import Artifact
from .dashboard import require
from .review_workspace import ReviewWorkspace, encode, sha

DIRECTORY = "reports/boxingvi/interval-reviews-v1"
CONFIG = "configs/interval_review_pilot_v1.json"
Pilot = Literal["original", "motion-v2"]
PILOTS = {
    "original": (CONFIG, DIRECTORY),
    "motion-v2": (
        "configs/interval_review_motion_v2.json",
        "reports/boxingvi/interval-reviews-motion-v2",
    ),
}


class IntervalDecision(StrictModel):
    request_id: str = Field(pattern=r"^[0-9a-f]{8}-[0-9a-f-]{27}$")
    expected_head: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["no_punch", "punch_present", "not_sure"]
    watched_entire: StrictBool = False
    visible_continuous: StrictBool = False
    notes: str = Field(default="", max_length=2000)
    tags: list[Literal["ordinary", "guard", "defense", "pause", "feint"]] = Field(
        default_factory=list, max_length=5
    )
    # Native decoded frames actually displayed, not a synthetic playback-complete flag.
    inspected_frames: list[StrictInt] = Field(default_factory=list, max_length=2000)
    unannotated_punches: Literal["yes", "no", "unknown"] = "unknown"

    @model_validator(mode="after")
    def absence(self):
        if self.decision == "no_punch" and not (
            self.watched_entire and self.visible_continuous and self.unannotated_punches == "no"
        ):
            raise ValueError("No punches requires explicit whole-interval visible review")
        if any(type(x) is not int or x < 0 for x in self.inspected_frames):
            raise ValueError("Native frame indices required")
        if len(set(self.inspected_frames)) != len(self.inspected_frames):
            raise ValueError("Duplicate inspected frame")
        return self


def interval_timing(arrays, start, stop, cuts=()):
    """Require a complete half-open interval and a known exact exclusive endpoint."""
    n = len(arrays["frame_index"])
    require(
        type(start) is int and type(stop) is int and 0 <= start < stop < n,
        "invalid_interval_bounds",
    )
    require(
        np.array_equal(arrays["frame_index"][start : stop + 1], np.arange(start, stop + 1)),
        "missing_interval_frames",
    )
    require(not any(start < cut < stop for cut in cuts), "interval_crosses_cut")
    for name in ("scene_id", "segment_id"):
        require(len(np.unique(arrays[name][start:stop])) == 1, "interval_discontinuity")
    times = [
        Fraction(
            int(arrays["pts"][i]) * int(arrays["time_base_num"][i]), int(arrays["time_base_den"][i])
        )
        for i in range(start, stop + 1)
    ]
    require(
        all(0 < b - a <= Fraction(1, 10) for a, b in zip(times, times[1:], strict=False)),
        "interval_timestamp_discontinuity",
    )
    require(times[-1] - times[0] <= 31, "interval_too_long")
    return {
        "start": str(times[0]),
        "stop_exclusive": str(times[-1]),
        "scene_id": int(arrays["scene_id"][start]),
        "segment_id": int(arrays["segment_id"][start]),
    }


class IntervalWorkspace:
    def __init__(self, base, frames, pilot: Pilot = "original"):
        require(pilot in PILOTS, "unknown_interval_pilot")
        self.base, self.frames, self.pilot = base, frames, pilot
        self.config_path, self.directory = PILOTS[pilot]

    def manifest(self):
        ref, _ = self.base.store.json(self.config_path)
        ref = Artifact.model_validate(ref)
        manifest, digest = self.base.store.json(ref.path, ref.sha256)
        require(manifest["schema_version"] == "interval-pilot-1.0", "invalid_interval_manifest")
        entries = manifest["entries"]
        require(0 < len(entries) <= 42, "invalid_interval_count")
        require(len({e["id"] for e in entries}) == len(entries), "duplicate_interval")
        require(
            sum(e["purpose"] == "background_candidate" for e in entries) <= 30,
            "background_budget_exceeded",
        )
        for e in entries:
            require(
                e["recording"] in RECORDINGS
                and e["id"].startswith(e["recording"] + "-")
                and re.fullmatch(r"V(?:4|5|6|7|8|10)-(?:bg|eval|motion)-[0-9]+", e["id"]),
                "reserved_or_invalid_interval",
            )
            role = manifest["roles"][e["recording"]]
            require(
                role in ("train_candidate", "development", "evaluation_reserved")
                and e["role"] == role,
                "interval_role_mismatch",
            )
            require(
                e["purpose"] in ("background_candidate", "evaluation_block"),
                "invalid_interval_purpose",
            )
            require(
                e["purpose"] != "evaluation_block" or role == "evaluation_reserved",
                "evaluation_split_leakage",
            )
            require(
                type(e["start"]) is int
                and type(e["stop"]) is int
                and 0 <= e["start"] < e["stop"] < e["frame_count"],
                "invalid_interval_bounds",
            )
        require(
            manifest["roles"]["V5"] == "development"
            and manifest["roles"]["V7"] == manifest["roles"]["V8"],
            "source_group_leakage",
        )
        duration = sum(
            Fraction(e["timing"]["stop_exclusive"]) - Fraction(e["timing"]["start"])
            for e in entries
            if e["purpose"] == "evaluation_block"
        )
        require(0 <= duration <= 360, "evaluation_budget_exceeded")
        if self.pilot == "motion-v2":
            require(
                len(entries) <= 12
                and all(
                    e["recording"] in ("V4", "V6")
                    and e["role"] == "train_candidate"
                    and e["purpose"] == "background_candidate"
                    and e["id"].startswith(e["recording"] + "-motion-")
                    for e in entries
                ),
                "invalid_motion_pilot_scope",
            )
        else:
            require(all("-motion-" not in e["id"] for e in entries), "wrong_interval_pilot")
        for i, a in enumerate(entries):
            for b in entries[i + 1 :]:
                if a["recording"] == b["recording"]:
                    require(
                        a["stop"] <= b["start"] or b["stop"] <= a["start"],
                        "overlapping_pilot_intervals",
                    )
        return manifest, digest

    def _history(self, fd, manifest_hash, manifest):
        head, records = manifest_hash, []
        names = sorted(n for n in os.listdir(fd) if n.startswith("v"))
        for number, name in enumerate(names, 1):
            require(name == f"v{number:08d}", "interval_history_gap", "stale")
            raw, digest = self.base.store.read(f"{self.directory}/{name}/review.json")
            record = strict_json(raw)
            require(
                record["parent"] == head
                and record["manifest_sha256"] == manifest_hash
                and record["source"] == "manual"
                and record["version"] == number,
                "interval_history_changed",
                "stale",
            )
            decision = IntervalDecision.model_validate(record["payload"])
            require(decision.expected_head == head, "interval_history_changed", "stale")
            entries = {e["id"]: e for e in manifest["entries"]}
            require(record["interval_id"] in entries, "invalid_history_interval", "stale")
            entry = entries[record["interval_id"]]
            require(
                record["provenance"] == entry
                and record["training_enabled"] is False
                and record["exhaustive_event_reference"] is False,
                "interval_history_changed",
                "stale",
            )
            expected = (
                "verified_background"
                if decision.decision == "no_punch"
                else "needs_event_labeling"
                if decision.decision == "punch_present"
                else "unresolved"
            )
            require(
                record["status"] == expected
                and bool(decision.inspected_frames)
                and all(entry["start"] <= i < entry["stop"] for i in decision.inspected_frames),
                "invalid_history_decision",
                "stale",
            )
            if decision.decision == "no_punch":
                require(
                    set(decision.inspected_frames) == set(range(entry["start"], entry["stop"])),
                    "incomplete_history_inspection",
                    "stale",
                )
            records.append(record | {"sha256": digest})
            head = digest
        return head, records

    def snapshot(self):
        manifest, digest = self.manifest()
        with ReviewWorkspace(self.base)._locked(self.directory) as fd:
            head, history = self._history(fd, digest, manifest)
        return dict(manifest=manifest, head=head, history=history)

    def detail(self, key):
        snap = self.snapshot()
        entries = [e for e in snap["manifest"]["entries"] if e["id"] == key]
        require(len(entries) == 1, "unknown_interval", "missing")
        e = entries[0]
        spec = self.base._recording(e["recording"])
        prep = self.base._preparation(e["recording"], spec)
        require(
            spec.inputs == e["source_inputs"]
            and prep["inputs"]["video_sha256"] == e["video_sha256"],
            "interval_source_changed",
            "stale",
        )
        with self.frames.lock:
            arrays = self.frames._load(e["recording"], spec, prep)
        timing = interval_timing(
            arrays, e["start"], e["stop"], prep.get("confirmed_scene_cuts", [])
        )
        require(timing == e["timing"], "interval_timing_changed", "stale")
        rows = self.base._rows(e["recording"], spec)
        refs = []
        for row in rows:
            bounds = row.get("source_interval_half_open")
            if bounds and bounds[0] < e["stop"] and bounds[1] > e["start"]:
                refs.append(row)
        history = [r for r in snap["history"] if r["interval_id"] == key]
        return dict(
            entry=e,
            head=snap["head"],
            history=history,
            annotations=refs,
            status=history[-1]["status"] if history else "unknown",
        )

    def save(self, key, decision):
        detail = self.detail(key)  # Source hashes, full interval timing and known cuts.
        e = detail["entry"]
        require(
            all(e["start"] <= f < e["stop"] for f in decision.inspected_frames),
            "inspection_outside_interval",
        )
        if decision.decision == "no_punch":
            require(
                set(decision.inspected_frames) == set(range(e["start"], e["stop"])),
                "watch_entire_interval_before_certifying",
            )
            # Existing positive source intervals are conflicting evidence, not silently relabeled.
            require(not detail["annotations"], "source_punch_conflict_needs_review")
        require(bool(decision.inspected_frames), "inspect_interval_before_saving")
        # Verify source video and actual boundary frames through the established exact decoder.
        for index in sorted({min(decision.inspected_frames), max(decision.inspected_frames)}):
            self.frames.frame(e["recording"], index)
        manifest, digest = self.manifest()
        payload = decision.model_dump(mode="json")
        with ReviewWorkspace(self.base)._locked(self.directory) as fd:
            head, records = self._history(fd, digest, manifest)
            for prior in records:
                if prior["payload"]["request_id"] == decision.request_id:
                    require(
                        prior["interval_id"] == key and prior["payload"] == payload,
                        "request_id_reused",
                        "stale",
                    )
                    return prior
            require(head == decision.expected_head, "interval_review_changed_reload", "stale")
            status = (
                "verified_background"
                if decision.decision == "no_punch"
                else "needs_event_labeling"
                if decision.decision == "punch_present"
                else "unresolved"
            )
            record = dict(
                schema_version="interval-review-1.0",
                version=len(records) + 1,
                manifest_sha256=digest,
                parent=head,
                interval_id=key,
                source="manual",
                reviewed_at=datetime.now(UTC).isoformat(),
                payload=payload,
                status=status,
                provenance=e,
                exhaustive_event_reference=False,
                training_enabled=False,
            )
            raw = encode(record)
            temporary = "pending-" + uuid.uuid4().hex
            os.mkdir(temporary, 0o700, dir_fd=fd)
            bundle = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                target = os.open(
                    "review.json",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=bundle,
                )
                with os.fdopen(target, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.fsync(bundle)
                os.rename(temporary, f"v{len(records) + 1:08d}", src_dir_fd=fd, dst_dir_fd=fd)
                os.fsync(fd)
            finally:
                os.close(bundle)
        return record | {"sha256": sha(raw)}
