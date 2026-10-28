"""Blind manual activity references for six reserved blocks. No model prediction loading."""

import hashlib
from fractions import Fraction
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, StrictBool, StrictInt, model_validator

from ..data.annotation_review import StrictModel
from ..data.interval_pilot import GROUPS
from .dashboard import require
from .interval_events import IntervalEvents
from .interval_review import IntervalDecision
from .review_workspace import encode, sha
from .viewing_coverage import PlaybackCoverage

SCOPE = tuple(f"{v}-eval-{i}" for v in ("V7", "V8", "V10") for i in (1, 2))
PROTOCOL = "configs/activity_eval_protocol_v1.json"
DIRECTORY = "reports/boxingvi/activity-evaluation-reviews-v1"


class ActivityRegion(StrictModel):
    event_id: Annotated[UUID, Field(strict=False)]
    start: StrictInt = Field(ge=0)
    stop: StrictInt = Field(gt=0)
    kind: Literal["activity", "uncertain"]

    @model_validator(mode="after")
    def bounds(self):
        if self.stop <= self.start:
            raise ValueError("End must follow start")
        return self


class ActivityDecision(StrictModel):
    request_id: Annotated[UUID, Field(strict=False)]
    expected_head: str = Field(pattern=r"^[0-9a-f]{64}$")
    operation: Literal["upsert", "remove", "complete"]
    event: ActivityRegion | None = None
    event_id: Annotated[UUID, Field(strict=False)] | None = None
    inspected_frames: list[StrictInt] = Field(default_factory=list, max_length=2000)
    playback_coverage: PlaybackCoverage | None = None
    watched_entire: StrictBool = False
    all_activity_labeled: StrictBool = False
    no_activity_observed: StrictBool = False

    @model_validator(mode="after")
    def fields(self):
        if (self.operation == "upsert") != (self.event is not None) or (
            self.operation == "remove"
        ) != (self.event_id is not None):
            raise ValueError("Operation requires exactly its matching event fields")
        if len(set(self.inspected_frames)) != len(self.inspected_frames):
            raise ValueError("Duplicate inspected frame")
        if self.operation != "complete" and (
            self.watched_entire
            or self.all_activity_labeled
            or self.no_activity_observed
            or self.playback_coverage is not None
        ):
            raise ValueError("Only explicit completion may attest coverage")
        return self


class ActivityEvaluation(IntervalEvents):
    directory = DIRECTORY
    decision_type = ActivityDecision
    review_schema = "human-activity-evaluation-review-1.0"

    def protocol(self):
        p, digest = self.base.store.json(PROTOCOL)
        require(
            p["schema_version"] == "activity-evaluation-protocol-1.0"
            and tuple(p["interval_ids"]) == SCOPE,
            "invalid_activity_protocol",
        )
        require(
            p["reference"]["short_pause_seconds"] == "1/4"
            and p["primary"]["uncertain_block_policy"] == "exclude_whole_block",
            "unsupported_activity_protocol",
        )
        for key in ("matching_code", "decoder_config", "checkpoint"):
            ref = p[key]
            with self.base.store.open_regular(ref["path"]) as stream:
                require(
                    hashlib.file_digest(stream, "sha256").hexdigest() == ref["sha256"],
                    "frozen_evaluation_dependency_changed",
                    "stale",
                )
        _, actual = self.intervals.manifest()
        require(
            actual == p["interval_manifest"]["sha256"],
            "evaluation_source_manifest_changed",
            "stale",
        )
        return p, digest

    def context(self, key):
        require(
            key in SCOPE and self.intervals.pilot == "original",
            "evaluation_interval_not_approved",
            "missing",
        )
        _, protocol_hash = self.protocol()
        detail = self.intervals.detail(key)
        e = detail["entry"]
        require(
            e["purpose"] == "evaluation_block"
            and e["role"] == "evaluation_reserved"
            and e["recording"] in ("V7", "V8", "V10")
            and e["source_group_guard"] == GROUPS[e["recording"]],
            "evaluation_split_leakage",
        )
        seed = dict(
            schema_version="activity-evaluation-binding-1.0",
            interval_id=key,
            provenance=e,
            protocol_sha256=protocol_hash,
        )
        return detail, seed, sha(encode(seed))

    def event_metadata(self, entry, event):
        require(
            entry["start"] <= event.start < event.stop <= entry["stop"],
            "activity_outside_reviewed_block",
        )
        metadata = super().event_metadata(entry, event)
        return metadata | dict(
            left_censored=event.kind == "activity" and event.start == entry["start"],
            right_censored=event.kind == "activity" and event.stop == entry["stop"],
        )

    def apply(self, entry, decision, events, complete):
        if decision.operation == "upsert":
            event = decision.event
            old = events.get(str(event.event_id))
            needed = set()
            if not old or event.start != old["start"]:
                needed.add(event.start)
            if not old or event.stop != old["stop"]:
                needed.add(event.stop - 1)
            require(needed.issubset(decision.inspected_frames), "inspect_both_event_boundaries")
            require(
                all(entry["start"] <= i < entry["stop"] for i in decision.inspected_frames),
                "inspection_outside_interval",
            )
            events[str(event.event_id)] = event.model_dump(mode="json") | self.event_metadata(
                entry, event
            )
            require(len(events) <= 100, "too_many_activity_regions")
            return False
        if decision.operation == "remove":
            require(str(decision.event_id) in events, "unknown_human_event", "missing")
            del events[str(decision.event_id)]
            return False
        require(
            decision.watched_entire and decision.all_activity_labeled,
            "explicit_activity_completion_required",
        )
        require(
            bool(events) or decision.no_activity_observed,
            "explicit_no_activity_confirmation_required",
        )
        require(
            not events or not decision.no_activity_observed,
            "contradictory_no_activity_confirmation",
        )
        require(
            all(entry["start"] <= i < entry["stop"] for i in decision.inspected_frames),
            "inspection_outside_interval",
        )
        proof = IntervalDecision(
            request_id=str(decision.request_id),
            expected_head=decision.expected_head,
            decision="punch_present",
            inspected_frames=decision.inspected_frames,
            playback_coverage=decision.playback_coverage,
        )
        require(
            self.intervals._viewing(entry, proof)["complete"],
            "watch_entire_interval_before_completing",
        )
        if not any(e["kind"] == "uncertain" for e in events.values()):
            ordered = sorted(events.values(), key=lambda e: e["start"])
            for first, second in zip(ordered, ordered[1:], strict=False):
                gap = Fraction(second["timing"]["start"]) - Fraction(
                    first["timing"]["stop_exclusive"]
                )
                require(gap >= Fraction(1, 4), "combine_activity_bouts_under_quarter_second")
        return True

    def detail(self, key):
        entry, seed, head = self.context(key)
        from .review_workspace import ReviewWorkspace

        with ReviewWorkspace(self.base)._locked(f"{self.directory}/{key}") as fd:
            state = self.history(fd, key, seed, head)
        uncertain = sum(e["kind"] == "uncertain" for e in state["events"])
        return state | dict(
            entry=entry["entry"],
            interval_id=key,
            uncertain_sections=uncertain,
            primary_eligible=state["complete"] and not uncertain,
            protocol_sha256=seed["protocol_sha256"],
            predictions_hidden=True,
        )

    def status(self):
        states = {key: self.detail(key) for key in SCOPE}
        return {
            key: {k: s[k] for k in ("complete", "events", "uncertain_sections", "primary_eligible")}
            for key, s in states.items()
        }
