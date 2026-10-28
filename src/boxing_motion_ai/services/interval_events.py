"""Human event revisions for the five approved V4 intervals, separate from source reviews."""

import os
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, StrictBool, StrictInt, model_validator

from ..data.annotation_review import StrictModel
from ..models import PunchType
from .artifacts import strict_json
from .dashboard import require
from .interval_review import IntervalDecision, interval_timing
from .review_workspace import ReviewWorkspace, append_review_bundle, encode, sha
from .viewing_coverage import PlaybackCoverage

SCOPE = tuple(f"V4-bg-{i}" for i in range(1, 6))
DIRECTORY = "reports/boxingvi/interval-events-v1"


class HumanPunch(StrictModel):
    event_id: Annotated[UUID, Field(strict=False)]
    start: StrictInt = Field(ge=0)
    stop: StrictInt = Field(gt=0)  # Native zero-based exclusive endpoint.
    punch_type: Annotated[PunchType, Field(strict=False)]
    type_uncertain: StrictBool

    @model_validator(mode="after")
    def valid_event(self):
        if self.stop <= self.start:
            raise ValueError("End must follow start")
        if self.type_uncertain != (self.punch_type == PunchType.UNKNOWN):
            raise ValueError("Uncertain type must remain unknown, not a verified six-class label")
        return self


class EventDecision(StrictModel):
    request_id: Annotated[UUID, Field(strict=False)]
    expected_head: str = Field(pattern=r"^[0-9a-f]{64}$")
    operation: Literal["upsert", "remove", "complete"]
    event: HumanPunch | None = None
    event_id: Annotated[UUID, Field(strict=False)] | None = None
    inspected_frames: list[StrictInt] = Field(default_factory=list, max_length=2000)
    playback_coverage: PlaybackCoverage | None = None
    watched_entire: StrictBool = False
    all_punches_labeled: StrictBool = False

    @model_validator(mode="after")
    def operation_fields(self):
        if (self.operation == "upsert") != (self.event is not None):
            raise ValueError("Add/change requires an event")
        if (self.operation == "remove") != (self.event_id is not None):
            raise ValueError("Remove requires an event ID")
        if len(set(self.inspected_frames)) != len(self.inspected_frames):
            raise ValueError("Duplicate inspected frame")
        if self.operation != "complete" and (
            self.watched_entire or self.all_punches_labeled or self.playback_coverage is not None
        ):
            raise ValueError("Only explicit completion can attest full coverage")
        return self


class IntervalEvents:
    directory = DIRECTORY
    decision_type = EventDecision
    review_schema = "human-interval-event-review-1.0"

    def __init__(self, intervals):
        self.intervals = intervals
        self.base, self.frames = intervals.base, intervals.frames

    def context(self, key):
        require(
            key in SCOPE and self.intervals.pilot == "original",
            "event_interval_not_approved",
            "missing",
        )
        detail = self.intervals.detail(key)
        require(
            detail["entry"]["recording"] == "V4" and detail["entry"]["role"] == "train_candidate",
            "event_source_not_training",
        )
        require(
            detail["history"] and detail["history"][-1]["status"] == "needs_event_labeling",
            "event_interval_needs_positive_review",
        )
        seed = dict(
            schema_version="interval-events-1.0",
            interval_id=key,
            provenance=detail["entry"],
            source_review_sha256=detail["history"][-1]["sha256"],
        )
        return detail, seed, sha(encode(seed))

    def event_metadata(self, entry, event):
        require(
            entry["context_start"] <= event.start < event.stop <= entry["context_end"] + 1,
            "event_outside_available_context",
        )
        require(
            event.start < entry["stop"] and event.stop > entry["start"], "event_outside_interval"
        )
        spec = self.base._recording(entry["recording"])
        prep = self.base._preparation(entry["recording"], spec)
        with self.frames.lock:
            arrays = self.frames._load(entry["recording"], spec, prep)
            timing = interval_timing(
                arrays, event.start, event.stop, prep.get("confirmed_scene_cuts", [])
            )
            pts = {
                name: {k: int(arrays[k][i]) for k in ("pts", "time_base_num", "time_base_den")}
                for name, i in (("start", event.start), ("stop_exclusive", event.stop))
            }
        return dict(
            timing=timing,
            endpoint_pts=pts,
            partial_overlap=event.start < entry["start"] or event.stop > entry["stop"],
        )

    def apply(self, entry, decision, events, complete):
        """Validate and replay every revision; never infer background from empty space."""
        if decision.operation == "upsert":
            event = decision.event
            metadata = self.event_metadata(entry, event)
            require(
                all(
                    type(i) is int and entry["context_start"] <= i <= entry["context_end"]
                    for i in decision.inspected_frames
                ),
                "event_inspection_outside_context",
            )
            previous = events.get(str(event.event_id))
            required = set()
            if not previous or previous["start"] != event.start:
                required.add(event.start)
            if not previous or previous["stop"] != event.stop:
                required.add(event.stop - 1)
            require(
                required.issubset(decision.inspected_frames),
                "inspect_both_event_boundaries",
            )
            events[str(event.event_id)] = event.model_dump(mode="json") | metadata
            require(len(events) <= 100, "too_many_interval_events")
            complete = False
        elif decision.operation == "remove":
            require(str(decision.event_id) in events, "unknown_human_event", "missing")
            del events[str(decision.event_id)]
            complete = False
        else:
            require(
                decision.watched_entire and decision.all_punches_labeled,
                "explicit_event_completion_required",
            )
            require(bool(events), "positive_interval_needs_events")
            coverage = IntervalDecision(
                request_id=str(decision.request_id),
                expected_head=decision.expected_head,
                decision="punch_present",
                inspected_frames=decision.inspected_frames,
                playback_coverage=decision.playback_coverage,
            )
            require(
                all(entry["start"] <= i < entry["stop"] for i in coverage.inspected_frames),
                "inspection_outside_interval",
            )
            if coverage.playback_coverage:
                require(
                    self.intervals._viewing(entry, coverage)["complete"],
                    "watch_entire_interval_before_completing",
                )
            else:
                require(
                    set(coverage.inspected_frames) == set(range(entry["start"], entry["stop"])),
                    "watch_entire_interval_before_completing",
                )
            complete = True
        return complete

    def history(self, fd, key, seed, head):
        events, records, complete = {}, [], False
        for number, name in enumerate(sorted(n for n in os.listdir(fd) if n.startswith("v")), 1):
            require(name == f"v{number:08d}", "event_history_gap", "stale")
            raw, digest = self.base.store.read(f"{self.directory}/{key}/{name}/review.json")
            record = strict_json(raw)
            require(
                record["parent"] == head
                and record["binding"] == seed
                and record["version"] == number
                and record["source"] == "manual"
                and record["schema_version"] == self.review_schema
                and record["background_inferred"] is False
                and record["training_enabled"] is False,
                "event_history_changed",
                "stale",
            )
            decision = self.decision_type.model_validate(record["payload"])
            require(decision.expected_head == head, "event_history_changed", "stale")
            complete = self.apply(seed["provenance"], decision, events, complete)
            require(
                record["events"] == list(events.values()) and record["complete"] == complete,
                "event_history_changed",
                "stale",
            )
            head = digest
            records.append(record | {"sha256": digest})
        return dict(head=head, events=list(events.values()), complete=complete, history=records)

    def detail(self, key):
        detail, seed, head = self.context(key)
        with ReviewWorkspace(self.base)._locked(f"{self.directory}/{key}") as fd:
            state = self.history(fd, key, seed, head)
        return state | dict(
            interval_id=key,
            uncertain_types=sum(e["type_uncertain"] for e in state["events"]),
            source_review_sha256=seed["source_review_sha256"],
        )

    def extra_record(self, entry, decision):
        return {}

    def save(self, key, decision):
        detail, seed, initial = self.context(key)
        entry = detail["entry"]
        if decision.event:
            self.event_metadata(entry, decision.event)
            for index in {decision.event.start, decision.event.stop - 1}:
                self.frames.frame(
                    entry["recording"], index
                )  # Actual decoder validates original source hash/PTS.
        else:
            for index in (entry["start"], entry["stop"] - 1):
                self.frames.frame(entry["recording"], index)
        with ReviewWorkspace(self.base)._locked(f"{self.directory}/{key}") as fd:
            state = self.history(fd, key, seed, initial)
            payload = decision.model_dump(mode="json")
            for prior in state["history"]:
                if prior["payload"]["request_id"] == payload["request_id"]:
                    require(prior["payload"] == payload, "request_id_reused", "stale")
                    return prior
            require(state["head"] == decision.expected_head, "event_review_changed_reload", "stale")
            events = {e["event_id"]: e for e in state["events"]}
            complete = self.apply(entry, decision, events, state["complete"])
            record = dict(
                schema_version=self.review_schema,
                version=len(state["history"]) + 1,
                parent=state["head"],
                binding=seed,
                source="manual",
                reviewed_at=datetime.now(UTC).isoformat(),
                payload=payload,
                events=list(events.values()),
                complete=complete,
                background_inferred=False,
                training_enabled=False,
            )
            record.update(self.extra_record(entry, decision))
            return append_review_bundle(fd, record["version"], record)
