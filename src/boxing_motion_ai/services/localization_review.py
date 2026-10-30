"""Bounded individual-punch review using the existing exact-frame event editor.

Separate append-only ledgers; no predictions, auto-completions or training side effects.
"""

from typing import Literal

from pydantic import StrictBool, model_validator

from .dashboard import require
from .interval_events import EventDecision, HumanPunch, IntervalEvents
from .interval_review import IntervalDecision
from .review_workspace import encode, sha


class LocalizationRegion(HumanPunch):
    kind: Literal["punch", "uncertain"] = "punch"

    @model_validator(mode="after")
    def uncertainty(self):
        if self.kind == "uncertain" and not self.type_uncertain:
            raise ValueError("Uncertain motion cannot carry a verified punch type")
        return self


class LocalizationDecision(EventDecision):
    event: LocalizationRegion | None = None
    no_punches_observed: StrictBool = False
    visible_continuous: StrictBool = False

    @model_validator(mode="after")
    def absence(self):
        if (self.no_punches_observed or self.visible_continuous) and self.operation != "complete":
            raise ValueError("Absence requires explicit full review")
        return self


class LocalizationReview(IntervalEvents):
    directory = "reports/boxingvi/localization-event-reviews-v1"
    decision_type = LocalizationDecision
    review_schema = "human-localization-review-1.0"

    def context(self, key):
        require(self.intervals.pilot == "localization-v1", "wrong_localization_pilot")
        detail = self.intervals.detail(key)
        _, manifest_hash = self.intervals.manifest()
        seed = dict(
            schema_version="localization-review-binding-1.0",
            interval_id=key,
            provenance=detail["entry"],
            source_review_sha256=manifest_hash,
        )
        return detail, seed, sha(encode(seed))

    def event_metadata(self, entry, event):
        return super().event_metadata(entry, event) | dict(
            left_censored=event.start <= entry["start"],
            right_censored=event.stop >= entry["stop"],
        )

    def apply(self, entry, decision, events, complete):
        if decision.operation != "complete":
            return super().apply(entry, decision, events, complete)
        require(
            decision.watched_entire and decision.all_punches_labeled,
            "explicit_event_completion_required",
        )
        require(
            decision.no_punches_observed == (not events),
            "explicit_no_punch_attestation_required",
        )
        require(
            bool(events) or decision.visible_continuous,
            "no_punch_review_requires_visible_boxer",
        )
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
        require(
            self.intervals._viewing(entry, coverage)["complete"]
            if coverage.playback_coverage
            else set(coverage.inspected_frames) == set(range(entry["start"], entry["stop"])),
            "watch_entire_interval_before_completing",
        )
        return True
