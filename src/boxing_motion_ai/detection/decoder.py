"""Past-only hysteresis and elapsed-time punch event decoding; no labels as input."""

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from pydantic import Field, model_validator

from ..data.annotation_review import StrictModel

VERSION = "causal-binary-events-1.0"


class DecoderConfig(StrictModel):
    schema_version: Literal["causal-binary-decoder-1.0"] = "causal-binary-decoder-1.0"
    activation_threshold: float = Field(default=0.65, ge=0, le=1)
    deactivation_threshold: float = Field(default=0.35, ge=0, le=1)
    minimum_activation_seconds: str = "2/25"
    termination_seconds: str = "2/25"
    maximum_timestamp_gap_seconds: str = "1/10"

    @model_validator(mode="after")
    def policy(self):
        try:
            values = [
                Fraction(getattr(self, field))
                for field in (
                    "minimum_activation_seconds",
                    "termination_seconds",
                    "maximum_timestamp_gap_seconds",
                )
            ]
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("Durations must be finite exact rational seconds") from exc
        if self.activation_threshold <= self.deactivation_threshold:
            raise ValueError("Activation threshold must exceed deactivation threshold")
        for value in values[:2]:
            if not 0 <= value <= 5:
                raise ValueError("Decoder durations must be exact seconds in [0,5]")
        if not 0 < values[2] <= 1:
            raise ValueError("Positive exact timestamp gap <=1 second required")
        return self


@dataclass(frozen=True)
class Observation:
    recording: str
    interval_id: str
    frame_index: int
    pts: int
    time_base_num: int
    time_base_den: int
    probability: float | None
    observable: bool = True
    reviewed: bool = True
    scene_id: int = 0
    segment_id: int = 0
    discontinuity_before: bool = False

    def __post_init__(self):
        if any(
            type(v) is not int
            for v in (
                self.frame_index,
                self.pts,
                self.time_base_num,
                self.time_base_den,
                self.scene_id,
                self.segment_id,
            )
        ):
            raise ValueError("Exact integer frame indices, PTS and timebases required")
        if self.frame_index < 0 or min(self.time_base_num, self.time_base_den) <= 0:
            raise ValueError("Invalid frame/timebase")
        if any(
            type(v) is not bool for v in (self.observable, self.reviewed, self.discontinuity_before)
        ):
            raise ValueError("Explicit boolean observation flags required")
        if self.probability is not None and (
            isinstance(self.probability, bool)
            or not math.isfinite(self.probability)
            or not 0 <= self.probability <= 1
        ):
            raise ValueError("Finite probability in [0,1] required")
        if self.observable and self.reviewed and self.probability is None:
            raise ValueError("Observable reviewed frames require a probability")

    @property
    def time(self):
        return Fraction(self.pts * self.time_base_num, self.time_base_den)

    @property
    def scope(self):
        return self.recording, self.interval_id

    def point(self):
        return dict(
            frame_index=self.frame_index,
            pts=self.pts,
            time_base_num=self.time_base_num,
            time_base_den=self.time_base_den,
            timestamp_exact=str(self.time),
        )

    @classmethod
    def from_row(cls, row):
        # Deliberately ignore reviewed_target, annotations, punch types and other metadata.
        return cls(
            recording=row["recording"],
            interval_id=row["interval_id"],
            frame_index=row["frame_index"],
            pts=row["pts"],
            time_base_num=row["time_base_num"],
            time_base_den=row["time_base_den"],
            probability=row["probability_punch"],
            observable=row["observable"],
            reviewed=row.get("reviewed", False),
            scene_id=row["scene_id"],
            segment_id=row["segment_id"],
            discontinuity_before=row.get("discontinuity_before", False),
        )


class CausalEventDecoder:
    """push() emits activation immediately; finalized records never backdate that emission.

    Chunk boundaries are not interval ends. finish() is an explicit end-of-coverage
    control marker, optionally carrying a known exact exclusive endpoint.
    """

    def __init__(self, config=None):
        self.config = config or DecoderConfig()
        self.previous = None
        self.samples = []
        self.low = []
        self.emission = None
        self.event_id = None
        self.left_censored = False
        self.segment_start = True
        self.events = []
        self.updates = []
        self.counter = 0
        self.discarded_candidates = 0

    def _clear(self):
        self.samples = []
        self.low = []
        self.emission = None
        self.event_id = None

    def _emit(self, now):
        self.counter += 1
        first = self.samples[0]
        self.event_id = f"{first.recording}:{first.interval_id}:punch-{self.counter}"
        self.emission = now
        return dict(
            kind="activated",
            event_id=self.event_id,
            recording=first.recording,
            interval_id=first.interval_id,
            estimated_start=first.point(),
            first_emission=now.point(),
            left_censored=self.left_censored,
            confidence_at_emission=max(s.probability for s in self.samples),
        )

    def _close(self, reason, now=None, *, censored=True, stop_frame=None, stop_time=None):
        if not self.samples:
            return []
        if self.emission is None:
            self.discarded_candidates += 1
            self._clear()
            return []
        first, last = self.samples[0], self.samples[-1]
        # Pending low observations are not included unless a subsequent observed
        # recovery already established that the dip was bridged.
        if self.low:
            stop_frame, stop_time = self.low[0].frame_index, self.low[0].time
        if stop_frame is None:
            stop_frame = last.frame_index + 1
        same_clock = now is not None and now.scope == first.scope
        if (
            stop_time is None
            and same_clock
            and now.frame_index == last.frame_index + 1
            and now.time - last.time <= Fraction(self.config.maximum_timestamp_gap_seconds)
        ):
            stop_time = now.time
        decision_point = (
            now
            if same_clock
            else (
                self.previous
                if self.previous is not None and self.previous.scope == first.scope
                else last
            )
        )
        final_time = decision_point.time
        observed = decision_point.frame_index
        values = [s.probability for s in self.samples]
        record = dict(
            schema_version=VERSION,
            source="model_prediction",
            event_id=self.event_id,
            recording=first.recording,
            interval_id=first.interval_id,
            estimated_start=first.point(),
            estimated_end_inclusive=last.point(),
            stop_frame_exclusive=stop_frame,
            stop_timestamp_exact=None if stop_time is None else str(stop_time),
            first_emission=self.emission.point(),
            activation_delay_seconds_exact=str(self.emission.time - first.time),
            finalized_at=dict(
                timestamp_exact=str(final_time),
                observed_through_frame=observed,
                clock_basis="current_source_observation"
                if same_clock
                else "last_observed_source_time",
                trigger=reason,
            ),
            left_censored=self.left_censored,
            right_censored=censored,
            confidence=dict(
                peak_probability=max(values),
                mean_probability=sum(values) / len(values),
                observed_samples=len(values),
                calibrated=False,
            ),
        )
        self.events.append(record)
        self._clear()
        return [dict(kind="finalized", event=record)]

    def push(self, frame):
        if not isinstance(frame, Observation):
            raise ValueError("Observation required")
        messages = []
        previous = self.previous
        reason = None
        if previous is not None:
            if frame.scope != previous.scope:
                reason = "recording_or_interval_boundary"
            elif frame.time <= previous.time or frame.frame_index <= previous.frame_index:
                raise ValueError("Nonmonotonic/repeated observations require a new stream")
            elif frame.discontinuity_before:
                reason = "explicit_discontinuity"
            elif frame.scene_id != previous.scene_id:
                reason = "scene_cut"
            elif frame.segment_id != previous.segment_id:
                reason = "segment_change"
            elif frame.frame_index != previous.frame_index + 1:
                reason = "missing_frames"
            elif frame.time - previous.time > Fraction(self.config.maximum_timestamp_gap_seconds):
                reason = "timestamp_gap"
        if reason:
            messages.extend(self._close(reason, frame))
            self.segment_start = True
        self.previous = frame
        if not frame.reviewed or not frame.observable:
            messages.extend(
                self._close(
                    "unreviewed_footage" if not frame.reviewed else "unreliable_pose", frame
                )
            )
            self.segment_start = True
        else:
            probability = frame.probability
            if (
                self.emission is not None
                and self.low
                and frame.time - self.low[0].time >= Fraction(self.config.termination_seconds)
            ):
                # A recovery at/after the deadline starts a fresh candidate; it
                # must not silently bridge a completed low-probability gap.
                messages.extend(self._close("sustained_low_probability", frame, censored=False))
            if not self.samples:
                if probability >= self.config.activation_threshold:
                    self.samples = [frame]
                    self.left_censored = self.segment_start
            elif self.emission is None:
                if probability <= self.config.deactivation_threshold:
                    self.discarded_candidates += 1
                    self._clear()
                else:
                    self.samples.append(frame)
            elif probability <= self.config.deactivation_threshold:
                self.low.append(frame)
                if frame.time - self.low[0].time >= Fraction(self.config.termination_seconds):
                    messages.extend(self._close("sustained_low_probability", frame, censored=False))
            else:
                self.samples.extend(self.low)
                self.low = []
                self.samples.append(frame)
            if (
                self.samples
                and self.emission is None
                and frame.time - self.samples[0].time
                >= Fraction(self.config.minimum_activation_seconds)
            ):
                messages.append(self._emit(frame))
            # A middle-band prefix does not establish an inactive signal before
            # the first high threshold crossing at a boundary.
            if self.samples or probability <= self.config.deactivation_threshold:
                self.segment_start = False
        self.updates.extend(messages)
        return messages

    def finish(self, *, stop_frame=None, stop_time=None):
        if (stop_frame is None) != (stop_time is None):
            raise ValueError("Both exact boundary fields required")
        if stop_frame is not None:
            if (
                self.previous is None
                or type(stop_frame) is not int
                or stop_frame != self.previous.frame_index + 1
                or not isinstance(stop_time, Fraction)
                or stop_time <= self.previous.time
            ):
                raise ValueError("Invalid exclusive coverage endpoint")
        messages = self._close("interval_end")
        if messages and stop_time is not None:
            event = messages[0]["event"]
            # A known end marker is not an observed future video frame. If a
            # trailing low run exists its earlier estimated stop is retained.
            if event["stop_timestamp_exact"] is None and stop_time - self.previous.time <= Fraction(
                self.config.maximum_timestamp_gap_seconds
            ):
                event["stop_timestamp_exact"] = str(stop_time)
            event["finalized_at"]["timestamp_exact"] = str(stop_time)
            event["finalized_at"]["clock_basis"] = "coverage_end_control"
        self.updates.extend(messages)
        self.previous = None
        self.segment_start = True
        return messages


def decode_batch(rows, boundaries, config=None):
    """Convenience batch adapter over exactly the same causal state machine."""
    decoder = CausalEventDecoder(config)
    scope = None
    for row in rows:
        frame = Observation.from_row(row)
        if scope is not None and frame.scope != scope:
            stop_frame, stop_time = boundaries[scope]
            decoder.finish(stop_frame=stop_frame, stop_time=stop_time)
        decoder.push(frame)
        scope = frame.scope
    if scope is not None:
        stop_frame, stop_time = boundaries[scope]
        decoder.finish(stop_frame=stop_frame, stop_time=stop_time)
    return dict(
        events=decoder.events,
        updates=decoder.updates,
        discarded_candidates=decoder.discarded_candidates,
    )
