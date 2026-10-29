"""Causal quarter-second bout grouping over the unchanged frozen event decoder."""

from copy import deepcopy
from fractions import Fraction

from .decoder import CausalEventDecoder, Observation

VERSION = "causal-activity-bouts-1.0"
PAUSE = Fraction(1, 4)  # Frozen activity-evaluation v1 rule, not a fitted threshold.


class ActivityDecoder:
    def __init__(self, config):
        self.decoder = CausalEventDecoder(config)
        self.current = None
        self.active = False
        self.counter = 0
        self.bouts = []
        self.updates = []

    def close(self, point, reason, *, censored=False, control=False):
        if self.current is None:
            return
        event = deepcopy(self.current)
        event["right_censored"] = event.get("right_censored", False) or censored
        event["finalized_at"] = dict(
            **point,
            trigger=reason,
            clock_basis="coverage_end_control" if control else "current_source_observation",
        )
        self.bouts.append(event)
        self.updates.append(dict(kind="finalized", bout=deepcopy(event)))
        self.current = None
        self.active = False

    def consume(self, messages, point):
        for message in messages:
            if message["kind"] == "activated":
                start = Fraction(message["estimated_start"]["timestamp_exact"])
                if self.current is not None:
                    stop = self.current.get("stop_timestamp_exact")
                    if stop is None or start - Fraction(stop) >= PAUSE:
                        self.close(point, "quarter_second_silence")
                if self.current is None:
                    self.counter += 1
                    self.current = dict(
                        schema_version=VERSION,
                        source="model_prediction",
                        unit="possible_punching_activity_bout",
                        verified=False,
                        bout_id=f"{message['recording']}:activity-{self.counter}",
                        estimated_start=message["estimated_start"],
                        first_emission=message["first_emission"],
                        left_censored=message["left_censored"],
                        member_event_ids=[],
                    )
                    self.updates.append(dict(kind="activated", bout=deepcopy(self.current)))
                self.active = True
            else:
                event = message["event"]
                if self.current is None:
                    raise ValueError("Activity finalization without a causal activation")
                self.current["member_event_ids"].append(event["event_id"])
                for name in (
                    "estimated_end_inclusive",
                    "stop_frame_exclusive",
                    "stop_timestamp_exact",
                ):
                    self.current[name] = event[name]
                self.active = False
                if event["right_censored"]:
                    self.close(
                        point,
                        event["finalized_at"]["trigger"],
                        censored=True,
                        control=event["finalized_at"]["clock_basis"] == "coverage_end_control",
                    )

    def push(self, row):
        frame = row if isinstance(row, Observation) else Observation.from_row(row)
        previous = self.decoder.previous
        interrupted = (
            not frame.observable
            or not frame.reviewed
            or frame.discontinuity_before
            or (
                previous is not None
                and (
                    frame.scope != previous.scope
                    or frame.scene_id != previous.scene_id
                    or frame.segment_id != previous.segment_id
                    or frame.frame_index != previous.frame_index + 1
                    or frame.time - previous.time
                    > Fraction(self.decoder.config.maximum_timestamp_gap_seconds)
                )
            )
        )
        messages = self.decoder.push(frame)
        self.consume(messages, frame.point())
        if interrupted:
            # Also interrupt a bout whose decoder event ended but whose silence is unresolved.
            self.close(frame.point(), "observation_interruption", censored=True)
        if self.current is not None and not self.active:
            stop = self.current.get("stop_timestamp_exact")
            candidate = self.decoder.samples[0].time if self.decoder.samples else None
            if stop is not None and frame.time - Fraction(stop) >= PAUSE:
                # A candidate begun before the deadline might still activate causally.
                if candidate is None or candidate - Fraction(stop) >= PAUSE:
                    self.close(frame.point(), "quarter_second_silence")
        return messages

    def finish(self, stop_frame, stop_time):
        point = dict(frame_index=stop_frame, timestamp_exact=str(stop_time))
        self.consume(self.decoder.finish(stop_frame=stop_frame, stop_time=stop_time), point)
        self.close(point, "coverage_end", censored=True, control=True)
        return dict(
            bouts=self.bouts,
            bout_updates=self.updates,
            decoder_events=self.decoder.events,
            decoder_updates=self.decoder.updates,
            discarded_candidates=self.decoder.discarded_candidates,
        )
