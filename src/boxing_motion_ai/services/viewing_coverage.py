"""Versioned continuous-viewing evidence, distinct from individual-frame inspection."""

from typing import Literal

from pydantic import Field, model_validator

from ..data.annotation_review import StrictModel
from .dashboard import require

EPSILON = 0.0001  # Numeric endpoint tolerance only; much smaller than one source frame.


class ViewingSpan(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    rate: Literal[1.0, 0.5, 0.25]
    wall_seconds: float = Field(gt=0, le=600)
    samples: int = Field(ge=2, le=100000)

    @model_validator(mode="after")
    def realtime(self):
        if self.end <= self.start or self.end - self.start > self.wall_seconds * self.rate + 0.02:
            raise ValueError(
                "Viewing progress must be bounded by elapsed real time and playback rate"
            )
        return self


class PlaybackCoverage(StrictModel):
    schema_version: Literal["continuous-viewing-1.0"]
    policy: Literal["visible-realtime-1.0"]
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    interval_id: str
    segments: list[ViewingSpan] = Field(max_length=256)


def covered(ranges, start, stop):
    end = start
    for left, right in sorted(ranges):
        if right <= end:
            continue
        if left > end + EPSILON:
            return False
        end = max(end, right)
        if end >= stop - EPSILON:
            return True
    return end >= stop - EPSILON


def validate_viewing(entry, descriptor, evidence, inspected):
    times = [float(t - descriptor["times"][0]) for t in descriptor["times"]]
    lookup = {index: i for i, index in enumerate(descriptor["indices"])}
    start = times[lookup[entry["start"]]]
    stop = times[entry["stop"] - descriptor["indices"][0]]
    continuous = []
    if evidence is not None:
        require(
            evidence.interval_id == entry["id"] and evidence.cache_key == descriptor["cache_key"],
            "viewing_source_or_interval_changed",
            "stale",
        )
        for segment in evidence.segments:
            require(segment.end <= times[-1] + EPSILON, "viewing_outside_preview")
            continuous.append((segment.start, segment.end))
    exact = [(times[lookup[i]], times[lookup[i] + 1]) for i in inspected]
    playback_complete = covered(continuous, start, stop)
    exact_complete = covered(exact, start, stop)
    complete = covered(continuous + exact, start, stop)
    return dict(
        schema_version="viewing-validation-1.0",
        cache_key=descriptor["cache_key"],
        media_start=start,
        media_stop=stop,
        complete=complete,
        playback_complete=playback_complete,
        exact_complete=exact_complete,
        has_content=any(b > start and a < stop for a, b in continuous + exact),
        completion_mode="continuous_playback"
        if playback_complete
        else "exact_frames"
        if exact_complete
        else "combined"
        if complete
        else "incomplete",
    )
