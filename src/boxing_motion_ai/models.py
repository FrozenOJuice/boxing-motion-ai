"""Versioned contracts. Unknown provenance stays null; no negatives are inferred."""

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["1.0"] = "1.0"


class PunchType(StrEnum):
    JAB = "jab"
    CROSS = "cross"
    LEAD_HOOK = "lead_hook"
    REAR_HOOK = "rear_hook"
    LEAD_UPPERCUT = "lead_uppercut"
    REAR_UPPERCUT = "rear_uppercut"
    UNKNOWN = "unknown"


class BackgroundState(StrEnum):
    NO_PUNCH = "no_punch"  # Only after explicit human review, never for unlabeled gaps.


class Event(Contract):
    event_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    boxer_id: str | None = None
    session_id: str | None = None
    frame_start: int = Field(ge=0, strict=True)
    frame_end: int = Field(ge=0, strict=True)
    frame_base: Literal[0, 1]
    end_convention: Literal["inclusive", "exclusive"]
    frame_onset: int | None = Field(default=None, ge=0, strict=True)
    frame_peak: int | None = Field(default=None, ge=0, strict=True)
    punch_type: PunchType
    hand: Literal["lead", "rear"] | None = None
    annotator: str | None = None
    source_dataset: Literal["boxingvi"] = "boxingvi"
    annotation_version: str = Field(min_length=1)
    review_status: Literal["unreviewed", "reviewed", "ambiguous"] = "unreviewed"
    source_workbook: str
    source_sheet: str
    source_row: int = Field(ge=1)
    source_class: str

    @model_validator(mode="after")
    def bounds(self):
        start, stop = self.zero_based_interval()
        if start < 0 or stop <= start:
            raise ValueError("Frame interval must be nonempty and respect frame_base")
        for frame in (self.frame_onset, self.frame_peak):
            if frame is not None and not start <= frame - self.frame_base < stop:
                raise ValueError("Onset/peak must lie within the event")
        if self.frame_onset is not None and self.frame_peak is not None:
            if self.frame_peak < self.frame_onset:
                raise ValueError("Peak precedes onset")
        return self

    def zero_based_interval(self) -> tuple[int, int]:
        return (
            self.frame_start - self.frame_base,
            self.frame_end - self.frame_base + (self.end_convention == "inclusive"),
        )


class VideoMetadata(Contract):
    video_id: str
    path: str  # Dataset-root-relative POSIX path.
    duration_seconds: float | None = Field(default=None, gt=0)
    fps: float | None = Field(default=None, gt=0)
    frame_rate_rational: str | None = None
    nominal_frame_rate: str | None = None
    time_base: str | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    frame_count: int | None = Field(default=None, gt=0)
    frame_count_source: str | None = None
    boxer_id: str | None = None
    session_id: str | None = None
    recording_source: str | None = None
    download_status: Literal["present"] = "present"
    licensing_status: Literal["unverified"] = "unverified"
    probe_status: Literal["ok", "partial", "failed"] = "failed"


class DatasetMetadata(Contract):
    name: Literal["boxingvi"] = "boxingvi"
    source_url: str = "https://github.com/Bikudebug/BoxingVI"
    found: bool
    licensing_status: Literal["unverified"] = "unverified"
    redistribution_status: Literal["unverified"] = "unverified"


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError("Use a portable dataset-root-relative POSIX path without '..'")
    return path.as_posix()
