"""Explicit inspected workbook layouts and provisional per-recording frame mappings.

No FPS conversion is inferred. A configured integer offset maps source frame coordinates
onto decoded indices; author FPS/base/end/trim remain unverified independently.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SOURCE_CLASSES = {
    "Jab": "jab",
    "Cross": "cross",
    "Lead Hook": "lead_hook",
    "Rear Hook": "rear_hook",
    "Lead Uppercut": "lead_uppercut",
    "Rear Uppercut": "rear_uppercut",
}


class RecordingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    video_id: str = Field(pattern=r"^V(?:[1-9]|10)$")
    header: list[str | None]
    helper_formulas: dict[str, str]
    class_map: dict[str, str]
    columns: dict[str, str] = Field(
        default_factory=lambda: {"start": "A", "end": "B", "class": "C"}
    )
    mirrored_columns: dict[str, str] = Field(default_factory=dict)
    frame_offset: int
    end_assumption: Literal["inclusive", "exclusive"]
    source_url: str | None = None
    boxer_id: str | None = None
    session_id: str | None = None
    mapping_status: Literal["provisional"] = "provisional"
    excluded_source_rows: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_layout(self):
        if len(self.header) < 3 or len(self.header) > 26:
            raise ValueError("Explicit inspected header must cover 3–26 columns")
        letters = {chr(65 + i) for i in range(len(self.header))}
        if (
            set(self.columns) != {"start", "end", "class"}
            or len(set(self.columns.values())) != 3
            or not set(self.columns.values()) <= letters
        ):
            raise ValueError("Distinct inspected start/end/class columns required within header")
        if not set(self.helper_formulas) <= letters or set(self.helper_formulas) & set(
            self.columns.values()
        ):
            raise ValueError("Helper columns must be separate from annotation columns")
        if (
            not set(self.mirrored_columns) <= letters
            or not set(self.mirrored_columns.values()) <= letters
        ):
            raise ValueError("Mirror columns must refer to inspected columns")
        if not self.class_map or not set(self.class_map.values()) <= set(SOURCE_CLASSES.values()):
            raise ValueError("Explicit literal mapping must contain only canonical punch classes")
        if any(n < 2 for n in self.excluded_source_rows):
            raise ValueError("Excluded rows must refer to data rows, starting at 2")
        return self


def load_recording(path: Path):
    import yaml

    return RecordingConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def v5_recording():
    return RecordingConfig(
        video_id="V5",
        header=[None] * 5,
        helper_formulas={"D": "=A{row}-24", "E": "=B{row}-24"},
        class_map=SOURCE_CLASSES,
        frame_offset=0,
        end_assumption="inclusive",
        excluded_source_rows=[272, 323, 442, 536],
    )
