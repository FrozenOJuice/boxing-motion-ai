"""Small validated YAML settings, with explicit CLI overrides."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PoseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    pose_mode: Literal["image", "video"] = "image"
    scene_cuts: list[int] = Field(default_factory=list)
    overlay_frames: list[int] = Field(default_factory=list)
    ranges: list[str] = Field(default_factory=list)
    max_width: int = Field(default=960, ge=64)
    confidence: float = Field(default=0.5, ge=0, le=1)
    overlay_every: int = Field(default=30, ge=1)
    overlay_limit: int = Field(default=18, ge=0)


def load_config(path: Path | None, **overrides) -> PoseConfig:
    data = {}
    if path is not None:
        import yaml

        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("Pose config must contain a YAML mapping")
    data.update({key: value for key, value in overrides.items() if value is not None})
    return PoseConfig.model_validate(data)
