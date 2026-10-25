"""Transport-independent dashboard query contracts, version 1."""

from typing import Literal

from pydantic import Field, JsonValue, field_validator

from boxing_motion_ai.data.annotation_review import StrictModel

from .artifacts import safe_relative


class Artifact(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    _path = field_validator("path")(safe_relative)


class Experiment(StrictModel):
    summary: Artifact
    format: Literal["velocity-study-result-1.0", "multi-recording-result-1.0", "rgb-legacy-1"]
    status: Literal["reference", "historical", "optional"]
    description: str
    metrics: dict[str, Artifact] = Field(default_factory=dict)


class Catalogue(StrictModel):
    schema_version: Literal["dashboard-catalogue-1.0"]
    curation: Artifact
    proposal: Artifact
    experiments: dict[str, Experiment]

    @field_validator("experiments")
    @classmethod
    def stable_ids(cls, value):
        import re

        if any(not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", key) for key in value):
            raise ValueError("Expected stable experiment IDs")
        return value


class QueryResult(StrictModel):
    schema_version: Literal["dashboard-query-1.0"] = "dashboard-query-1.0"
    status: Literal["available", "missing", "invalid", "stale", "incomplete"]
    issues: list[str] = Field(default_factory=list)
    data: JsonValue = None
    verification: Literal["metadata_only"] = "metadata_only"
