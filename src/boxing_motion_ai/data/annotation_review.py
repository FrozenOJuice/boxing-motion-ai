"""Versioned review decisions, separate from immutable source annotations."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

RECORDINGS = ("V4", "V5", "V6", "V7", "V8", "V10")
LABELS = ("jab", "cross", "lead_hook", "rear_hook", "lead_uppercut", "rear_uppercut")
Label = Literal["jab", "cross", "lead_hook", "rear_hook", "lead_uppercut", "rear_uppercut"]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Bounds(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    convention: Literal["zero_based_decoded_inclusive"]

    @model_validator(mode="after")
    def ordered(self):
        if self.end < self.start:
            raise ValueError("Never repair reversed boundaries automatically")
        return self


class Evidence(StrictModel):
    kind: Literal["diagnostic_observation", "human_frame_review", "author_guidance"]
    reference: str = Field(min_length=1)
    description: str = Field(min_length=1)
    frame_indices: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def usable(self):
        if (
            not self.reference.strip()
            or not self.description.strip()
            or self.description.startswith("REPLACE")
        ):
            raise ValueError("Evidence cannot be blank")
        if any(f < 0 for f in self.frame_indices):
            raise ValueError("Evidence uses nonnegative native frame indices")
        return self


AXES = ("punch_type", "hand", "timing")


class VerificationSelection(StrictModel):
    schema_version: Literal["annotation-verification-1.0"] = "annotation-verification-1.0"
    punch_type: bool
    hand: bool
    timing: bool

    def selected(self):
        return {axis for axis in AXES if getattr(self, axis)}


class Verification(VerificationSelection):
    label: Label
    bounds: Bounds | None
    origins: dict[str, str]

    @model_validator(mode="after")
    def scoped(self):
        if set(self.origins) != self.selected() or any(not v for v in self.origins.values()):
            raise ValueError("Each verified aspect needs its human revision origin")
        if self.timing and self.bounds is None:
            raise ValueError("Timing confirmation requires explicit boundaries")
        return self


def verified_axes(revision):
    if not revision or revision.actor_type != "human":
        return set()
    if revision.verification is not None:
        return revision.verification.selected()
    # Read legacy human confirmations without rewriting their serialized history.
    return ({"punch_type", "hand"} if revision.label_status == "confirmed" else set()) | (
        {"timing"} if revision.boundary_status == "confirmed" else set()
    )


class Revision(StrictModel):
    verification: Verification | None = None
    revision_id: str = Field(min_length=1)
    supersedes_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    actor_type: Literal["assistant", "human"]
    reviewer: str = Field(min_length=1)
    reviewed_at: str
    disposition: Literal["provisional", "hold", "exclude", "accept"]
    label_status: Literal["unreviewed", "ambiguous", "confirmed"]
    confirmed_label: Label | None = None
    lead_rear_evidence: str | None = None
    boundary_status: Literal["unreviewed", "ambiguous", "mixed_action", "confirmed"]
    confirmed_bounds: Bounds | None = None
    reasons: list[str] = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    unresolved_uncertainty: list[str]

    @model_serializer(mode="wrap")
    def preserve_legacy_hashes(self, handler):
        data = handler(self)
        if self.verification is None:
            data.pop("verification", None)
        return data

    @model_validator(mode="after")
    def honest_confirmation(self):
        checks = self.verification
        if checks is not None:
            if self.actor_type != "human":
                raise ValueError("Independent confirmations require a human reviewer")
            if (self.label_status == "confirmed") != (checks.punch_type and checks.hand):
                raise ValueError("Six-class labels require both punch type AND lead/rear hand")
            if self.confirmed_label is not None and self.confirmed_label != checks.label:
                raise ValueError("Confirmed label must match the reviewed target")
            if (self.boundary_status == "confirmed") != checks.timing:
                raise ValueError("Boundary status must match independent timing verification")
            if self.confirmed_bounds is not None and self.confirmed_bounds != checks.bounds:
                raise ValueError("Confirmed boundaries must match the reviewed target")
            if self.disposition == "accept" and checks.selected() != set(AXES):
                raise ValueError("Full acceptance requires all three confirmations")
            if checks.selected() and not any(
                e.kind == "human_frame_review" and e.frame_indices for e in self.evidence
            ):
                raise ValueError("Each human confirmation requires exact-frame evidence")
        when = datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00"))
        if when.utcoffset() is None or not self.reviewer.strip():
            raise ValueError("Named reviewer and timezone-aware review time required")
        if any(not r.strip() for r in self.reasons + self.unresolved_uncertainty):
            raise ValueError("Reasons cannot be empty")
        if (self.label_status == "confirmed") != (self.confirmed_label is not None):
            raise ValueError("Only confirmed labels may carry a replacement/confirmed value")
        if (self.boundary_status == "confirmed") != (self.confirmed_bounds is not None):
            raise ValueError("Only confirmed boundaries may carry an interval")
        confirming = self.label_status == "confirmed" or self.boundary_status == "confirmed"
        if confirming:
            if self.actor_type != "human" or not any(
                e.kind == "human_frame_review" and e.frame_indices for e in self.evidence
            ):
                raise ValueError("Confirmation requires explicit human native-frame evidence")
        if self.label_status == "confirmed" and not (
            self.lead_rear_evidence and self.lead_rear_evidence.strip()
        ):
            raise ValueError("Six-class confirmation requires stance/lead-rear evidence")
        if self.disposition == "accept" and not (
            self.label_status == self.boundary_status == "confirmed"
        ):
            raise ValueError("Acceptance requires confirmed label AND boundaries")
        if self.disposition in ("hold", "exclude") and not self.unresolved_uncertainty:
            # A definitive human exclusion may have no uncertainty, but must have reasons.
            if not (self.actor_type == "human" and self.disposition == "exclude"):
                raise ValueError("Unresolved holds must record their uncertainty")
        return self


class ReviewCase(StrictModel):
    source_candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    origin: Literal["diagnostic_79", "precautionary_neighbor"]
    revisions: list[Revision]

    @model_validator(mode="after")
    def chain(self):
        previous, ids, seen = None, set(), {}
        for r in self.revisions:
            if r.revision_id in ids or r.supersedes_sha256 != previous:
                raise ValueError("Review history must be an append-only hash-linked chain")
            if r.verification is not None:
                for axis, origin in r.verification.origins.items():
                    if origin == r.revision_id:
                        continue
                    ancestor = seen.get(origin)
                    if axis not in verified_axes(ancestor):
                        raise ValueError(
                            "Inherited confirmation must reference a verified human origin"
                        )
                    old_label = (
                        ancestor.verification.label
                        if ancestor.verification
                        else ancestor.confirmed_label
                    )
                    old_bounds = (
                        ancestor.verification.bounds
                        if ancestor.verification
                        else ancestor.confirmed_bounds
                    )
                    if axis in ("punch_type", "hand") and (
                        old_label != r.verification.label or old_bounds != r.verification.bounds
                    ):
                        raise ValueError("Cannot inherit a label/hand check for another target")
                    if axis == "timing" and (
                        old_bounds != r.verification.bounds
                        or old_label is not None
                        and old_label != r.verification.label
                    ):
                        raise ValueError("Cannot inherit timing for another target")
                    if ancestor.verification and ancestor.verification.origins[axis] != origin:
                        raise ValueError("Origin must identify the actual human assertion")
            ids.add(r.revision_id)
            seen[r.revision_id] = r
            previous = digest(r.model_dump(mode="json"))
        return self


class ReviewLedger(StrictModel):
    schema_version: Literal["annotation-review-1.0"]
    ledger_version: int = Field(ge=1)
    parent_ledger_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: dict[str, ReviewCase]

    @model_validator(mode="after")
    def scope(self):
        for key in self.cases:
            v, sep, row = key.partition(":")
            if (
                v not in RECORDINGS
                or not sep
                or not row.isdigit()
                or int(row) < 1
                or key != f"{v}:{int(row)}"
            ):
                raise ValueError("Only explicit non-V9 recording/source-row identities allowed")
        if (self.ledger_version == 1) != (self.parent_ledger_sha256 is None):
            raise ValueError("Subsequent ledger versions require a parent hash")
        return self


def load_ledger(path):
    return ReviewLedger.model_validate(json.loads(Path(path).read_text()))


def write_new(path, value):
    """Exclusive creation: never overwrite source annotations, prior ledgers or reports."""
    path = Path(path)
    resolved = path.resolve()
    if not any(resolved.is_relative_to((Path.cwd() / d).resolve()) for d in ("configs", "reports")):
        raise ValueError("Write only new review configs or ignored reports, never original data")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append_review(ledger_path, decision_path, output, *, attest_human=False):
    """Publish a new ledger version; an explicit decision never mutates its parent."""
    ledger = load_ledger(ledger_path)
    decision = json.loads(Path(decision_path).read_text())
    if set(decision) != {"case_id", "revision"} or decision["case_id"] not in ledger.cases:
        raise ValueError("Decision must name an existing pinned case and one revision")
    revision = Revision.model_validate(decision["revision"])
    if revision.actor_type == "human" and not attest_human:
        raise ValueError("Human attribution requires --attest-human-review; never auto-certify")
    data = ledger.model_dump(mode="json")
    data["cases"][decision["case_id"]]["revisions"].append(revision.model_dump(mode="json"))
    data["ledger_version"] += 1
    data["parent_ledger_sha256"] = file_hash(ledger_path)
    updated = ReviewLedger.model_validate(data)
    write_new(output, updated.model_dump(mode="json"))
    return updated
