"""Local append-only review transactions using the existing curation contracts."""

import fcntl
import hashlib
import json
import os
import re
import stat
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, field_validator, model_serializer, model_validator

from boxing_motion_ai.data.annotation_review import (
    AXES,
    Bounds,
    Evidence,
    Label,
    ReviewLedger,
    Revision,
    StrictModel,
    Verification,
    VerificationSelection,
    digest,
    verified_axes,
)
from boxing_motion_ai.data.curation import CurationConfig, build_plan, latest

from .artifacts import ServiceError, strict_json
from .dashboard import DashboardService, query, require

DIRECTORY = "reports/boxingvi/dashboard-reviews-v1"


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def append_review_bundle(fd, version, record):
    """Publish one fsynced immutable revision while the caller holds the ledger lock."""
    raw = encode(record)
    temporary = "pending-" + uuid.uuid4().hex
    os.mkdir(temporary, 0o700, dir_fd=fd)
    bundle = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
    try:
        target = os.open(
            "review.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=bundle,
        )
        with os.fdopen(target, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(bundle)
        os.rename(temporary, f"v{version:08d}", src_dir_fd=fd, dst_dir_fd=fd)
        os.fsync(fd)
    finally:
        os.close(bundle)
    return record | {"sha256": sha(raw)}


class Decision(StrictModel):
    verification: VerificationSelection | None = None
    request_id: str = Field(pattern=r"^[0-9a-f]{8}-[0-9a-f-]{27}$")
    expected_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_revision_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    source_candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    action: Literal["confirm", "verify", "ambiguous", "hold", "exclude"]
    reviewer: str | None = Field(default=None, min_length=1, max_length=160)
    notes: str = Field(min_length=1, max_length=4000)
    uncertainty: str = Field(max_length=4000)
    lead_rear_evidence: str = Field(max_length=2000)
    proposed_label: Label | None = None
    proposed_bounds: Bounds | None = None
    inspected_frames: list[int] = Field(max_length=512)
    attest_human: Literal[True]

    @model_serializer(mode="wrap")
    def preserve_legacy_payload(self, handler):
        data = handler(self)
        for key in ("verification", "reviewer"):
            if getattr(self, key) is None:
                data.pop(key, None)
        return data

    @model_validator(mode="after")
    def verification_action(self):
        if (self.action == "verify") != (self.verification is not None):
            raise ValueError("Independent checks require the verify action")
        return self

    @field_validator("attest_human", mode="before")
    @classmethod
    def explicit_attestation(cls, value):
        if value is not True:
            raise ValueError("Explicit human attestation required")
        return value

    @field_validator("reviewer", "notes")
    @classmethod
    def not_blank(cls, value):
        if value is None:
            return None
        if not value.strip():
            raise ValueError("Legacy reviewer and notes cannot be blank")
        return value.strip()

    @field_validator("inspected_frames")
    @classmethod
    def indices(cls, value):
        if any(i < 0 for i in value) or len(value) != len(set(value)):
            raise ValueError("Unique nonnegative native frame indices required")
        return sorted(value)


def action_of(revision):
    if revision and revision.actor_type == "human":
        for action in ("confirm", "verify", "ambiguous", "hold", "exclude"):
            if f"dashboard_{action}" in revision.reasons:
                return action
    return None


class LiveDashboardService(DashboardService):
    """One immutable ledger snapshot per request; original catalogue/results stay pinned."""

    def __init__(self, base, config, ledger, ledger_hash, receipts):
        self.store, self.catalogue = base.store, base.catalogue
        self.catalogue_sha256 = base.catalogue_sha256
        self.config, self.ledger, self.ledger_hash = config, ledger, ledger_hash
        self.receipts = receipts

    def _config(self):
        return self.config

    def _ledger(self, config):
        return self.ledger

    def review(self, recording_id, source_row):
        result = super().review(recording_id, source_row)
        if result.data:
            revision = latest(self.ledger, f"{recording_id}:{source_row}")
            result.data["latest_action"] = action_of(revision)
            axes = verified_axes(revision)
            result.data["verification_status"] = (
                "fully_verified"
                if axes == set(AXES)
                else "partially_verified"
                if axes
                else "unverified"
            )
        return result

    def review_status(self):
        result = super().review_status()
        if result.data:
            result.data["human_reviewed"] = sum(
                bool(latest(self.ledger, key) and latest(self.ledger, key).actor_type == "human")
                for key in self.ledger.cases
            )
            result.data["ledger_sha256"] = self.ledger_hash
        return result

    @query
    def edit_state(self, recording_id, source_row):
        checked = self.review(recording_id, source_row)
        require(checked.status == "available", "review_source_unavailable", checked.status)
        key = f"{recording_id}:{source_row}"
        revision = latest(self.ledger, key)
        return dict(
            verification={axis: axis in verified_axes(revision) for axis in AXES},
            ledger_sha256=self.ledger_hash,
            revision_sha256=digest(revision.model_dump(mode="json")) if revision else None,
            source_candidate_sha256=self.ledger.cases[key].source_candidate_sha256,
            ledger_version=self.ledger.ledger_version,
            history=[r.model_dump(mode="json") for r in self.ledger.cases[key].revisions],
            decisions=[r for r in self.receipts if r["case_id"] == key],
        )

    @query
    def curation_status(self):
        sources = {}
        for v, spec in self.config.sources.items():
            rows = self._rows(v, spec)
            policy = self._policy(v, spec, rows)
            prep = self._preparation(v, spec)
            sources[v] = dict(
                rows={r["source_row"]: r for r in rows},
                approved=set(policy["selected_source_rows"]),
                frame_count=prep["source_frame_count"],
            )
        plan = build_plan(self.config, self.ledger, sources)
        return dict(
            proposal=plan,
            ledger_sha256=self.ledger_hash,
            verification_scope="current_review_and_pinned_metadata_only",
            preservation_manifest_reverified=False,
            source_pose_bytes_verified=False,
        )


class ReviewWorkspace:
    """Serialize writers across threads/processes; atomic immutable version directories.

    No mutable head pointer: only complete contiguous version directories count.
    A crash before rename leaves a pending directory, never a published decision.
    """

    def __init__(self, base):
        self.base = base

    @contextmanager
    def _locked(self, directory=DIRECTORY):
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        lock = None
        try:
            for part in self.base.store.root.parts[1:]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
            for part in directory.split("/"):
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
            lock = os.open(
                ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=fd
            )
            require(stat.S_ISREG(os.fstat(lock).st_mode), "invalid_review_lock")
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield fd
        except OSError as exc:
            raise ServiceError("invalid", "review_storage_unavailable_or_unsafe") from exc
        finally:
            if lock is not None:
                os.close(lock)
            os.close(fd)

    def _snapshot(self, directory_fd):
        config = self.base._config()
        ledger = self.base._ledger(config)
        head = config.review_ledger.sha256
        receipts = []
        names = sorted(name for name in os.listdir(directory_fd) if name.startswith("v"))
        for name in names:
            require(re.fullmatch(r"v[0-9]{8}", name), "invalid_review_version", "stale")
            require(int(name[1:]) == ledger.ledger_version + 1, "review_history_gap", "stale")
            prefix = f"{DIRECTORY}/{name}"
            receipt, _ = self.base.store.json(f"{prefix}/receipt.json")
            raw, new_hash = self.base.store.read(f"{prefix}/ledger.json", receipt["ledger_sha256"])
            updated = ReviewLedger.model_validate(strict_json(raw))
            require(
                updated.parent_ledger_sha256 == head
                and updated.ledger_version == ledger.ledger_version + 1,
                "review_parent_changed",
                "stale",
            )
            decision = Decision.model_validate(receipt["decision"])
            key = receipt["case_id"]
            require(
                key in ledger.cases
                and receipt["schema_version"] == "dashboard-review-save-1.0"
                and receipt["base_config_sha256"] == self.base.catalogue.curation.sha256
                and decision.expected_ledger_sha256 == head,
                "review_binding_changed",
                "stale",
            )
            previous = latest(ledger, key)
            previous_hash = digest(previous.model_dump(mode="json")) if previous else None
            revision = updated.cases[key].revisions[-1]
            if decision.action == "verify":
                require(
                    revision.verification is not None
                    and all(
                        getattr(revision.verification, axis) == getattr(decision.verification, axis)
                        for axis in AXES
                    )
                    and revision.verification.label == decision.proposed_label
                    and revision.verification.bounds == decision.proposed_bounds,
                    "review_verification_changed",
                    "stale",
                )
            audit = {
                k: receipt[k]
                for k in (
                    "case_id",
                    "decision",
                    "verified_frames",
                    "source_inputs",
                    "video_sha256",
                    "base_config_sha256",
                )
            }
            require(
                decision.expected_revision_sha256 == previous_hash
                and decision.source_candidate_sha256 == ledger.cases[key].source_candidate_sha256
                and revision.revision_id == decision.request_id
                and revision.actor_type == "human"
                and revision.reviewer == decision.reviewer
                and receipt.get("source") == revision.source
                and revision.source in (None, "manual")
                and receipt["saved_at"] == revision.reviewed_at
                and revision.evidence[0].description
                == decision.notes + "\nDashboard audit SHA256: " + digest(audit),
                "review_audit_changed",
                "stale",
            )
            expected = ledger.model_dump(mode="json")
            require(
                len(updated.cases[key].revisions) == len(ledger.cases[key].revisions) + 1,
                "review_history_changed",
                "stale",
            )
            expected["cases"][key]["revisions"].append(
                updated.cases[key].revisions[-1].model_dump(mode="json")
            )
            expected.update(ledger_version=updated.ledger_version, parent_ledger_sha256=head)
            require(expected == updated.model_dump(mode="json"), "review_history_changed", "stale")
            config_data, _ = self.base.store.json(f"{prefix}/config.json", receipt["config_sha256"])
            expected_config = config.model_dump(mode="json")
            expected_config["review_ledger"] = dict(path=f"{prefix}/ledger.json", sha256=new_hash)
            require(config_data == expected_config, "review_config_changed", "stale")
            config = CurationConfig.model_validate(config_data)
            ledger, head = updated, new_hash
            receipts.append(receipt)
        return LiveDashboardService(self.base, config, ledger, head, receipts)

    def snapshot(self):
        with self._locked() as fd:
            try:
                return self._snapshot(fd)
            except (KeyError, ValueError, TypeError) as exc:
                if isinstance(exc, ServiceError):
                    raise
                raise ServiceError("stale", "invalid_review_history") from exc

    def save(self, recording, row, decision, frame_service):
        # Validate ID and candidate before opening any write directory.
        checked = self.base.review(recording, row)
        require(checked.status == "available", "review_source_unavailable", checked.status)
        key = f"{recording}:{row}"
        with self._locked() as fd:
            live = self._snapshot(fd)
            payload = decision.model_dump(mode="json")
            for receipt in live.receipts:
                if receipt["decision"]["request_id"] == decision.request_id:
                    require(
                        receipt["case_id"] == key and receipt["decision"] == payload,
                        "request_id_reused",
                        "stale",
                    )
                    return receipt
            require(
                decision.expected_ledger_sha256 == live.ledger_hash, "stale_review_reload", "stale"
            )
            previous = latest(live.ledger, key)
            previous_hash = digest(previous.model_dump(mode="json")) if previous else None
            require(
                decision.expected_revision_sha256 == previous_hash
                and decision.source_candidate_sha256
                == live.ledger.cases[key].source_candidate_sha256,
                "stale_review_source_or_revision",
                "stale",
            )
            spec = live.config.sources[recording]
            prep = live._preparation(recording, spec)
            count = prep["source_frame_count"]
            require(all(i < count for i in decision.inspected_frames), "evidence_out_of_range")
            bounds = decision.proposed_bounds
            require(bounds is None or bounds.end < count, "proposed_bounds_out_of_range")
            confirm = decision.action == "confirm"
            checks = None
            evidence_indices = set(decision.inspected_frames)
            if decision.action == "verify":
                selected = decision.verification.selected()
                require(decision.proposed_label is not None, "verification_target_label_required")
                prior_axes = verified_axes(previous)
                require(selected or prior_axes, "select_at_least_one_confirmation")
                prior_label = (
                    previous.verification.label
                    if previous and previous.verification
                    else previous.confirmed_label
                    if previous
                    else None
                ) or checked.data["source"]["punch_type_provisional"]
                prior_bounds = (
                    previous.verification.bounds
                    if previous and previous.verification
                    else previous.confirmed_bounds
                    if previous
                    else None
                )
                same_target = prior_label == decision.proposed_label and prior_bounds == bounds
                carried = selected & prior_axes if same_target else set()
                origins = {}
                for axis in selected:
                    origins[axis] = (
                        (
                            previous.verification.origins[axis]
                            if previous.verification
                            else previous.revision_id
                        )
                        if axis in carried
                        else decision.request_id
                    )
                checks = Verification(
                    **decision.verification.model_dump(),
                    label=decision.proposed_label,
                    bounds=bounds,
                    origins=origins,
                )
                by_id = {r.revision_id: r for r in live.ledger.cases[key].revisions}
                for axis in carried:
                    for item in by_id[origins[axis]].evidence:
                        if item.kind == "human_frame_review":
                            evidence_indices.update(item.frame_indices)
                if (selected - carried) & {"punch_type", "hand"}:
                    require(
                        any(
                            bounds is None or bounds.start <= i <= bounds.end
                            for i in decision.inspected_frames
                        ),
                        "inspect_punch_before_verifying",
                    )
            timing = confirm or bool(checks and checks.timing)
            if timing:
                require(
                    bounds is not None and decision.proposed_label is not None,
                    "confirmation_requires_label_and_bounds",
                )
                require(
                    bounds.start in evidence_indices and bounds.end in evidence_indices,
                    "inspect_both_proposed_boundaries",
                )
                if confirm:
                    require(decision.lead_rear_evidence.strip(), "lead_rear_evidence_required")
                require(
                    not any(
                        bounds.start < cut <= bounds.end
                        for cut in prep.get("confirmed_scene_cuts", [])
                    ),
                    "confirmation_crosses_known_cut",
                )
            if decision.action in ("hold", "ambiguous"):
                require(decision.uncertainty.strip(), "unresolved_uncertainty_required")
            verified = []
            for index in sorted(evidence_indices):
                frame, _ = frame_service.frame(recording, index)
                verified.append(
                    {
                        k: frame[k]
                        for k in (
                            "frame_index",
                            "pts",
                            "time_base",
                            "exact_timestamp",
                            "scene_id",
                            "segment_id",
                        )
                    }
                )
            if timing:
                ends = [f for f in verified if f["frame_index"] in (bounds.start, bounds.end)]
                require(
                    len({(f["scene_id"], f["segment_id"]) for f in ends}) == 1,
                    "confirmation_crosses_scene_or_segment",
                )
            rechecked = live.review(recording, row)
            require(rechecked.status == "available", "source_changed_during_review", "stale")
            audit = dict(
                case_id=key,
                decision=payload,
                verified_frames=verified,
                source_inputs=spec.inputs,
                video_sha256=prep.get("inputs", {}).get("video_sha256"),
                base_config_sha256=self.base.catalogue.curation.sha256,
            )
            version = live.ledger.ledger_version + 1
            name = f"v{version:08d}"
            prefix = f"{DIRECTORY}/{name}"
            fully_verified = confirm or bool(checks and checks.selected() == set(AXES))
            label_verified = confirm or bool(checks and checks.punch_type and checks.hand)
            unresolved = [decision.uncertainty.strip()] if decision.uncertainty.strip() else []
            if checks and not fully_verified:
                unresolved += [
                    f"Not verified: {axis}" for axis in AXES if axis not in checks.selected()
                ]
            revision = Revision(
                verification=checks,
                revision_id=decision.request_id,
                supersedes_sha256=previous_hash,
                actor_type="human",
                source="manual",
                reviewer=decision.reviewer,
                reviewed_at=datetime.now(UTC).isoformat(),
                disposition="accept"
                if fully_verified
                else "exclude"
                if decision.action == "exclude"
                else "hold",
                label_status="confirmed"
                if label_verified
                else "ambiguous"
                if decision.action == "ambiguous"
                else "unreviewed",
                confirmed_label=decision.proposed_label if label_verified else None,
                lead_rear_evidence=(
                    f"Explicit human hand confirmation; origin revision {checks.origins['hand']}"
                    if checks and checks.hand
                    else None
                )
                if checks
                else decision.lead_rear_evidence.strip() or None,
                boundary_status="confirmed"
                if timing
                else "ambiguous"
                if decision.action == "ambiguous"
                else "unreviewed",
                confirmed_bounds=bounds if timing else None,
                reasons=[f"dashboard_{decision.action}", decision.notes],
                evidence=[
                    Evidence(
                        kind="human_frame_review" if verified else "diagnostic_observation",
                        reference=f"{prefix}/receipt.json",
                        description=decision.notes + "\nDashboard audit SHA256: " + digest(audit),
                        frame_indices=sorted(evidence_indices),
                    )
                ],
                unresolved_uncertainty=unresolved,
            )
            data = live.ledger.model_dump(mode="json")
            data["cases"][key]["revisions"].append(revision.model_dump(mode="json"))
            data.update(ledger_version=version, parent_ledger_sha256=live.ledger_hash)
            ledger = ReviewLedger.model_validate(data)
            ledger_raw = encode(ledger.model_dump(mode="json"))
            config_data = live.config.model_dump(mode="json")
            config_data["review_ledger"] = dict(
                path=f"{prefix}/ledger.json", sha256=sha(ledger_raw)
            )
            config_raw = encode(CurationConfig.model_validate(config_data).model_dump(mode="json"))
            receipt = dict(
                schema_version="dashboard-review-save-1.0",
                source="manual",
                case_id=key,
                base_config_sha256=self.base.catalogue.curation.sha256,
                ledger_sha256=sha(ledger_raw),
                config_sha256=sha(config_raw),
                decision=payload,
                saved_at=revision.reviewed_at,
                verified_frames=verified,
                source_inputs=spec.inputs,
                video_sha256=prep.get("inputs", {}).get("video_sha256"),
                corrections_applied=False,
                training_enabled=False,
                ledger_path=f"{prefix}/ledger.json",
                config_path=f"{prefix}/config.json",
            )
            temporary = "pending-" + uuid.uuid4().hex
            os.mkdir(temporary, 0o700, dir_fd=fd)
            bundle = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                for filename, raw in (
                    ("ledger.json", ledger_raw),
                    ("config.json", config_raw),
                    ("receipt.json", encode(receipt)),
                ):
                    target = os.open(
                        filename,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=bundle,
                    )
                    with os.fdopen(target, "wb") as stream:
                        stream.write(raw)
                        stream.flush()
                        os.fsync(stream.fileno())
                os.fsync(bundle)
                require(name not in os.listdir(fd), "review_version_exists", "stale")
                os.rename(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
                os.fsync(fd)
            finally:
                os.close(bundle)
            return receipt
