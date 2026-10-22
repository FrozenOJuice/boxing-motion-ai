"""Hash-pinned, exclusion-only curation plans. No training or source rewriting."""

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .annotation_review import (
    LABELS,
    RECORDINGS,
    ReviewLedger,
    StrictModel,
    digest,
    file_hash,
    write_new,
)

GROUPS = {
    "V4": "fightcamp-unresolved",
    "V5": "self-improvement-boxer-unresolved",
    "V6": "natebowerfitness-unresolved",
    "V7": "precision-striking-unresolved",
    "V8": "precision-striking-unresolved",
    "V10": "utah-lee-unresolved",
}
GUARDS = {
    v: ["V1", "V6"] if v == "V6" else ["V7", "V8"] if v in ("V7", "V8") else [v] for v in RECORDINGS
}


class PinnedFile(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def read_bytes(self):
        if file_hash(self.path) != self.sha256:
            raise ValueError(f"Stale pinned input: {self.path}")
        return Path(self.path).read_bytes()


class Source(StrictModel):
    dataset: str
    policy: PinnedFile
    inputs: dict[str, str]
    source_group: str
    related_recordings_guard: list[str]
    role: Literal["training", "development"]


class CurationConfig(StrictModel):
    schema_version: Literal["annotation-curation-plan-1.0"]
    status: Literal["proposal_no_training"]
    review_ledger: PinnedFile
    diagnostic: PinnedFile
    cases: PinnedFile
    preservation_manifest: PinnedFile
    sources: dict[str, Source]
    unreviewed_policy: Literal["retain_previously_approved_as_provisional"]
    development_policy: Literal["unchanged_frozen_benchmark_with_shadow_holds"]
    training_representation: Literal["xy_velocity"]
    mirror_probability: Literal[0.5]
    visibility_probability: Literal[0.0]
    context_seconds: Literal[0.0]
    priority_disputes: list[str]
    verification_anchors: list[str]

    @model_validator(mode="after")
    def fixed_scope(self):
        if set(self.sources) != set(RECORDINGS):
            raise ValueError("Exactly V4/V5/V6/V7/V8/V10; V9 reserved")
        for v, source in self.sources.items():
            if (
                source.source_group != GROUPS[v]
                or source.related_recordings_guard != GUARDS[v]
                or source.role != ("development" if v == "V5" else "training")
            ):
                raise ValueError("Preserve development/source-group/identity guards")
            if source.dataset != f"data/processed/boxingvi/{v}/candidates-full-v1":
                raise ValueError("Only existing approved preparation paths")
            if set(source.inputs) != {"preparation.json", "poses.npz", "candidates.jsonl"}:
                raise ValueError("Pin all prepared inputs, including pose bytes")
        for queue in (self.priority_disputes, self.verification_anchors):
            if len(queue) != len(set(queue)):
                raise ValueError("Duplicate review queue cases")
        return self


def load_config(path):
    return CurationConfig.model_validate(json.loads(Path(path).read_text()))


def read_sources(config):
    """Read existing versioned policy selections; never recompute/change quality thresholds."""
    sources = {}
    for v, spec in config.sources.items():
        policy = json.loads(spec.policy.read_bytes())
        if (
            policy["video_id"] != v
            or policy["schema_version"] != "recording-mask-policy-index-1.1"
            or policy["policy"] != "mask_aware70"
            or policy["source_mask_threshold"] != 0.5
            or policy["inputs"] != spec.inputs
        ):
            raise ValueError("Unchanged approved mask-aware policy required")
        root = Path(spec.dataset)
        for name, expected in spec.inputs.items():
            if file_hash(root / name) != expected:
                raise ValueError(f"Prepared input changed: {v}/{name}")
        rows = [json.loads(line) for line in (root / "candidates.jsonl").read_text().splitlines()]
        lookup = {r["source_row"]: r for r in rows}
        approved = policy["selected_source_rows"]
        if len(lookup) != len(rows) or len(set(approved)) != len(approved):
            raise ValueError("Duplicate source or policy rows")
        if not set(approved) <= set(lookup):
            raise ValueError("Policy references absent rows")
        for row in approved:
            r = lookup[row]
            if (
                r["video_id"] != v
                or r.get("recording_id", r["video_id"]) != v
                or not r["annotation_eligible"]
                or r["exclusion_reasons"]
                or r["punch_type_provisional"] not in LABELS
            ):
                raise ValueError("Invalid previously approved candidate")
        prep = json.loads((root / "preparation.json").read_text())
        if prep["npz_sha256"] != spec.inputs["poses.npz"] or prep["video_id"] != v:
            raise ValueError("Preparation identity/pose hash mismatch")
        sources[v] = dict(
            rows=lookup, approved=set(approved), frame_count=prep["source_frame_count"]
        )
    return sources


def latest(ledger, case_id):
    case = ledger.cases.get(case_id)
    return case.revisions[-1] if case and case.revisions else None


def validate_bindings(config, ledger, cases, diagnostic, sources):
    if (
        ledger.cases_sha256 != config.cases.sha256
        or ledger.diagnostic_sha256 != config.diagnostic.sha256
    ):
        raise ValueError("Ledger refers to different diagnostic evidence")
    ids = {c["id"] for c in cases}
    diagnostic_ids = {k for k, c in ledger.cases.items() if c.origin == "diagnostic_79"}
    if len(ids) != len(cases) or ids != diagnostic_ids or ids != set(diagnostic["assessments"]):
        raise ValueError("Diagnostic case sets must match exactly")
    if (
        config.priority_disputes != diagnostic["priority_disputes"]
        or config.verification_anchors != diagnostic["candidate_evaluation_subset"]
    ):
        raise ValueError("Preserve the original priority queue and verification anchors")
    if not set(config.priority_disputes + config.verification_anchors) <= set(ledger.cases):
        raise ValueError("Queue case missing from ledger")
    for key, review in ledger.cases.items():
        v, row = key.split(":")
        original = sources[v]["rows"].get(int(row))
        if original is None or digest(original) != review.source_candidate_sha256:
            raise ValueError(f"Source candidate changed: {key}")
        for revision in review.revisions:
            for evidence in revision.evidence:
                if any(i >= sources[v]["frame_count"] for i in evidence.frame_indices):
                    raise ValueError("Review evidence exceeds decoded video length")
            if (
                revision.confirmed_bounds
                and revision.confirmed_bounds.end >= sources[v]["frame_count"]
            ):
                raise ValueError("Confirmed boundaries exceed decoded video length")


def changed_confirmation(revision, original):
    if not revision:
        return False
    bounds = revision.confirmed_bounds
    return bool(
        revision.confirmed_label
        and revision.confirmed_label != original["punch_type_provisional"]
        or bounds
        and [bounds.start, bounds.end]
        != [original["candidate_start_index"], original["candidate_end_index"]]
    )


def build_plan(config, ledger, sources):
    """Only remove approved rows. Confirmed changes require explicit new preparation."""
    records, excluded, materialization, unapplied_changes = {}, [], [], []
    for v, source in sources.items():
        before, after, diagnostic_after = Counter(), Counter(), Counter()
        selected, shadow_held = [], []
        for row_id, row in sorted(source["rows"].items()):
            key = f"{v}:{row_id}"
            revision = latest(ledger, key)
            base = row_id in source["approved"]
            held = revision is not None and revision.disposition in ("hold", "exclude")
            changed = changed_confirmation(revision, row)
            if changed:
                unapplied_changes.append(key)
            if changed and base and revision.disposition == "accept":
                materialization.append(key)
            omit = held or changed
            label = row["punch_type_provisional"]
            if base:
                before[label] += 1
                if not omit:
                    diagnostic_after[label] += 1
                elif key in ledger.cases and ledger.cases[key].origin == "precautionary_neighbor":
                    diagnostic_after[label] += 1  # Comparison: 79-case-only scenario.
                if v == "V5" or not omit:
                    selected.append(row_id)
                    after[label] += 1
                if omit and v == "V5":
                    shadow_held.append(row_id)
            if not base or omit:
                excluded.append(
                    dict(
                        case_id=key,
                        source_label=label,
                        previously_approved=base,
                        applied_to_training=base and omit and v != "V5",
                        shadow_only=v == "V5" and base,
                        reasons=(list(revision.reasons) if held else [])
                        + (["confirmed_change_requires_new_preparation"] if changed else [])
                        + (
                            []
                            if base
                            else ["existing_policy_exclusion", *row["exclusion_reasons"]]
                            + (
                                []
                                if row["exclusion_reasons"]
                                else ["not_selected_by_pinned_mask_aware70_policy"]
                            )
                        ),
                        unresolved_uncertainty=list(revision.unresolved_uncertainty)
                        if revision
                        else [],
                    )
                )
        records[v] = dict(
            role=config.sources[v].role,
            source_group=config.sources[v].source_group,
            related_recordings_guard=config.sources[v].related_recordings_guard,
            boxer_id=None,
            session_id=None,
            selected_source_rows=selected,
            development_shadow_hold_rows=shadow_held,
            by_class={
                label: dict(
                    before=before[label],
                    after=after[label],
                    removed=before[label] - after[label],
                    diagnostic_79_only_after=diagnostic_after[label],
                )
                for label in LABELS
            },
            before=sum(before.values()),
            after=sum(after.values()),
        )
    pending = []
    verified_anchor_support = Counter()
    for key in config.priority_disputes + config.verification_anchors:
        revision = latest(ledger, key)
        if not revision or revision.actor_type != "human" or revision.disposition == "provisional":
            pending.append(key)
        if key in config.verification_anchors and revision and revision.disposition == "accept":
            v, row = key.split(":")
            original = sources[v]["rows"][int(row)]
            if int(row) in sources[v]["approved"] and not changed_confirmation(revision, original):
                verified_anchor_support[revision.confirmed_label] += 1
    train = [v for v in RECORDINGS if v != "V5"]
    support = {
        label: sum(records[v]["by_class"][label]["after"] for v in train) for label in LABELS
    }
    blockers = []
    if pending:
        blockers.append("targeted_human_review_pending")
    if any(verified_anchor_support[label] == 0 for label in LABELS):
        blockers.append("human_verified_anchor_class_coverage_incomplete")
    if materialization:
        blockers.append("confirmed_changes_need_separate_preparation")
    if any(n == 0 for n in support.values()):
        blockers.append("training_class_absent")
    return dict(
        schema_version="curated-selection-proposal-1.0",
        ledger_version=ledger.ledger_version,
        status="blocked_pending_review" if blockers else "ready_for_provisional_experiment_review",
        training_enabled=False,
        human_verified_anchor_support={label: verified_anchor_support[label] for label in LABELS},
        blockers=blockers,
        pending_human_queue=pending,
        confirmed_changes_not_applied=unapplied_changes,
        accepted_changes_requiring_preparation=materialization,
        training_class_support=support,
        human_accepted_cases=[
            k
            for k in ledger.cases
            if latest(ledger, k) and latest(ledger, k).disposition == "accept"
        ],
        records=records,
        excluded=excluded,
        labels_and_coordinates_rewritten=False,
        v9_accessed=False,
        unreviewed_examples="provisional, not certified; unknown footage is not no_punch",
    )


def verify_preserved(pinned):
    protected = json.loads(pinned.read_bytes())
    if any(
        any(part == "V9" or part.startswith(("V9.", "V9-")) for part in Path(p).parts)
        for p in protected
    ):
        raise ValueError("V9 must remain untouched")
    for path, expected in protected.items():
        if file_hash(path) != expected:
            raise ValueError(f"Previously preserved input changed: {path}")
    return len(protected)


def run_curation(config_path, output):
    import yaml

    output = Path(output)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a fresh ignored reports output")
    config = load_config(config_path)
    ledger = ReviewLedger.model_validate(json.loads(config.review_ledger.read_bytes()))
    cases = json.loads(config.cases.read_bytes())
    diagnostic = yaml.safe_load(config.diagnostic.read_bytes())
    sources = read_sources(config)
    validate_bindings(config, ledger, cases, diagnostic, sources)
    preserved = verify_preserved(config.preservation_manifest)
    result = build_plan(config, ledger, sources)
    result["provenance"] = dict(
        config_sha256=file_hash(config_path),
        ledger_sha256=config.review_ledger.sha256,
        previous_files_verified=preserved,
    )
    output.mkdir(parents=True)
    write_new(output / "curated-selection.json", result)
    lines = [
        "# Proposed annotation curation; no training",
        "",
        "Unreviewed labels remain provisional. V5 benchmark is unchanged; holds are shadow-only.",
        "",
        "| Recording | Class | Approved before | Proposed after | Removed |",
        "|---|---|---:|---:|---:|",
    ]
    for v, record in result["records"].items():
        for label, counts in record["by_class"].items():
            lines.append(
                f"| {v} | {label} | {counts['before']} | {counts['after']} | {counts['removed']} |"
            )
    lines += [
        "",
        "Blockers: " + ", ".join(result["blockers"]),
        "",
        "Human queue: " + ", ".join(result["pending_human_queue"]),
    ]
    (output / "CURATION.md").write_text("\n".join(lines) + "\n")
    for key in config.priority_disputes + config.verification_anchors:
        current = latest(ledger, key)
        template = dict(
            case_id=key,
            revision=dict(
                revision_id=f"human-{key}-ledger-{ledger.ledger_version + 1}",
                supersedes_sha256=digest(current.model_dump(mode="json")) if current else None,
                actor_type="human",
                reviewer=None,
                reviewed_at=None,
                disposition="hold",
                label_status="ambiguous",
                confirmed_label=None,
                lead_rear_evidence=None,
                boundary_status="ambiguous",
                confirmed_bounds=None,
                reasons=["unresolved_after_review"],
                evidence=[
                    dict(
                        kind="human_frame_review",
                        reference=f"reports/boxingvi/level1-label-pose-review-v1/index.html#{key}",
                        description="REPLACE with what you actually inspected",
                        frame_indices=[],
                    )
                ],
                unresolved_uncertainty=[
                    "REPLACE with unresolved question or explicit reason to hold"
                ],
            ),
        )
        write_new(output / "decision-templates" / f"{key.replace(':', '-')}.json", template)
    verify_preserved(config.preservation_manifest)
    return result
