"""Explicit exclusion-only approved subset; never changes review decisions or candidates."""

import json
from pathlib import Path

import yaml

from .annotation_review import LABELS, ReviewLedger, digest, file_hash
from .curation import (
    CurationConfig,
    PinnedFile,
    build_plan,
    read_sources,
    validate_bindings,
)


def conservative_selection(old, current, ledger, approved_unresolved):
    """Only the approved unresolved human holds may bypass the queue-completion gate."""
    if set(current["blockers"]) - {"targeted_human_review_pending"}:
        raise ValueError("Other curation readiness blockers remain")
    if set(current["pending_human_queue"]) != set(approved_unresolved):
        raise ValueError("Unresolved queue differs from explicit approval")
    for key in approved_unresolved:
        revision = ledger.cases[key].revisions[-1]
        if revision.actor_type != "human" or revision.disposition not in ("hold", "exclude"):
            raise ValueError("Unresolved approval requires an explicit human hold/exclusion")
    selected = {}
    for v, record in current["records"].items():
        previous = old["records"][v]["selected_source_rows"]
        now = record["selected_source_rows"]
        if v == "V5":
            if previous != now:
                raise ValueError("Frozen V5 evaluation membership changed")
            continue
        selected[v] = sorted(set(previous) & set(now))
        if any(f"{v}:{row}" in approved_unresolved for row in selected[v]):
            raise ValueError("Unresolved case entered training")
    return selected


def prepare(config_path, output):
    """Validate pins and history, then write a new manifest under ignored reports only."""
    from ..services.dashboard import DashboardService
    from ..services.review_workspace import ReviewWorkspace

    c = json.loads(Path(config_path).read_text())
    if c["schema_version"] != "approved-curated-subset-1.0":
        raise ValueError("Unsupported subset approval")
    old = CurationConfig.model_validate_json(PinnedFile(**c["baseline_config"]).read_bytes())
    current = CurationConfig.model_validate_json(PinnedFile(**c["reviewed_config"]).read_bytes())
    a, b = old.model_dump(), current.model_dump()
    a.pop("review_ledger")
    b.pop("review_ledger")
    if a != b:
        raise ValueError("Only review ledger may differ from original proposal")
    ledger = ReviewLedger.model_validate_json(current.review_ledger.read_bytes())
    initial = ReviewLedger.model_validate_json(old.review_ledger.read_bytes())
    workspace = ReviewWorkspace(DashboardService(Path.cwd()))
    live = workspace.snapshot()  # Replays hashes, receipts, verification origins and histories.
    if live.ledger_hash != current.review_ledger.sha256:
        raise ValueError("Live reviews changed since approval; inspect before new preparation")
    backup = json.loads(PinnedFile(**c["backup_manifest"]).read_bytes())
    protected = dict(backup)
    backup_root = Path(c["backup_manifest"]["path"]).parent / "backup"
    for path, expected in backup.items():
        if file_hash(path) != expected or file_hash(backup_root / path) != expected:
            raise ValueError("Review original or backup differs")
        protected[str(backup_root / path)] = expected
    sources = read_sources(current)
    for cfg, log in ((old, initial), (current, ledger)):
        validate_bindings(
            cfg,
            log,
            json.loads(cfg.cases.read_bytes()),
            yaml.safe_load(cfg.diagnostic.read_bytes()),
            sources,
        )
    old_plan, new_plan = [
        build_plan(cfg, log, sources) for cfg, log in ((old, initial), (current, ledger))
    ]
    selected = conservative_selection(old_plan, new_plan, ledger, c["approved_unresolved_holds"])
    baseline = {v: sorted(sources[v]["approved"]) for v in selected}
    counts = {label: 0 for label in LABELS}
    removed = []
    for v, rows in selected.items():
        if not set(rows) <= sources[v]["approved"]:
            raise ValueError("Original exclusion restored")
        for row in rows:
            counts[sources[v]["rows"][row]["punch_type_provisional"]] += 1
        for row in sorted(set(baseline[v]) - set(rows)):
            key = f"{v}:{row}"
            removed.append(
                dict(
                    case_id=key,
                    label=sources[v]["rows"][row]["punch_type_provisional"],
                    reason="preserved_historical_curation_hold"
                    if row not in old_plan["records"][v]["selected_source_rows"]
                    else "approved_unresolved_manual_hold",
                    current_disposition=ledger.cases[key].revisions[-1].disposition,
                )
            )
    if (
        sum(map(len, baseline.values())) != c["baseline_count"]
        or sum(map(len, selected.values())) != c["selected_count"]
        or counts != c["class_counts"]
        or digest(selected) != c["selection_sha256"]
    ):
        raise ValueError("Approved subset identity/counts changed")
    for spec in current.sources.values():
        protected.update({str(Path(spec.dataset) / name): h for name, h in spec.inputs.items()})
        protected[spec.policy.path] = spec.policy.sha256
    protected[str(config_path)] = file_hash(config_path)
    result = dict(
        schema_version="curated-training-subset-1.0",
        status="approved_subset_ready",
        approval_config_sha256=file_hash(config_path),
        review_ledger=current.review_ledger.model_dump(),
        selected=selected,
        baseline=baseline,
        class_counts=counts,
        removed=removed,
        development_rows=sorted(sources["V5"]["approved"]),
        unresolved_excluded=c["approved_unresolved_holds"],
        original_readiness=new_plan["status"],
        review_states_changed=False,
        source_groups={v: s.source_group for v, s in current.sources.items()},
        guards={v: s.related_recordings_guard for v, s in current.sources.items()},
        protected=protected,
    )
    output = Path(output)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a fresh ignored reports destination")
    output.mkdir(parents=True)
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
