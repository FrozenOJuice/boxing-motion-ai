"""Inspect readiness or explicitly freeze completed activity references; never run predictions."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

from ..services import activity_evaluation as activity
from ..services.dashboard import require
from ..services.review_workspace import encode, sha
from .verified_background import workspace_at

OUTPUT = "reports/boxingvi/activity-reference-v1"


def collect(root):
    service = activity.ActivityEvaluation(workspace_at(root, "original"))
    protocol, digest = service.protocol()
    states = [service.detail(key) for key in activity.SCOPE]
    clean = [s for s in states if s["primary_eligible"]]
    groups = {tuple(s["entry"]["source_group_guard"]) for s in clean}
    activity_count = sum(e["kind"] == "activity" for s in clean for e in s["events"])
    blockers = []
    if not all(s["complete"] for s in states):
        blockers.append("All six blocks need explicit complete human reviews before freezing")
    if not {("V7", "V8"), ("V10",)}.issubset(groups):
        blockers.append("Primary scoring needs at least one clean block in each recording group")
    if not activity_count:
        blockers.append("No clean reference activity bouts are available for recall")
    return dict(
        schema_version="activity-reference-1.0",
        protocol_sha256=digest,
        protocol=protocol,
        blocks=states,
        summary=dict(
            total_blocks=len(states),
            completed=sum(s["complete"] for s in states),
            primary_blocks=len(clean),
            primary_seconds=str(
                sum(
                    (
                        Fraction(s["entry"]["timing"]["stop_exclusive"])
                        - Fraction(s["entry"]["timing"]["start"])
                        for s in clean
                    ),
                    Fraction(),
                )
            ),
            primary_activity_bouts=activity_count,
            omitted_uncertain_blocks=[
                s["interval_id"] for s in states if s["complete"] and not s["primary_eligible"]
            ],
            ready_to_freeze=all(s["complete"] for s in states),
            ready_for_primary_scoring=not blockers,
            blockers=blockers,
        ),
    )


def freeze(root):
    """Immutable snapshot and byte-for-byte ledger backups. Future edits invalidate validation."""
    root = Path(root)
    output = root / OUTPUT
    require(not output.exists(), "activity_reference_already_exists")
    result = collect(root)
    require(result["summary"]["ready_to_freeze"], "activity_reviews_incomplete")
    service = activity.ActivityEvaluation(workspace_at(root, "original"))
    backup = {}
    paths = [
        activity.PROTOCOL,
        service.intervals.config_path,
        result["protocol"]["interval_manifest"]["path"],
    ]
    for state in result["blocks"]:
        entry = state["entry"]
        for index in (entry["start"], entry["stop"] - 1):
            service.frames.frame(entry["recording"], index)
        paths += [
            f"{service.directory}/{state['interval_id']}/v{r['version']:08d}/review.json"
            for r in state["history"]
        ]
    for path in paths:
        text, digest = service.base.store.read(path)
        backup[path] = (text.encode("utf-8"), digest)
    require(collect(root) == result, "activity_reviews_changed_retry", "stale")
    output.mkdir(parents=True, exist_ok=False)
    for path, (raw, _) in backup.items():
        target = output / "review-backup" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    receipts = {path: digest for path, (_, digest) in sorted(backup.items())}
    (output / "review-backup-sha256.json").write_bytes(encode(receipts))
    (output / "manifest.json").write_bytes(encode(result))
    (output / "receipt.json").write_bytes(
        encode(dict(manifest_sha256=sha(encode(result)), backup_sha256=sha(encode(receipts))))
    )
    return result


def validate(root):
    service = activity.ActivityEvaluation(workspace_at(root, "original"))
    receipt, _ = service.base.store.json(f"{OUTPUT}/receipt.json")
    frozen, _ = service.base.store.json(f"{OUTPUT}/manifest.json", receipt["manifest_sha256"])
    require(frozen == collect(root), "activity_reference_stale", "stale")
    for state in frozen["blocks"]:
        entry = state["entry"]
        for index in (entry["start"], entry["stop"] - 1):
            service.frames.frame(entry["recording"], index)
    backup, _ = service.base.store.json(
        f"{OUTPUT}/review-backup-sha256.json", receipt["backup_sha256"]
    )
    for path, digest in backup.items():
        service.base.store.read(f"{OUTPUT}/review-backup/{path}", digest)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--freeze", action="store_true", help="Freeze only after all six human reviews"
    )
    mode.add_argument("--check", action="store_true", help="Validate an existing reference freeze")
    args = parser.parse_args()
    result = (freeze if args.freeze else validate if args.check else collect)(Path.cwd())
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
