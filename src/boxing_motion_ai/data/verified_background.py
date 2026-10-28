"""Freeze reviewed absence intervals; never infer labels for other footage or train models."""

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path

from ..api.media import FrameService
from ..services.dashboard import DashboardService, require
from ..services.interval_review import IntervalDecision, IntervalWorkspace
from ..services.review_workspace import encode, sha
from .interval_pilot import GROUPS, ROLES

OUTPUT = "reports/boxingvi/verified-background-v1"
PIN = "configs/verified_background_v1.json"


def collect(workspace):
    """Revalidate the live ledger, sources, PTS and conflicts before exporting latest decisions."""
    require(workspace.pilot == "motion-v2", "wrong_background_pilot")
    snap = workspace.snapshot()  # Validates every revision, not just latest decisions.
    manifest, manifest_hash = workspace.manifest()
    require(manifest["roles"] == ROLES, "background_roles_changed")
    latest = {r["interval_id"]: r for r in snap["history"]}
    intervals = []
    for entry in manifest["entries"]:
        v = entry["recording"]
        require(
            v in ("V4", "V6")
            and entry["role"] == "train_candidate"
            and entry["source_group_guard"] == GROUPS[v],
            "background_source_group_changed",
        )
        detail = workspace.detail(entry["id"])
        record = latest.get(entry["id"])
        require(record is not None, "background_review_missing")
        decision = IntervalDecision.model_validate(record["payload"])
        require(
            record["source"] == "manual"
            and record["status"] == "verified_background"
            and decision.decision == "no_punch"
            and decision.watched_entire
            and decision.visible_continuous
            and decision.unannotated_punches == "no",
            "background_not_verified",
        )
        require(not detail["annotations"], "background_source_punch_conflict")
        spec = workspace.base._recording(v)
        prep = workspace.base._preparation(v, spec)
        with workspace.frames.lock:
            arrays = workspace.frames._load(v, spec, prep)
            endpoints = {
                name: {
                    "pts": int(arrays["pts"][i]),
                    "time_base_num": int(arrays["time_base_num"][i]),
                    "time_base_den": int(arrays["time_base_den"][i]),
                }
                for name, i in (("start", entry["start"]), ("stop_exclusive", entry["stop"]))
            }
        require(endpoints == entry["endpoint_pts"], "background_endpoint_changed")
        # Verify actual video hash and decode native boundaries, not just metadata references.
        for index in (entry["start"], entry["stop"] - 1):
            workspace.frames.frame(v, index)
        duration = Fraction(entry["timing"]["stop_exclusive"]) - Fraction(entry["timing"]["start"])
        intervals.append(
            dict(
                id=entry["id"],
                background_state="no_punch",
                provenance=entry,  # Original candidate remains unknown in its own manifest.
                endpoint_pts=endpoints,
                duration_exact_seconds=str(duration),
                review=dict(
                    path=f"{workspace.directory}/v{record['version']:08d}/review.json",
                    sha256=record["sha256"],
                    version=record["version"],
                    reviewed_at=record["reviewed_at"],
                    source="manual",
                    watched_entire=decision.watched_entire,
                    visible_continuous=decision.visible_continuous,
                    unannotated_punches=decision.unannotated_punches,
                    movement_tags=decision.tags,
                    notes=decision.notes,
                    playback_coverage=record["payload"].get("playback_coverage"),
                    validation=record.get("viewing_validation"),
                    inspected_frames=decision.inspected_frames,
                ),
            )
        )
    require(workspace.snapshot()["head"] == snap["head"], "background_reviews_changed")
    seconds = sum((Fraction(e["duration_exact_seconds"]) for e in intervals), Fraction())
    result = dict(
        schema_version="verified-background-1.0",
        dataset_id="boxingvi-motion-v2-background-v1",
        frame_convention="zero_based_half_open",
        supervision="explicitly_reviewed_intervals_only",
        default_unreviewed_state="unknown",
        context_is_verified=False,
        detector_training_authorized=False,
        exhaustive_evaluation_reference=False,
        roles=ROLES,
        source_group_guards=GROUPS,
        pilot_manifest_sha256=manifest_hash,
        review_head=snap["head"],
        review_revision_count=len(snap["history"]),
        summary=dict(
            intervals=len(intervals),
            frames=sum(e["provenance"]["stop"] - e["provenance"]["start"] for e in intervals),
            duration_exact_seconds=str(seconds),
            duration_seconds=float(seconds),
            recordings=dict(Counter(e["provenance"]["recording"] for e in intervals)),
            discovery_strata=dict(Counter(e["provenance"]["selection_stratum"] for e in intervals)),
            human_movement_tags=dict(
                Counter(t for e in intervals for t in e["review"]["movement_tags"])
            ),
        ),
        intervals=intervals,
    )
    return result


def workspace_at(root, pilot="motion-v2"):
    base = DashboardService(root)
    return IntervalWorkspace(base, FrameService(base), pilot)


def prepare(root, *, expected_intervals=12):
    """Create a new immutable export and byte-for-byte review backup; refuse replacement."""
    root = Path(root)
    output, pin = root / OUTPUT, root / PIN
    require(not output.exists() and not pin.exists(), "background_version_already_exists")
    ws = workspace_at(root)
    result = collect(ws)
    require(
        1 <= expected_intervals <= 12 and len(result["intervals"]) == expected_intervals,
        "motion_pilot_not_complete",
    )
    # Back up both interval pilots, including superseded records and original pins/manifests.
    backup = {}
    for pilot in ("motion-v2", "original"):
        other = ws if pilot == "motion-v2" else workspace_at(root, pilot)
        snapshot = other.snapshot()
        ref, _ = other.base.store.json(other.config_path)
        paths = [other.config_path, ref["path"]] + [
            f"{other.directory}/v{r['version']:08d}/review.json" for r in snapshot["history"]
        ]
        for path in paths:
            raw, digest = other.base.store.read(path)
            backup[path] = (raw.encode("utf-8"), digest)
    require(ws.snapshot()["head"] == result["review_head"], "background_reviews_changed")
    output.mkdir(parents=True, exist_ok=False)
    for path, (raw, _) in backup.items():
        target = output / "review-backup" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    receipt = {path: digest for path, (_, digest) in sorted(backup.items())}
    (output / "review-backup-sha256.json").write_bytes(encode(receipt))
    raw = encode(result)
    (output / "manifest.json").write_bytes(raw)
    with pin.open("xb") as stream:
        stream.write(
            encode(
                dict(
                    path=f"{OUTPUT}/manifest.json",
                    sha256=sha(raw),
                    backup_receipt_sha256=sha(encode(receipt)),
                )
            )
        )
    return result


def validate(root):
    """Reject stale exports after any new review revision or changed source metadata."""
    ws = workspace_at(root)
    ref, _ = ws.base.store.json(PIN)
    require(ref["path"] == f"{OUTPUT}/manifest.json", "unexpected_background_path")
    frozen, _ = ws.base.store.json(ref["path"], ref["sha256"])
    require(frozen == collect(ws), "background_snapshot_stale")
    receipt, _ = ws.base.store.json(
        f"{OUTPUT}/review-backup-sha256.json", ref["backup_receipt_sha256"]
    )
    for path, digest in receipt.items():
        ws.base.store.read(f"{OUTPUT}/review-backup/{path}", digest)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate the existing frozen export")
    args = parser.parse_args()
    result = validate(Path.cwd()) if args.check else prepare(Path.cwd())
    print(result["summary"])
    print("Review head:", result["review_head"])


if __name__ == "__main__":
    main()
