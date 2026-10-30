"""Bounded localization review queue and adapter over existing human evidence.

Preparation never creates human decisions. Snapshot export is separate from training.
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

from ..api.clips import ClipService
from ..data.annotation_review import digest, file_hash, verified_axes
from ..services.dashboard import require
from ..services.interval_review import interval_timing
from ..services.localization_review import LocalizationReview
from ..services.review_workspace import ReviewWorkspace, encode, sha
from .continuous_dataset import PIN as CONTINUOUS_PIN
from .continuous_dataset import validate as validate_continuous
from .interval_pilot import GROUPS, ROLES
from .verified_background import workspace_at

OUTPUT = "reports/boxingvi/localization-review-v1"
PIN = "configs/localization_review_v1.json"
DATA_PIN = "configs/localization_evidence_v1.json"
SELECTION = "reports/boxingvi/level1-curation-study-v1/dataset/selection.json"
APPROVAL = "configs/level1_curated_subset_v1.json"
# Source suggestions choose review work, never supervision. No learned outputs used.
WINDOWS = (
    ("V6", 450, 642, "Isolated trajectories, guard and recovery"),
    ("V6", 6400, 6592, "Repeated lead-arm trajectories and recovery"),
    ("V6", 8280, 8472, "Rear-hook anchor and surrounding movement"),
    ("V6", 25050, 25242, "Uppercut trajectories and intervening movement"),
    ("V6", 30490, 30682, "Both-arm hook/uppercut combination"),
    ("V5", 8100, 8460, "Development-only straight/hook combinations"),
)


def overlap(a, b):
    return a["start"] < b["stop"] and b["start"] < a["stop"]


def canonical_events(base_intervals, anchors, additional):
    """Deduplicate exact human references; reject background or anchor contradictions."""
    grouped, background, pending = {}, [], []
    for interval in base_intervals:
        v = interval["provenance"]["recording"]
        require(v in ("V4", "V6"), "base_source_leakage")
        pending.extend((v, e, interval["id"]) for e in interval["events"])
        background.extend((v, r) for r in interval["supervision_runs"] if r["activity"] == 0)
    pending.extend((e["recording"], e, e["source_id"]) for e in anchors)
    for interval in additional:
        p = interval["provenance"]
        v = p["recording"]
        require(
            v in ("V5", "V6") and p["role"] == ROLES[v] and p["source_group_guard"] == GROUPS[v],
            "localization_source_group_leakage",
        )
        pending.extend(
            (v, e, interval["id"]) for e in interval["events"] if e.get("kind") != "uncertain"
        )
        background.extend((v, r) for r in interval["supervision_runs"] if r["activity"] == 0)
    for v, event, origin in pending:
        require(
            not any(bv == v and overlap(event, r) for bv, r in background),
            "human_positive_background_conflict",
        )
        key = (v, event["start"], event["stop"])
        if key not in grouped:
            grouped[key] = dict(
                recording=v,
                start=event["start"],
                stop=event["stop"],
                role=ROLES[v],
                source_group_guard=GROUPS[v],
                origins=[],
            )
        grouped[key]["origins"].append(dict(reference=origin, event_id=event["event_id"]))
        for anchor in anchors:
            if (
                v == anchor["recording"]
                and origin != anchor["source_id"]
                and overlap(event, anchor)
            ):
                require(
                    (event["start"], event["stop"]) == (anchor["start"], anchor["stop"]),
                    "conflicting_anchor_boundaries_need_explicit_reconciliation",
                )
    return [grouped[k] for k in sorted(grouped)]


def supervision(entry, events, complete):
    """Independent activity/background/boundary contracts; observations never define truth."""
    lo, hi = entry["start"], entry["stop"]
    require(all(e["stop"] > e["start"] for e in events), "invalid_localization_event")
    edges = sorted({lo, hi} | {max(lo, min(hi, e[k])) for e in events for k in ("start", "stop")})
    result = []
    for start, stop in zip(edges, edges[1:], strict=False):
        active = [e for e in events if e["start"] < stop and e["stop"] > start]
        uncertain = any(e.get("kind") == "uncertain" for e in active)
        punches = [e for e in active if e.get("kind") != "uncertain"]
        target = None if uncertain else 1 if punches else 0 if complete else None
        result.append(
            dict(
                start=start,
                stop=stop,
                activity=target,
                activity_supervised=target is not None,
                background_supervised=target == 0,
            )
        )
    boundaries = []
    for event in events:
        if event.get("kind") == "uncertain":
            continue
        ambiguous = any(e.get("kind") == "uncertain" and overlap(e, event) for e in events)
        boundaries.append(
            dict(
                event_id=event["event_id"],
                start=event["start"],
                stop=event["stop"],
                onset_supervised=not ambiguous and lo < event["start"] < hi,
                end_supervised=not ambiguous and lo < event["stop"] < hi,
                left_censored=event["start"] <= lo,
                right_censored=event["stop"] >= hi,
            )
        )
    return dict(
        supervision_runs=result,
        event_boundaries=boundaries,
        boundary_absence_domain=[
            dict(start=r["start"], stop=r["stop"])
            for r in result
            if complete and r["activity_supervised"]
        ],
        boundary_edge_policy=(
            "Exclude censored boundaries and boundary-support bands at domain edges"
        ),
        default_state="unknown",
        pose_masks_are_label_masks=False,
    )


def audit(ws):
    """Intersect actual full human checks with the conservative, still-approved source pool."""
    base = ws.base
    selection, selection_hash = base.store.json(SELECTION)
    approval, approval_hash = base.store.json(APPROVAL)
    require(digest(selection["selected"]) == approval["selection_sha256"], "selection_changed")
    live = ReviewWorkspace(base).snapshot()  # Validates original append-only transactions.
    require(live.ledger_hash == selection["review_ledger"]["sha256"], "anchor_review_changed")
    anchors, held = [], []
    receipts = {
        SELECTION: selection_hash,
        APPROVAL: approval_hash,
        selection["review_ledger"]["path"]: live.ledger_hash,
    }
    for key, case in live.ledger.cases.items():
        v, source_row = key.split(":")
        if v not in ("V4", "V6") or not case.revisions:
            continue
        r = case.revisions[-1]
        if not (
            r.actor_type == "human"
            and r.disposition == "accept"
            and verified_axes(r) == {"timing", "punch_type", "hand"}
            and not r.unresolved_uncertainty
        ):
            continue
        bounds = r.confirmed_bounds
        inspected = {
            i for e in r.evidence if e.kind == "human_frame_review" for i in e.frame_indices
        }
        require(
            bounds.start in inspected and bounds.end in inspected,
            "anchor_boundary_evidence_missing",
        )
        if int(source_row) not in selection["selected"][v]:
            held.append(dict(id=key, reason="preserved_conservative_exclusion"))
            continue
        spec = base._recording(v)
        row = next(x for x in base._rows(v, spec) if x["source_row"] == int(source_row))
        require(digest(row) == case.source_candidate_sha256, "anchor_source_annotation_changed")
        prep = base._preparation(v, spec)
        with ws.frames.lock:
            arrays = ws.frames._load(v, spec, prep)
            timing = interval_timing(
                arrays, bounds.start, bounds.end + 1, prep.get("confirmed_scene_cuts", [])
            )
            pts = points(arrays, bounds.start, bounds.end + 1)
        refs = []
        for evidence in r.evidence:
            if evidence.kind == "human_frame_review":
                _, h = base.store.read(evidence.reference)
                receipts[evidence.reference] = h
                refs.append(dict(path=evidence.reference, sha256=h))
        anchors.append(
            dict(
                event_id=f"human-anchor:{key}",
                source_id=key,
                recording=v,
                role="train_candidate",
                source_group_guard=GROUPS[v],
                start=bounds.start,
                stop=bounds.end + 1,
                kind="punch",
                punch_type=r.confirmed_label,
                type_uncertain=False,
                timing=timing,
                endpoint_pts=pts,
                source_inputs=spec.inputs,
                video_sha256=prep["inputs"]["video_sha256"],
                revision=r.model_dump(mode="json"),
                review_receipts=refs,
                source_candidate_sha256=case.source_candidate_sha256,
                surrounding_state="unknown",
                background_supervised=False,
                onset_supervised=True,
                end_supervised=True,
            )
        )
    require(len(anchors) == 4 and len(held) == 2, "unexpected_anchor_inventory_reaudit_required")
    v5_verified = sum(
        bool(case.revisions)
        and case.revisions[-1].actor_type == "human"
        and case.revisions[-1].disposition == "accept"
        and "timing" in verified_axes(case.revisions[-1])
        for key, case in live.ledger.cases.items()
        if key.startswith("V5:")
    )
    return dict(
        anchors=anchors,
        held_boundary_verified=held,
        source_receipts=receipts,
        v5_verified_events=v5_verified,
        provisional_annotations_are_labels=False,
    )


def points(arrays, start, stop):
    return {
        name: {k: int(arrays[k][i]) for k in ("pts", "time_base_num", "time_base_den")}
        for name, i in (("start", start), ("stop_exclusive", stop))
    }


def queue(ws, evidence):
    selection, _ = ws.base.store.json(SELECTION)
    entries = []
    for number, (v, start, stop, reason) in enumerate(WINDOWS):
        spec = ws.base._recording(v)
        prep = ws.base._preparation(v, spec)
        with ws.frames.lock:
            a = ws.frames._load(v, spec, prep)
            padding = 30 if v == "V5" else 24
            left, right = start - padding, stop + padding
            interval_timing(a, left, right + 1, prep.get("confirmed_scene_cuts", []))
            timing = interval_timing(a, start, stop, prep.get("confirmed_scene_cuts", []))
            endpoint_pts = points(a, start, stop)
        refs = []
        for row in ws.base._rows(v, spec):
            b = row.get("source_interval_half_open") or row.get("array_slice_half_open")
            if b and b[0] <= right and b[1] > left:
                allowed = selection["development_rows"] if v == "V5" else selection["selected"][v]
                require(row["source_row"] in allowed, "queue_context_touches_historical_exclusion")
                refs.append(
                    dict(
                        source_row=row["source_row"],
                        start=b[0],
                        stop=b[1],
                        verified=False,
                        overlaps_core=b[0] < stop and b[1] > start,
                    )
                )
        entries.append(
            dict(
                id=f"{v}-motion-{101 + number if v == 'V6' else 101}",
                recording=v,
                role=ROLES[v],
                purpose="localization_review",
                start=start,
                stop=stop,
                context_start=left,
                context_end=right,
                frame_count=len(a["frame_index"]),
                timing=timing,
                endpoint_pts=endpoint_pts,
                source_inputs=spec.inputs,
                video_sha256=prep["inputs"]["video_sha256"],
                source_group_guard=GROUPS[v],
                boxer_id=None,
                session_id=None,
                state="unknown",
                discovery_hint=reason,
                suggestions=refs,
                selection="bounded_source_suggestions_not_model_predictions",
                known_anchor_ids=[
                    e["source_id"]
                    for e in evidence["anchors"]
                    if e["recording"] == v and e["start"] < stop and start < e["stop"]
                ],
            )
        )
    return dict(
        schema_version="interval-pilot-1.0",
        roles=ROLES,
        entries=entries,
        source="automated",
        human_decisions=0,
        training_enabled=False,
        default_state="unknown",
        context_is_exhaustive=False,
    )


def collect(root):
    """Validate original v1 and compose a pending/reviewed manifest; never changes either."""
    old = validate_continuous(root)
    ws = workspace_at(root, "localization-v1")
    manifest, queue_hash = ws.manifest()
    saved, _ = ws.base.store.json(f"{OUTPUT}/audit.json")
    current = audit(ws)
    require(saved == current, "localization_anchor_audit_stale")
    require(manifest == queue(ws, current), "localization_queue_stale")
    ref, _ = ws.base.store.json(CONTINUOUS_PIN)
    service = LocalizationReview(ws)
    reviewed, heads, receipts = [], {}, dict(current["source_receipts"])
    for path in (PIN, f"{OUTPUT}/manifest.json", f"{OUTPUT}/audit.json", CONTINUOUS_PIN):
        _, receipts[path] = ws.base.store.read(path)
    for entry in manifest["entries"]:
        detail = ws.detail(entry["id"])
        state = service.detail(entry["id"])
        heads[entry["id"]] = state["head"]
        for r in state["history"]:
            receipts[f"{service.directory}/{entry['id']}/v{r['version']:08d}/review.json"] = r[
                "sha256"
            ]
        contract = supervision(entry, state["events"], state["complete"])
        # Sparse human anchors cannot silently become background or uncertain under a new review.
        for anchor in current["anchors"]:
            if anchor["recording"] != entry["recording"]:
                continue
            for run in contract["supervision_runs"]:
                if overlap(anchor, run) and (
                    run["activity"] == 0 or (state["complete"] and run["activity"] is None)
                ):
                    raise ValueError(
                        "New review conflicts with existing human anchor; resolve explicitly"
                    )
        reviewed.append(
            dict(
                id=entry["id"],
                provenance=detail["entry"],
                complete=state["complete"],
                events=state["events"],
                review_head=state["head"],
                **contract,
            )
        )
    require(
        heads == {e["id"]: service.detail(e["id"])["head"] for e in manifest["entries"]},
        "localization_reviews_changed",
    )
    train = [i for i in reviewed if i["provenance"]["recording"] == "V6"]
    dev = [i for i in reviewed if i["provenance"]["recording"] == "V5"]
    canonical = canonical_events(old["intervals"], current["anchors"], reviewed)
    for interval in reviewed:
        p = interval["provenance"]
        spec = ws.base._recording(p["recording"])
        prep = ws.base._preparation(p["recording"], spec)
        with ws.frames.lock:
            arrays = ws.frames._load(p["recording"], spec, prep)
            for run in interval["supervision_runs"]:
                run["timing"] = interval_timing(arrays, run["start"], run["stop"])
                run["endpoint_pts"] = points(arrays, run["start"], run["stop"])
    return dict(
        schema_version="localization-evidence-1.0",
        dataset_id="localization-evidence-v1",
        frame_convention="zero_based_half_open",
        base_continuous_dataset=ref,
        base_intervals=old["intervals"],
        preserved_exclusions=old["excluded_intervals"],
        held_boundary_verified=current["held_boundary_verified"],
        sparse_positive_anchors=current["anchors"],
        additional_intervals=reviewed,
        canonical_event_references=canonical,
        roles=ROLES,
        source_group_guards=GROUPS,
        review_queue_sha256=queue_hash,
        review_heads=heads,
        source_receipts=receipts,
        default_state="unknown",
        context_is_exhaustive=False,
        training_authorized=False,
        deduplication=(
            "Union activity by recording/frame; identical ranges are one reference "
            "with multiple origins. Never count anchors twice."
        ),
        readiness=dict(
            training_review_complete=all(i["complete"] for i in train),
            development_review_complete=all(i["complete"] for i in dev),
            experiment_ready=False,
            reason="Human review and new experiment approval required",
        ),
        summary=dict(
            original_events=11,
            additional_verified_events=len(current["anchors"]),
            pending_intervals=sum(not i["complete"] for i in reviewed),
            reviewed_intervals=sum(i["complete"] for i in reviewed),
            review_seconds=str(
                sum(
                    (
                        Fraction(e["timing"]["stop_exclusive"]) - Fraction(e["timing"]["start"])
                        for e in manifest["entries"]
                    ),
                    Fraction(),
                )
            ),
        ),
    )


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encode(value))


def prepare(root):
    root = Path(root)
    require(not (root / OUTPUT).exists() and not (root / PIN).exists(), "queue_already_exists")
    ws = workspace_at(root, "original")
    validate_continuous(root)
    evidence = audit(ws)
    manifest = queue(ws, evidence)
    write_new(root / OUTPUT / "audit.json", evidence)
    write_new(root / OUTPUT / "manifest.json", manifest)
    write_new(root / PIN, dict(path=f"{OUTPUT}/manifest.json", sha256=sha(encode(manifest))))
    result = collect(root)
    snapshot(root, root / OUTPUT / "initial-evidence", result)
    write_new(
        root / DATA_PIN,
        dict(path=f"{OUTPUT}/initial-evidence/manifest.json", sha256=sha(encode(result))),
    )
    return result


def snapshot(root, directory, result):
    """Exclusive immutable snapshot with durable human-review byte backups."""
    root, directory = Path(root), Path(directory)
    require(
        directory.resolve().is_relative_to((root / "reports").resolve()), "snapshot_outside_reports"
    )
    require(not directory.exists(), "snapshot_already_exists")
    receipts = dict(result["source_receipts"])
    # Original continuous v1 already has durable backups, validated separately on every check.
    for path, expected in receipts.items():
        raw = (root / path).read_bytes()
        require(sha(raw) == expected, "review_changed_during_snapshot")
        target = directory / "review-backup" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    write_new(directory / "manifest.json", result)


def check(root):
    root = Path(root)
    result = collect(root)
    pin = json.loads((root / DATA_PIN).read_text())
    require(file_hash(root / pin["path"]) == pin["sha256"], "initial_evidence_changed")
    original = json.loads((root / pin["path"]).read_text())
    backup = (root / pin["path"]).parent / "review-backup"
    for p, expected in original["source_receipts"].items():
        require(file_hash(backup / p) == expected, "review_backup_changed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--previews", action="store_true", help="Prepare existing buffered clips, no review writes"
    )
    parser.add_argument(
        "--snapshot", type=Path, help="Write a new immutable evidence snapshot under reports"
    )
    args = parser.parse_args()
    root = Path.cwd()
    result = check(root) if args.check or args.previews or args.snapshot else prepare(root)
    if args.snapshot:
        snapshot(root, args.snapshot, result)
    if args.previews:
        ws = workspace_at(root, "localization-v1")
        clips = ClipService(ws, ws.frames)
        for e in ws.manifest()[0]["entries"]:
            clips.prepare(e["id"])
    print(json.dumps(dict(summary=result["summary"], readiness=result["readiness"]), indent=2))


if __name__ == "__main__":
    main()
