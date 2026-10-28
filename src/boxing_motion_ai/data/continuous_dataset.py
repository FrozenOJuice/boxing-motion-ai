"""Freeze explicitly reviewed activity supervision; unknown footage stays unknown."""

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path

from ..services.dashboard import require
from ..services.interval_events import DIRECTORY, SCOPE, HumanPunch, IntervalEvents
from ..services.interval_review import IntervalDecision
from ..services.review_workspace import encode, sha
from .interval_pilot import GROUPS, ROLES
from .verified_background import PIN as BACKGROUND_PIN
from .verified_background import validate as validate_background
from .verified_background import workspace_at

OUTPUT = "reports/boxingvi/continuous-dataset-v1"
PIN = "configs/continuous_dataset_v1.json"
LABELS = ("jab", "cross", "lead_hook", "rear_hook", "lead_uppercut", "rear_uppercut")


def duration(timing):
    return Fraction(timing["stop_exclusive"]) - Fraction(timing["start"])


def runs(start, stop, events):
    """Only call for an explicitly exhaustive interval. No supervision outside its bounds."""
    edges = sorted(
        {start, stop} | {max(start, min(stop, e[k])) for e in events for k in ("start", "stop")}
    )
    result = []
    for left, right in zip(edges, edges[1:], strict=False):
        active = [e for e in events if e["start"] < right and e["stop"] > left]
        result.append(
            dict(
                start=left,
                stop=right,
                activity=int(bool(active)),
                event_ids=[e["event_id"] for e in active],
                type_target=active[0]["punch_type"]
                if len(active) == 1 and not active[0]["type_uncertain"]
                else None,
                type_loss_enabled=len(active) == 1 and not active[0]["type_uncertain"],
            )
        )
    return result


def collect(root):
    background = validate_background(root)
    ws = workspace_at(root, "original")
    service = IntervalEvents(ws)
    intervals, excluded, heads, events, receipts = [], [], {}, [], {}

    def reference(path, expected=None):
        _, digest = ws.base.store.read(path, expected)
        receipts[path] = digest
        return dict(path=path, sha256=digest)

    background_ref = reference(BACKGROUND_PIN)
    bg_pin, _ = ws.base.store.json(BACKGROUND_PIN)
    reference(bg_pin["path"], bg_pin["sha256"])
    backup_root = str(Path(bg_pin["path"]).parent)
    backup_receipt_path = f"{backup_root}/review-backup-sha256.json"
    reference(backup_receipt_path, bg_pin["backup_receipt_sha256"])
    background_backup, _ = ws.base.store.json(backup_receipt_path, bg_pin["backup_receipt_sha256"])
    for path, digest in background_backup.items():
        reference(f"{backup_root}/review-backup/{path}", digest)
    ref, _ = ws.base.store.json(ws.config_path)
    reference(ws.config_path)
    reference(ref["path"], ref["sha256"])
    original = ws.snapshot()
    for record in original["history"]:
        reference(f"{ws.directory}/v{record['version']:08d}/review.json", record["sha256"])
    for b in background["intervals"]:
        p = b["provenance"]
        require(
            p["recording"] in ("V4", "V6")
            and p["role"] == "train_candidate"
            and p["source_group_guard"] == GROUPS[p["recording"]],
            "continuous_split_leakage",
        )
        reference(b["review"]["path"], b["review"]["sha256"])
        intervals.append(
            dict(
                id=b["id"],
                kind="explicit_no_punch",
                provenance=p,
                review=b["review"],
                exhaustive=True,
                events=[],
                supervision_runs=runs(p["start"], p["stop"], []),
            )
        )
    for key in SCOPE:
        state = service.detail(key)  # Replays all edits, removals and completion evidence.
        heads[key] = state["head"]
        for r in state["history"]:
            reference(f"{DIRECTORY}/{key}/v{r['version']:08d}/review.json", r["sha256"])
        if key == "V4-bg-5" or not state["complete"]:
            excluded.append(
                dict(
                    id=key,
                    reason="user_held_incomplete"
                    if key == "V4-bg-5"
                    else "no_current_explicit_completion",
                    head=state["head"],
                    saved_events=len(state["events"]),
                )
            )
            continue
        detail = ws.detail(key)
        entry = detail["entry"]
        require(
            entry["recording"] == "V4"
            and entry["role"] == "train_candidate"
            and entry["source_group_guard"] == GROUPS["V4"],
            "continuous_split_leakage",
        )
        completion = state["history"][-1]
        payload = completion["payload"]
        require(
            payload["operation"] == "complete"
            and payload["watched_entire"]
            and payload["all_punches_labeled"]
            and completion["source"] == "manual",
            "continuous_completion_missing",
        )
        coverage = ws._viewing(
            entry,
            IntervalDecision(
                request_id=payload["request_id"],
                expected_head=payload["expected_head"],
                decision="punch_present",
                inspected_frames=payload["inspected_frames"],
                playback_coverage=payload["playback_coverage"],
            ),
        )
        require(coverage["complete"], "continuous_viewing_incomplete")
        current = []
        for event in state["events"]:
            typed = HumanPunch.model_validate({k: event[k] for k in HumanPunch.model_fields})
            require(
                service.event_metadata(entry, typed)
                == {k: event[k] for k in ("timing", "endpoint_pts", "partial_overlap")},
                "continuous_event_timing_changed",
            )
            for index in (event["start"], event["stop"] - 1):
                ws.frames.frame("V4", index)
            upsert = next(
                r
                for r in reversed(state["history"])
                if r["payload"]["operation"] == "upsert"
                and r["payload"]["event"]["event_id"] == event["event_id"]
            )
            item = event | dict(
                recording="V4",
                interval_id=key,
                duration_exact_seconds=str(duration(event["timing"])),
                review=reference(
                    f"{DIRECTORY}/{key}/v{upsert['version']:08d}/review.json", upsert["sha256"]
                ),
            )
            current.append(item)
            events.append(item)
        for index in (entry["start"], entry["stop"] - 1):
            ws.frames.frame("V4", index)
        intervals.append(
            dict(
                id=key,
                kind="exhaustive_punch_interval",
                provenance=entry,
                review=reference(
                    f"{DIRECTORY}/{key}/v{completion['version']:08d}/review.json",
                    completion["sha256"],
                ),
                source_review_sha256=state["source_review_sha256"],
                exhaustive=True,
                viewing_validation=coverage,
                events=current,
                supervision_runs=runs(entry["start"], entry["stop"], current),
            )
        )
    require(len({e["event_id"] for e in events}) == len(events), "duplicate_human_event_id")
    # Conflicting supervision is a blocker, never resolved by priority or silent relabeling.
    for i, a in enumerate(intervals):
        for b in intervals[i + 1 :]:
            x, y = a["provenance"], b["provenance"]
            require(
                x["recording"] != y["recording"]
                or x["stop"] <= y["start"]
                or y["stop"] <= x["start"],
                "continuous_supervision_overlap",
            )
    for e in events:
        for b in background["intervals"]:
            p = b["provenance"]
            require(
                p["recording"] != e["recording"]
                or p["stop"] <= e["start"]
                or e["stop"] <= p["start"],
                "positive_conflicts_with_verified_background",
            )
    overlaps = [
        dict(
            first=a["event_id"],
            second=b["event_id"],
            start=max(a["start"], b["start"]),
            stop=min(a["stop"], b["stop"]),
        )
        for i, a in enumerate(events)
        for b in events[i + 1 :]
        if a["recording"] == b["recording"] and a["start"] < b["stop"] and b["start"] < a["stop"]
    ]
    counts = Counter(e["punch_type"] for e in events)
    totals = Counter()
    for interval in intervals:
        p = interval["provenance"]
        spec = ws.base._recording(p["recording"])
        prep = ws.base._preparation(p["recording"], spec)
        with ws.frames.lock:
            arrays = ws.frames._load(p["recording"], spec, prep)

            def point(i, arrays=arrays):
                return dict(
                    pts=int(arrays["pts"][i]),
                    time_base_num=int(arrays["time_base_num"][i]),
                    time_base_den=int(arrays["time_base_den"][i]),
                )

            for run in interval["supervision_runs"]:
                endpoints = {"start": point(run["start"]), "stop_exclusive": point(run["stop"])}
                times = [
                    Fraction(v["pts"] * v["time_base_num"], v["time_base_den"])
                    for v in endpoints.values()
                ]
                run["endpoint_pts"] = endpoints
                run["duration_exact_seconds"] = str(times[1] - times[0])
                totals[f"activity_{run['activity']}_seconds"] += times[1] - times[0]
                totals[f"activity_{run['activity']}_frames"] += run["stop"] - run["start"]
    require(
        heads == {key: service.detail(key)["head"] for key in SCOPE}, "continuous_reviews_changed"
    )
    return dict(
        schema_version="continuous-dataset-1.0",
        dataset_id="boxingvi-continuous-v1",
        frame_convention="zero_based_half_open",
        roles=ROLES,
        source_group_guards=GROUPS,
        default_unreviewed_state="unknown",
        context_is_exhaustive=False,
        detector_training_authorized=False,
        development_intervals=[],
        evaluation_intervals=[],
        background_reference=background_ref,
        background_review_head=background["review_head"],
        event_review_heads=heads,
        source_receipts=receipts,
        readiness=dict(
            engineering_smoke_possible=bool(events and background["intervals"]),
            class_coverage_complete=all(counts[k] for k in LABELS),
            development_reference_available=False,
            evaluation_reference_available=False,
            requested_four_completions_present=sum(
                i["kind"] == "exhaustive_punch_interval" for i in intervals
            )
            == 4,
            detector_experiment_ready=False,
        ),
        summary=dict(
            positive_events=len(events),
            explicit_no_punch_intervals=len(background["intervals"]),
            completed_positive_intervals=sum(
                i["kind"] == "exhaustive_punch_interval" for i in intervals
            ),
            expected_completed_positive_intervals=4,
            class_counts={k: counts[k] for k in LABELS},
            class_duration_exact_seconds={
                k: str(
                    sum((duration(e["timing"]) for e in events if e["punch_type"] == k), Fraction())
                )
                for k in LABELS
            },
            explicit_negative_seconds=str(
                sum(
                    (duration(b["provenance"]["timing"]) for b in background["intervals"]),
                    Fraction(),
                )
            ),
            completed_interval_seconds=str(
                sum(
                    (
                        duration(i["provenance"]["timing"])
                        for i in intervals
                        if i["kind"] == "exhaustive_punch_interval"
                    ),
                    Fraction(),
                )
            ),
            uncertain_types=counts["unknown"],
            overlap_pairs=len(overlaps),
            partial_events=sum(e["partial_overlap"] for e in events),
            **{k: str(v) if k.endswith("seconds") else v for k, v in totals.items()},
        ),
        intervals=intervals,
        excluded_intervals=excluded,
        overlaps=overlaps,
    )


def prepare(root):
    root = Path(root)
    output, pin = root / OUTPUT, root / PIN
    require(not output.exists() and not pin.exists(), "continuous_version_already_exists")
    result = collect(root)
    require(result["summary"]["positive_events"] > 0, "continuous_no_verified_positive_events")
    ws = workspace_at(root, "original")
    backup = {
        path: ws.base.store.read(path, digest)[0].encode("utf-8")
        for path, digest in result["source_receipts"].items()
    }
    require(
        result["event_review_heads"]
        == {key: IntervalEvents(ws).detail(key)["head"] for key in SCOPE},
        "continuous_reviews_changed",
    )
    output.mkdir(parents=True, exist_ok=False)
    for path, raw in backup.items():
        target = output / "review-backup" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    raw = encode(result)
    (output / "manifest.json").write_bytes(raw)
    with pin.open("xb") as stream:
        stream.write(encode(dict(path=f"{OUTPUT}/manifest.json", sha256=sha(raw))))
    return result


def validate(root):
    ws = workspace_at(root, "original")
    ref, _ = ws.base.store.json(PIN)
    require(ref["path"] == f"{OUTPUT}/manifest.json", "unexpected_continuous_path")
    frozen, _ = ws.base.store.json(ref["path"], ref["sha256"])
    require(frozen == collect(root), "continuous_snapshot_stale")
    for path, digest in frozen["source_receipts"].items():
        ws.base.store.read(f"{OUTPUT}/review-backup/{path}", digest)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Revalidate the immutable dataset snapshot"
    )
    args = parser.parse_args()
    result = validate(Path.cwd()) if args.check else prepare(Path.cwd())
    print(result["summary"])
    print("Excluded:", result["excluded_intervals"])


if __name__ == "__main__":
    main()
