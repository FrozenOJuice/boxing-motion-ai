"""Explicit blue-reference experiment subset; source suggestions alone never become labels."""

import argparse
import hashlib
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

from ..analysis.common import sha, write_json
from ..services.dashboard import require
from ..services.localization_review import LocalizationReview
from ..vision.observation_study import fresh, read
from .annotation_review import digest
from .continuous_dataset import PIN as BASE_PIN
from .continuous_dataset import validate as validate_base
from .interval_pilot import GROUPS, ROLES
from .localization_evidence import audit, overlap
from .verified_background import workspace_at

OUTPUT = "reports/boxingvi/blue-localization-dataset-v1"
PIN = "configs/blue_localization_dataset_v1.json"
POLICY = dict(
    version="human-blue-native-bounds-1.0",
    association="unique exact native bounds in current human event and prepared source suggestion",
    explicit_source_click_id=False,
    labels="manual upsert, inspected endpoints; source type never supplies target",
    scope="new queue positives only; gaps and excluded/conflicting motion Unknown",
    conflicts="excluded overlaps Unknown; boundary-support masks; conflicted references unscored",
    new_exhaustive=False,
)


def select(entry, state, sources):
    """Deterministic association, not proximity/IoU matching or automatic suggestion approval."""
    require(
        entry["recording"] in ("V5", "V6")
        and entry["role"] == ROLES[entry["recording"]]
        and entry["source_group_guard"] == GROUPS[entry["recording"]],
        "blue_source_role",
    )
    require(
        state["complete"] and state["history"][-1]["payload"]["operation"] == "complete",
        "blue_review_incomplete",
    )
    selected, selected_ids = [], set()
    for source in sources:
        bounds = source.get("source_interval_half_open") or source.get("array_slice_half_open")
        require(bounds and len(bounds) == 2 and bounds[0] < bounds[1], "blue_mapping_missing")
        matches = [
            e
            for e in state["events"]
            if e.get("kind") == "punch" and [e["start"], e["stop"]] == bounds
        ]
        require(len(matches) == 1, "blue_missing_or_ambiguous_human_event")
        e = deepcopy(matches[0])
        require(e["event_id"] not in selected_ids, "blue_duplicate_association")
        origin = next(
            r
            for r in reversed(state["history"])
            if r["payload"]["operation"] == "upsert"
            and r["payload"]["event"]["event_id"] == e["event_id"]
        )
        require(
            origin["source"] == "manual"
            and e["start"] in origin["payload"]["inspected_frames"]
            and e["stop"] - 1 in origin["payload"]["inspected_frames"],
            "blue_boundary_not_inspected",
        )
        e.update(
            source_row=source["source_row"],
            source_candidate_sha256=digest(source),
            source_type_provisional=source["punch_type_provisional"],
            review=dict(version=origin["version"], sha256=origin["sha256"]),
            association=POLICY["association"],
        )
        selected.append(e)
        selected_ids.add(e["event_id"])
    excluded = [deepcopy(e) for e in state["events"] if e["event_id"] not in selected_ids]
    for e in selected:
        conflicts = [x["event_id"] for x in excluded if overlap(e, x)]
        e.update(
            excluded_overlap_ids=conflicts,
            reference_eligible=not conflicts,
            conflict_policy="overlap Unknown; retain full human bounds in provenance",
        )
    return dict(
        id=entry["id"],
        provenance=entry,
        events=selected,
        excluded_events=excluded,
        exhaustive=False,
        supervision="sparse_positive_only",
        review_head=state["head"],
        latest_completion_sha256=state["history"][-1]["sha256"],
    )


def collect(root):
    old = validate_base(root)
    ws = workspace_at(root, "localization-v1")
    anchors = audit(ws)
    service = LocalizationReview(ws)
    intervals, receipts = [], dict(anchors["source_receipts"])
    manifest, manifest_hash = ws.manifest()
    for p in (
        "configs/localization_review_v1.json",
        "reports/boxingvi/localization-review-v1/manifest.json",
        BASE_PIN,
    ):
        _, receipts[p] = ws.base.store.read(p)
    for entry in manifest["entries"]:
        state = service.detail(entry["id"])
        detail = ws.detail(entry["id"])
        intervals.append(select(entry, state, detail["annotations"]))
        for r in state["history"]:
            receipts[f"{service.directory}/{entry['id']}/v{r['version']:08d}/review.json"] = r[
                "sha256"
            ]
    require([len(i["events"]) for i in intervals] == [3, 2, 3, 2, 4, 8], "blue_inventory_changed")
    # Reuse uncontested V4 sparse positives. V6 anchors are already selected and not duplicated.
    for a in anchors["anchors"]:
        if a["recording"] == "V6":
            selected = [
                e
                for i in intervals
                if i["provenance"]["recording"] == "V6"
                for e in i["events"]
                if (e["start"], e["stop"]) == (a["start"], a["stop"])
            ]
            require(len(selected) == 1, "blue_anchor_conflict")
            selected[0]["earlier_anchor_reference"] = a["source_id"]
            continue
        # One second of annotation-free visual context either side; sparse labels only.
        spec = ws.base._recording("V4")
        prep = ws.base._preparation("V4", spec)
        with ws.frames.lock:
            arrays = ws.frames._load("V4", spec, prep)
        p = dict(
            recording="V4",
            role="train_candidate",
            source_group_guard=GROUPS["V4"],
            start=a["start"],
            stop=a["stop"],
            context_start=a["start"] - 24,
            context_end=a["stop"] + 24,
            source_inputs=spec.inputs,
            video_sha256=prep["inputs"]["video_sha256"],
            frame_count=len(arrays["frame_index"]),
        )
        event = a | dict(reference_eligible=True, excluded_overlap_ids=[])
        intervals.append(
            dict(
                id=a["source_id"].replace(":", "-anchor-"),
                provenance=p,
                events=[event],
                excluded_events=[],
                exhaustive=False,
                supervision="sparse_positive_only",
            )
        )
    require(
        all(
            i.get("review_head") == service.detail(i["id"])["head"]
            for i in intervals
            if i.get("review_head")
        ),
        "blue_reviews_changed",
    )
    return dict(
        schema_version="blue-localization-dataset-1.0",
        policy=POLICY,
        base_continuous_dataset=read(BASE_PIN),
        base_intervals=old["intervals"],
        intervals=intervals,
        source_receipts=receipts,
        review_queue_sha256=manifest_hash,
        roles=ROLES,
        source_group_guards=GROUPS,
        unknown_outside_positive_spans=True,
        training_recordings=["V4", "V6"],
        development_recordings=["V5"],
        training_authorized_for="bounded_three_arm_blue_localization_v1_only",
        preserved_exclusions=old["excluded_intervals"],
        summary=dict(
            original_v4_events=11,
            sparse_v4_anchors=2,
            selected_v6=14,
            selected_v5=8,
            excluded_manual=18,
            conflicted_v6=sum(
                not e["reference_eligible"]
                for i in intervals
                if i["provenance"]["recording"] == "V6"
                for e in i["events"]
            ),
        ),
    )


def targets(interval, video, offset, radius):
    """Sparse positive activity and within-instance boundaries; excluded motion remains Unknown."""
    times = [Q(f["timestamp_exact"]) for f in video["frames"]]
    n = len(times)
    y, mask = np.zeros((n, 3), np.float32), np.zeros((n, 3), bool)
    excluded = np.zeros(n, bool)
    for e in interval["excluded_events"]:
        a, z = max(0, e["start"] - offset), min(n, e["stop"] - offset)
        if a < z:
            excluded[a:z] = True
    censoring = []
    for e in interval["events"]:
        a, z = e["start"] - offset, e["stop"] - offset
        require(0 <= a < z <= n, "blue_event_outside_context")
        mask[a:z] = True
        y[a:z, 0] = 1
        for channel, point in ((1, a), (2, z - 1)):
            if excluded[point]:
                continue
            for i in range(a, z):
                distance = abs(times[i] - times[point])
                if distance < radius:
                    y[i, channel] = max(y[i, channel], float(1 - distance / radius))
        censoring.append(
            dict(
                event_id=e["event_id"],
                onset_available=not bool(excluded[a]),
                end_available=not bool(excluded[z - 1]),
                primary_reference_eligible=e["reference_eligible"],
            )
        )
    mask[excluded] = False
    for j in np.flatnonzero(excluded):
        for i in range(n):
            if abs(times[i] - times[j]) <= radius:
                mask[i, 1:] = False
    # No boundary/background negatives in unreviewed gaps or outside selected events.
    return y, mask, censoring


def prepare(root):
    root = Path(root)
    require(not (root / PIN).exists(), "blue_pin_already_exists")
    result = collect(root)
    out = fresh(root / OUTPUT)
    for path, expected in result["source_receipts"].items():
        raw = (root / path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, "blue_receipt_changed")
        dest = out / "review-backup" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    write_json(out / "manifest.json", result)
    write_json(root / PIN, dict(path=f"{OUTPUT}/manifest.json", sha256=sha(out / "manifest.json")))
    return result


def validate(root):
    root = Path(root)
    pin = read(root / PIN)
    require(sha(root / pin["path"]) == pin["sha256"], "blue_manifest_changed")
    frozen = read(root / pin["path"])
    require(frozen == collect(root), "blue_snapshot_stale")
    for path, h in frozen["source_receipts"].items():
        require(
            sha((root / pin["path"]).parent / "review-backup" / path) == h, "blue_backup_changed"
        )
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print((validate(Path.cwd()) if args.check else prepare(Path.cwd()))["summary"])
