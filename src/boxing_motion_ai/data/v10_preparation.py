"""V10-only provisional preparation evidence; no training or annotation corrections."""

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from ..vision.annotations import read_annotations
from ..vision.pipeline import sha256, write_json
from ..vision.recording import load_recording
from .pose_quality import POLICIES, retain, score_validity
from .preparation import SOURCE_LABELS
from .source_review import decode_frames, render_case, write_player

ROOT = Path("reports/boxingvi/v10-full-v1")
DATASET = Path("data/processed/boxingvi/V10/candidates-full-v1")
POSE_RUN = Path("data/processed/boxingvi/V10/pose-full-image-v1")
REVIEW = Path("configs/v10_alignment_review.yaml")
RECORDING = Path("configs/v10_recording_provisional_v1.yaml")
QUARANTINED = [9, 10, 123, *range(130, 138)]


def check_timing(arrays, frames):
    """Every decoded index and rational timestamp must match the source inventory."""
    import numpy as np

    n = len(frames)
    if not np.array_equal(arrays["frame_index"], np.arange(n)):
        raise ValueError("Full source index inventory differs")
    observed = [
        Fraction(int(p) * int(a), int(b))
        for p, a, b in zip(
            arrays["pts"], arrays["time_base_num"], arrays["time_base_den"], strict=True
        )
    ]
    if observed != [Fraction(f["pts_time"]) for f in frames]:
        raise ValueError("Exact source timestamps differ; no FPS approximation allowed")


def summarize_selection(rows, selected):
    """Count verified policy choices; reject restoration, duplicates and unknown IDs."""
    lookup = {r["source_row"]: r for r in rows}
    if len(lookup) != len(rows) or len(set(selected)) != len(selected):
        raise ValueError("Duplicate source rows in selection")
    if any(n not in lookup for n in selected):
        raise ValueError("Unknown source row in selection")
    if set(selected) & set(QUARANTINED) or any(
        not lookup[n]["annotation_eligible"] or lookup[n]["exclusion_reasons"] for n in selected
    ):
        raise ValueError("Excluded annotation cannot be restored by pose quality")
    result = {}
    for label in SOURCE_LABELS:
        subset = [r for r in rows if r["punch_type_provisional"] == label]
        eligible = [r for r in subset if r["annotation_eligible"]]
        kept = [r for r in subset if r["source_row"] in selected]
        result[label] = {
            "source": len(subset),
            "annotation_eligible": len(eligible),
            "annotation_excluded": len(subset) - len(eligible),
            "pose_rejected": len(eligible) - len(kept),
            "retained": len(kept),
            "retained_source_rows": [r["source_row"] for r in kept],
        }
    return result


def source_inputs():
    """Access only the explicit V10 inventory and its hash-matched original files."""
    inventory_path = Path("reports/boxingvi/rear-punch-priorities-v1/V10/inventory.json")
    inventory = json.loads(inventory_path.read_text())
    inputs = inventory["inputs"]
    for key in ("video", "workbook"):
        if sha256(inputs[f"{key}_path"]) != inputs[f"{key}_sha256"]:
            raise ValueError("V10 source differs from reviewed source")
    probe_path = Path("reports/boxingvi/development/V10.ffprobe.json")
    if sha256(probe_path) != inputs["frame_inventory_sha256"]:
        raise ValueError("Source timing inventory changed")
    probe = json.loads(probe_path.read_text())
    return inventory_path, inventory, probe_path, probe


def make_review(output):
    """Targeted class/phase review, reusing exact-PTS strips without rewriting old evidence."""
    output = Path(output)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a fresh review directory under reports/")
    inventory_path, inventory, _, probe = source_inputs()
    rows = read_annotations(inventory["inputs"]["workbook_path"], load_recording(RECORDING))["rows"]
    lookup = {r["source_row"]: r for r in rows}
    selected = {c["source_row"] for c in inventory["review_cases"]}
    selected.update(range(130, 138))
    # Pair adjacent opposite-hand labels; inspect transition phases beyond event bounds.
    selected.update([33, 41, 48, 79, 87, 129, 139, 141, 143, 145, 146, 147, 148, 150])
    times = [Fraction(f["pts_time"]) for f in probe["frames"]]
    cases = []
    for n in sorted(selected):
        row = lookup[n]
        a, b = int(row["raw_start"]), int(row["raw_end"])
        lo, hi = min(a, b), max(a, b)
        # Broad but short context around start/end of the floor-exercise sequence.
        margin = 50 if n in (129, 130, 137, 138, 146, 153) else 8
        cases.append(
            {
                "id": f"V10-row-{n}-raw",
                "source_row": n,
                "source_class": row["source_class"],
                "hypothesis": "raw; reversed interval diagnostic only" if b < a else "raw",
                "indices": list(range(max(0, lo - margin), min(len(times), hi + margin + 1))),
                "neighbor_bounds": [
                    [r["source_row"], r["raw_start"], r["raw_end"]]
                    for r in rows
                    if r["raw_start"] <= hi + margin and r["raw_end"] >= lo - margin
                ],
                "training_restored": False,
            }
        )
    decode_frames(
        inventory["inputs"]["video_path"],
        times,
        {i for c in cases for i in c["indices"]},
        output / "frames",
    )
    for case in cases:
        render_case(case, times, output)
    write_player(cases, times, output)
    write_json(
        output / "review.json",
        {
            "schema_version": "v10-targeted-review-1.0",
            "video_id": "V10",
            "inputs": inventory["inputs"],
            "prior_inventory_path": str(inventory_path),
            "prior_inventory_sha256": sha256(inventory_path),
            "recording_config_sha256": sha256(RECORDING),
            "selected_source_rows": sorted(selected),
            "quarantined_source_rows": QUARANTINED,
            "annotation_corrections": [],
            "training_performed": False,
        },
    )


def finish(output):
    """Freeze unchanged policy decisions and provenance; never authorize/run training."""
    import numpy as np
    import yaml

    output = Path(output)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a fresh readiness directory under reports/")
    _, inventory, probe_path, probe = source_inputs()
    review = yaml.safe_load(REVIEW.read_text())
    if (
        review["video_id"] != "V10"
        or review["status"] != "raw_indices_supported_for_provisional_preparation"
        or review["excluded_source_rows"] != QUARANTINED
        or review["mapping"].get("frame_offset") != 0
        or review["mapping"].get("end_assumption") != "inclusive"
    ):
        raise ValueError("Explicit V10 review with preserved quarantines required")
    evidence_path = Path(review["evidence"])
    evidence = json.loads(evidence_path.read_text())
    if (
        evidence["inputs"] != inventory["inputs"]
        or sha256(evidence_path) != review["evidence_sha256"]
        or evidence["recording_config_sha256"] != sha256(RECORDING)
    ):
        raise ValueError("Review source provenance differs")
    manifest_path = POSE_RUN / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = load_recording(RECORDING).model_dump()
    if (
        manifest["status"] != "complete"
        or manifest["scope"] != "full_video"
        or manifest["recording_config"] != config
        or config["excluded_source_rows"] != [123, *range(130, 138)]
        or config["frame_offset"] != 0
        or config["end_assumption"] != "inclusive"
        or manifest["config"]["pose_mode"] != "image"
        or manifest["config"]["confidence"] != 0.5
    ):
        raise ValueError("Incomplete extraction or changed configuration")
    for key in ("video", "workbook"):
        if manifest["inputs"][key]["sha256"] != inventory["inputs"][f"{key}_sha256"]:
            raise ValueError("Extraction belongs to different source files")
    if manifest["poses_sha256"] != sha256(POSE_RUN / "poses.jsonl"):
        raise ValueError("Pose journal changed")
    prep = json.loads((DATASET / "preparation.json").read_text())
    index_path = ROOT / "quality/V10/policy-index.json"
    index = json.loads(index_path.read_text())
    if (
        prep["status"] != "complete"
        or prep["scope"] != "full_video"
        or index["video_id"] != "V10"
        or index["policy"] != "mask_aware70"
        or index["definition"] != POLICIES["mask_aware70"]
        or index["source_mask_threshold"] != 0.5
        or index["max_missing_frame_fraction"] != 0.1
        or prep["config"]["excluded_source_rows"] != QUARANTINED
        or prep["npz_sha256"] != sha256(DATASET / "poses.npz")
        or prep["inputs"]["poses_sha256"] != manifest["poses_sha256"]
        or prep["extracted_frame_count"] != len(probe["frames"])
        or manifest["source_frame_count"] != len(probe["frames"])
    ):
        raise ValueError("Incomplete preparation or changed quality policy")
    for name, digest in index["inputs"].items():
        if sha256(DATASET / name) != digest:
            raise ValueError("Policy inputs changed")
    if prep["inputs"]["pose_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Preparation belongs to another extraction")
    rows = [json.loads(s) for s in (DATASET / "candidates.jsonl").read_text().splitlines()]
    with np.load(DATASET / "poses.npz", allow_pickle=False) as arrays:
        check_timing(arrays, probe["frames"])
        if not np.array_equal(
            arrays["valid_mask"], score_validity(arrays["coordinates"], arrays["confidence"], 0.5)
        ):
            raise ValueError("Saved masks differ from the original fixed mask contract")
        selected = []
        for row in rows:
            if row["annotation_eligible"]:
                a, b = row["array_slice_half_open"]
                if retain(
                    arrays["valid_mask"][a:b],
                    arrays["pose_status"][a:b] == 2,
                    POLICIES["mask_aware70"],
                ):
                    selected.append(row["source_row"])
        if selected != index["selected_source_rows"]:
            raise ValueError("Policy index differs from recomputed retention")
        statuses = dict(Counter(int(i) for i in arrays["pose_status"]))
    counts = summarize_selection(rows, selected)
    result = {
        "schema_version": "v10-preparation-readiness-1.0",
        "provisional_preparation_ready": all(c["retained"] for c in counts.values()),
        "training_performed": False,
        "training_authorized": False,
        "source_group_id": "utah-lee-unresolved",
        "boxer_id": None,
        "session_id": None,
        "mapping": review["mapping"],
        "uncertainties": review["uncertainties"],
        "exact_timing_verified_frames": len(probe["frames"]),
        "class_counts": counts,
        "pose_status_counts": statuses,
        "excluded_annotations": [
            {
                "source_row": r["source_row"],
                "class": r["punch_type_provisional"],
                "raw_bounds": [r["raw_start"], r["raw_end"]],
                "reasons": r["exclusion_reasons"],
            }
            for r in rows
            if not r["annotation_eligible"]
        ],
        "pose_rejected_source_rows": [
            r["source_row"]
            for r in rows
            if r["annotation_eligible"] and r["source_row"] not in selected
        ],
        "unknown_outside_annotation_bounds_frames": prep[
            "unknown_outside_annotation_bounds_frames"
        ],
        "hashes": {
            str(p): sha256(p)
            for p in [
                REVIEW,
                RECORDING,
                evidence_path,
                probe_path,
                manifest_path,
                DATASET / "preparation.json",
                DATASET / "poses.npz",
                DATASET / "candidates.jsonl",
                index_path,
            ]
        },
        "source_inputs": inventory["inputs"],
        "comparison": comparison(),
    }
    output.mkdir(parents=True)
    write_json(output / "readiness.json", result)
    lines = [
        "# V10 provisional preparation",
        "",
        "No training performed or authorized.",
        "",
        "| Class | Source | Annotation excluded | Pose rejected | Retained |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, c in counts.items():
        lines.append(
            f"| {label} | {c['source']} | {c['annotation_excluded']} | "
            f"{c['pose_rejected']} | {c['retained']} |"
        )
    lines += [
        "",
        "All source labels/boundaries remain provisional. Unknown footage stays unknown.",
        "See readiness.json for exact-PTS checks, hashes, exclusions and comparison evidence.",
    ]
    (output / "readiness.md").write_text("\n".join(lines) + "\n")
    return result


def comparison():
    """Full-data original-mask quality only; no reserved recording paths are opened."""
    result = {}
    for v in ("V4", "V5", "V7", "V8", "V10"):
        dataset = Path(f"data/processed/boxingvi/{v}/candidates-full-v1")
        if v in ("V4", "V5"):
            path = Path(f"reports/boxingvi/expansion/quality/{v}/quality.json")
        elif v == "V10":
            path = ROOT / "quality/V10/quality.json"
        else:
            path = Path(f"reports/boxingvi/v7-v8-full-v1/quality/{v}/quality.json")
        q = json.loads(path.read_text())
        if q["scope"] != "full_video" or q["policies"]["mask_aware70"] != POLICIES["mask_aware70"]:
            raise ValueError("Comparison must use full recordings with unchanged policy")
        for name, digest in q["inputs"].items():
            if sha256(dataset / name) != digest:
                raise ValueError("Comparison source changed")
        result[v] = {
            "quality_path": str(path),
            "quality_sha256": sha256(path),
            "by_class": q["by_class"],
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["review", "finish"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (make_review if args.action == "review" else finish)(args.output)
