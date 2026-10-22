"""Freeze approved V6 preparation and descriptive representation diagnostics; no training."""

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from ..vision.contracts import JOINTS
from ..vision.pipeline import sha256, write_json
from ..vision.recording import load_recording
from .pose_quality import ANCHORS, POLICIES, retain, score_validity
from .preparation import SOURCE_LABELS
from .v6_review import RECOMMENDED_HOLDS

ROOT = Path("reports/boxingvi/v6-full-v1")
DATASET = Path("data/processed/boxingvi/V6/candidates-full-v1")
POSE_RUN = Path("data/processed/boxingvi/V6/pose-full-image-v1")
RECORDING = Path("configs/v6_recording_provisional_v1.yaml")
REVIEW = Path("configs/v6_alignment_review.yaml")


def check_arrays(arrays, frames):
    """Require every source frame, exact rational PTS and unchanged original masks."""
    import numpy as np

    if not np.array_equal(arrays["frame_index"], np.arange(len(frames))):
        raise ValueError("Full source index inventory differs")
    observed = [
        Fraction(int(p) * int(a), int(b))
        for p, a, b in zip(
            arrays["pts"], arrays["time_base_num"], arrays["time_base_den"], strict=True
        )
    ]
    if observed != [Fraction(f["pts_time"]) for f in frames]:
        raise ValueError("Exact source timestamps differ")
    if not np.array_equal(
        arrays["valid_mask"], score_validity(arrays["coordinates"], arrays["confidence"], 0.5)
    ):
        raise ValueError("Original 0.5 masks differ")


def selection_counts(rows, selected):
    """Separate annotation exclusions from pose rejection; never restore flagged rows."""
    lookup = {r["source_row"]: r for r in rows}
    if len(lookup) != len(rows) or len(set(selected)) != len(selected):
        raise ValueError("Duplicate source rows")
    if any(n not in lookup for n in selected):
        raise ValueError("Unknown source row")
    if set(selected) & set(RECOMMENDED_HOLDS) or any(
        not lookup[n]["annotation_eligible"] or lookup[n]["exclusion_reasons"] for n in selected
    ):
        raise ValueError("Excluded annotation cannot be restored")
    result = {}
    for label in SOURCE_LABELS:
        subset = [r for r in rows if r["punch_type_provisional"] == label]
        eligible = [r for r in subset if r["annotation_eligible"]]
        kept = [r for r in subset if r["source_row"] in selected]
        result[label] = {
            "source": len(subset),
            "annotation_excluded": len(subset) - len(eligible),
            "annotation_eligible": len(eligible),
            "pose_rejected": len(eligible) - len(kept),
            "retained": len(kept),
            "retained_source_rows": [r["source_row"] for r in kept],
        }
    return result


def check_annotation_preservation(rows, reviewed):
    """No source bounds, classes, helper cells or old flags may change during preparation."""
    old = {r["source_row"]: r for r in reviewed}
    if len(old) != len(reviewed) or len(rows) != len(old):
        raise ValueError("Annotation inventory changed")
    seen = set()
    for row in rows:
        n = row["source_row"]
        if n not in old or n in seen:
            raise ValueError("Annotation inventory changed")
        seen.add(n)
        before = old[n]
        keys = ("raw_start", "raw_end", "source_class", "source_cells", "punch_type_provisional")
        if any(row[k] != before[k] for k in keys) or not set(before["flags"]) <= set(row["flags"]):
            raise ValueError("Original annotation fields or flags changed")
        expected = before["structurally_eligible"] and n not in RECOMMENDED_HOLDS
        if row["annotation_eligible"] != expected:
            raise ValueError("Original exclusions or approved holds changed")


def quantiles(values):
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return np.quantile(values, [0.1, 0.5, 0.9]).tolist() if len(values) else None


def whole_video_quality(arrays):
    """Include rest/title footage; missingness is recorded, never filled or labeled background."""
    import numpy as np

    missing = arrays["pose_status"] == 2
    edges = np.diff(np.r_[False, missing, False].astype(int))
    runs = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):

        def time(i):
            return str(
                Fraction(
                    int(arrays["pts"][i]) * int(arrays["time_base_num"][i]),
                    int(arrays["time_base_den"][i]),
                )
            )

        runs.append(
            {
                "frame_start": int(a),
                "frame_stop_exclusive": int(b),
                "frames": int(b - a),
                "first_pts_seconds": time(a),
                "last_pts_seconds": time(b - 1),
            }
        )
    valid = arrays["valid_mask"]
    return {
        "scope": "all source frames including unknown rest/title footage; not a retention policy",
        "status_counts": dict(Counter(int(s) for s in arrays["pose_status"])),
        "invalid_fraction_by_joint": {
            name: float((~valid[:, j]).mean()) for j, name in enumerate(JOINTS)
        },
        "anchor_valid_frame_fraction": float(valid[:, ANCHORS].all(1).mean()),
        "missing_run_count": len(runs),
        "longest_missing_runs": sorted(runs, key=lambda r: -r["frames"])[:10],
    }


def policy_failure_reasons(metrics):
    reasons = [k for k, minimum in POLICIES["mask_aware70"].items() if metrics[k] < minimum]
    if metrics["missing_frames"] > 0.1:
        reasons.append("missing_frames")
    return reasons


def geometry_profile(arrays, rows, selected, aspect):
    """Descriptive only: valid-only image-plane geometry, not implemented model features.

    Summaries are quantiles of clip medians; angle/speed coverage is per clip.
    No orientation canonicalization, resampling, interpolation or feature mutation.
    """
    import numpy as np

    # NpzFile reads/decompresses on each subscript; materialize each array once.
    arrays = {
        k: arrays[k]
        for k in (
            "coordinates",
            "valid_mask",
            "pts",
            "time_base_num",
            "time_base_den",
            "frame_index",
            "scene_id",
            "segment_id",
        )
    }
    clips = []
    for row in rows:
        if row["source_row"] not in selected:
            continue
        a, b = row["array_slice_half_open"]
        xy = arrays["coordinates"][a:b, :, :2].copy()
        xy[..., 0] *= aspect
        valid = arrays["valid_mask"][a:b]
        hip = xy[:, [23, 24]].mean(1)
        torso = xy[:, [11, 12]].mean(1) - hip
        scale = np.linalg.norm(torso, axis=-1)
        anchors = valid[:, ANCHORS].all(1) & np.isfinite(scale) & (scale > 1e-6)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = (xy - hip[:, None]) / scale[:, None, None]
        usable = valid & anchors[:, None]
        times = arrays["pts"][a:b] * arrays["time_base_num"][a:b] / arrays["time_base_den"][a:b]
        pair = (
            (np.diff(arrays["frame_index"][a:b]) == 1)
            & (np.diff(arrays["scene_id"][a:b]) == 0)
            & (np.diff(arrays["segment_id"][a:b]) == 0)
            & (np.diff(times) > 0)
        )
        shoulder = norm[:, 12] - norm[:, 11]
        values = {
            "frames": len(xy),
            "center_span_seconds": float(times[-1] - times[0]),
            "anchor_fraction": float(anchors.mean()),
            "shoulder_width_over_torso": np.linalg.norm(shoulder[anchors], axis=-1),
            "directed_shoulder_x_over_torso": shoulder[anchors, 0],
            "torso_tilt_degrees": np.degrees(np.arctan2(torso[anchors, 0], -torso[anchors, 1])),
            "narrow_shoulder_fraction": float(
                (np.linalg.norm(shoulder[anchors], axis=-1) < 0.1).mean()
            )
            if anchors.any()
            else None,
        }
        for side, joints in (("left", [11, 13, 15]), ("right", [12, 14, 16])):
            s, e, w = joints
            u, v = xy[:, s] - xy[:, e], xy[:, w] - xy[:, e]
            denom = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
            angle_valid = valid[:, joints].all(1) & np.isfinite(denom) & (denom > 1e-12)
            angle = np.degrees(
                np.arccos(
                    np.clip((u[angle_valid] * v[angle_valid]).sum(1) / denom[angle_valid], -1, 1)
                )
            )
            speed_valid = pair & usable[:-1, w] & usable[1:, w]
            speed = np.linalg.norm(np.diff(norm[:, w], axis=0)[speed_valid], axis=-1)
            speed /= np.diff(times)[speed_valid]
            values[f"{side}_elbow_angle_degrees"] = angle
            values[f"{side}_elbow_angle_coverage"] = float(angle_valid.mean())
            values[f"{side}_wrist_speed_torso_per_second"] = speed
            values[f"{side}_wrist_speed_coverage"] = (
                float(speed_valid.mean()) if len(pair) else None
            )
        clip = {"source_row": row["source_row"], "class": row["punch_type_provisional"]}
        for key, value in values.items():
            clip[key] = (
                float(np.median(value))
                if isinstance(value, np.ndarray) and len(value)
                else (None if isinstance(value, np.ndarray) else value)
            )
        clips.append(clip)
    fields = [k for k in clips[0] if k not in ("class", "source_row")] if clips else []
    return {
        "scope": "policy-retained provisional positives only; no model inputs changed",
        "quantiles": [0.1, 0.5, 0.9],
        "aggregation": "quantiles across clip medians or per-clip coverage; "
        "conditional on validity",
        "by_class": {
            label: {
                "count": sum(c["class"] == label for c in clips),
                **{k: quantiles([c[k] for c in clips if c["class"] == label]) for k in fields},
            }
            for label in SOURCE_LABELS
        },
        "clips": clips,
    }


def finish(output):
    """Verify frozen extraction/preparation/policy and write a new readiness report."""
    import numpy as np
    import yaml

    output = Path(output)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a fresh directory under reports/")
    review = yaml.safe_load(REVIEW.read_text())
    evidence_path = Path(review["evidence"])
    if sha256(evidence_path) != review["evidence_sha256"]:
        raise ValueError("Approved review evidence changed")
    evidence = json.loads(evidence_path.read_text())
    if sha256("configs/v6_recording.yaml") != evidence["recording_config_sha256"]:
        raise ValueError("Original recording configuration changed")
    config = load_recording(RECORDING).model_dump()
    original = load_recording("configs/v6_recording.yaml").model_dump()
    expected = {**original, "excluded_source_rows": RECOMMENDED_HOLDS}
    if (
        config != expected
        or review["video_id"] != "V6"
        or review["proposed_additional_quarantine"] != RECOMMENDED_HOLDS
        or review["source_group_id"] != "natebowerfitness-unresolved"
        or review["boxer_id"] is not None
        or review["session_id"] is not None
    ):
        raise ValueError("Changed approved V6 mapping, identities or holds")
    for kind in ("video", "workbook"):
        if sha256(evidence["inputs"][kind + "_path"]) != review[kind + "_sha256"]:
            raise ValueError("Original source changed")
    probe_path = Path("reports/boxingvi/development/V6.ffprobe.json")
    if sha256(probe_path) != evidence["inputs"]["frame_inventory_sha256"]:
        raise ValueError("Source timing inventory changed")
    probe = json.loads(probe_path.read_text())
    manifest = json.loads((POSE_RUN / "manifest.json").read_text())
    prep = json.loads((DATASET / "preparation.json").read_text())
    policy_path = ROOT / "quality/V6/policy-index.json"
    index = json.loads(policy_path.read_text())
    quality_path = ROOT / "quality/V6/quality.json"
    quality = json.loads(quality_path.read_text())
    if (
        manifest["status"] != "complete"
        or manifest["scope"] != "full_video"
        or manifest["recording_config"] != config
        or manifest["config"]["pose_mode"] != "image"
        or manifest["config"]["confidence"] != 0.5
        or manifest["config"]["max_width"] != 960
        or manifest["config"]["ranges_half_open"]
        or manifest["config"]["scene_cuts"]
        or manifest["source_frame_count"] != len(probe["frames"])
        or manifest["processed_frame_count"] != len(probe["frames"])
        or manifest["poses_sha256"] != sha256(POSE_RUN / "poses.jsonl")
        or prep["status"] != "complete"
        or prep["scope"] != "full_video"
        or prep["video_id"] != "V6"
        or prep["extracted_frame_count"] != len(probe["frames"])
        or prep["npz_sha256"] != sha256(DATASET / "poses.npz")
        or prep["config"]["excluded_source_rows"] != RECOMMENDED_HOLDS
        or prep["inputs"]["pose_manifest_sha256"] != sha256(POSE_RUN / "manifest.json")
        or prep["inputs"]["poses_sha256"] != manifest["poses_sha256"]
        or index["video_id"] != "V6"
        or index["policy"] != "mask_aware70"
        or index["definition"] != POLICIES["mask_aware70"]
        or index["source_mask_threshold"] != 0.5
        or index["max_missing_frame_fraction"] != 0.1
    ):
        raise ValueError("Incomplete data or changed extraction/preparation/policy")
    for kind in ("video", "workbook"):
        if manifest["inputs"][kind]["sha256"] != review[kind + "_sha256"]:
            raise ValueError("Extraction source mismatch")
    for name, digest in index["inputs"].items():
        if sha256(DATASET / name) != digest or quality["inputs"][name] != digest:
            raise ValueError("Dataset or quality input changed")
    rows = [json.loads(s) for s in (DATASET / "candidates.jsonl").read_text().splitlines()]
    check_annotation_preservation(rows, evidence["inventory"]["rows"])
    comparison, geometry = {}, {}
    for video in ("V4", "V5", "V6"):
        dataset = (
            DATASET
            if video == "V6"
            else Path(f"data/processed/boxingvi/{video}/candidates-full-v1")
        )
        qpath = (
            quality_path
            if video == "V6"
            else Path(f"reports/boxingvi/expansion/quality/{video}/quality.json")
        )
        q = json.loads(qpath.read_text())
        if q["scope"] != "full_video" or q["policies"]["mask_aware70"] != POLICIES["mask_aware70"]:
            raise ValueError("Comparison requires unchanged full-data policy")
        for name, digest in q["inputs"].items():
            if sha256(dataset / name) != digest:
                raise ValueError("Comparison inputs changed")
        crows = [json.loads(s) for s in (dataset / "candidates.jsonl").read_text().splitlines()]
        cindex = json.loads((qpath.parent / "policy-index.json").read_text())
        expected_selected = [
            c["source_row"] for c in q["candidate_decisions"] if c["retained"]["mask_aware70"]
        ]
        if (
            cindex["inputs"] != q["inputs"]
            or cindex["video_id"] != video
            or cindex["definition"] != POLICIES["mask_aware70"]
            or cindex["source_mask_threshold"] != 0.5
            or cindex["selected_source_rows"] != expected_selected
        ):
            raise ValueError("Comparison selection inputs changed")
        if video == "V6":
            aspect = probe["streams"][0]["width"] / probe["streams"][0]["height"]
        else:
            cmanifest = json.loads((dataset / "preparation.json").read_text())
            candidates_pose_hash = cmanifest["inputs"]["poses_sha256"]
            source_run = Path(f"data/processed/boxingvi/{video}/pose-full-image-v1")
            if sha256(source_run / "poses.jsonl") != candidates_pose_hash:
                raise ValueError("Comparison source poses changed")
            with (source_run / "poses.jsonl").open() as stream:
                first = json.loads(next(stream))
            aspect = first["source_width"] / first["source_height"]
        with np.load(dataset / "poses.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
            whole = whole_video_quality(arrays)
            if video == "V6":
                check_arrays(arrays, probe["frames"])
                selected = [
                    r["source_row"]
                    for r in rows
                    if r["annotation_eligible"]
                    and retain(
                        arrays["valid_mask"][slice(*r["array_slice_half_open"])],
                        arrays["pose_status"][slice(*r["array_slice_half_open"])] == 2,
                        POLICIES["mask_aware70"],
                    )
                ]
                if selected != index["selected_source_rows"]:
                    raise ValueError("Policy retention differs on recomputation")
                statuses = dict(Counter(int(s) for s in arrays["pose_status"]))
            geometry[video] = geometry_profile(
                arrays, crows, cindex["selected_source_rows"], aspect
            )
        comparison[video] = {
            "quality_path": str(qpath),
            "sha256": sha256(qpath),
            "by_class": q["by_class"],
            "whole_video": whole,
        }
    counts = selection_counts(rows, selected)
    report = {
        "schema_version": "v6-preparation-readiness-1.0",
        "provisional_preparation_complete": True,
        "training_performed": False,
        "representation_changed": False,
        "training_ready": "requires separately reviewed experiment; labels remain provisional",
        "source_group_id": "natebowerfitness-unresolved",
        "related_recordings_guard": ["V1", "V6"],
        "boxer_id": None,
        "session_id": None,
        "review_sha256": sha256(REVIEW),
        "review_evidence_sha256": sha256(evidence_path),
        "inputs": index["inputs"],
        "pose_run_manifest_sha256": sha256(POSE_RUN / "manifest.json"),
        "policy_index_sha256": sha256(policy_path),
        "source_inputs": manifest["inputs"],
        "exact_timing_verified_frames": len(probe["frames"]),
        "class_counts": counts,
        "pose_status_counts": statuses,
        "exclusion_reason_counts_nonexclusive": dict(
            Counter(reason for r in rows for reason in r["exclusion_reasons"])
        ),
        "excluded_annotations": [r for r in rows if not r["annotation_eligible"]],
        "pose_rejected_rows": [
            r["source_row"]
            for r in rows
            if r["annotation_eligible"] and r["source_row"] not in selected
        ],
        "quality_candidate_decisions": quality["candidate_decisions"],
        "pose_rejection_details": [
            {**d, "failed_requirements": policy_failure_reasons(d["metrics"])}
            for d in quality["candidate_decisions"]
            if not d["retained"]["mask_aware70"]
        ],
        "unknown_outside_annotation_bounds_frames": prep[
            "unknown_outside_annotation_bounds_frames"
        ],
        "mapping": review["mapping"],
        "uncertainties": review["uncertainties"],
        "comparison": comparison,
        "geometry": geometry,
        "diagnostic_code_sha256": sha256(__file__),
        "feature_code_sha256": sha256(Path(__file__).parents[1] / "training/data.py"),
    }
    output.mkdir(parents=True)
    write_json(output / "readiness.json", report)
    lines = [
        "# V6 provisional preparation",
        "",
        "No training or model-feature change.",
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
        "All labels/boundaries provisional; unlabeled footage stays unknown.",
        "See readiness.json for all exclusions, exact-PTS checks and valid-only geometry.",
    ]
    (output / "readiness.md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    finish(parser.parse_args().output)
