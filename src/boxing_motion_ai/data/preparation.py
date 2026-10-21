"""Provisional V5 positive candidates referencing a single immutable pose sequence."""

import json
from bisect import bisect_right
from collections import Counter
from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..vision.annotations import read_annotations
from ..vision.contracts import BODY_JOINTS, JOINTS
from ..vision.pipeline import code_provenance, sha256, write_json
from ..vision.recording import v5_recording

# Literal source names only; no stance/hand inference or numeric class guessing.
CLASS_MAP = {
    "Jab": "jab",
    "Cross": "cross",
    "Lead Hook": "lead_hook",
    "Rear Hook": "rear_hook",
    "Lead Uppercut": "lead_uppercut",
    "Rear Uppercut": "rear_uppercut",
}
SOURCE_LABELS = tuple(CLASS_MAP.values())
GLOBAL_UNCERTAINTIES = {"provisional_mapping", "class_unverified", "boundaries_unverified"}


class PreparationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    excluded_source_rows: list[int] = Field(default_factory=lambda: [272, 323, 442, 536])
    scene_guard_frames: int = Field(default=2, ge=0)


def _load_poses(pose_run, manifest):
    """One JSONL pass, bounded arrays. Null becomes NaN, never an invented coordinate."""
    import numpy as np

    full = manifest.get("scope") == "full_video"
    n = manifest["processed_frame_count"]
    if type(n) is not int or n <= 0 or (full and n != manifest["source_frame_count"]):
        raise ValueError("Inconsistent complete source frame count")
    video_id = manifest.get("video_id", "V5")
    expected_indices = (
        list(range(n))
        if full
        else [i for a, b in manifest["config"]["ranges_half_open"] for i in range(a, b)]
    )
    if len(expected_indices) != n or expected_indices != sorted(set(expected_indices)):
        raise ValueError("Pose ranges disagree with processed frame count")
    config = manifest["config"]
    if config.get("pose_mode") != "image":
        raise ValueError("Preparation requires independent IMAGE poses to prevent cross-cut state")
    cuts = config.get("scene_cuts", [])
    if cuts != sorted(set(cuts)) or any(
        type(c) is not int or not 0 < c or (full and c >= n) for c in cuts
    ):
        raise ValueError("Invalid scene cut inventory")
    xyz = np.full((n, 33, 3), np.nan, dtype=np.float64)
    confidence = np.full((n, 33, 2), np.nan, dtype=np.float64)
    mask = np.zeros((n, 33), dtype=bool)
    pts = np.empty(n, dtype=np.int64)
    num = np.empty(n, dtype=np.int64)
    den = np.empty(n, dtype=np.int64)
    scene = np.empty(n, dtype=np.int64)
    segment = np.empty(n, dtype=np.int64)
    status = np.empty(n, dtype=np.uint8)
    previous_time, previous_segment, count = None, None, 0
    with (pose_run / "poses.jsonl").open() as stream:
        for i, line in enumerate(stream):
            record = json.loads(line)
            if (
                i >= n
                or type(record["frame_index"]) is not int
                or record["frame_index"] != expected_indices[i]
            ):
                raise ValueError(
                    "Pose records must contain every decoded frame exactly once in order"
                )
            if record["schema_version"] != "pose-1.1" or record["video_id"] != video_id:
                raise ValueError("Expected matching recording pose-1.1 records")
            if "recording_config" in manifest and (
                record.get("recording_id") != video_id
                or record.get("boxer_id") != manifest.get("boxer_id")
                or record.get("session_id") != manifest.get("session_id")
            ):
                raise ValueError("Pose recording/boxer/session identity disagrees with manifest")
            base = Fraction(record["time_base"])
            if type(record["pts"]) is not int or base <= 0:
                raise ValueError("Invalid exact pose timing")
            time = record["pts"] * base
            if previous_time is not None and time <= previous_time:
                raise ValueError("Non-increasing pose timestamps")
            previous_time = time
            source_index = expected_indices[i]
            gap = i > 0 and source_index != expected_indices[i - 1] + 1
            sid = bisect_right(cuts, source_index)
            if record["scene_id"] != sid or record["scene_cut_before"] != (source_index in cuts):
                raise ValueError("Pose scene metadata disagrees with cut configuration")
            if (source_index in cuts or gap) and record["segment_id"] == previous_segment:
                raise ValueError("Pose estimator segment did not reset at the scene cut")
            if (
                i
                and source_index not in cuts
                and not gap
                and record["segment_id"] != previous_segment
            ):
                raise ValueError("Unexpected full-video estimator segment change")
            previous_segment = record["segment_id"]
            points, valid = record["landmarks"], record["valid_mask"]
            if len(points) != 33 or len(valid) != 33 or any(type(v) is not bool for v in valid):
                raise ValueError("Expected 33 points and boolean validity entries")
            for j, point in enumerate(points):
                if point is not None:
                    xyz[i, j] = [
                        point[k] if point[k] is not None else np.nan for k in ("x", "y", "z")
                    ]
                    confidence[i, j] = [
                        point[k] if point[k] is not None else np.nan
                        for k in ("visibility", "presence")
                    ]
            finite = np.isfinite(xyz[i]).all(axis=1) & np.isfinite(confidence[i]).all(axis=1)
            expected_valid = (
                finite
                & (xyz[i, :, :2] >= 0).all(axis=1)
                & (xyz[i, :, :2] <= 1).all(axis=1)
                & (confidence[i] >= config["confidence"]).all(axis=1)
                & (confidence[i] <= 1).all(axis=1)
            )
            if not np.array_equal(expected_valid, valid):
                raise ValueError(f"Inconsistent validity mask at frame {i}")
            mask[i] = valid
            body_count = int(mask[i, list(BODY_JOINTS)].sum())
            name = record["status"]
            if name == "missing":
                if any(p is not None for p in points) or any(valid):
                    raise ValueError("Missing poses must contain null points and false masks")
            elif name != ("ok" if body_count == 22 else "unreliable"):
                raise ValueError("Inconsistent pose quality status")
            if record["valid_body_joint_count"] != body_count:
                raise ValueError("Inconsistent valid body joint count")
            status[i] = {"ok": 0, "unreliable": 1, "missing": 2}[name]
            pts[i], num[i], den[i] = record["pts"], base.numerator, base.denominator
            scene[i], segment[i] = sid, record["segment_id"]
            count += 1
    if count != n:
        raise ValueError("Truncated pose sequence")
    return {
        "frame_index": np.array(expected_indices, dtype=np.int64),
        "pts": pts,
        "time_base_num": num,
        "time_base_den": den,
        "coordinates": xyz,
        "confidence": confidence,
        "valid_mask": mask,
        "scene_id": scene,
        "segment_id": segment,
        "pose_status": status,
    }


def _unknown_intervals(intervals, n):
    """Complement of plausible source intervals; not a reviewed negative label."""
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    result, cursor = [], 0
    for a, b in merged + [[n, n]]:
        if cursor < a:
            result.append({"frame_start": cursor, "frame_stop_exclusive": a, "state": "unknown"})
        cursor = max(cursor, b)
    return result


def _candidates(annotations, arrays, config, source_frame_count=None, scene_cuts=None):
    import numpy as np

    frame_ids = arrays["frame_index"]
    n = source_frame_count
    prefix_stop = int(frame_ids[-1]) + 1
    recording = annotations.get("recording_config", v5_recording().model_dump())
    video_id = annotations["video_id"]
    class_map = recording["class_map"]
    # Derive confirmed boundaries from scene IDs, not nominal FPS or video timestamps.
    boundaries = [
        int(frame_ids[i])
        for i in range(1, len(frame_ids))
        if arrays["scene_id"][i] != arrays["scene_id"][i - 1]
    ]
    if scene_cuts is not None:
        boundaries = scene_cuts
    candidates, plausible = [], []
    for row in annotations["rows"]:
        a, end = row["candidate_start_index"], row["candidate_end_index"]
        reasons = sorted(set(row["flags"]) - GLOBAL_UNCERTAINTIES)
        if row["source_row"] in config.excluded_source_rows:
            reasons.append("known_questionable_source_row")
        label = class_map.get(row["source_class"])
        if label is None:
            reasons.append("no_exact_class_mapping")
        stop = end + int(recording["end_assumption"] == "inclusive") if end is not None else None
        bounds_ok = (
            a is not None and stop is not None and 0 <= a < stop and (n is None or stop <= n)
        )
        if not bounds_ok:
            reasons.append("invalid_or_out_of_bounds_interval")
        quality = None
        array_slice = None
        sampled = False
        if bounds_ok:
            if a < prefix_stop:
                plausible.append((a, min(stop, prefix_stop)))
            if any(a < cut < stop for cut in boundaries):
                reasons.append("crosses_confirmed_scene_cut")
            guard = config.scene_guard_frames
            if any(a < cut + guard and stop > cut - guard for cut in boundaries):
                reasons.append("near_confirmed_scene_cut")
            lo, hi = np.searchsorted(frame_ids, [a, stop]).tolist()
            sampled = np.array_equal(frame_ids[lo:hi], np.arange(a, stop))
            array_slice = [lo, hi] if sampled else None
            values = arrays["pose_status"][lo:hi] if sampled else []
            counts = Counter(int(v) for v in values)
            quality = {
                "frame_count": stop - a,
                "ok_frames": counts[0],
                "unreliable_frames": counts[1],
                "missing_frames": counts[2],
                "invalid_body_joint_entries": int((~arrays["valid_mask"][lo:hi, 11:]).sum()),
            }
            if not sampled:
                quality = None
        eligible = not reasons
        clean = (
            eligible
            and sampled
            and quality["unreliable_frames"] == 0
            and quality["missing_frames"] == 0
        )
        candidates.append(
            {
                **row,
                "schema_version": "recording-candidate-1.1",
                "video_id": video_id,
                "recording_id": video_id,
                "boxer_id": recording["boxer_id"],
                "session_id": recording["session_id"],
                "punch_type_provisional": label,
                "review_status": "unreviewed",
                "split_group": video_id,
                "split": None,
                "source_interval_half_open": [a, stop] if bounds_ok else None,
                "array_slice_half_open": array_slice,
                "quality_assessable": eligible and sampled,
                "pose_coverage": "complete" if sampled else "not_fully_extracted",
                "annotation_end_assumption": recording["end_assumption"]
                + "; author convention unverified",
                "exclusion_reasons": sorted(set(reasons)),
                "annotation_eligible": eligible,
                "pose_clean_candidate": clean,
                "pose_quality": quality,
            }
        )
    return candidates, _unknown_intervals(plausible, prefix_stop)


def prepare_recording(pose_run, workbook, output, config=None, allow_sample=False):
    import numpy as np

    pose_run, workbook, output = map(Path, (pose_run, workbook, output))
    if output.exists():
        raise ValueError("Output already exists; choose a new preparation directory")
    if (
        output.resolve().is_relative_to(pose_run.resolve())
        or any(p.name == "raw" for p in output.resolve().parents)
        or output.resolve().is_relative_to(workbook.resolve().parent)
    ):
        raise ValueError(
            "Preparation output must be outside pose inputs and raw dataset directories"
        )
    manifest = json.loads((pose_run / "manifest.json").read_text())
    if manifest["status"] != "complete" or (
        not allow_sample and manifest.get("scope") != "full_video"
    ):
        raise ValueError("Preparation requires a complete full-video pose run")
    if manifest["inputs"]["workbook"]["sha256"] != sha256(workbook):
        raise ValueError(
            "Workbook hash differs from the pose run; no implicit annotation update allowed"
        )
    from ..vision.recording import RecordingConfig

    recording = (
        RecordingConfig.model_validate(manifest["recording_config"])
        if "recording_config" in manifest
        else v5_recording()
    )
    video_id = recording.video_id
    if manifest.get("video_id", "V5") != video_id:
        raise ValueError("Recording configuration and manifest video IDs disagree")
    config = config or PreparationConfig(excluded_source_rows=recording.excluded_source_rows)
    annotations = read_annotations(workbook, recording)
    arrays = _load_poses(pose_run, manifest)
    source_count = manifest.get("source_frame_count")
    candidates, unknown = _candidates(
        annotations, arrays, config, source_count, manifest["config"]["scene_cuts"]
    )
    balance = {}
    for label in [*SOURCE_LABELS, "unmapped"]:
        subset = [c for c in candidates if (c["punch_type_provisional"] or "unmapped") == label]
        eligible = [c for c in subset if c["annotation_eligible"]]
        assessed = [c for c in eligible if c["quality_assessable"]]
        invalid_counts = np.zeros(33, dtype=np.int64)
        event_frames = 0
        arms_clean = 0
        for candidate in assessed:
            a, stop = candidate["array_slice_half_open"]
            valid = arrays["valid_mask"][a:stop]
            invalid_counts += (~valid).sum(axis=0)
            event_frames += stop - a
            arms_clean += bool(valid[:, 11:17].all())
        worst = sorted(range(11, 33), key=lambda i: -int(invalid_counts[i]))[:5]
        balance[label] = {
            "eligible_all_arm_joints_valid": arms_clean,
            "worst_body_joints": [
                {"joint": JOINTS[i], "invalid_fraction": float(invalid_counts[i]) / event_frames}
                for i in worst
                if event_frames
            ],
            "quality_assessable": len(assessed),
            "source_rows": len(subset),
            "annotation_eligible": len(eligible),
            "excluded_annotation_rows": len(subset) - len(eligible),
            "eligible_with_unreliable_pose": sum(
                c["pose_quality"]["unreliable_frames"] > 0 for c in assessed
            ),
            "eligible_with_missing_pose": sum(
                c["pose_quality"]["missing_frames"] > 0 for c in assessed
            ),
            "eligible_with_any_pose_issue": sum(
                c["pose_quality"]["unreliable_frames"] > 0
                or c["pose_quality"]["missing_frames"] > 0
                for c in assessed
            ),
            "pose_clean_candidates": sum(c["pose_clean_candidate"] for c in eligible),
        }
    no_clean = [
        label for label in CLASS_MAP.values() if balance[label]["pose_clean_candidates"] == 0
    ]
    output.mkdir(parents=True)
    report = {
        "schema_version": "recording-preparation-1.1",
        "status": "writing",
        "video_id": video_id,
        "recording_id": video_id,
        "boxer_id": recording.boxer_id,
        "session_id": recording.session_id,
        "scope": manifest.get("scope"),
        "config": config.model_dump(),
        "class_map": recording.class_map,
        "class_counts": balance,
        "inputs": {
            "pose_manifest_sha256": sha256(pose_run / "manifest.json"),
            "poses_sha256": sha256(pose_run / "poses.jsonl"),
            "workbook_sha256": sha256(workbook),
            "video_sha256": manifest["inputs"]["video"]["sha256"],
        },
        "code": code_provenance(),
        "source_frame_count": manifest.get("source_frame_count"),
        "extracted_frame_count": len(arrays["pts"]),
        "unknown_inventory_scope": (
            "full_video" if manifest.get("scope") == "full_video" else "decoded prefix only"
        ),
        "pose_contract": {
            "joint_order": JOINTS,
            "coordinates": "raw image x/y and hip-relative z",
            "confidence_order": ["visibility", "presence"],
            "missing": "NaN coordinates/confidences and false masks; never impute",
            "status_codes": {"ok": 0, "unreliable": 1, "missing": 2},
        },
        "annotation_row_count": len(candidates),
        "source_class_counts": dict(Counter(r["source_class"] for r in annotations["rows"])),
        "excluded_rows_by_reason": dict(
            Counter(reason for c in candidates for reason in c["exclusion_reasons"])
        ),
        "confirmed_scene_cuts": manifest["config"]["scene_cuts"],
        "unknown_outside_annotation_bounds_frames": sum(
            u["frame_stop_exclusive"] - u["frame_start"] for u in unknown
        ),
        "mapping_uncertainty": {k: v for k, v in annotations.items() if k != "rows"},
        "sequence_policy": (
            f"raw bounds + {recording.frame_offset}; provisional "
            f"{recording.end_assumption} end; no resampling or imputation"
        ),
        "pose_clean_definition": "annotation eligible; all body joints valid on every event frame",
        "uncertainty": "clean is structural/pose quality only, never verified annotation truth",
        "scene_inventory": "confirmed listed cuts only; IMAGE inference has no cross-frame state",
        "split_status": (
            f"recording group {video_id}; boxer/session identities separate; splits unassigned"
        ),
        "classes_without_pose_clean_candidates": no_clean,
        "training_readiness": (
            "needs_pose_quality_review_and_independent_split"
            if no_clean
            else "provisional_development_only; independent split unresolved"
        ),
        "licensing_and_redistribution": "unverified",
    }
    report["code"]["sha256"][
        Path(__file__).relative_to(Path(__file__).resolve().parents[3]).as_posix()
    ] = sha256(__file__)
    write_json(output / "preparation.json", report)
    try:
        np.savez_compressed(output / "poses.npz", **arrays)
        with (output / "candidates.jsonl").open("w") as stream:
            for candidate in candidates:
                stream.write(json.dumps(candidate, allow_nan=False) + "\n")
        write_json(
            output / "unknown-intervals.json",
            {
                "state": "unknown",
                "intervals": unknown,
                "note": "Outside plausible source bounds; never reviewed negatives",
            },
        )
        report["status"] = "complete"
        report["npz_sha256"] = sha256(output / "poses.npz")
        write_json(output / "preparation.json", report)
        _write_markdown(report, output)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "preparation.json", report)
        raise
    return report


def _write_markdown(report, output):
    lines = [
        f"# {report['video_id']} provisional candidate dataset",
        "",
        "No training was performed. All labels and boundaries remain provisional.",
        "",
        "| Class | Source rows | Annotation eligible | Assessed | With unreliable pose "
        "| With missing pose | Pose-clean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, count in report["class_counts"].items():
        lines.append(
            f"| {label} | {count['source_rows']} | {count['annotation_eligible']} | "
            f"{count['quality_assessable']} | "
            f"{count['eligible_with_unreliable_pose']} | {count['eligible_with_missing_pose']} | "
            f"{count['pose_clean_candidates']} |"
        )
    lines += [
        "",
        "Pose issue counts apply only to assessed eligible rows and may overlap. Clean means all",
        "22 body joints valid for every event frame, not certified anatomical/label accuracy.",
        "",
        f"Unknown outside source bounds ({report['unknown_inventory_scope']}): "
        f"{report['unknown_outside_annotation_bounds_frames']}; "
        "state remains unknown.",
        "",
        "Classes without strict pose-clean candidates: "
        + ", ".join(report["classes_without_pose_clean_candidates"]),
        "No class is automatically dropped and no confidence threshold is relaxed.",
        "",
        f"All examples belong to {report['video_id']}; splits are unassigned. "
        "No independent held-out evaluation",
        "is possible from this single source group. See preparation.json for assumptions, source",
        "hashes and exact policies; candidates.jsonl retains every excluded row and its reasons.",
        "",
        "poses.npz contains one shared sequence. Use array_slice_half_open to select a candidate.",
        "Null coordinates are NaN under false masks; no imputation, padding or smoothing is added.",
        "Keep invalid raw coordinates masked in any later feature/model code.",
        "",
    ]
    (output / "preparation.md").write_text("\n".join(lines))


def prepare_v5(pose_run, workbook, output, config=None):
    """Compatibility entry point, still requires full extraction."""
    if Path(workbook).stem.upper() != "V5":
        raise ValueError("prepare-v5 requires V5; use prepare-recording for other recordings")
    return prepare_recording(pose_run, workbook, output, config)
