"""Dataset-wide evidence inventory. Pose retention never certifies annotation alignment."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from ..vision.pipeline import code_provenance, sha256, write_json
from .pose_quality import POLICIES, RECOMMENDED
from .preparation import CLASS_MAP


def summarize_datasets(datasets, quality_reports, alignment, output, review=None):
    output = Path(output)
    paths = [Path(p) for p in datasets]
    if (
        output.exists()
        or any(output.resolve().is_relative_to(p.resolve()) for p in paths)
        or any(p.name == "raw" for p in output.resolve().parents)
    ):
        raise ValueError("Use a new summary directory outside raw/prepared datasets")
    evidence = json.loads(Path(alignment).read_text())["recordings"]
    reviews = json.loads(Path(review).read_text()) if review else {}
    qualities = {}
    for path in quality_reports:
        path = Path(path)
        q = json.loads(path.read_text())
        video_id = q["video_id"]
        if video_id in qualities:
            raise ValueError("Duplicate recording quality report")
        qualities[video_id] = (q, path)
    records, families, media_hashes = {}, defaultdict(list), defaultdict(list)
    for dataset in paths:
        prep = json.loads((dataset / "preparation.json").read_text())
        video_id = prep["video_id"]
        if video_id in records:
            raise ValueError(
                "Duplicate recording dataset; distinguish full versus sample explicitly"
            )
        if video_id not in evidence or video_id not in qualities:
            raise ValueError(f"Missing alignment/quality evidence for {video_id}")
        q, qpath = qualities[video_id]
        hashes = {
            name: sha256(dataset / name)
            for name in ["preparation.json", "poses.npz", "candidates.jsonl"]
        }
        if (
            prep["status"] != "complete"
            or prep["npz_sha256"] != hashes["poses.npz"]
            or hashes != q["inputs"]
        ):
            raise ValueError(
                f"Changed/incomplete prepared inputs or stale quality report: {video_id}"
            )
        source = evidence[video_id]
        if (
            prep["inputs"]["video_sha256"] != source["inputs"]["video_sha256"]
            or prep["inputs"]["workbook_sha256"] != source["inputs"]["workbook_sha256"]
        ):
            raise ValueError("Alignment and preparation refer to different original inputs")
        if q["policies"] != POLICIES or q["source_mask_threshold"] != 0.5:
            raise ValueError("Dataset-wide comparison requires the unchanged approved policies")
        counts = {}
        for label in CLASS_MAP.values():
            p = prep["class_counts"][label]
            v = q["by_class"][label]
            counts[label] = {
                "source_rows": p["source_rows"],
                "annotation_eligible": p["annotation_eligible"],
                "pose_assessed": v["eligible"],
                "mask_retained_assessed": v["retention"][RECOMMENDED],
                "strict_body_retained_assessed": v["retention"]["strict_body"],
                "assessed_with_unreliable_pose": p["eligible_with_unreliable_pose"],
                "assessed_with_missing_pose": p["eligible_with_missing_pose"],
                "left_elbow_invalid_fraction": v["joints"]["left_elbow"]["invalid_frame_fraction"],
                "right_elbow_invalid_fraction": v["joints"]["right_elbow"][
                    "invalid_frame_fraction"
                ],
            }
        row_review = reviews.get(
            video_id, {"decision": "unreviewed", "reason": "No visual alignment review supplied"}
        )
        if row_review["decision"] not in {"provisional", "investigate", "unreviewed"}:
            raise ValueError("Review decision must remain provisional/investigate/unreviewed")
        full = prep.get("scope", "full_video") == "full_video"
        class_missing = [k for k, v in counts.items() if not v["mask_retained_assessed"]]
        status = (
            "provisional_development_pool"
            if full and not class_missing and row_review["decision"] == "provisional"
            else "requires_alignment_investigation"
            if row_review["decision"] != "provisional"
            else "sample_only_requires_full_extraction_or_class_review"
        )
        channel = source.get("channel_id")
        families[channel or f"unknown-source:{video_id}"].append(video_id)
        media_hashes[source["inputs"]["video_sha256"]].append(video_id)
        records[video_id] = {
            "recording_id": video_id,
            "boxer_id": prep.get("boxer_id"),
            "session_id": prep.get("session_id"),
            "source_channel_id": channel,
            "source_channel_name": source.get("channel_name"),
            "identity_warning": "Publisher is a conservative grouping proxy, not boxer identity",
            "scope": "full_video" if full else "selected_windows",
            "source_frame_count": source["video"]["frame_count"],
            "extracted_frames": prep.get("extracted_frame_count", prep["source_frame_count"]),
            "class_counts": counts,
            "unmapped_source_rows": prep["class_counts"]["unmapped"]["source_rows"],
            "classes_absent_in_source": [k for k, v in counts.items() if not v["source_rows"]],
            "classes_without_retained_assessed_candidates": class_missing,
            "annotation_flag_counts": dict(
                Counter(f for r in source["issues"] for f in r["flags"])
            ),
            "strict_overlap_pairs": len(source.get("strict_overlap_pairs", [])),
            "shared_endpoint_pairs": len(source.get("shared_endpoint_pairs", [])),
            "pose_rejected_rows": [
                c["source_row"] for c in q["candidate_decisions"] if not c["retained"][RECOMMENDED]
            ],
            "visual_review": row_review,
            "readiness": status,
            "dataset": dataset.as_posix(),
            "quality_report": qpath.as_posix(),
            "input_hashes": hashes,
            "background": "unknown; no verified no_punch examples established",
            "alignment": "provisional source coordinates; no certified author FPS/base/end/trim",
        }
    totals = {
        key: sum(sum(c[key] for c in r["class_counts"].values()) for r in records.values())
        for key in ["source_rows", "annotation_eligible", "pose_assessed", "mask_retained_assessed"]
    }
    report = {
        "schema_version": "dataset-readiness-1.0",
        "recordings": records,
        "totals": totals,
        "source_annotation_rows_including_unmapped": totals["source_rows"]
        + sum(r["unmapped_source_rows"] for r in records.values()),
        "provisional_development_recordings": [
            vid for vid, r in records.items() if r["readiness"] == "provisional_development_pool"
        ],
        "provisional_development_mask_retained": sum(
            sum(c["mask_retained_assessed"] for c in r["class_counts"].values())
            for r in records.values()
            if r["readiness"] == "provisional_development_pool"
        ),
        "diagnostic_sample_mask_retained": sum(
            sum(c["mask_retained_assessed"] for c in r["class_counts"].values())
            for r in records.values()
            if r["scope"] != "full_video"
        ),
        "source_family_guard_groups": dict(families),
        "exact_duplicate_media_groups": [v for v in media_hashes.values() if len(v) > 1],
        "identity_status": (
            "Unknown boxer/session identities remain null; no boxer-independent split certified"
        ),
        "split_assignment": None,
        "count_limit": (
            "Full-run counts and sampled counts are explicitly separate. "
            "Sample retention is not extrapolated to the source recording."
        ),
        "no_punch": "No reviewed negative examples supplied; unlabeled footage remains unknown",
        "licensing_and_redistribution": "unverified",
        "model_training_performed": False,
        "policy": RECOMMENDED,
        "policies": POLICIES,
        "inputs": {
            "alignment_sha256": sha256(alignment),
            "review_sha256": sha256(review) if review else None,
        },
        "code": code_provenance(),
        "summary_code_sha256": sha256(__file__),
    }
    output.mkdir(parents=True)
    write_json(output / "dataset.json", report)
    lines = [
        "# Dataset readiness — provisional labels, no training",
        "",
        report["count_limit"],
        "",
        "| Recording | Scope | Extracted/source frames | Source events | "
        "Eligible source events | Assessed | Mask retained | Readiness |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for video_id, r in records.items():
        c = r["class_counts"]
        total = {
            key: sum(v[key] for v in c.values())
            for key in [
                "source_rows",
                "annotation_eligible",
                "pose_assessed",
                "mask_retained_assessed",
            ]
        }
        lines.append(
            f"| {video_id} | {r['scope']} | {r['extracted_frames']}/{r['source_frame_count']} | "
            f"{total['source_rows'] + r['unmapped_source_rows']} | "
            f"{total['annotation_eligible']} | "
            f"{total['pose_assessed']} | {total['mask_retained_assessed']} | {r['readiness']} |"
        )
    lines += [
        "",
        "Provisional full-video development pool: "
        f"{report['provisional_development_mask_retained']} "
        "positive candidates. Diagnostic sample counts are not automatically training data.",
        "",
        "## Class support",
        "",
        "Each cell is source rows / assessed / mask-retained. Missing classes stay explicit.",
        "",
        "| Recording | " + " | ".join(CLASS_MAP.values()) + " |",
        "| --- | " + " | ".join("---:" for _ in CLASS_MAP) + " |",
    ]
    for video_id, r in records.items():
        lines.append(
            f"| {video_id} | "
            + " | ".join(
                f"{v['source_rows']}/{v['pose_assessed']}/{v['mask_retained_assessed']}"
                for v in r["class_counts"].values()
            )
            + " |"
        )
    lines += ["", "## Recording review", ""]
    for video_id, r in records.items():
        lines += [
            f"### {video_id}",
            "",
            r["visual_review"]["reason"],
            "",
            f"Strict overlap pairs: {r['strict_overlap_pairs']}; "
            f"shared endpoints: {r['shared_endpoint_pairs']}. "
            "Shared endpoints are a convention question, not automatically annotation errors.",
            "",
            f"Boxer: {r['boxer_id']}; session: {r['session_id']}; "
            f"publisher: {r['source_channel_name']}.",
            "",
        ]
    lines += [
        "## Split constraints",
        "",
        report["identity_status"],
        "",
        "Keep whole recordings together. Conservatively keep shared publisher groups together; "
        "this reduces source/session leakage risk without assigning identities. Near-duplicate "
        "and boxer/session review is still required. Fit all preprocessing and class weights "
        "on training data only. Never randomly split frames or overlapping clips.",
        "",
        "No final split is assigned. No untouched six-class test benchmark or verified "
        "background set has been established. See the readiness note for the proposed "
        "baseline path.",
        "",
    ]
    (output / "dataset.md").write_text("\n".join(lines))
    return report
