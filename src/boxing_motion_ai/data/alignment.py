"""Source-index/timestamp hypothesis diagnostics, never an automatic annotation correction."""

import json
from bisect import bisect_left
from collections import Counter
from fractions import Fraction
from pathlib import Path

from ..vision.annotations import read_annotations
from ..vision.pipeline import sha256, write_json
from ..vision.recording import load_recording


def nearest_index(times, target):
    right = bisect_left(times, target)
    return min(
        {max(0, right - 1), min(len(times) - 1, right)}, key=lambda n: (abs(times[n] - target), n)
    )


def diagnose_alignment(root, evidence, output, recordings, plans=None):
    root, evidence, output = map(Path, (root, evidence, output))
    if output.exists() or output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Use a new diagnostic directory outside raw inputs")
    registry = json.loads((evidence / "source-manifest.json").read_text())
    original = {r["id"]: r for r in registry["recordings"]}
    reports = {}
    for path in recordings:
        config = load_recording(Path(path))
        video_id = config.video_id
        source = original[video_id]
        workbook = root / source["workbook"]
        video = root / source["media_path"]
        if sha256(video) != source["media_sha256"] or sha256(workbook) != source["workbook_sha256"]:
            raise ValueError(f"{video_id}: cached frame evidence belongs to different inputs")
        annotations = read_annotations(workbook, config)
        probe_path = evidence / f"{video_id}.ffprobe.json"
        probe = json.loads(probe_path.read_text())
        times = [Fraction(f["pts_time"]) for f in probe["frames"]]
        if not times or any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError(f"{video_id}: non-monotonic or absent frame evidence")
        stream = probe["streams"][0]
        deltas = Counter(str(b - a) for a, b in zip(times, times[1:], strict=False))
        if plans is not None and (Path(plans) / video_id / "sample-plan.json").exists():
            selected = json.loads((Path(plans) / video_id / "sample-plan.json").read_text())[
                "selected_rows"
            ]
        else:
            rows = annotations["rows"]
            selected = [rows[i] for i in sorted({0, len(rows) // 2, len(rows) - 1})]
        anchors = []
        for row in selected:
            a, b = row["candidate_start_index"], row["candidate_end_index"]
            if a is None or b is None:
                continue
            hypotheses = {}
            for name, index in [("raw_zero_base", a), ("raw_one_base", a - 1)]:
                hypotheses[name] = {
                    "decoded_index": index,
                    "pts_seconds": float(times[index]) if 0 <= index < len(times) else None,
                    "in_bounds": 0 <= index < len(times),
                }
            for rate in [
                Fraction(24),
                Fraction(24000, 1001),
                Fraction(25),
                Fraction(30),
                Fraction(30000, 1001),
            ]:
                target = Fraction(int(row["raw_start"]), 1) / rate
                index = nearest_index(times, target)
                hypotheses[f"time_at_{rate}_fps"] = {
                    "target_seconds": float(target),
                    "nearest_decoded_index": index,
                    "nearest_pts_seconds": float(times[index]),
                    "outside_source_time": not times[0] <= target <= times[-1],
                }
            anchors.append(
                {
                    "source_row": row["source_row"],
                    "source_class": row["source_class"],
                    "raw_bounds": [row["raw_start"], row["raw_end"]],
                    "flags": row["flags"],
                    "hypotheses": hypotheses,
                }
            )
        issues = []
        for row in annotations["rows"]:
            a, b = row["candidate_start_index"], row["candidate_end_index"]
            flags = list(row["flags"][3:])
            duration = None
            if a is not None and b is not None:
                if not 0 <= a <= b < len(times):
                    flags.append("invalid_or_out_of_video_bounds")
                else:
                    duration = float(times[b] - times[a])
                    if duration > 2:
                        flags.append("interval_over_two_seconds_review")
            if flags:
                issues.append(
                    {
                        "source_row": row["source_row"],
                        "flags": flags,
                        "raw_bounds": [row["raw_start"], row["raw_end"]],
                        "span_seconds_end_minus_start": duration,
                    }
                )
        row_lookup = {r["source_row"]: r for r in annotations["rows"]}
        pairs = {
            tuple(sorted((r["source_row"], n)))
            for r in annotations["rows"]
            for n in r.get("overlapping_source_rows", [])
        }
        touching = []
        strict = []
        for left, right in sorted(pairs):
            a, b = row_lookup[left], row_lookup[right]
            span = min(a["raw_end"], b["raw_end"]) - max(a["raw_start"], b["raw_start"])
            (touching if span == 0 else strict).append([left, right])
        info_path = root / "RGB_videos" / f"{video_id}.info.json"
        info = json.loads(info_path.read_text())
        if info["id"] != source["youtube_id"]:
            raise ValueError("Source download metadata ID mismatch")
        reports[video_id] = {
            "recording_id": video_id,
            "boxer_id": config.boxer_id,
            "session_id": config.session_id,
            "source_url": source["original_url"],
            "source_title": info.get("title"),
            "channel_id": info.get("channel_id"),
            "channel_name": info.get("channel"),
            "identity_note": "Channel/title are provenance, not verified boxer/session identity",
            "inputs": {
                "video_sha256": source["media_sha256"],
                "workbook_sha256": source["workbook_sha256"],
                "ffprobe_sha256": sha256(probe_path),
                "info_sha256": sha256(info_path),
            },
            "video": {
                "frame_count": len(times),
                "duration_seconds": probe["format"]["duration"],
                "width": stream["width"],
                "height": stream["height"],
                "avg_frame_rate": stream["avg_frame_rate"],
                "nominal_frame_rate": stream["r_frame_rate"],
                "time_base": stream["time_base"],
                "first_pts_seconds": float(times[0]),
                "last_pts_seconds": float(times[-1]),
                "pts_delta_histogram": dict(deltas),
            },
            "source_event_rows": len(annotations["rows"]),
            "source_class_counts": dict(Counter(r["source_class"] for r in annotations["rows"])),
            "auxiliary_only_rows": annotations["auxiliary_only_rows"],
            "layout": config.model_dump(),
            "issues": issues,
            "strict_overlap_pairs": strict,
            "shared_endpoint_pairs": touching,
            "anchors": anchors,
            "mapping_status": "raw-index hypothesis only; no verified author FPS/base/end/trim",
            "prior_visual_status": source.get("temporal_alignment_status"),
            "note": (
                "Rate alternatives are hypotheses, not measured alignment errors. "
                "URL timestamps and helper rebases are not automatically applied."
            ),
        }
    report = {"schema_version": "alignment-diagnostics-1.0", "recordings": reports}
    output.mkdir(parents=True)
    write_json(output / "alignment.json", report)
    lines = [
        "# Frame-coordinate hypotheses — not corrections",
        "",
        "No offset, FPS or class convention is inferred from pose confidence. "
        "Source inputs match the cached frame inventory hashes.",
        "",
        "| Recording | Source FPS (average) | Source frames | Event rows | "
        "Flagged rows | Classes |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for video_id, r in reports.items():
        lines.append(
            f"| {video_id} | {r['video']['avg_frame_rate']} | {r['video']['frame_count']} | "
            f"{r['source_event_rows']} | {len(r['issues'])} | {r['source_class_counts']} |"
        )
        detail = [
            f"# {video_id} selected source anchors",
            "",
            "Raw decoded indices are provisional. Alternatives map A/rate to nearest actual PTS; "
            "none is verified.",
            "",
            "| Row | Source class | Raw bounds | Raw zero-base PTS | A/30 time | "
            "A/30 nearest index | Flags |",
            "| ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
        for a in r["anchors"]:
            h = a["hypotheses"]
            t = h["time_at_30_fps"]
            detail.append(
                f"| {a['source_row']} | {a['source_class']} | {a['raw_bounds']} | "
                f"{h['raw_zero_base']['pts_seconds']} | {t['target_seconds']:.3f} | "
                f"{t['nearest_decoded_index']} | {', '.join(a['flags'][3:])} |"
            )
        (output / f"{video_id}-anchors.md").write_text("\n".join(detail) + "\n")
    (output / "alignment.md").write_text("\n".join(lines) + "\n")
    return report
