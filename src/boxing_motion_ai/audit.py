"""Deterministic local discovery, validation and evidence reports."""

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from .boxingvi import Mapping, inspect_and_parse
from .media import probe
from .models import DatasetMetadata, Event, PunchType

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".mts"}
MANUAL_CHECKS = [
    "Verify footage and annotation licenses/redistribution with documented author evidence.",
    "Review source frame base, inclusive/exclusive end, FPS/timebase and visual alignment.",
    "Verify continuous source recordings, edits and reviewed non-punch/feint footage for Level 2.",
    "Verify boxer/session identities and freeze independent groups; source-video fallback alone "
    "does not establish boxer independence.",
    "Manually review onset/peak boundaries and representative classes/viewpoints before modeling.",
]


def validate_events(events: list[Event], videos: dict, issue):
    """Return interval-union coverage only for events with inspected in-bounds media."""
    intervals = defaultdict(list)
    seen_ids, seen_events = set(), set()
    by_video = defaultdict(list)
    for event in events:
        if event.event_id in seen_ids:
            issue("duplicate_id", "error", event.event_id, "Repeated event identifier")
        seen_ids.add(event.event_id)
        start, stop = event.zero_based_interval()
        key = (event.video_id, start, stop, event.punch_type, event.boxer_id)
        if key in seen_events:
            issue("duplicate_event", "error", event.event_id, "Repeated video/bounds/class/boxer")
        seen_events.add(key)
        by_video[event.video_id].append((start, stop, event))
        if not event.boxer_id or not event.session_id:
            issue(
                "missing_identity",
                "warning",
                event.event_id,
                "Boxer/session IDs absent; do not guess identities from filenames",
            )
        video = videos.get(event.video_id)
        if video is None:
            issue("missing_media", "error", event.event_id, f"Missing media: {event.video_id}")
        elif video.frame_count is None:
            issue(
                "unknown_frame_count",
                "error",
                event.event_id,
                "Cannot validate interval against an exact video frame count",
            )
        elif stop > video.frame_count:
            issue(
                "out_of_bounds",
                "error",
                event.event_id,
                f"Interval [{start}, {stop}) exceeds {video.frame_count} frames",
            )
        else:
            intervals[event.video_id].append((start, stop))
    for rows in by_video.values():
        # Sweep by start: flag each event intersecting a previous interval. No quadratic pairs.
        active_end = -1
        for start, stop, event in sorted(rows, key=lambda row: (row[0], row[1], row[2].event_id)):
            if start < active_end:
                issue(
                    "suspect_overlap",
                    "warning",
                    event.event_id,
                    "Overlapping punch intervals; review duplicate/boxer attribution",
                )
            active_end = max(active_end, stop)
    coverage = []
    for video_id, video in sorted(videos.items()):
        merged = []
        for start, stop in sorted(intervals[video_id]):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        frames = sum(stop - start for start, stop in merged)
        coverage.append(
            {
                "video_id": video_id,
                "frame_count": video.frame_count,
                "in_bounds_labeled_frames": frames,
                "labeled_fraction": frames / video.frame_count if video.frame_count else None,
                "unreviewed_or_unlabeled_frames": video.frame_count - frames
                if video.frame_count
                else None,
                "reviewed_no_punch_frames": None,
            }
        )
    return coverage


def run_audit(root: Path, mapping_path: Path | None = None) -> dict:
    root = root.resolve()
    issues = []

    def issue(code, severity, location, message):
        issues.append(dict(code=code, severity=severity, location=location, message=message))

    mapping, mapping_info = None, None
    if mapping_path is not None:
        try:
            contents = mapping_path.read_bytes()
            mapping = Mapping.model_validate_json(contents)
            mapping_info = {
                "sha256": hashlib.sha256(contents).hexdigest(),
                "config": mapping.model_dump(mode="json"),
                "verification": "operator supplied; not independently verified",
            }
        except (OSError, ValueError) as exc:
            issue("invalid_mapping", "error", mapping_path.name, str(exc))
    files, paths = [], []
    if not root.is_dir():
        issue("missing_root", "error", ".", "Dataset directory is missing or is not a directory")
    else:

        def walk_error(exc):
            issue("discovery_error", "error", ".", str(exc))

        for directory, dirs, names in os.walk(root, onerror=walk_error, followlinks=False):
            for name in list(dirs):
                path = Path(directory) / name
                if path.is_symlink():
                    issue(
                        "symlink_skipped",
                        "warning",
                        path.relative_to(root).as_posix(),
                        "Place actual local dataset files beneath root; symlink skipped",
                    )
                    dirs.remove(name)
            for name in sorted(names):
                path = Path(directory) / name
                rel = path.relative_to(root).as_posix()
                if path.is_symlink():
                    issue("symlink_skipped", "warning", rel, "Symlink skipped")
                    continue
                try:
                    files.append(
                        {"path": rel, "bytes": path.stat().st_size, "suffix": path.suffix.lower()}
                    )
                    paths.append(path)
                except OSError as exc:
                    issue("unreadable_file", "error", rel, str(exc))
    paths.sort()
    files.sort(key=lambda item: item["path"])
    workbooks = [p for p in paths if p.suffix.lower() == ".xlsx"]
    media = [p for p in paths if p.suffix.lower() in VIDEO_SUFFIXES]
    if not workbooks:
        issue("missing_annotations", "error", ".", "Obtain original Annotation_files/*.xlsx")
    if not media:
        issue("missing_videos", "error", ".", "Obtain matching original untrimmed RGB_videos media")
    if mapping:
        discovered = {p.relative_to(root).as_posix() for p in workbooks}
        for config in mapping.sheets:
            if config.workbook not in discovered:
                issue(
                    "missing_workbook",
                    "error",
                    config.workbook,
                    "Mapped workbook is absent or not a local .xlsx",
                )
    videos = {p.relative_to(root).as_posix(): probe(p, root, issue) for p in media}
    events, sheets, raw_classes = [], [], Counter()
    for path in workbooks:
        inventory, parsed, counts = inspect_and_parse(path, root, mapping, issue)
        sheets.extend(inventory)
        events.extend(parsed)
        raw_classes.update(counts)
    if not events:
        issue(
            "no_parsed_events",
            "error",
            ".",
            "No valid events parsed. Inspect XLSX previews and supply a verified --mapping",
        )
    if mapping:
        for video_id, video in videos.items():
            configs = [s for s in mapping.sheets if s.video == video_id]
            for field in ("boxer_id", "session_id"):
                values = {getattr(c, field) for c in configs if getattr(c, field)}
                if len(values) == 1:
                    setattr(video, field, next(iter(values)))
                elif len(values) > 1:
                    issue(
                        "conflicting_identity",
                        "error",
                        video_id,
                        f"Multiple {field} values; video-level identity cannot be established",
                    )
    coverage = validate_events(events, videos, issue)
    for video_id, video in videos.items():
        if not any(e.video_id == video_id for e in events):
            issue("unannotated_video", "error", video_id, "No parsed events link to this media")
        if not video.boxer_id or not video.session_id:
            issue(
                "missing_video_identity",
                "warning",
                video_id,
                "Boxer/session provenance is incomplete",
            )
    groups = {}
    for field in ("boxer_id", "session_id", "video_id"):
        candidates = defaultdict(set)
        for video in videos.values():
            key = getattr(video, field)
            if key:
                candidates[key].add(video.video_id)
        groups[field] = {key: sorted(value) for key, value in sorted(candidates.items())}
    issue(
        "manual_review_needed",
        "warning",
        ".",
        "Licensing, continuity, identities and visual alignment remain unverified; "
        "see manual_checks",
    )
    errors = any(i["severity"] == "error" for i in issues)
    return {
        "schema_version": "1.0",
        "dataset": DatasetMetadata(found=bool(workbooks or media)).model_dump(mode="json"),
        "root_directory_exists": root.is_dir(),
        "status": "missing_input"
        if not (workbooks or media)
        else "incomplete"
        if errors
        else "needs_review",
        "structural_checks_passed": not errors,
        "phase_0_gate": "not_verified",
        "files": files,
        "counts": {
            "files": len(files),
            "xlsx": len(workbooks),
            "videos": len(media),
            "parsed_events": len(events),
        },
        "sheets": sheets,
        "mapping": mapping_info,
        "raw_class_counts_in_mapped_rows": dict(sorted(raw_classes.items())),
        "class_balance_parsed_rows": {
            p.value: sum(e.punch_type == p for e in events) for p in PunchType
        },
        "class_balance_note": "Parsed rows include duplicates and out-of-bounds events; "
        "not a count of usable training samples. Unmapped rows are excluded.",
        "videos": [v.model_dump(mode="json") for _, v in sorted(videos.items())],
        "events": [e.model_dump(mode="json") for e in events],
        "annotation_coverage": coverage,
        "coverage_note": "Union of parsed, in-bounds intervals only; "
        "unreviewed gaps are not no_punch.",
        "candidate_split_groups": groups,
        "split_status": "candidates only; identities and cross-recording duplicates need review",
        "level_2_continuity": "unverified; metadata alone cannot establish contiguous negatives",
        "manual_checks": MANUAL_CHECKS,
        "issues": issues,
    }


def md(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_markdown(report: dict) -> str:
    lines = [
        "# BoxingVI local dataset audit",
        "",
        f"Status: **{report['status']}**. Phase 0 gate: **not verified**.",
        "",
        f"Dataset found: **{report['dataset']['found']}**. "
        f"Root directory exists: **{report['root_directory_exists']}**.",
        "",
        "Licensing: **unverified**. Redistribution: **unverified**.",
        "",
        f"Source: {report['dataset']['source_url']}",
        "",
        "## Actual local counts",
        "",
        *[f"- {k}: {v}" for k, v in report["counts"].items()],
        "",
        "## Files (paths relative to dataset root)",
        "",
        "| File | Bytes |",
        "| --- | ---: |",
        *[f"| {md(f['path'])} | {f['bytes']} |" for f in report["files"]],
        "",
        "## Class balance",
        "",
        report["class_balance_note"],
        "",
        "| Canonical class | Parsed rows |",
        "| --- | ---: |",
        *[f"| {k} | {v} |" for k, v in report["class_balance_parsed_rows"].items()],
        "",
        "## Source class values in mapped rows",
        "",
        "```json",
        json.dumps(report["raw_class_counts_in_mapped_rows"], indent=2),
        "```",
        "",
        "## Annotation coverage",
        "",
        report["coverage_note"],
        "",
        "```json",
        json.dumps(report["annotation_coverage"], indent=2),
        "```",
        "",
        "## Video metadata",
        "",
        "```json",
        json.dumps(report["videos"], indent=2),
        "```",
        "",
        "## Workbook layout inspection",
        "",
        "```json",
        json.dumps(report["sheets"], indent=2),
        "```",
        "",
        "## Mapping provenance",
        "",
        "```json",
        json.dumps(report["mapping"], indent=2),
        "```",
        "",
        "## Candidate split groups",
        "",
        report["split_status"],
        "",
        "```json",
        json.dumps(report["candidate_split_groups"], indent=2),
        "```",
        "",
        "## Level 2 footage continuity",
        "",
        report["level_2_continuity"],
        "",
        "## Diagnostics",
        "",
        *[
            f"- **{i['severity']} / {i['code']}** ({md(i['location'])}): {md(i['message'])}"
            for i in report["issues"]
        ],
        "",
        "## Outstanding manual checks",
        "",
        *[f"- {check}" for check in report["manual_checks"]],
        "",
    ]
    return "\n".join(lines)


def write_report(report: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "audit.md").write_text(render_markdown(report), encoding="utf-8")
