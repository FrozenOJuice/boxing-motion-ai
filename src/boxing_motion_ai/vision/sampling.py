"""Deterministic class/time-spread sample planning, before seeing any pose quality."""

from .annotations import read_annotations
from .pipeline import sha256, write_json

UNCERTAINTIES = {"provisional_mapping", "class_unverified", "boundaries_unverified"}


def select_rows(rows, class_map, per_class=5, allow_flagged=False):
    if type(per_class) is not int or per_class < 2:
        raise ValueError("At least two samples per class are required")
    selected = []
    for label in dict.fromkeys(class_map.values()):
        group = sorted(
            [
                r
                for r in rows
                if class_map.get(r["source_class"]) == label
                and not (set(r["flags"]) - UNCERTAINTIES)
            ],
            key=lambda r: (r["candidate_start_index"], r["source_row"]),
        )
        if not group and allow_flagged:
            group = sorted(
                [
                    dict(r, selected_for_review_only=True)
                    for r in rows
                    if class_map.get(r["source_class"]) == label
                    and r["candidate_start_index"] is not None
                    and r["candidate_end_index"] > r["candidate_start_index"]
                ],
                key=lambda r: (r["candidate_start_index"], r["source_row"]),
            )
        if not group:
            raise ValueError(f"No structurally eligible rows for {label}")
        count = min(per_class, len(group))
        positions = (
            [round(i * (len(group) - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
        )
        selected.extend(group[i] for i in positions)
    return sorted(selected, key=lambda r: r["source_row"])


def plan_sample(workbook, recording, output, per_class=5):
    from pathlib import Path

    output = Path(output)
    if (
        output.exists()
        or any(p.name == "raw" for p in output.resolve().parents)
        or output.resolve().is_relative_to(Path(workbook).resolve().parent)
    ):
        raise ValueError("Use a new plan directory outside raw data")
    annotations = read_annotations(workbook, recording)
    allowed = [
        r for r in annotations["rows"] if r["source_row"] not in recording.excluded_source_rows
    ]
    rows = select_rows(allowed, recording.class_map, per_class, allow_flagged=True)
    intervals = sorted(
        (
            r["candidate_start_index"],
            r["candidate_end_index"] + int(recording.end_assumption == "inclusive"),
        )
        for r in rows
    )
    merged = []
    for a, b in intervals:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    frames = sorted({n for a, b in intervals for n in [a, (a + b - 1) // 2, b - 1]})
    plan = {
        "schema_version": "pose-sample-plan-1.0",
        "video_id": recording.video_id,
        "recording": recording.model_dump(),
        "workbook_sha256": sha256(workbook),
        "method": (
            "Equally spaced ordinal positions per literal class among structurally "
            "eligible rows; fallback to flagged diagnostic-only rows if a class has none; "
            "no pose-based selection"
        ),
        "per_class_requested": per_class,
        "selected_rows": rows,
        "annotation_uncertainties": {k: v for k, v in annotations.items() if k != "rows"},
    }
    output.mkdir(parents=True)
    write_json(output / "sample-plan.json", plan)
    write_json(
        output / "pose-config.json",
        {
            "pose_mode": "image",
            "scene_cuts": [],
            "ranges": [f"{a}:{b}" for a, b in merged],
            "max_width": 960,
            "confidence": 0.5,
            "overlay_frames": frames,
            "overlay_limit": len(frames),
        },
    )
    return plan
