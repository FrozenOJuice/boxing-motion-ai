"""Descriptive, class-balanced cross-recording diagnostics; never a model evaluation."""

import json
from pathlib import Path

from ..vision.pipeline import code_provenance, sha256, write_json
from ..vision.sampling import select_rows
from .pose_quality import analyze_arrays


def compare_recordings(datasets, output, per_class=5):
    import numpy as np

    output = Path(output)
    datasets = [Path(p) for p in datasets]
    if output.exists() or any(output.resolve().is_relative_to(p.resolve()) for p in datasets):
        raise ValueError("Use a new comparison directory outside prepared datasets")
    if any(p.name == "raw" for p in output.resolve().parents):
        raise ValueError("Comparison reports must stay outside raw data")
    if len(datasets) < 2:
        raise ValueError("Compare at least two distinct recording datasets")
    results = {}
    for dataset in datasets:
        manifest = json.loads((dataset / "preparation.json").read_text())
        if (
            manifest["status"] != "complete"
            or sha256(dataset / "poses.npz") != manifest["npz_sha256"]
        ):
            raise ValueError("Incomplete dataset or changed pose array hash")
        video_id = manifest["video_id"]
        if video_id in results:
            raise ValueError("Comparison requires distinct recording IDs")
        rows = [json.loads(s) for s in (dataset / "candidates.jsonl").read_text().splitlines()]
        assessed = [
            r for r in rows if r["annotation_eligible"] and r.get("quality_assessable", True)
        ]
        available = {r["punch_type_provisional"] for r in assessed}
        mapping = {k: v for k, v in manifest["class_map"].items() if v in available}
        selected = select_rows(assessed, mapping, per_class)
        with np.load(dataset / "poses.npz", allow_pickle=False) as arrays:
            quality = analyze_arrays(arrays, selected, permutations=19, required_classes=())
        results[video_id] = {
            "selection_method": (
                "Equally spaced class ordinal positions among assessable rows; "
                "no mask-based selection"
            ),
            "scope": manifest.get("scope", "full_video"),
            "boxer_id": manifest.get("boxer_id"),
            "session_id": manifest.get("session_id"),
            "selected_source_rows": [r["source_row"] for r in selected],
            "inputs": {
                name: sha256(dataset / name)
                for name in ["preparation.json", "poses.npz", "candidates.jsonl"]
            },
            "quality": quality,
        }
    report = {
        "schema_version": "recording-quality-comparison-1.0",
        "code": code_provenance(),
        "comparison_code_sha256": sha256(__file__),
        "recordings": results,
        "limitation": (
            "Small class-balanced samples of dependent events; different viewpoints and "
            "frame rates. Provisional source classes/coordinates. No generalization "
            "accuracy, identity separation or occlusion causation established."
        ),
    }
    output.mkdir(parents=True)
    write_json(output / "comparison.json", report)
    lines = [
        "# Cross-recording pose sample comparison",
        "",
        report["limitation"],
        "",
        "| Recording | Class | Assessed | Strict body | Mask-aware70 | "
        "Left elbow invalid | Right elbow invalid |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for video_id, result in results.items():
        for label, values in result["quality"]["by_class"].items():
            joints, retention = values["joints"], values["retention"]
            left = joints["left_elbow"]["invalid_frame_fraction"]
            right = joints["right_elbow"]["invalid_frame_fraction"]
            left_text = f"{left:.1%}" if left is not None else "unassessed"
            right_text = f"{right:.1%}" if right is not None else "unassessed"
            lines.append(
                f"| {video_id} | {label} | {values['eligible']} | "
                f"{retention['strict_body']} | {retention['mask_aware70']} | "
                f"{left_text} | "
                f"{right_text} |"
            )
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    return report
