"""Read-only V6 alignment/overlap review using exact frames and existing sample poses."""

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from ..vision.annotations import read_annotations
from ..vision.pipeline import sha256, write_json
from ..vision.recording import load_recording
from .expansion_review import summarize_rows
from .source_review import decode_frames, hypotheses, overlap_relations, render_case, write_player

RECORDING = Path("configs/v6_recording.yaml")
ANCHORS = [2, 41, 288, 321, 336, 598, 642, 685, 686]
RECOMMENDED_HOLDS = [321, 598, 620, *range(675, 687)]
TARGETS = [
    2,
    3,
    5,
    6,
    7,
    8,
    9,
    10,
    25,
    41,
    49,
    58,
    78,
    80,
    100,
    101,
    102,
    288,
    315,
    321,
    327,
    328,
    329,
    330,
    331,
    332,
    333,
    334,
    335,
    336,
    337,
    338,
    339,
    340,
    379,
    380,
    411,
    598,
    620,
    642,
    *range(675, 687),
]


def helper_audit(rows, auxiliary):
    """Literal spreadsheet expressions only; no evaluation or timing inference."""
    counts = Counter()
    anomalies = []
    for row in rows:
        n, cells = row["source_row"], row["source_cells"]
        expected = {"D": f"=B{n}-A{n}", "E": f"=A{n}-157", "F": f"=B{n}-157"}
        for col, formula in expected.items():
            if cells.get(col) == formula:
                counts[col] += 1
            else:
                anomalies.append(
                    {
                        "source_row": n,
                        "column": col,
                        "value": cells.get(col),
                        "expected_literal": formula,
                    }
                )
    combined = rows + auxiliary
    ordinal = [
        r["source_row"] for r in combined if r["source_cells"].get("H") != r["source_row"] - 2
    ]
    return {
        "matching_formula_counts": dict(counts),
        "anomalies": anomalies,
        "H_row_minus_2_mismatches": ordinal,
        "auxiliary_only_rows": len(auxiliary),
        "G_nonempty_rows": [
            r["source_row"] for r in combined if r["source_cells"].get("G") is not None
        ],
        "interpretation": "D is coordinate difference, not verified inclusive frame count. "
        "E/F subtract157 literally, not evidence of source trim. H follows row ordinal, "
        "not a verified frame, boxer or session ID. No correction applied.",
    }


def pair_inventory(rows):
    pairs = []
    for row in rows:
        for other in overlap_relations(row, rows):
            if row["source_row"] < other["source_row"]:
                pairs.append(
                    {
                        "source_rows": [row["source_row"], other["source_row"]],
                        "classes": [row["source_class"], other["source_class"]],
                        "raw_bounds": [[row["raw_start"], row["raw_end"]], other["raw_bounds"]],
                        "kind": other["kind"],
                        "overlap_coordinate_span": other["end_minus_start_overlap"],
                        "training_restored": False,
                    }
                )
    return {
        "geometry_counts": dict(Counter(p["kind"] for p in pairs)),
        "pairs": pairs,
        "interpretation": "Geometry is not an action/label verdict. All existing flags persist.",
    }


def display_intervals(row, name, bounds, rows):
    """Alternative displays mark their hypothetical target, never raw coordinates elsewhere."""
    if name != "raw":
        return [[row["source_row"], *bounds]]
    return [
        [row["source_row"], row["raw_start"], row["raw_end"]],
        *[[r["source_row"], *r["raw_bounds"]] for r in overlap_relations(row, rows)],
    ]


def conditional_capacity(inventory, holds):
    """Exact review-constrained ceilings; 80% is a planning assumption, not observed yield."""
    result = {}
    held = set(holds)
    for label, counts in inventory["by_class"].items():
        ids = [
            r["source_row"]
            for r in inventory["rows"]
            if r["punch_type_provisional"] == label
            and r["structurally_eligible"]
            and r["source_row"] not in held
        ]
        result[label] = {
            "source": counts["source"],
            "structural_ceiling": counts["structurally_eligible"],
            "review_constrained_ceiling": len(ids),
            "capacity_scenario_80_percent": len(ids) * 8 // 10,
            "candidate_source_rows": ids,
            "approved_training_examples": 0,
        }
    return {
        "by_class": result,
        "assumed_fraction": 0.8,
        "limitation": "Conditional capacity only, not measured full-policy retention or "
        "independently verified label counts. Actual usable yield can be lower, including zero.",
    }


def existing_pose_evidence(probe):
    """Recompute the original masks/policy on saved samples, never run inference."""
    import numpy as np

    from .pose_quality import POLICIES, analyze_arrays, score_validity

    dataset = Path("data/processed/boxingvi/V6/candidates-sample-expansion-v1")
    quality_path = Path("reports/boxingvi/expansion/quality/V6/quality.json")
    quality = json.loads(quality_path.read_text())
    for name, digest in quality["inputs"].items():
        if sha256(dataset / name) != digest:
            raise ValueError("Existing sample changed")
    if quality["policies"]["mask_aware70"] != POLICIES["mask_aware70"]:
        raise ValueError("Quality thresholds changed")
    rows = [json.loads(s) for s in (dataset / "candidates.jsonl").read_text().splitlines()]
    with np.load(dataset / "poses.npz", allow_pickle=False) as arrays:
        if not np.array_equal(
            arrays["valid_mask"], score_validity(arrays["coordinates"], arrays["confidence"], 0.5)
        ):
            raise ValueError("Stored confidence/mask contract differs")
        observed = [
            Fraction(int(p) * int(a), int(b))
            for p, a, b in zip(
                arrays["pts"], arrays["time_base_num"], arrays["time_base_den"], strict=True
            )
        ]
        expected = [Fraction(probe["frames"][int(i)]["pts_time"]) for i in arrays["frame_index"]]
        if observed != expected:
            raise ValueError("Sample exact timestamps differ from source")
        assessed = [r for r in rows if r["quality_assessable"]]
        for row in assessed:
            if row["source_row"] in RECOMMENDED_HOLDS:
                row["annotation_eligible"] = False  # in-memory diagnostic only
                row["exclusion_reasons"] = ["suspected_class_phase_mismatch"]
        conservative = analyze_arrays(arrays, assessed, permutations=19, required_classes=())
    return {
        "new_pose_inference": False,
        "scope": "selected_windows",
        "extracted_frames": len(expected),
        "sample_events_before_review": 20,
        "quality_path": str(quality_path),
        "quality_sha256": sha256(quality_path),
        "original_sample_quality": quality,
        "diagnostic_without_review_holds": conservative,
        "recommended_review_holds": RECOMMENDED_HOLDS,
        "limitation": "Nonrandom five ordinal samples/class; row685 policy pass is not "
        "label evidence. Reassessment is in memory and changes no candidate files.",
    }


def run(output):
    output = Path(output)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a new report directory under reports/")
    inventory_path = Path("reports/boxingvi/rear-punch-priorities-v1/V6/inventory.json")
    prior = json.loads(inventory_path.read_text())
    inputs = prior["inputs"]
    for key in ("video", "workbook"):
        if sha256(inputs[f"{key}_path"]) != inputs[f"{key}_sha256"]:
            raise ValueError("V6 source changed")
    probe_path = Path("reports/boxingvi/development/V6.ffprobe.json")
    if sha256(probe_path) != inputs["frame_inventory_sha256"]:
        raise ValueError("Exact source inventory changed")
    probe = json.loads(probe_path.read_text())
    times = [Fraction(f["pts_time"]) for f in probe["frames"]]
    config = load_recording(RECORDING)
    annotations = read_annotations(inputs["workbook_path"], config)
    rows = annotations["rows"]
    lookup = {r["source_row"]: r for r in rows}
    cases, timing = [], {}
    for n in TARGETS:
        row = lookup[n]
        alternatives = hypotheses(row, times, "V6")
        timing[n] = {
            name: {
                "indices": pair,
                "start_pts": str(times[pair[0]]),
                "end_pts": str(times[pair[1]]),
            }
            for name, pair in alternatives.items()
        }
        names = ["raw"]
        if n in ANCHORS:
            names += [k for k in ["clock_24", "helper_minus_157", "clock_30"] if k in alternatives]
        for name in names:
            a, b = alternatives[name]
            neighbors = display_intervals(row, name, [a, b], rows)
            if name == "raw":
                a, b = min(int(x[1]) for x in neighbors), max(int(x[2]) for x in neighbors)
            margin = 36 if n >= 675 else 12
            cases.append(
                {
                    "id": f"V6-row-{n}-{name}",
                    "source_row": n,
                    "source_class": row["source_class"],
                    "hypothesis": f"{name}; raw {row['raw_start']}:{row['raw_end']}; "
                    f"target {alternatives[name]}; unverified",
                    "hypothesis_bounds": alternatives[name],
                    "indices": list(range(max(0, a - margin), min(len(times), b + margin + 1))),
                    "neighbor_bounds": neighbors,
                    "training_restored": False,
                }
            )
            if n in ANCHORS or n >= 675:
                left, right = alternatives[name]
                cases.append(
                    {
                        **cases[-1],
                        "id": f"V6-row-{n}-{name}-tight",
                        "indices": list(range(left, right + 1)),
                        "neighbor_bounds": [[n, left, right]],
                    }
                )
    requested = {i for c in cases for i in c["indices"]}
    decode_frames(inputs["video_path"], times, requested, output / "frames")
    for case in cases:
        render_case(case, times, output)
    write_player(cases, times, output)
    report = {
        "schema_version": "v6-targeted-review-1.0",
        "video_id": "V6",
        "inputs": inputs,
        "recording_config_sha256": sha256(RECORDING),
        "prior_inventory_sha256": sha256(inventory_path),
        "source_frame_count": len(times),
        "source_clock_residuals_seconds": {
            str(rate): float(max(abs(t - Fraction(i, 1) / rate) for i, t in enumerate(times)))
            for rate in (Fraction(24), Fraction(24000, 1001))
        },
        "review_decoded_frames": len(requested),
        "review_windows": len(cases),
        "selected_source_rows": TARGETS,
        "inventory": summarize_rows(rows, len(times), config.class_map),
        "helpers": helper_audit(rows, annotations["auxiliary_only_rows"]),
        "overlaps": pair_inventory(rows),
        "timing_hypotheses": timing,
        "pose_evidence": existing_pose_evidence(probe),
        "source_group_id": "natebowerfitness-unresolved",
        "boxer_id": None,
        "session_id": None,
        "annotations_restored": [],
        "training_performed": False,
        "full_extraction_performed": False,
        "review_code_sha256": sha256(__file__),
    }
    report["conditional_capacity"] = conditional_capacity(report["inventory"], RECOMMENDED_HOLDS)
    write_json(output / "review.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
