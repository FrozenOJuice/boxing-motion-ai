"""Gated V4 versus V4/V7/V8 mirror-only experiment; V5 development, no V9 access."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml

from ..vision.pipeline import sha256
from .experiment import load_config, run_experiment, write_json
from .groups import GROUPS, source_groups, validate_split
from .mirror_study import aggregate

SEEDS = [20261021, 20261022, 20261023]


def freeze_gate(review_path, specs, output):
    """Freeze explicit visual-review decision and complete, hash-checked selections."""
    from .groups import GroupedClips

    review_path, output = Path(review_path), Path(output)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a fresh gate below reports/")
    review = yaml.safe_load(review_path.read_text())
    if (
        review["status"] != "raw_indices_supported_for_provisional_expansion"
        or review["source_group"] != GROUPS["V7"]
        or review["mapping"]["frame_offset"] != 0
        or review["mapping"]["end_assumption"] != "inclusive"
    ):
        raise ValueError("Explicit provisional alignment decision required")
    evidence_path = Path(review["evidence"])
    evidence = json.loads(evidence_path.read_text())
    dataset = GroupedClips(specs)
    sources = {}
    for spec, ds in zip(specs, dataset.datasets, strict=True):
        v = spec["recording"]
        if v == "V4":
            continue
        prior = evidence["recordings"][v]["inputs"]
        manifest = json.loads((Path(spec["pose_run"]) / "manifest.json").read_text())
        for key in ("video", "workbook"):
            actual = manifest["inputs"][key]
            if actual["sha256"] != prior[key]["sha256"]:
                raise ValueError("Review evidence belongs to different source inputs")
        if manifest["inputs"]["workbook"]["sha256"] != review["recordings"][v]["workbook_sha256"]:
            raise ValueError("Review decision belongs to a different workbook")
        if (
            manifest["status"] != "complete"
            or manifest["scope"] != "full_video"
            or manifest["source_frame_count"] != evidence["recordings"][v]["source_frame_count"]
            or manifest["recording_config"]["frame_offset"] != 0
            or manifest["recording_config"]["end_assumption"] != "inclusive"
        ):
            raise ValueError("Incomplete extraction or changed mapping")
        # Verify every decoded index and PTS against the source inventory, not a nominal FPS.
        probe = json.loads(Path(f"reports/boxingvi/development/{v}.ffprobe.json").read_text())
        with np.load(Path(spec["dataset"]) / "poses.npz", allow_pickle=False) as arrays:
            observed = [
                Fraction(int(p) * int(n), int(d))
                for p, n, d in zip(
                    arrays["pts"], arrays["time_base_num"], arrays["time_base_den"], strict=True
                )
            ]
        if observed != [Fraction(frame["pts_time"]) for frame in probe["frames"]]:
            raise ValueError("Full pose timestamps differ from source decoded inventory")
        sources[v] = ds.provenance
    result = {
        "schema_version": "expansion-training-gate-1.0",
        "training_ready": True,
        "readiness": "provisional; sampled timing support, labels/boundaries unverified",
        "source_groups": GROUPS,
        "review_path": str(review_path),
        "review_sha256": sha256(review_path),
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256(evidence_path),
        "recordings": sources,
        "training_specs": specs,
        "uncertainties": review["uncertainties"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_report(specs, output.parent)
    write_json(output, result)
    return result


def dataset_report(specs, output, development=None):
    """Full inventory and unchanged-policy quality, with explicit per-row exclusions."""
    from .data import CLASSES

    report = {
        "schema_version": "expansion-preparation-summary-1.0",
        "recordings": {},
        "source_groups": GROUPS,
        "boxer_id": None,
        "session_id": None,
    }
    lines = [
        "# V7/V8 full preparation and V4/V5 quality comparison",
        "",
        "Raw-index mapping remains provisional. V7/V8 share one unresolved source group. "
        "Unlabeled/excluded footage is unknown. No no_punch examples are created.",
        "",
        "| Recording | Class | Source rows | Annotation eligible | mask_aware70 | "
        "Invalid left elbow | Invalid right elbow |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    sources = {s["recording"]: s for s in specs}
    sources["V5"] = development or {
        "recording": "V5",
        "dataset": "data/processed/boxingvi/V5/candidates-full-v1",
        "policy": "reports/boxingvi/expansion/quality/V5/policy-index.json",
    }
    for v in ["V4", "V5", "V7", "V8"]:
        spec = sources[v]
        quality_path = Path(spec["policy"]).with_name("quality.json")
        quality = json.loads(quality_path.read_text())
        prep = json.loads((Path(spec["dataset"]) / "preparation.json").read_text())
        rows = [
            json.loads(line)
            for line in (Path(spec["dataset"]) / "candidates.jsonl").read_text().splitlines()
        ]
        if quality["scope"] != "full_video":
            raise ValueError("Cannot present a pose sample as full-video quality")
        classes = {}
        for label in CLASSES:
            d = quality["by_class"][label]
            joints = {j: x["invalid_frame_fraction"] for j, x in d["joints"].items()}
            count = prep["class_counts"][label]
            classes[label] = {
                "source_rows": count["source_rows"],
                "annotation_eligible": d["eligible"],
                "mask_aware70": d["retention"]["mask_aware70"],
                "pose_quality_rejected": d["eligible"] - d["retention"]["mask_aware70"],
                "frame_weighted_invalid_joint_fraction": joints,
            }
            values = [
                "n/a" if joints.get(j) is None else f"{joints[j]:.1%}"
                for j in ["left_elbow", "right_elbow"]
            ]
            lines.append(
                f"| {v} | {label} | {count['source_rows']} | {d['eligible']} | "
                f"{d['retention']['mask_aware70']} | {' | '.join(values)} |"
            )
        report["recordings"][v] = {
            "source_frame_count": prep["source_frame_count"],
            "classes": classes,
            "unknown_outside_annotation_bounds_frames": prep[
                "unknown_outside_annotation_bounds_frames"
            ],
            "source_groups": GROUPS[v],
            "boxer_id": prep.get("boxer_id"),
            "session_id": prep.get("session_id"),
            "quality_sha256": sha256(quality_path),
            "excluded_annotation_rows": [
                {
                    "source_row": r["source_row"],
                    "source_class": r["source_class"],
                    "raw_bounds": [r["raw_start"], r["raw_end"]],
                    "reasons": r["exclusion_reasons"],
                }
                for r in rows
                if not r["annotation_eligible"]
            ],
        }
        if v in ("V7", "V8"):
            report["recordings"][v]["diagnostic_all_source_rows"] = diagnostic_visibility(
                rows, Path(spec["dataset"]) / "poses.npz"
            )
    lines += [
        "",
        "Joint fractions use all annotation-eligible event frames, before policy "
        "selection. Low confidence is not proof of inaccurate coordinates. V7 rear "
        "uppercuts have no eligible rows; V8 has no rear-hook labels, so their class "
        "quality is unassessed. Full JSON lists exclusions without correcting them.",
    ]
    write_json(output / "dataset-summary.json", report)
    (output / "dataset-summary.md").write_text("\n".join(lines) + "\n")


def diagnostic_visibility(rows, path):
    """Include flagged source intervals for diagnosis only; never a training selection."""
    from ..data.pose_quality import CORE, POLICIES, retain
    from ..vision.contracts import JOINTS
    from .data import CLASSES

    result = {}
    with np.load(path, allow_pickle=False) as arrays:
        valid, status = arrays["valid_mask"], arrays["pose_status"]
        for label in CLASSES:
            group = [
                r
                for r in rows
                if r["punch_type_provisional"] == label and r["array_slice_half_open"] is not None
            ]
            masks = [valid[slice(*r["array_slice_half_open"])] for r in group]
            if not masks:
                result[label] = {"source_rows_assessed": 0, "visibility": None}
                continue
            combined = np.concatenate(masks)
            passed = sum(
                retain(m, status[slice(*r["array_slice_half_open"])] == 2, POLICIES["mask_aware70"])
                for r, m in zip(group, masks, strict=True)
            )
            result[label] = {
                "source_rows_assessed": len(group),
                "annotation_excluded_rows": sum(not r["annotation_eligible"] for r in group),
                "diagnostic_mask_policy_passes": int(passed),
                "visibility": {JOINTS[j]: float((~combined[:, j]).mean()) for j in CORE},
                "interpretation": "event-frame-weighted masks, including flagged annotations; "
                "not clean examples or evidence that labels are correct",
            }
    return result


def check_gate(path, specs):
    if [s.get("recording") for s in specs] == ["V4", "V10"]:
        from .v10_study import check_v10_gate

        return check_v10_gate(path, specs)
    gate = json.loads(path.read_text())
    if (
        gate.get("schema_version") != "expansion-training-gate-1.0"
        or gate.get("training_ready") is not True
        or gate.get("training_specs") != specs
        or gate.get("source_groups") != GROUPS
    ):
        raise ValueError("Expansion has not passed its frozen provisional data gate")
    for kind in ("review", "evidence"):
        if sha256(Path(gate[f"{kind}_path"])) != gate[f"{kind}_sha256"]:
            raise ValueError("Expansion review evidence changed")
    for spec in specs:
        v = spec["recording"]
        if v == "V4":
            continue
        p = gate["recordings"][v]
        for name, digest in p["hashes"].items():
            if sha256(Path(spec["dataset"]) / name) != digest:
                raise ValueError("Expansion candidate dataset changed since review")
        if sha256(Path(spec["policy"])) != p["policy_sha256"]:
            raise ValueError("Expansion quality selection changed since review")
    return gate


def run_study(config_path, output, smoke=False):
    config_path, output = Path(config_path), Path(output)
    study = yaml.safe_load(config_path.read_text())
    if set(study) != {"schema_version", "base_config", "reference", "train", "gate", "seeds"}:
        raise ValueError("Unexpected expansion study fields")
    if study["schema_version"] != "expansion-study-1.0" or study["seeds"] != SEEDS:
        raise ValueError("Frozen expansion version and three matched seeds required")
    base = load_config(Path(study["base_config"]))
    validate_split(study["train"], base["development"])
    check_gate(Path(study["gate"]), study["train"])
    reference = Path(study["reference"])
    frozen = yaml.safe_load((reference / f"seed-{SEEDS[0]}/mirrored/config.yaml").read_text())
    expected = {
        **base,
        "schema_version": "level1-experiment-1.1",
        "seed": SEEDS[0],
        "mirror_probability": 0.5,
        "runs": ["tcn", "tcn_mask_only"],
    }
    if frozen != expected:
        raise ValueError("Base training configuration differs from mirror-only reference")
    original = {str(p): sha256(p) for p in reference.rglob("*") if p.is_file()}
    if (
        not original
        or output.exists()
        or not output.resolve().is_relative_to(Path("reports").resolve())
    ):
        raise ValueError("Reference required; use a fresh reports directory")
    # Changing loading support must not change feature construction, collation or architecture.
    import ast

    current = Path("src/boxing_motion_ai/training")
    prior = reference / f"seed-{SEEDS[0]}" / "mirrored/source/src/boxing_motion_ai/training"
    for name in ("features", "collate", "exact_times", "validate_arrays"):
        functions = []
        for path in (prior / "data.py", current / "data.py"):
            tree = ast.parse(path.read_text())
            functions.append(
                ast.dump(next(n for n in tree.body if getattr(n, "name", None) == name))
            )
        if functions[0] != functions[1]:
            raise ValueError(f"Frozen feature/timing function changed: {name}")
    if sha256(prior / "models.py") != sha256(current / "models.py"):
        raise ValueError("Architecture changed")
    output.mkdir(parents=True)
    (output / "study.yaml").write_text(config_path.read_text())
    write_json(output / "preserved-reference.json", original)
    seeds = SEEDS[:1] if smoke else SEEDS
    runs = {}
    for seed in seeds:
        runs[str(seed)] = {}
        for arm in ("v4_only", "expanded"):
            config = {
                **base,
                "schema_version": "level1-experiment-1.1",
                "seed": seed,
                "mirror_probability": 0.5,
                "runs": ["tcn", "tcn_mask_only"],
            }
            if arm == "expanded":
                config.update(
                    schema_version="level1-experiment-1.4",
                    train=study["train"],
                    expansion_gate=study["gate"],
                )
            if smoke:
                config.update(epochs=2, limit_per_class=2)
            path = output / f"seed-{seed}-{arm}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            directory = output / f"seed-{seed}" / arm
            print(f"=== {seed}: {arm}; smoke={smoke} ===", flush=True)
            result = run_experiment(path, directory, lambda s: print(s, flush=True))
            runs[str(seed)][arm] = result
            logs = [
                (directory / name / "augmentation.json").read_text()
                for name in ("tcn", "tcn_mask_only")
            ]
            if logs[0] != logs[1]:
                raise ValueError("Full/mask controls differ in shuffle/augmentation")
            if not smoke and arm == "v4_only":
                previous = json.loads(
                    (reference / f"seed-{seed}/mirrored/results.json").read_text()
                )
                for model in ("tcn", "tcn_mask_only"):
                    for key in ("selected_epoch", "train", "development", "last_epoch"):
                        if previous[model][key] != result[model][key]:
                            raise ValueError("Mirror-only control failed exact reproduction")
        for model in ("tcn", "tcn_mask_only"):
            if (
                runs[str(seed)]["v4_only"][model]["initial_model_sha256"]
                != runs[str(seed)]["expanded"][model]["initial_model_sha256"]
            ):
                raise ValueError("Paired initial model mismatch")
    summary = aggregate(runs, seeds, arms=("v4_only", "expanded"))
    summary.update(
        schema_version="expansion-study-result-1.0",
        smoke=smoke,
        configuration_sha256=sha256(config_path),
        source_groups=source_groups(study["train"]),
        training_recordings=[s["recording"] for s in study["train"]],
        reference_metrics_reproduced=not smoke,
        budget=f"{2 if smoke else base['epochs']} epochs each; expanded data produces "
        "more optimizer steps per epoch",
        warning="V5 is repeatedly used development; no unbiased test or boxer-independent claim",
    )
    write_json(output / "summary.json", summary)
    write_json(output / "runs.json", runs)
    write_summary(output, summary, runs)
    if any(sha256(Path(p)) != digest for p, digest in original.items()):
        raise ValueError("Previous experiment artifacts changed")
    return summary


def write_summary(output, summary, runs):
    from .data import CLASSES

    lines = [
        "# V4-only versus "
        + "/".join(summary.get("training_recordings", ["V4", "V7", "V8"]))
        + " mirror-only TCN",
        "",
        summary["warning"],
        "",
        summary["budget"],
        "",
        f"Seeds: {summary['seeds']}; smoke: {summary['smoke']}",
        "",
        "| Seed | Input | Training | Selected epoch | Train F1 | V5 F1 | V5 accuracy | "
        "Final train F1 | Final V5 F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in summary["seeds"]:
        for arm in ("v4_only", "expanded"):
            for model in ("tcn", "tcn_mask_only"):
                r = runs[str(seed)][arm][model]
                lines.append(
                    f"| {seed} | {model} | {arm} | {r['selected_epoch']} | "
                    f"{r['train']['macro_f1']:.4f} | {r['development']['macro_f1']:.4f} | "
                    f"{r['development']['accuracy']:.4f} | "
                    f"{r['last_epoch']['train']['macro_f1']:.4f} | "
                    f"{r['last_epoch']['development']['macro_f1']:.4f} |"
                )
    lines += [
        "",
        "## Per-class recall (mean over seeds)",
        "",
        "| Class | V4 full | Expanded full | V4 mask-only | Expanded mask-only |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in CLASSES:
        values = [
            summary["by_model"][model][arm]["per_class_recall"][label]["mean"]
            for model in ("tcn", "tcn_mask_only")
            for arm in ("v4_only", "expanded")
        ]
        lines.append(f"| {label} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += [
        "",
        "Each run retains confusion matrices, probabilities, recording-qualified source "
        "rows, learning curves, configs, source snapshots and best/final checkpoints. "
        "V9 is not opened. Labels/boundaries and person identities remain provisional.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--freeze-gate",
        type=Path,
        help="Freeze gate from this reviewed YAML instead of training; output is JSON",
    )
    args = parser.parse_args()
    if args.freeze_gate:
        study = yaml.safe_load(args.config.read_text())
        base = load_config(Path(study["base_config"]))
        validate_split(study["train"], base["development"])
        freeze_gate(args.freeze_gate, study["train"], args.output)
    else:
        run_study(args.config, args.output, args.smoke)
