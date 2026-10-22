"""Frozen checkpoint transfer diagnostics. No optimization, tuning or source-data writes."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..data.pose_quality import mutual_information
from ..vision.pipeline import sha256
from .data import CLASSES, FEATURE_VERSION, PoseClips, collate, exact_times
from .experiment import evaluate, seed_all, write_json
from .mirror_study import summarize
from .models import CausalTCN
from .velocity import VELOCITY_VERSION, VelocityClips

RECORDINGS = ("V6", "V7", "V8", "V10")
SEEDS = (20261021, 20261022, 20261023)
ARMS = ("xy", "xy_velocity")
MODELS = ("tcn", "tcn_mask_only")
GROUPS = {
    "V6": ("natebowerfitness-unresolved", ["V1", "V6"]),
    "V7": ("precision-striking-unresolved", ["V7", "V8"]),
    "V8": ("precision-striking-unresolved", ["V7", "V8"]),
    "V10": ("utah-lee-unresolved", ["V10"]),
}
POLICY_PATHS = {
    "V6": "reports/boxingvi/v6-full-v1/quality/V6/policy-index.json",
    "V7": "reports/boxingvi/v7-v8-full-v1/quality/V7/policy-index.json",
    "V8": "reports/boxingvi/v7-v8-full-v1/quality/V8/policy-index.json",
    "V10": "reports/boxingvi/v10-full-v1/quality/V10/policy-index.json",
}


def load_transfer_config(path):
    c = yaml.safe_load(Path(path).read_text())
    fields = {
        "schema_version",
        "reference_registry",
        "frozen_study",
        "seeds",
        "checkpoint",
        "tail_threshold_source",
        "severe_tail_threshold_torso_per_second",
        "recordings",
    }
    if not isinstance(c, dict) or set(c) != fields:
        raise ValueError("Invalid transfer configuration")
    if (
        c["schema_version"] != "velocity-transfer-1.0"
        or c["seeds"] != list(SEEDS)
        or c["checkpoint"] != "best"
        or c["frozen_study"] != "reports/boxingvi/level1-velocity-v1"
        or c["reference_registry"] != "configs/level1_reference.yaml"
        or c["tail_threshold_source"] != "accepted_V4_training_velocity_p99"
        or c["severe_tail_threshold_torso_per_second"] != 50.0
        or [s.get("recording") for s in c["recordings"]] != list(RECORDINGS)
    ):
        raise ValueError(
            "Frozen V4-only study, best V5 checkpoints and diagnostic recordings required"
        )
    for spec in c["recordings"]:
        v = spec["recording"]
        if (
            set(spec)
            != {
                "recording",
                "source_group_id",
                "related_recordings_guard",
                "dataset",
                "policy",
                "pose_run",
            }
            or spec["dataset"] != f"data/processed/boxingvi/{v}/candidates-full-v1"
            or spec["pose_run"] != f"data/processed/boxingvi/{v}/pose-full-image-v1"
            or spec["policy"] != POLICY_PATHS[v]
            or (spec["source_group_id"], spec["related_recordings_guard"]) != GROUPS[v]
        ):
            raise ValueError("Recording paths or unresolved source guards changed")
    return c


def supported_metrics(result):
    """Keep the original six-class result; separately describe evaluable class support."""
    active = [c for c in CLASSES if result["per_class"][c]["support"]]
    if not active:
        raise ValueError("No eligible labeled examples")
    return {
        **result,
        "supported_classes": active,
        "supported_macro_f1": float(np.mean([result["per_class"][c]["f1"] for c in active])),
        "recall_with_absence": {
            c: result["per_class"][c]["recall"] if c in active else None for c in CLASSES
        },
    }


def source_support(dataset):
    root = Path(dataset.provenance["dataset"])
    rows = [json.loads(s) for s in (root / "candidates.jsonl").read_text().splitlines()]
    result = {}
    for label in CLASSES:
        source = [r for r in rows if r["punch_type_provisional"] == label]
        eligible = [r for r in source if r["annotation_eligible"]]
        retained = dataset.provenance["class_counts"][label]
        result[label] = {
            "source_annotations": len(source),
            "annotation_eligible": len(eligible),
            "retained": retained,
            "annotation_excluded": len(source) - len(eligible),
            "pose_rejected": len(eligible) - retained,
            "status": "present"
            if retained
            else (
                "absent_from_source_annotations"
                if not source
                else "no_retained_examples_after_exclusions"
            ),
        }
    return result


def clip_diagnostics(dataset, threshold, severe):
    """Descriptive tails only: never mask, clip, smooth, relabel or select model inputs."""
    rows, largest = [], []
    for c in dataset.clips:
        v = c["x"][:, 41:].numpy().reshape(-1, 8, 3)
        t, j = np.nonzero(v[..., 2])
        speed = np.linalg.norm(v[t, j, :2], axis=-1)
        times = exact_times(c)
        coverage = c["mask_x"][:, :8].numpy().mean(0)
        rows.append(
            {
                "source_row": c["source_row"],
                "class": CLASSES[c["label"]],
                "frames": len(v),
                "center_span_seconds": float(times[-1] - times[0]),
                "original_mask_coverage": coverage.tolist(),
                "velocity_valid_fraction": float(v[..., 2].sum() / max(1, (len(v) - 1) * 8)),
                "valid_joint_pairs": len(speed),
                "tail_pairs": int((speed > threshold).sum()),
                "severe_tail_pairs": int((speed > severe).sum()),
                "max_speed": float(speed.max()) if len(speed) else None,
                "median_speed": float(np.median(speed)) if len(speed) else None,
                "speed_p99": float(np.quantile(speed, 0.99)) if len(speed) else None,
            }
        )
        for k in np.argsort(speed)[-3:]:
            i = int(t[k])
            largest.append(
                {
                    "source_row": c["source_row"],
                    "class": CLASSES[c["label"]],
                    "core_joint_index": int(j[k]),
                    "previous_frame_index": int(c["raw"]["frame_index"][i - 1]),
                    "frame_index": int(c["raw"]["frame_index"][i]),
                    "pts_seconds_exact": str(times[i]),
                    "dt_seconds_exact": str(times[i] - times[i - 1]),
                    "speed_torso_per_second": float(speed[k]),
                }
            )

    def describe(group):
        pairs = sum(r["valid_joint_pairs"] for r in group)
        return {
            "clips": len(group),
            "valid_joint_pairs": pairs,
            "tail_pairs": sum(r["tail_pairs"] for r in group),
            "tail_pair_fraction": sum(r["tail_pairs"] for r in group) / pairs if pairs else None,
            "severe_tail_pairs": sum(r["severe_tail_pairs"] for r in group),
            "clips_with_tail": sum(r["tail_pairs"] > 0 for r in group),
            "clips_with_severe_tail": sum(r["severe_tail_pairs"] > 0 for r in group),
            "max_speed": max(
                (r["max_speed"] for r in group if r["max_speed"] is not None), default=None
            ),
            "median_clip_span_seconds": float(np.median([r["center_span_seconds"] for r in group]))
            if group
            else None,
            "median_clip_frames": float(np.median([r["frames"] for r in group])) if group else None,
        }

    labels = np.array([r["class"] for r in rows])
    coverage = np.array([r["original_mask_coverage"] for r in rows])
    length_bins = np.digitize([r["frames"] for r in rows], [8, 15])
    duration_bins = np.digitize([r["center_span_seconds"] for r in rows], [0.3, 0.6, 1.0])
    mi = [
        mutual_information(np.minimum((coverage[:, j] * 4).astype(int), 3), labels)
        for j in range(8)
    ]
    return {
        "threshold": threshold,
        "severe_threshold": severe,
        "threshold_origin": "frozen accepted V4 training-pair p99; severe 50 predeclared",
        "all": describe(rows),
        "by_class": {c: describe([r for r in rows if r["class"] == c]) for c in CLASSES},
        "clip_metadata": rows,
        "largest_pairs": sorted(largest, key=lambda r: -r["speed_torso_per_second"])[:12],
        "shortcut_associations": {
            "length_bins": [8, 15],
            "duration_bins_seconds": [0.3, 0.6, 1.0],
            "length_class_mi_bits": mutual_information(length_bins, labels),
            "duration_class_mi_bits": mutual_information(duration_bins, labels),
            "joint_mask_coverage_class_mi_bits": mi,
            "limitation": (
                "Descriptive fixed-bin associations in dependent scripted clips; "
                "no causal/independent significance claim."
            ),
        },
    }


def error_associations(metadata, predictions):
    """Conditional error rates on unchanged inputs, stratified by class."""
    lookup = {p["source_row"]: p for p in predictions}
    if len(lookup) != len(predictions) or set(lookup) != {r["source_row"] for r in metadata}:
        raise ValueError("Diagnostic metadata/prediction examples differ")

    def describe(rows):
        return {
            "n": len(rows),
            "errors": sum(
                lookup[r["source_row"]]["target"] != lookup[r["source_row"]]["prediction"]
                for r in rows
            ),
        }

    def group(rows):
        return {
            "all": describe(rows),
            "with_tail": describe([r for r in rows if r["tail_pairs"]]),
            "without_tail": describe([r for r in rows if not r["tail_pairs"]]),
            "with_severe_tail": describe([r for r in rows if r["severe_tail_pairs"]]),
            "shorter_than_15_frames": describe([r for r in rows if r["frames"] < 15]),
            "at_least_15_frames": describe([r for r in rows if r["frames"] >= 15]),
            "all_original_core_valid": describe(
                [r for r in rows if min(r["original_mask_coverage"]) == 1]
            ),
            "some_original_core_invalid": describe(
                [r for r in rows if min(r["original_mask_coverage"]) < 1]
            ),
        }

    return {
        "all": group(metadata),
        "by_class": {c: group([r for r in metadata if r["class"] == c]) for c in CLASSES},
    }


def ast_item(path, name):
    nodes = ast.parse(Path(path).read_text()).body
    for n in nodes:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.dump(n)
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in n.targets
        ):
            return ast.dump(n.value)
    raise ValueError(f"Missing frozen implementation: {name}")


def load_frozen_models(study):
    models, frozen = {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            folder = study / f"seed-{seed}" / arm
            source = folder / "source/src/boxing_motion_ai/training"
            if sha256(source / "models.py") != sha256(Path(__file__).with_name("models.py")):
                raise ValueError("Frozen architecture differs")
            if ast_item(source / "data.py", "features") != ast_item(
                Path(__file__).with_name("data.py"), "features"
            ):
                raise ValueError("Frozen normalization differs")
            if arm == "xy_velocity":
                for name in ("joint_velocities", "source_gap_limit", "VELOCITY_POLICY"):
                    if ast_item(source / "velocity.py", name) != ast_item(
                        Path(__file__).with_name("velocity.py"), name
                    ):
                        raise ValueError("Frozen velocity handling differs")
            old = json.loads((folder / "results.json").read_text())
            provenance = json.loads((folder / "provenance.json").read_text())
            for name in MODELS:
                checkpoint = folder / name / "best.pt"
                state = torch.load(checkpoint, weights_only=True)
                cfg = state["config"]
                if (
                    cfg["train"]["recording"] != "V4"
                    or cfg["development"]["recording"] != "V5"
                    or cfg["seed"] != seed
                    or cfg["feature_representation"] != arm
                    or cfg["mirror_probability"] != 0.5
                    or cfg.get("visibility_probability", 0) != 0
                    or cfg.get("context_seconds", 0) != 0
                    or state["epoch"] != old[name]["selected_epoch"]
                    or state["provenance_sha256"] != sha256(folder / "provenance.json")
                    or state["feature_version"]
                    != (VELOCITY_VERSION if arm == "xy_velocity" else FEATURE_VERSION)
                ):
                    raise ValueError("Checkpoint source, selection or representation differs")
                channels = (
                    (65 if arm == "xy_velocity" else 41)
                    if name == "tcn"
                    else (16 if arm == "xy_velocity" else 8)
                )
                model = CausalTCN(channels, cfg["hidden"], cfg["dropout"])
                model.load_state_dict(state["model_state"])
                model.eval()
                key = f"{seed}/{arm}/{name}"
                models[key] = model
                frozen[key] = {
                    "checkpoint": str(checkpoint),
                    "sha256": sha256(checkpoint),
                    "epoch": state["epoch"],
                    "channels": channels,
                    "provenance_sha256": sha256(folder / "provenance.json"),
                    "source_train_hashes": provenance["train"]["hashes"],
                    "source_development_hashes": provenance["development"]["hashes"],
                }
    return models, frozen


def run(config_path, output, progress=print):
    config_path, output = Path(config_path), Path(output)
    cfg = load_transfer_config(config_path)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a fresh output below reports/")
    registry = yaml.safe_load(Path(cfg["reference_registry"]).read_text())
    if (
        registry["accepted_study"] != cfg["frozen_study"]
        or registry["representation"] != "xy_velocity"
        or registry["training_recordings"] != ["V4"]
        or registry["development_recordings"] != ["V5"]
        or registry["checkpoint"] != "best"
        or registry["seeds"] != list(SEEDS)
    ):
        raise ValueError("Reference must be original V4-only, V5-selected velocity family")
    study = Path(cfg["frozen_study"])
    preserved = json.loads((study / "preserved-inputs.json").read_text())
    preserved = {p: d for p, d in preserved.items() if not p.startswith("src/")}
    preserved.update({str(p): sha256(p) for p in study.rglob("*") if p.is_file()})
    if any("/V9/" in p or "/V9." in p for p in preserved):
        raise ValueError("Reserved input outside transfer scope")
    if any(sha256(p) != d for p, d in preserved.items()):
        raise ValueError("Prior artifacts changed")
    seed_all(0, 2)
    models, frozen = load_frozen_models(study)
    # Reproduce frozen V5 predictions first; never reselect checkpoints on new recordings.
    old_cfg = yaml.safe_load((study / f"seed-{SEEDS[0]}-xy.yaml").read_text())
    train = PoseClips(old_cfg["train"], "V4")
    weights = torch.tensor(
        [train.provenance["class_counts"][c] for c in CLASSES], dtype=torch.float32
    )
    weights = len(train) / (6 * weights)
    base_dev = PoseClips(old_cfg["development"], "V5")
    devs = {"xy": base_dev, "xy_velocity": VelocityClips(base_dev)}
    verification = {}
    for key, model in models.items():
        seed, arm, name = key.split("/")
        _, detail = evaluate(
            model,
            DataLoader(devs[arm], batch_size=32, collate_fn=collate),
            "x" if name == "tcn" else "mask_x",
            weights,
        )
        previous = json.loads(
            (study / f"seed-{seed}" / arm / name / "development_predictions.json").read_text()
        )
        error = float(
            np.abs(
                np.array([r["probabilities"] for r in detail])
                - np.array([r["probabilities"] for r in previous])
            ).max()
        )
        if error > 1e-6 or [r["prediction"] for r in detail] != [r["prediction"] for r in previous]:
            raise ValueError("Frozen V5 predictions did not reproduce")
        verification[key] = error
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(config_path.read_text())
    write_json(output / "preserved-inputs.json", preserved)
    write_json(output / "reference-checkpoints.json", frozen)
    write_json(output / "V5-reproduction.json", verification)
    threshold = json.loads((study / "velocity-diagnostics.json").read_text())["V4"]["speed"]["p99"]
    result = {
        "schema_version": "velocity-transfer-result-1.0",
        "training_performed": False,
        "reference_registry_sha256": sha256(cfg["reference_registry"]),
        "selection": "Frozen best V5 checkpoints; no external selection or tuning",
        "recordings": {},
        "warning": (
            "Repeated diagnostic sources with provisional labels, unknown identities. "
            "V7/V8 one source group; V1/V6 guard preserved. V9 untouched."
        ),
    }
    for spec in cfg["recordings"]:
        video = spec["recording"]
        progress(f"Evaluating frozen models on {video}")
        ds = PoseClips(
            {k: spec[k] for k in ("dataset", "policy", "pose_run", "recording")},
            video,
            diagnostic=True,
        )
        vel = VelocityClips(ds, diagnostic=True)
        if ds.provenance["source_rows"] != vel.provenance["source_rows"]:
            raise ValueError("Paired examples differ")
        d = {
            "source_group_id": spec["source_group_id"],
            "related_recordings_guard": spec["related_recordings_guard"],
            "provenance": vel.provenance,
            "source_support": source_support(ds),
            "diagnostics": clip_diagnostics(
                vel, threshold, cfg["severe_tail_threshold_torso_per_second"]
            ),
            "runs": {},
        }
        for seed in SEEDS:
            d["runs"][str(seed)] = {}
            for arm, data in (("xy", ds), ("xy_velocity", vel)):
                d["runs"][str(seed)][arm] = {}
                for name in MODELS:
                    metric, detail = evaluate(
                        models[f"{seed}/{arm}/{name}"],
                        DataLoader(data, batch_size=32, collate_fn=collate),
                        "x" if name == "tcn" else "mask_x",
                        weights,
                    )
                    d["runs"][str(seed)][arm][name] = supported_metrics(metric)
                    path = output / video / f"seed-{seed}" / arm
                    path.mkdir(parents=True, exist_ok=True)
                    write_json(path / f"{name}-predictions.json", detail)
                    write_json(
                        path / f"{name}-associations.json",
                        error_associations(d["diagnostics"]["clip_metadata"], detail),
                    )
        d["aggregate"] = {}
        for name in MODELS:
            per_model = {}
            for arm in ARMS:
                rs = [d["runs"][str(seed)][arm][name] for seed in SEEDS]
                per_model[arm] = {
                    k: summarize([r[k] for r in rs])
                    for k in ("accuracy", "macro_f1", "supported_macro_f1")
                }
                per_model[arm]["per_class_recall"] = {
                    c: summarize([r["per_class"][c]["recall"] for r in rs])
                    if rs[0]["per_class"][c]["support"]
                    else None
                    for c in CLASSES
                }
                per_model[arm]["mean_confusion_counts"] = np.mean(
                    [r["confusion_matrix"] for r in rs], axis=0
                ).tolist()
            for measure in ("accuracy", "macro_f1", "supported_macro_f1"):
                per_model[f"paired_{measure}_gain"] = summarize(
                    [
                        d["runs"][str(s)]["xy_velocity"][name][measure]
                        - d["runs"][str(s)]["xy"][name][measure]
                        for s in SEEDS
                    ]
                )
            d["aggregate"][name] = per_model
        result["recordings"][video] = d
        write_json(output / f"{video}-result.json", d)
        for name, digest in ds.provenance["hashes"].items():
            if sha256(Path(spec["dataset"]) / name) != digest:
                raise ValueError("Evaluation data changed")
    write_json(output / "summary.json", result)
    write_report(output, result)
    if any(sha256(p) != d for p, d in preserved.items()):
        raise ValueError("Previous artifacts changed")
    return result


def write_report(output, result):
    lines = [
        "# Frozen velocity transfer diagnostics",
        "",
        result["warning"],
        "",
        "Macro-F1 six averages all six classes. Supported averages only classes with "
        "eligible ground truth; predictions into absent classes remain errors.",
        "",
        "| Recording | Input | XY accuracy | Velocity accuracy | XY F1 six / supported "
        "| Velocity F1 six / supported |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for v, d in result["recordings"].items():
        for name, a in d["aggregate"].items():
            x, y = a["xy"], a["xy_velocity"]
            lines.append(
                f"| {v} | {name} | {x['accuracy']['mean']:.4f} | {y['accuracy']['mean']:.4f} | "
                f"{x['macro_f1']['mean']:.4f} / {x['supported_macro_f1']['mean']:.4f} | "
                f"{y['macro_f1']['mean']:.4f} / {y['supported_macro_f1']['mean']:.4f} |"
            )
    for v, d in result["recordings"].items():
        lines += [
            "",
            f"## {v} class support and full-feature recall",
            "",
            "| Class | Source | Retained | Status | XY recall | Velocity recall |",
            "|---|---:|---:|---|---:|---:|",
        ]
        for c, s in d["source_support"].items():
            recalls = [d["aggregate"]["tcn"][a]["per_class_recall"][c] for a in ARMS]
            cells = ["absent" if r is None else f"{r['mean']:.4f}" for r in recalls]
            lines.append(
                f"| {c} | {s['source_annotations']} | {s['retained']} | {s['status']} | "
                f"{cells[0]} | {cells[1]} |"
            )
        for arm in ARMS:
            lines += [
                "",
                f"### {arm}: mean confusion counts (not extra independent examples)",
                "",
                "Rows true / columns predicted: " + ", ".join(CLASSES),
                "",
                "```json",
                json.dumps(d["aggregate"]["tcn"][arm]["mean_confusion_counts"]),
                "```",
            ]
    lines += [
        "",
        "No fitting, fine-tuning, test-time augmentation, threshold changes or model selection. "
        "See summary.json and per-recording prediction/association files.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
