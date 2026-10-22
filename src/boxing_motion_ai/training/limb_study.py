"""Matched partial-observation arm-geometry study; V4 train, V5 select, V9 unopened."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..vision.pipeline import sha256
from .data import CLASSES, PoseClips, collate
from .experiment import evaluate, load_config, run_experiment, seed_all, write_json
from .limbs import LIMB_POLICY, LimbClips
from .mirror_study import summarize
from .models import CausalTCN
from .velocity import VelocityClips
from .velocity_transfer import (
    MODELS,
    SEEDS,
    load_frozen_models,
    load_transfer_config,
    source_support,
    supported_metrics,
)

ARMS = ("xy_velocity", "xy_velocity_limb")


def load_study(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg != {
        "schema_version": "limb-study-1.0",
        "base_config": "configs/level1_initial.yaml",
        "frozen_reference": "reports/boxingvi/level1-velocity-v1",
        "coverage_report": "reports/boxingvi/level1-limb-readiness-v2/summary.json",
        "transfer_config": "configs/level1_velocity_transfer.yaml",
        "seeds": list(SEEDS),
        "mirror_probability": 0.5,
        "visibility_probability": 0.0,
        "context_seconds": 0.0,
    }:
        raise ValueError("Use frozen V4-only three-seed mirror study; V9 reserved")
    return cfg


def aggregate(runs):
    result = {}
    for name in MODELS:
        a = {}
        for arm in ARMS:
            rs = [runs[str(seed)][arm][name] for seed in SEEDS]
            a[arm] = {
                k: summarize([r[k] for r in rs])
                for k in ("accuracy", "macro_f1", "supported_macro_f1")
            }
            a[arm]["per_class_recall"] = {
                c: summarize([r["per_class"][c]["recall"] for r in rs])
                if rs[0]["per_class"][c]["support"]
                else None
                for c in CLASSES
            }
            a[arm]["mean_confusion_counts"] = np.mean(
                [r["confusion_matrix"] for r in rs], axis=0
            ).tolist()
        for measure in ("accuracy", "macro_f1", "supported_macro_f1"):
            a[f"paired_{measure}_gain"] = summarize(
                [
                    runs[str(s)][ARMS[1]][name][measure] - runs[str(s)][ARMS[0]][name][measure]
                    for s in SEEDS
                ]
            )
        result[name] = a
    return result


def run(config_path, output, smoke=False, progress=print):
    config_path, output = Path(config_path), Path(output)
    cfg = load_study(config_path)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Choose a fresh ignored output below reports/")
    base = load_config(Path(cfg["base_config"]))
    reference = Path(cfg["frozen_reference"])
    transfer = load_transfer_config(Path(cfg["transfer_config"]))
    coverage = json.loads(Path(cfg["coverage_report"]).read_text())
    if list(coverage["recordings"]) != ["V4", "V5", "V6", "V7", "V8", "V10"]:
        raise ValueError("Coverage investigation incomplete")
    # Allow partial-observation testing, without a new quality/selection threshold.
    # Never mistake either-arm coverage for recovered hidden arm measurements.
    preserved_root = Path("reports/boxingvi/level1-velocity-transfer-v1")
    preserved = json.loads((preserved_root / "preserved-inputs.json").read_text())
    for root in (preserved_root, Path(cfg["coverage_report"]).parent):
        preserved.update({str(p): sha256(p) for p in root.rglob("*") if p.is_file()})
    if any("/V9/" in p or "/V9." in p for p in preserved):
        raise ValueError("Reserved recording outside study")
    if any(sha256(p) != d for p, d in preserved.items()):
        raise ValueError("Prior artifacts changed")
    for seed in SEEDS:
        old = yaml.safe_load((reference / f"seed-{seed}/xy_velocity/config.yaml").read_text())
        expected = {
            **base,
            "schema_version": "level1-experiment-1.5",
            "seed": seed,
            "feature_representation": "xy_velocity",
            "mirror_probability": 0.5,
            "runs": list(MODELS),
        }
        if old != expected:
            raise ValueError("Frozen training configuration differs")
    output.mkdir(parents=True)
    (output / "study.yaml").write_text(config_path.read_text())
    write_json(output / "preserved-inputs.json", preserved)
    trained = {}
    for seed in SEEDS[:1] if smoke else SEEDS:
        config = {
            **base,
            "schema_version": "level1-experiment-1.6",
            "seed": seed,
            "feature_representation": ARMS[1],
            "mirror_probability": 0.5,
            "runs": list(MODELS),
        }
        if smoke:
            config.update(epochs=2, limit_per_class=2)
        path = output / f"seed-{seed}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        folder = output / f"seed-{seed}" / ARMS[1]
        trained[str(seed)] = run_experiment(path, folder, progress)
        if not smoke:
            old_folder = reference / f"seed-{seed}/xy_velocity"
            old = json.loads((old_folder / "results.json").read_text())
            for model in MODELS:
                if (
                    trained[str(seed)][model]["shared_initial_model_sha256"]
                    != old[model]["shared_initial_model_sha256"]
                ):
                    raise ValueError("Shared initial parameters differ")
                if json.loads((folder / model / "augmentation.json").read_text()) != json.loads(
                    (old_folder / model / "augmentation.json").read_text()
                ):
                    raise ValueError("Data order/mirror decisions differ from reference")
            provenance = json.loads((folder / "provenance.json").read_text())
            prior = json.loads((old_folder / "provenance.json").read_text())
            for split in ("train", "development"):
                if (
                    provenance[split]["hashes"] != prior[split]["hashes"]
                    or provenance[split]["source_rows"] != prior[split]["source_rows"]
                ):
                    raise ValueError("Paired training/development examples differ")
    write_json(output / "training.json", trained)
    if smoke:
        return trained
    # All epochs are selected on V5 before any external evaluation.
    seed_all(0, 2)
    frozen, frozen_manifest = load_frozen_models(reference)
    models = {k: m for k, m in frozen.items() if k.split("/")[1] == ARMS[0]}
    checkpoints = {k: m for k, m in frozen_manifest.items() if k.split("/")[1] == ARMS[0]}
    for seed in SEEDS:
        for name in MODELS:
            path = output / f"seed-{seed}" / ARMS[1] / name / "best.pt"
            state = torch.load(path, weights_only=True)
            model = CausalTCN(99 if name == "tcn" else 28, base["hidden"], base["dropout"])
            model.load_state_dict(state["model_state"])
            model.eval()
            key = f"{seed}/{ARMS[1]}/{name}"
            models[key] = model
            checkpoints[key] = {
                "checkpoint": str(path),
                "sha256": sha256(path),
                "epoch": state["epoch"],
            }
    write_json(output / "frozen-checkpoints.json", checkpoints)
    train = PoseClips(base["train"], "V4")
    weights = len(train) / (
        6
        * torch.tensor([train.provenance["class_counts"][c] for c in CLASSES], dtype=torch.float32)
    )
    result = {
        "schema_version": "limb-study-result-1.0",
        "policy": LIMB_POLICY,
        "reference_promoted": False,
        "recordings": {},
        "selection": "V4-only training, best V5 macro-F1 per seed; external frozen evaluation",
        "warning": "Partial observed-arm geometry only; unverified labels/orientation. "
        "V7/V8 one source group; V1/V6 guard retained. V9 unopened.",
    }
    for spec in (base["development"], *transfer["recordings"]):
        v = spec["recording"]
        diag = v != "V5"
        xy = PoseClips(spec, v, diagnostic=diag)
        vel = VelocityClips(xy, diagnostic=diag)
        limb = LimbClips(vel)
        if xy.provenance["hashes"] != coverage["recordings"][v]["provenance"]["hashes"]:
            raise ValueError("Data differs from investigated inputs")
        if any(
            not torch.equal(a["x"], b["x"][:, :65])
            or not torch.equal(a["mask_x"], b["mask_x"][:, :16])
            for a, b in zip(vel.clips, limb.clips, strict=True)
        ):
            raise ValueError("Existing input features changed")
        d = {
            "source_support": source_support(xy),
            "provenance": limb.provenance,
            "source_group_id": spec.get("source_group_id", "V5-development"),
            "related_recordings_guard": spec.get("related_recordings_guard", [v]),
            "runs": {},
        }
        for seed in SEEDS:
            d["runs"][str(seed)] = {}
            for arm, ds in ((ARMS[0], vel), (ARMS[1], limb)):
                d["runs"][str(seed)][arm] = {}
                for name in MODELS:
                    metric, detail = evaluate(
                        models[f"{seed}/{arm}/{name}"],
                        DataLoader(ds, batch_size=base["batch_size"], collate_fn=collate),
                        "x" if name == "tcn" else "mask_x",
                        weights,
                    )
                    if arm == ARMS[0]:
                        old_path = (
                            reference
                            / f"seed-{seed}/xy_velocity/{name}/development_predictions.json"
                            if v == "V5"
                            else preserved_root
                            / v
                            / f"seed-{seed}/xy_velocity/{name}-predictions.json"
                        )
                        previous = json.loads(old_path.read_text())
                        if detail != previous:
                            raise ValueError(
                                "Frozen reference predictions did not reproduce exactly"
                            )
                    d["runs"][str(seed)][arm][name] = supported_metrics(metric)
                    dest = output / v / f"seed-{seed}" / arm
                    dest.mkdir(parents=True, exist_ok=True)
                    write_json(dest / f"{name}-predictions.json", detail)
        d["aggregate"] = aggregate(d["runs"])
        result["recordings"][v] = d
        write_json(output / f"{v}-result.json", d)
        progress(f"{v}: all frozen paired predictions complete")
    write_json(output / "summary.json", result)
    write_report(output, result, trained)
    if any(sha256(p) != d for p, d in preserved.items()):
        raise ValueError("Prior artifacts changed")
    write_json(output / "preservation.json", {"verified_files": len(preserved), "changed": []})
    return result


def write_report(output, result, trained):
    lines = [
        "# Additional relative-arm features: matched Level 1 comparison",
        "",
        result["warning"],
        "",
        "Original six-class F1 and supported-class F1 are separate. "
        "Unsupported-class predictions still count as errors.",
        "",
        "| Recording | Model | Accuracy reference → limbs | F1 six reference → limbs | "
        "F1 supported reference → limbs | Paired supported gains |",
        "|---|---|---:|---:|---:|---|",
    ]
    for v, d in result["recordings"].items():
        for name, a in d["aggregate"].items():
            x, y = a[ARMS[0]], a[ARMS[1]]
            lines.append(
                f"| {v} | {name} | {x['accuracy']['mean']:.4f} → {y['accuracy']['mean']:.4f} | "
                f"{x['macro_f1']['mean']:.4f} → {y['macro_f1']['mean']:.4f} | "
                f"{x['supported_macro_f1']['mean']:.4f} → "
                f"{y['supported_macro_f1']['mean']:.4f} | "
                + ", ".join(f"{g:+.4f}" for g in a["paired_supported_macro_f1_gain"]["values"])
                + " |"
            )
        lines += [
            "",
            f"## {v} full-feature recalls",
            "",
            "| Class | Eligible | Status | Reference → limb recall |",
            "|---|---:|---|---:|",
        ]
        for c, s in d["source_support"].items():
            recalls = [d["aggregate"]["tcn"][arm]["per_class_recall"][c] for arm in ARMS]
            cell = (
                "unsupported"
                if recalls[0] is None
                else f"{recalls[0]['mean']:.4f} → {recalls[1]['mean']:.4f}"
            )
            lines.append(f"| {c} | {s['retained']} | {s['status']} | {cell} |")
        for name in MODELS:
            for arm in ARMS:
                lines += [
                    "",
                    f"### {name} {arm} mean confusion counts",
                    "",
                    "Rows true / columns predicted: " + ", ".join(CLASSES),
                    "",
                    "```json",
                    json.dumps(d["aggregate"][name][arm]["mean_confusion_counts"]),
                    "```",
                ]
    lines += [
        "",
        "## Selected epochs and overfitting",
        "",
        "| Seed | Model | Epoch | V4 F1 | V5 F1 | Final V4/V5 F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for seed, rs in trained.items():
        for name in MODELS:
            m = rs[name]
            lines.append(
                f"| {seed} | {name} | {m['selected_epoch']} | {m['train']['macro_f1']:.4f} | "
                f"{m['development']['macro_f1']:.4f} | "
                f"{m['last_epoch']['train']['macro_f1']:.4f}/"
                f"{m['last_epoch']['development']['macro_f1']:.4f} |"
            )
    (output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, args.smoke)
