"""Fixed-budget CPU experiments. V5 selects epochs; V9 is never opened."""

import csv
import json
import platform
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ..vision.pipeline import code_provenance, sha256
from .data import CLASSES, FEATURE_VERSION, PoseClips, collate
from .models import CausalTCN, StaticBaseline


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metrics(target, prediction):
    target, prediction = np.asarray(target, dtype=int), np.asarray(prediction, dtype=int)
    if not len(target) or target.shape != prediction.shape:
        raise ValueError("Metrics require nonempty matching label vectors")
    if not (np.isin(target, range(6)).all() and np.isin(prediction, range(6)).all()):
        raise ValueError("Invalid six-class prediction")
    cm = np.bincount(target * 6 + prediction, minlength=36).reshape(6, 6)
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(cm.diagonal(), predicted, out=np.zeros(6), where=predicted != 0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(6), where=support != 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros(6), where=precision + recall != 0
    )
    return {
        "accuracy": float(cm.trace() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": cm.tolist(),
        "labels": list(CLASSES),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASSES)
        },
    }


def seed_all(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def evaluate(model, loader, feature_key, weights):
    model.eval()
    targets, predictions, rows, probabilities = [], [], [], []
    numerator, denominator = 0.0, 0.0
    with torch.no_grad():
        for b in loader:
            logits = model(b[feature_key], b["time_mask"])
            labels = b["labels"]
            numerator += float(
                nn.functional.cross_entropy(logits, labels, weight=weights, reduction="sum")
            )
            denominator += float(weights[labels].sum())
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(-1).tolist())
            probabilities.extend(logits.softmax(-1).tolist())
            rows.extend(b["source_rows"])
    result = metrics(targets, predictions)
    result["weighted_loss"] = numerator / denominator
    detail = [
        {"source_row": r, "target": CLASSES[t], "prediction": CLASSES[p], "probabilities": probs}
        for r, t, p, probs in zip(rows, targets, predictions, probabilities, strict=True)
    ]
    return result, detail


def load_config(path):
    config = yaml.safe_load(path.read_text())
    required = {
        "schema_version",
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "hidden",
        "dropout",
        "threads",
        "limit_per_class",
        "train",
        "development",
        "runs",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Unexpected or missing experiment configuration fields")
    if config["schema_version"] != "level1-experiment-1.0":
        raise ValueError("Unsupported experiment version")
    for k in ("seed", "epochs", "batch_size", "hidden", "threads"):
        if type(config[k]) is not int or config[k] < (0 if k == "seed" else 1):
            raise ValueError(f"Invalid {k}")
    if not all(np.isfinite(config[k]) for k in ("learning_rate", "weight_decay", "dropout")):
        raise ValueError("Nonfinite optimizer configuration")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0 or not 0 <= config["dropout"] < 1:
        raise ValueError("Invalid optimizer/dropout configuration")
    if config["limit_per_class"] is not None and (
        type(config["limit_per_class"]) is not int or config["limit_per_class"] < 1
    ):
        raise ValueError("Invalid smoke subset limit")
    if config["runs"] != ["static", "tcn", "static_mask_only", "tcn_mask_only"]:
        raise ValueError("Both baselines and both matched mask ablations are required")
    for split, recording in (("train", "V4"), ("development", "V5")):
        spec = config[split]
        if (
            set(spec) != {"recording", "dataset", "policy", "pose_run"}
            or spec["recording"] != recording
        ):
            raise ValueError("Experiments require V4 train / V5 development; V9 reserved")
    return config


def run_experiment(config_path, output, progress=print):
    config_path, output = Path(config_path), Path(output)
    config = load_config(config_path)
    # Outputs are constrained to the ignored report tree; never overwrite data or prior runs.
    report_root = Path("reports").resolve()
    if not output.resolve().is_relative_to(report_root) or output.resolve() == report_root:
        raise ValueError("Experiment output must be a new directory below reports/")
    if output.exists():
        raise ValueError("Output already exists; choose a fresh run directory")
    progress("Checking immutable datasets and approved policy selections...")
    train = PoseClips(config["train"], "V4", config["limit_per_class"])
    dev = PoseClips(config["development"], "V5", config["limit_per_class"])
    if train.provenance["hashes"]["poses.npz"] == dev.provenance["hashes"]["poses.npz"]:
        raise ValueError("Train and development source poses are identical")
    output.mkdir(parents=True)
    shutil.copyfile(config_path, output / "config.yaml")
    provenance = {
        "schema_version": "level1-run-1.0",
        "feature_version": FEATURE_VERSION,
        "config_sha256": sha256(config_path),
        "code": code_provenance(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "seed": config["seed"],
        "deterministic_algorithms": True,
        "train": train.provenance,
        "development": dev.provenance,
        "reserved_evaluation": "V9: not opened or used",
        "scope": "provisional positive-clip Level 1; unknown background omitted",
        "selection": "maximum V5 macro-F1; ties retain earlier epoch",
        "normalization": "frame-local aspect-correct XY, hip center / torso length",
        "features": "8 core joints: XY, visibility/presence, usable mask; torso-valid bit",
        "ablation": "8 original binary core masks only; no coordinates/confidence/timing",
        "timing": "native frames; integer PTS/time base retained, never model inputs",
        "limitations": [
            "unknown boxer/session identities; no boxer-independent claim",
            "provisional labels/boundaries and incomplete scene inventory",
            "one seed; V5 repeatedly selected development, not unbiased test",
            "complete-clip pooling; no event detection or early recognition",
            "native FPS differs: 15-frame TCN field has different duration",
            "clip duration and mask coverage may act as shortcuts",
        ],
    }
    source_root = Path(__file__).resolve().parents[3]
    source_files = sorted((source_root / "src/boxing_motion_ai").rglob("*.py"))
    snapshot = output / "source"
    for source in source_files:
        relative = source.relative_to(source_root)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        provenance["code"]["sha256"][relative.as_posix()] = sha256(source)
    # Preserve lock/config even when experimenting from a dirty working tree.
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(source_root / name, snapshot / name)
    write_json(output / "provenance.json", provenance)
    counts = torch.tensor(
        [train.provenance["class_counts"][c] for c in CLASSES], dtype=torch.float32
    )
    weights = len(train) / (6 * counts)  # V4 only. No oversampling/double reweighting.
    results = {}
    for name in config["runs"]:
        seed_all(config["seed"], config["threads"])
        feature_key = "mask_x" if name.endswith("mask_only") else "x"
        cls = StaticBaseline if name.startswith("static") else CausalTCN
        channels = train[0][feature_key].shape[1]
        model = cls(channels, config["hidden"], config["dropout"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        generator = torch.Generator().manual_seed(config["seed"])
        shuffled = DataLoader(
            train,
            batch_size=config["batch_size"],
            shuffle=True,
            generator=generator,
            collate_fn=collate,
            num_workers=0,
        )
        train_eval = DataLoader(train, batch_size=config["batch_size"], collate_fn=collate)
        dev_eval = DataLoader(dev, batch_size=config["batch_size"], collate_fn=collate)
        directory = output / name
        directory.mkdir()
        curves, best = [], -1.0
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            for b in shuffled:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(
                    model(b[feature_key], b["time_mask"]), b["labels"], weight=weights
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tr, _ = evaluate(model, train_eval, feature_key, weights)
            va, _ = evaluate(model, dev_eval, feature_key, weights)
            curves.append(
                {
                    "epoch": epoch,
                    "train_loss": tr["weighted_loss"],
                    "dev_loss": va["weighted_loss"],
                    "train_accuracy": tr["accuracy"],
                    "dev_accuracy": va["accuracy"],
                    "train_macro_f1": tr["macro_f1"],
                    "dev_macro_f1": va["macro_f1"],
                }
            )
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "seed": config["seed"],
                "config": config,
                "run": name,
                "feature_version": FEATURE_VERSION,
                "torch_rng_state": torch.get_rng_state(),
                "loader_rng_state": generator.get_state(),
                "provenance_sha256": sha256(output / "provenance.json"),
            }
            if va["macro_f1"] > best:
                best, best_epoch = va["macro_f1"], epoch
                torch.save(checkpoint, directory / "best.pt")
            if epoch == config["epochs"]:
                torch.save(checkpoint, directory / "last.pt")
                final = {"train": tr, "development": va}
            if epoch == 1 or epoch % 10 == 0 or epoch == config["epochs"]:
                progress(
                    f"{name} epoch {epoch}: train F1={tr['macro_f1']:.3f}; "
                    f"V5 F1={va['macro_f1']:.3f}"
                )
        model.load_state_dict(torch.load(directory / "best.pt", weights_only=True)["model_state"])
        tr, tr_detail = evaluate(model, train_eval, feature_key, weights)
        va, va_detail = evaluate(model, dev_eval, feature_key, weights)
        result = {
            "selected_epoch": best_epoch,
            "train": tr,
            "development": va,
            "last_epoch": final,
            "parameters": sum(p.numel() for p in model.parameters()),
            "class_weights_train_only": weights.tolist(),
        }
        write_json(directory / "metrics.json", result)
        write_json(directory / "train_predictions.json", tr_detail)
        write_json(directory / "development_predictions.json", va_detail)
        with (directory / "curves.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(curves[0]))
            writer.writeheader()
            writer.writerows(curves)
        plot(directory, curves, va["confusion_matrix"])
        results[name] = result
    majority = int(counts.argmax())
    results["v4_majority_reference"] = metrics(
        [c["label"] for c in dev.clips], [majority] * len(dev)
    )
    write_json(output / "results.json", results)
    report(output, results, provenance)
    # Detect concurrent accidental input changes; never write source artifacts.
    for split in (train, dev):
        for filename, digest in split.provenance["hashes"].items():
            if sha256(Path(split.provenance["dataset"]) / filename) != digest:
                raise ValueError("Dataset changed during experiment")
    progress(f"Complete: {output / 'results.md'}")
    return results


def plot(directory, curves, confusion):
    import os

    os.environ.setdefault("MPLCONFIGDIR", str((directory.parent / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ("train", "dev"):
        axes[0].plot(
            [c["epoch"] for c in curves], [c[f"{split}_loss"] for c in curves], label=split
        )
        axes[1].plot(
            [c["epoch"] for c in curves], [c[f"{split}_macro_f1"] for c in curves], label=split
        )
    for ax, title in zip(axes, ("Weighted loss", "Macro-F1"), strict=True):
        ax.set(xlabel="Epoch", title=title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "curves.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(confusion, cmap="Blues")
    ax.set(
        xticks=range(6),
        yticks=range(6),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted",
        ylabel="V5 provisional source label",
        title="Development confusion",
    )
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(confusion[i][j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(directory / "confusion.png")
    plt.close(fig)


def report(output, results, provenance):
    lines = [
        "# Initial Level 1 experiment",
        "",
        "V4 training; V5 development. V9 was not opened. Labels and boundaries remain "
        "provisional. This is positive-clip recognition, with no no_punch examples.",
        "",
        "| Run | Selected epoch | V4 accuracy | V4 macro-F1 | V5 accuracy | V5 macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        if "selected_epoch" not in r:
            continue
        tr, va = r["train"], r["development"]
        lines.append(
            f"| {name} | {r['selected_epoch']} | {tr['accuracy']:.4f} | "
            f"{tr['macro_f1']:.4f} | {va['accuracy']:.4f} | {va['macro_f1']:.4f} |"
        )
    majority = results["v4_majority_reference"]
    lines += [
        "",
        f"V4-majority reference on V5: accuracy {majority['accuracy']:.4f}; "
        f"macro-F1 {majority['macro_f1']:.4f}.",
        "",
        "Each run contains per-class precision/recall/F1/support, confusion matrix, "
        "predictions with source rows, loss/F1 curves, best and final checkpoints. "
        "Checkpoint selection uses V5 macro-F1 only (ties keep the earlier epoch).",
        "",
        "## Limits",
        "",
    ]
    lines += [f"- {s}" for s in provenance["limitations"]]
    (output / "results.md").write_text("\n".join(lines) + "\n")
