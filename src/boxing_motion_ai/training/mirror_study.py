"""Matched three-seed mirror comparison; no other data/model intervention."""

import json
from pathlib import Path

import numpy as np
import yaml

from ..vision.pipeline import sha256
from .data import CLASSES
from .experiment import load_config, run_experiment, write_json


def summarize(values):
    a = np.asarray(values, dtype=float)
    return {
        "values": a.tolist(),
        "mean": float(a.mean()),
        "sample_std": float(a.std(ddof=1)) if len(a) > 1 else None,
    }


def aggregate(runs, seeds, arms=("unaugmented", "mirrored")):
    result = {"seeds": seeds, "by_model": {}}
    for model in ("tcn", "tcn_mask_only"):
        result["by_model"][model] = {}
        for arm in arms:
            selected = [runs[str(seed)][arm][model] for seed in seeds]
            result["by_model"][model][arm] = {
                "accuracy": summarize([r["development"]["accuracy"] for r in selected]),
                "macro_f1": summarize([r["development"]["macro_f1"] for r in selected]),
                "train_macro_f1": summarize([r["train"]["macro_f1"] for r in selected]),
                "last_train_macro_f1": summarize(
                    [r["last_epoch"]["train"]["macro_f1"] for r in selected]
                ),
                "last_dev_macro_f1": summarize(
                    [r["last_epoch"]["development"]["macro_f1"] for r in selected]
                ),
                "selected_epochs": [r["selected_epoch"] for r in selected],
                "per_class_recall": {
                    c: summarize([r["development"]["per_class"][c]["recall"] for r in selected])
                    for c in CLASSES
                },
                "confusion_mean_counts": np.mean(
                    [r["development"]["confusion_matrix"] for r in selected], axis=0
                ).tolist(),
            }
        original = result["by_model"][model][arms[0]]
        mirrored = result["by_model"][model][arms[1]]
        result["by_model"][model]["paired_macro_f1_gain"] = summarize(
            np.asarray(mirrored["macro_f1"]["values"]) - original["macro_f1"]["values"]
        )
        result["by_model"][model]["paired_recall_gains"] = {
            c: summarize(
                np.asarray(mirrored["per_class_recall"][c]["values"])
                - original["per_class_recall"][c]["values"]
            )
            for c in CLASSES
        }
    full = result["by_model"]["tcn"]["paired_macro_f1_gain"]["values"]
    masks = result["by_model"]["tcn_mask_only"]["paired_macro_f1_gain"]["values"]
    result["full_minus_mask_paired_gain"] = summarize(np.asarray(full) - masks)
    return result


def run_study(config_path, output, smoke=False, progress=print):
    config_path, output = Path(config_path), Path(output)
    study = yaml.safe_load(config_path.read_text())
    fields = {
        "schema_version",
        "base_config",
        "reference_experiment",
        "seeds",
        "mirror_probability",
    }
    if not isinstance(study, dict) or set(study) != fields:
        raise ValueError("Invalid mirror study configuration")
    if study["schema_version"] != "mirror-study-1.0" or study["mirror_probability"] != 0.5:
        raise ValueError("Study requires version 1.0 and fixed 50% mirror probability")
    seeds = study["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or len(set(seeds)) != 3
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("Specify exactly three distinct nonnegative integer seeds")
    base = load_config(Path(study["base_config"]))
    reference = Path(study["reference_experiment"])
    if base != yaml.safe_load((reference / "config.yaml").read_text()):
        raise ValueError("Base configuration differs from the frozen initial experiment")
    original_provenance = json.loads((reference / "provenance.json").read_text())
    for name, digest in original_provenance["code"]["sha256"].items():
        if name.endswith(("training/models.py", "training/data.py")) and sha256(name) != digest:
            raise ValueError("Architecture or feature representation changed")
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Choose a fresh output directory under reports/")
    preserved = {str(p): sha256(p) for p in reference.rglob("*") if p.is_file()}
    output.mkdir(parents=True)
    (output / "study.yaml").write_text(config_path.read_text())
    write_json(output / "preserved-reference.json", preserved)
    runs, audit = {}, {}
    seeds = seeds[:1] if smoke else seeds
    for seed in seeds:
        runs[str(seed)], audit[str(seed)] = {}, {}
        for arm, probability in (("unaugmented", 0.0), ("mirrored", 0.5)):
            config = {
                **base,
                "schema_version": "level1-experiment-1.1",
                "seed": seed,
                "mirror_probability": probability,
                "runs": ["tcn", "tcn_mask_only"],
            }
            if smoke:
                config.update(epochs=2, limit_per_class=2)
            config_file = output / f"seed-{seed}-{arm}.yaml"
            config_file.write_text(yaml.safe_dump(config, sort_keys=False))
            directory = output / f"seed-{seed}" / arm
            progress(f"=== seed {seed}: {arm}; smoke={smoke} ===")
            runs[str(seed)][arm] = run_experiment(config_file, directory, progress)
            logs = {
                name: json.loads((directory / name / "augmentation.json").read_text())
                for name in ("tcn", "tcn_mask_only")
            }
            # Same per-clip decisions and data order across full/mask-only controls.
            if logs["tcn"] != logs["tcn_mask_only"]:
                raise ValueError("Full/mask-only augmentation or shuffle decisions differ")
            audit[str(seed)][arm] = {
                "seen": sum(e["seen"] for e in logs["tcn"]),
                "mirrored": sum(len(e["mirrored_source_rows"]) for e in logs["tcn"]),
                "shuffle_digests": [e["shuffle_sha256"] for e in logs["tcn"]],
            }
        for model in ("tcn", "tcn_mask_only"):
            a, b = [runs[str(seed)][arm][model] for arm in ("unaugmented", "mirrored")]
            if a["initial_model_sha256"] != b["initial_model_sha256"]:
                raise ValueError("Paired initialization mismatch")
        if (
            audit[str(seed)]["unaugmented"]["shuffle_digests"]
            != audit[str(seed)]["mirrored"]["shuffle_digests"]
        ):
            raise ValueError("Paired training data order mismatch")
    summary = aggregate(runs, seeds)
    summary.update(
        schema_version="mirror-study-result-1.0",
        smoke=smoke,
        augmentation_audit=audit,
        config_sha256=sha256(config_path),
        reserved_evaluation="V9 not opened; V4 train / V5 development",
        selection="V5 macro-F1 independently selects each run; earlier ties",
        warning="Three seeds on the same recordings are not independent boxer tests; "
        "V5 is repeatedly selected development, not an unbiased test.",
    )
    if not smoke and original_provenance["seed"] in seeds:
        previous = json.loads((reference / "results.json").read_text())
        control = runs[str(original_provenance["seed"])]["unaugmented"]
        reproduced = all(
            control[name][key] == previous[name][key]
            for name in ("tcn", "tcn_mask_only")
            for key in ("selected_epoch", "train", "development", "last_epoch")
        )
        summary["original_seed_metrics_reproduced_exactly"] = reproduced
        if not reproduced:
            raise ValueError("Unaugmented original-seed control did not reproduce prior metrics")
    write_json(output / "summary.json", summary)
    write_report(output, summary, runs)
    plot_summary(output, summary)
    if any(sha256(path) != digest for path, digest in preserved.items()):
        raise ValueError("Previous experiment artifacts changed")
    progress(f"Study complete: {output / 'summary.md'}")
    return summary


def write_report(output, summary, runs):
    lines = [
        "# Mirror augmentation comparison",
        "",
        summary["warning"],
        "",
        f"Seeds: {summary['seeds']}; smoke: {summary['smoke']}. "
        "All evaluation uses original, unaugmented clips. Labels stay unchanged.",
        "",
        "| Seed | Input | Mirror | Best epoch | V4 F1 | V5 accuracy | V5 F1 | "
        "Final V4 F1 | Final V5 F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in summary["seeds"]:
        for arm in ("unaugmented", "mirrored"):
            for model in ("tcn", "tcn_mask_only"):
                r = runs[str(seed)][arm][model]
                lines.append(
                    f"| {seed} | {model} | {arm} | {r['selected_epoch']} | "
                    f"{r['train']['macro_f1']:.4f} | "
                    f"{r['development']['accuracy']:.4f} | "
                    f"{r['development']['macro_f1']:.4f} | "
                    f"{r['last_epoch']['train']['macro_f1']:.4f} | "
                    f"{r['last_epoch']['development']['macro_f1']:.4f} |"
                )
    lines += [
        "",
        "## Per-class development recall (mean across seeds)",
        "",
        "| Class | Full original | Full mirrored | Mask original | Mask mirrored |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in CLASSES:
        values = [
            summary["by_model"][model][arm]["per_class_recall"][c]["mean"]
            for model in ("tcn", "tcn_mask_only")
            for arm in ("unaugmented", "mirrored")
        ]
        lines.append(f"| {c} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += [
        "",
        "## Artifacts and interpretation",
        "",
        "Every seed/arm/model has metrics with all six per-class precision/recall/F1, "
        "confusion matrix, original-clip source-row predictions, best/final checkpoints, "
        "curves and the realized augmentation row log. `summary.json` contains individual "
        "seed values, means, sample standard deviations and paired changes. "
        "`confusions.png` uses mean row-normalized confusion across runs; the same V5 "
        "clips repeat across seeds, not extra independent examples.",
        "",
        "Private augmentation RNG is isolated from initialization/dropout/shuffle. Both "
        "arms have matching initial weights and example order; full/mask-only controls "
        "share augmentation decisions. Actual realized rates fluctuate around 50%. "
        "Original source clips, labels, indices, timestamps and policy remain unchanged.",
        "",
        "Interpret full-feature gains alongside mask-only gains and every class recall. "
        "An accuracy increase or strong mask-only gain does not establish six-class "
        "movement recognition. Full-feature advantage does not alone prove temporal "
        "motion reasoning. No visibility augmentation or pre-clip context is used.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def plot_summary(output, summary, arms=("unaugmented", "mirrored")):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 5))
    entries = [(model, arm) for model in ("tcn", "tcn_mask_only") for arm in arms]
    for i, (model, arm) in enumerate(entries):
        d = summary["by_model"][model][arm]["per_class_recall"]
        means = [d[c]["mean"] for c in CLASSES]
        std = [d[c]["sample_std"] or 0 for c in CLASSES]
        ax.bar(
            np.arange(6) + (i - 1.5) * 0.2,
            means,
            width=0.2,
            yerr=std,
            capsize=2,
            label=f"{model}: {arm}",
        )
    ax.set(
        xticks=range(6),
        xticklabels=CLASSES,
        ylim=(0, 1.15),
        ylabel="V5 recall",
        title="Per-class recall: mean ± sample SD across seeds (same V5 clips)",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "recall-comparison.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, (model, arm) in zip(axes.flat, entries, strict=True):
        counts = np.asarray(summary["by_model"][model][arm]["confusion_mean_counts"])
        normalized = counts / counts.sum(1, keepdims=True)
        ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        ax.set(
            xticks=range(6),
            xticklabels=CLASSES,
            yticks=range(6),
            yticklabels=CLASSES,
            title=f"{model}: {arm}",
            xlabel="Predicted",
            ylabel="Source label",
        )
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{normalized[i, j]:.0%}", ha="center", va="center", fontsize=8)
    fig.suptitle("Mean V5 confusion over seeds, row normalized; same clips across seeds")
    fig.tight_layout()
    fig.savefig(output / "confusions.png")
    plt.close(fig)
