"""Descriptive matched-seed V10 results; no fitting, checkpoint selection or source access."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import CLASSES
from .experiment import write_json
from .mirror_study import summarize


def compare_runs(runs, seeds):
    result = {"seeds": seeds, "by_model": {}}
    for model in ("tcn", "tcn_mask_only"):
        arms = {}
        for arm in ("v4_only", "expanded"):
            selected = [runs[str(s)][arm][model] for s in seeds]
            arms[arm] = {
                "accuracy": summarize([r["development"]["accuracy"] for r in selected]),
                "macro_f1": summarize([r["development"]["macro_f1"] for r in selected]),
                "selected_epochs": [r["selected_epoch"] for r in selected],
                "per_class": {
                    c: {
                        m: summarize([r["development"]["per_class"][c][m] for r in selected])
                        for m in ("precision", "recall", "f1")
                    }
                    for c in CLASSES
                },
                "mean_confusion_matrix": np.mean(
                    [r["development"]["confusion_matrix"] for r in selected], axis=0
                ).tolist(),
                "selected_train_f1": summarize([r["train"]["macro_f1"] for r in selected]),
                "final_train_f1": summarize(
                    [r["last_epoch"]["train"]["macro_f1"] for r in selected]
                ),
                "final_development_f1": summarize(
                    [r["last_epoch"]["development"]["macro_f1"] for r in selected]
                ),
                "final_train_dev_gap": summarize(
                    [
                        r["last_epoch"]["train"]["macro_f1"]
                        - r["last_epoch"]["development"]["macro_f1"]
                        for r in selected
                    ]
                ),
            }
        deltas = [
            runs[str(s)]["expanded"][model]["development"]["macro_f1"]
            - runs[str(s)]["v4_only"][model]["development"]["macro_f1"]
            for s in seeds
        ]
        arms["paired_macro_f1_change"] = summarize(deltas)
        arms["positive_macro_f1_seeds"] = sum(d > 0 for d in deltas)
        arms["per_class_recall_change"] = {
            c: summarize(
                [
                    runs[str(s)]["expanded"][model]["development"]["per_class"][c]["recall"]
                    - runs[str(s)]["v4_only"][model]["development"]["per_class"][c]["recall"]
                    for s in seeds
                ]
            )
            for c in CLASSES
        }
        result["by_model"][model] = arms
    full, masks = [result["by_model"][m] for m in ("tcn", "tcn_mask_only")]
    result["full_gain_minus_mask_gain"] = summarize(
        [
            f - m
            for f, m in zip(
                full["paired_macro_f1_change"]["values"],
                masks["paired_macro_f1_change"]["values"],
                strict=True,
            )
        ]
    )
    result["limitations"] = [
        "Three matched random seeds, not independent recordings or a significance test",
        "V5 repeatedly used for epoch/model development; not unbiased evaluation",
        "Mask-only inputs also retain real sequence lengths and padding/boundary cues",
        "Full-minus-mask gain is descriptive, not causal attribution to movement",
        "Extra data changes class weights and optimizer steps per epoch",
        "Unknown boxer/session identities; no boxer-independent claim",
    ]
    return result


def report(directory):
    directory = Path(directory)
    if (directory / "comparison.json").exists():
        raise ValueError("Comparison already exists; preserve completed analysis")
    summary = json.loads((directory / "summary.json").read_text())
    if summary["smoke"] or summary["training_recordings"] != ["V4", "V10"]:
        raise ValueError("Requires completed three-seed V4/V10 study")
    runs = json.loads((directory / "runs.json").read_text())
    result = compare_runs(runs, summary["seeds"])
    result["reference_metrics_reproduced"] = summary["reference_metrics_reproduced"]
    write_json(directory / "comparison.json", result)
    lines = [
        "# V10 matched-seed comparison",
        "",
        *result["limitations"],
        "",
        "| Input | Training | Accuracy | Macro-F1 ± seed SD | Final train F1 | "
        "Final V5 F1 | Final gap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, arms in result["by_model"].items():
        if model == "tcn_mask_only":
            lines += [
                "",
                "## Mask-only aggregate",
                "",
                "| Input | Training | Accuracy | Macro-F1 ± seed SD | Final train F1 | "
                "Final V5 F1 | Final gap |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        for arm in ("v4_only", "expanded"):
            r = arms[arm]
            lines.append(
                f"| {model} | {arm} | {r['accuracy']['mean']:.4f} | "
                f"{r['macro_f1']['mean']:.4f} ± {r['macro_f1']['sample_std']:.4f} | "
                f"{r['final_train_f1']['mean']:.4f} | "
                f"{r['final_development_f1']['mean']:.4f} | "
                f"{r['final_train_dev_gap']['mean']:.4f} |"
            )
        lines += [
            "",
            f"## {model}: per-class precision / recall",
            "",
            "| Class | V4 precision | Expanded precision | V4 recall | Expanded recall | "
            "Recall change |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for c in CLASSES:
            a, b = [arms[arm]["per_class"][c] for arm in ("v4_only", "expanded")]
            lines.append(
                f"| {c} | {a['precision']['mean']:.4f} | {b['precision']['mean']:.4f} | "
                f"{a['recall']['mean']:.4f} | {b['recall']['mean']:.4f} | "
                f"{arms['per_class_recall_change'][c]['mean']:+.4f} |"
            )
        lines += [
            "",
            "Paired macro-F1 changes: " + str(arms["paired_macro_f1_change"]["values"]) + ".",
            "",
        ]
    (directory / "comparison.md").write_text("\n".join(lines) + "\n")
    plot_comparison(directory, result)
    return result


def plot_comparison(directory, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for i, model in enumerate(("tcn", "tcn_mask_only")):
        for j, arm in enumerate(("v4_only", "expanded")):
            ax = axes[i, j]
            matrix = np.array(result["by_model"][model][arm]["mean_confusion_matrix"])
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=226)
            ax.set(
                xticks=range(6),
                yticks=range(6),
                xticklabels=CLASSES,
                yticklabels=CLASSES,
                title=f"{model}: {arm}",
                xlabel="Predicted",
                ylabel="Source class",
            )
            ax.tick_params(axis="x", labelrotation=60)
            for y in range(6):
                for x in range(6):
                    ax.text(x, y, f"{matrix[y, x]:.1f}", ha="center", va="center")
    fig.suptitle("V5 development confusion: mean counts over three seeds (not extra test samples)")
    fig.savefig(directory / "confusion-comparison.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    report(parser.parse_args().directory)
