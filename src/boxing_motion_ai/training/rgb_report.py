"""Readable pilot metrics, class support and frozen control comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import CLASSES

MODES = ("pose", "rgb", "hybrid", "mask", "background")


def render(result):
    lines = [
        "# Frozen RGB + pose feasibility pilot",
        "",
        "One seed (20261021); checkpoints selected only on V5. Provisional source labels; "
        "unknown athlete identities; V9 unopened. No boxer-independent claim.",
        "",
        "Background control repeats one fixed boxer-erased source embedding per recording; "
        "it tests static recording appearance plus clip-duration shortcuts, not every "
        "dynamic background cue. RGB crops retain some background.",
        "",
        "| Fold | Evaluation | Mode | Accuracy | Six-class F1 | Supported F1 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for case, d in result["cases"].items():
        targets = ["V5", *d["fold"]["heldout"]]
        if d["fold"]["heldout"]:
            targets.append("heldout_group")
        for v in targets:
            for mode in MODES:
                m = d["runs"][mode][v]
                lines.append(
                    f"| {case} | {v} | {mode} | {m['accuracy']:.4f} | "
                    f"{m['macro_f1']:.4f} | {m['supported_macro_f1']:.4f} |"
                )
            duration = d["duration_control"][v]
            lines.append(
                f"| {case} | {v} | duration | {duration['accuracy']:.4f} | "
                f"{duration['macro_f1']:.4f} | {duration['supported_macro_f1']:.4f} |"
            )
    for case, d in result["cases"].items():
        for v in ["V5", *d["fold"]["heldout"]]:
            lines += [
                "",
                f"## {case}: {v}",
                "",
                "| Class | Support | Pose recall | RGB recall | Hybrid recall |",
                "|---|---:|---:|---:|---:|",
            ]
            for c in CLASSES:
                m = d["runs"]["pose"][v]["per_class"][c]
                recalls = [
                    d["runs"][mode][v]["per_class"][c]["recall"]
                    for mode in ("pose", "rgb", "hybrid")
                ]
                cells = [f"{r:.4f}" if m["support"] else "absent" for r in recalls]
                lines.append(f"| {c} | {m['support']} | " + " | ".join(cells) + " |")
            for mode in ("pose", "rgb", "hybrid"):
                m = d["runs"][mode][v]
                lines += [
                    "",
                    f"### {mode}: per-class precision/recall and confusion",
                    "",
                    "Rows=true, columns=predicted; class order " + ", ".join(CLASSES),
                    "",
                    "| Class | Precision | Recall | F1 | Support |",
                    "|---|---:|---:|---:|---:|",
                ]
                for c in CLASSES:
                    p = m["per_class"][c]
                    lines.append(
                        f"| {c} | {p['precision']:.4f} | {p['recall']:.4f} | "
                        f"{p['f1']:.4f} | {p['support']} |"
                    )
                lines += ["", "```text", *[str(row) for row in m["confusion_matrix"]], "```"]
    groups = [d for d in result["cases"].values() if d["fold"]["heldout"]]
    if groups:
        lines += [
            "",
            "## Group-balanced descriptive mean (one seed)",
            "",
            "Unweighted across four held-out source groups; not an athlete population estimate.",
        ]
        for mode in MODES:
            a = np.mean([d["runs"][mode]["heldout_group"]["macro_f1"] for d in groups])
            b = np.mean([d["runs"][mode]["heldout_group"]["supported_macro_f1"] for d in groups])
            lines.append(f"- {mode}: six-class F1 {a:.4f}; supported F1 {b:.4f}.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not a.output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Generated reports must remain below ignored reports/")
    result = json.loads((a.output / "summary.json").read_text())
    (a.output / "REPORT.md").write_text(render(result))
