"""Matched mirror-only versus real pre-clip context, with frozen shortcut diagnostics."""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..vision.pipeline import sha256
from .context import CONTEXT_POLICY, ContextClips, context_collate, pooling_mask
from .data import CLASSES, PoseClips
from .diagnostics import load_model
from .experiment import load_config, metrics, run_experiment, write_json
from .mirror_study import aggregate, plot_summary

ARMS = ("mirror_only", "mirror_context")


def constant_length_probe(model, prefix):
    logits = []
    with torch.no_grad():
        for n in range(3, 33):
            logits.append(
                model(torch.ones(1, prefix + n, 8), torch.arange(prefix + n)[None] >= prefix)[
                    0
                ].numpy()
            )
    values = np.asarray(logits)
    return {
        "prefix_frames": prefix,
        "punch_lengths": list(range(3, 33)),
        "predicted_classes": [CLASSES[i] for i in values.argmax(1)],
        "max_absolute_logit_change": float(np.abs(values - values[0]).max()),
    }


def frozen_predictions(model, ds, mask_only, kind="original"):
    key = "mask_x" if mask_only else "x"
    clips = []
    for c in ds.clips:
        clone = dict(c)
        offset = c.get("context_length", 0)
        if kind == "constant_validity":
            if not mask_only:
                raise ValueError("Constant validity probe requires a mask-only model")
            clone[key] = torch.ones_like(c[key])
        elif kind == "neutral_context":
            if not mask_only:
                raise ValueError("Neutral context probe requires a mask-only model")
            clone[key] = c[key].clone()
            clone[key][:offset] = 1
        elif kind == "remove_context":
            for name in ("x", "mask_x"):
                clone[name] = c[name][offset:]
            clone["pool_mask"] = torch.ones(len(clone["x"]), dtype=torch.bool)
        elif kind != "original":
            raise ValueError("Unknown frozen diagnostic")
        clips.append(clone)
    probabilities = []
    with torch.no_grad():
        for b in DataLoader(clips, batch_size=64, collate_fn=context_collate):
            probabilities.extend(model(b[key], pooling_mask(b)).softmax(-1).tolist())
    p = np.asarray(probabilities)
    return metrics([c["label"] for c in ds.clips], p.argmax(1)), p


def shortcut_diagnostics(directory):
    config = load_config(directory / "config.yaml")
    result = {
        "warning": "Frozen interventions can be out of distribution; no new fitting. "
        "Constant mask tokens are synthetic diagnostic inputs, not filled pose coordinates."
    }
    for video, split in (("V4", "train"), ("V5", "development")):
        ds = PoseClips(config[split], video, config["limit_per_class"])
        if config["context_seconds"]:
            ds = ContextClips(ds, config["context_seconds"])
        result[video] = {}
        for name in ("tcn", "tcn_mask_only"):
            model, mask_only, _ = load_model(directory, name, "best")
            original, original_p = frozen_predictions(model, ds, mask_only)
            saved = json.loads((directory / name / f"{split}_predictions.json").read_text())
            saved_p = np.asarray([p["probabilities"] for p in saved])
            if np.abs(original_p - saved_p).max() > 1e-5:
                raise ValueError("Context-aware frozen predictions do not reproduce saved results")
            d = {
                "original": original,
                "saved_probability_max_error": float(np.abs(original_p - saved_p).max()),
            }
            kinds = ["remove_context"] + (
                ["constant_validity", "neutral_context"] if mask_only else []
            )
            for kind in kinds:
                measured, p = frozen_predictions(model, ds, mask_only, kind)
                measured["changed_prediction_fraction"] = float(
                    (p.argmax(1) != original_p.argmax(1)).mean()
                )
                measured["predicted_class_counts"] = np.bincount(p.argmax(1), minlength=6).tolist()
                d[kind] = measured
            if mask_only:
                d["constant_token_length_probes"] = [
                    constant_length_probe(model, n) for n in (0, 15, 19)
                ]
            result[video][name] = d
    return result


def run_study(config_path, output, smoke=False, progress=print):
    config_path, output = Path(config_path), Path(output)
    study = yaml.safe_load(config_path.read_text())
    fields = {
        "schema_version",
        "reference_mirror_study",
        "seeds",
        "mirror_probability",
        "visibility_probability",
        "context_seconds",
    }
    if not isinstance(study, dict) or set(study) != fields:
        raise ValueError("Invalid context study configuration")
    if (
        study["schema_version"] != "context-study-1.0"
        or study["mirror_probability"] != 0.5
        or study["visibility_probability"] != 0
        or type(study["context_seconds"]) not in (int, float)
        or not np.isfinite(study["context_seconds"])
        or not 0 < study["context_seconds"] <= 2
    ):
        raise ValueError("Require 50% mirroring, visibility disabled and a finite >0..2s context")
    reference = Path(study["reference_mirror_study"])
    previous_summary = json.loads((reference / "summary.json").read_text())
    seeds = study["seeds"]
    if seeds != previous_summary["seeds"] or len(seeds) != 3 or previous_summary["smoke"]:
        raise ValueError("Use the same three seeds as the full mirror study")
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Choose a fresh output directory under reports/")
    preserved = {str(p): sha256(p) for p in reference.rglob("*") if p.is_file()}
    output.mkdir(parents=True)
    (output / "study.yaml").write_text(config_path.read_text())
    write_json(output / "preserved-reference.json", preserved)
    seeds = seeds[:1] if smoke else seeds
    runs, audit = {}, {}
    for seed in seeds:
        old = reference / f"seed-{seed}" / "mirrored"
        base = load_config(old / "config.yaml")
        prior_provenance = json.loads((old / "provenance.json").read_text())
        for name, digest in prior_provenance["code"]["sha256"].items():
            if name.endswith(("training/data.py", "training/models.py")) and sha256(name) != digest:
                raise ValueError("Source feature contract or model architecture changed")
        runs[str(seed)], audit[str(seed)] = {}, {}
        for arm, seconds in zip(ARMS, (0.0, study["context_seconds"]), strict=True):
            config = {
                **base,
                "schema_version": "level1-experiment-1.3",
                "visibility_probability": 0.0,
                "context_seconds": seconds,
            }
            if smoke:
                config.update(epochs=2, limit_per_class=2)
            config_file = output / f"seed-{seed}-{arm}.yaml"
            config_file.write_text(yaml.safe_dump(config, sort_keys=False))
            directory = output / f"seed-{seed}" / arm
            progress(f"=== seed {seed}: {arm}; smoke={smoke} ===")
            runs[str(seed)][arm] = run_experiment(config_file, directory, progress)
            provenance = json.loads((directory / "provenance.json").read_text())
            for split in ("train", "development"):
                for key in ("hashes", "policy_sha256"):
                    if provenance[split][key] != prior_provenance[split][key]:
                        raise ValueError("Source dataset or quality policy changed")
                if (
                    not smoke
                    and provenance[split]["source_rows"] != prior_provenance[split]["source_rows"]
                ):
                    raise ValueError("Eligible examples differ from reference")
            logs = {
                name: json.loads((directory / name / "augmentation.json").read_text())
                for name in ("tcn", "tcn_mask_only")
            }
            if logs["tcn"] != logs["tcn_mask_only"] or any(
                e["visibility_events"] for e in logs["tcn"]
            ):
                raise ValueError("Augmentation schedules differ or visibility was enabled")
            schedule = [
                {k: e[k] for k in ("epoch", "seen", "mirrored_source_rows", "shuffle_sha256")}
                for e in logs["tcn"]
            ]
            audit[str(seed)][arm] = schedule
            if not smoke:
                old_log = json.loads((old / "tcn/augmentation.json").read_text())
                if schedule != [{k: e[k] for k in schedule[0]} for e in old_log]:
                    raise ValueError("Training order or mirror choices differ from reference")
            progress(f"Frozen shortcut diagnostics: seed {seed} / {arm}")
            write_json(directory / "shortcuts.json", shortcut_diagnostics(directory))
        if audit[str(seed)][ARMS[0]] != audit[str(seed)][ARMS[1]]:
            raise ValueError("Paired mirror or batch schedules differ")
        for name in ("tcn", "tcn_mask_only"):
            baseline, context = [runs[str(seed)][arm][name] for arm in ARMS]
            if baseline["initial_model_sha256"] != context["initial_model_sha256"]:
                raise ValueError("Initial model weights differ")
            if not smoke:
                previous = json.loads((old / name / "metrics.json").read_text())
                if any(
                    baseline[k] != previous[k]
                    for k in ("selected_epoch", "train", "development", "last_epoch")
                ):
                    raise ValueError("Mirror-only control did not reproduce prior metrics")
    summary = aggregate(runs, seeds, ARMS)
    summary.update(
        schema_version="context-study-result-1.0",
        smoke=smoke,
        context_policy={**CONTEXT_POLICY, "seconds": study["context_seconds"]},
        mirror_only_reference_reproduced_exactly=not smoke,
        config_sha256=sha256(config_path),
        warning="V4 train / V5 repeatedly selected development, no V9. "
        "Same examples across seeds, not independent boxer tests. "
        "Context may contain scripted prior actions; those frames are not supervised.",
    )
    write_json(output / "summary.json", summary)
    write_report(output, summary, runs)
    plot_summary(output, summary, ARMS)
    if any(sha256(path) != digest for path, digest in preserved.items()):
        raise ValueError("Previous experiment files changed")
    progress(f"Study complete: {output / 'summary.md'}")
    return summary


def write_report(output, summary, runs):
    lines = [
        "# Real pre-clip context comparison",
        "",
        summary["warning"],
        "",
        "| Seed | Input | Arm | Epoch | V4 F1 | V5 accuracy | V5 F1 | Final V4 F1 | Final V5 F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in summary["seeds"]:
        for arm in ARMS:
            for name in ("tcn", "tcn_mask_only"):
                r = runs[str(seed)][arm][name]
                lines.append(
                    f"| {seed} | {name} | {arm} | {r['selected_epoch']} | "
                    f"{r['train']['macro_f1']:.4f} | "
                    f"{r['development']['accuracy']:.4f} | "
                    f"{r['development']['macro_f1']:.4f} | "
                    f"{r['last_epoch']['train']['macro_f1']:.4f} | "
                    f"{r['last_epoch']['development']['macro_f1']:.4f} |"
                )
    lines += [
        "",
        "## Per-class development recall (seed mean)",
        "",
        "| Class | Full mirror | Full context | Mask mirror | Mask context |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in CLASSES:
        values = [
            summary["by_model"][model][arm]["per_class_recall"][c]["mean"]
            for model in ("tcn", "tcn_mask_only")
            for arm in ARMS
        ]
        lines.append(f"| {c} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += [
        "",
        "Each run includes best/final checkpoints, original-source predictions, "
        "metrics/confusions and curves. Context input is real, unfilled and causal; "
        "only annotated punch states are pooled. `provenance.json` records exact "
        "context boundaries/PTS, truncations and neighboring-annotation overlaps. "
        "`shortcuts.json` contains frozen context-removal, constant-mask and fixed-prefix "
        "length probes; synthetic controls are not recognition improvements. "
        "Visibility augmentation is disabled. Architecture and quality policy unchanged.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
