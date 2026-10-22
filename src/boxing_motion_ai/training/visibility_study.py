"""Matched mirror-only versus mirror + temporary-visibility study, V4/V5 only."""

import json
from pathlib import Path

import yaml

from ..vision.pipeline import sha256
from .augmentation import VISIBILITY_POLICY
from .data import CLASSES
from .experiment import load_config, run_experiment, write_json
from .mirror_study import aggregate, plot_summary

ARMS = ("mirror_only", "mirror_visibility")


def run_study(config_path, output, smoke=False, progress=print):
    config_path, output = Path(config_path), Path(output)
    study = yaml.safe_load(config_path.read_text())
    fields = {
        "schema_version",
        "reference_mirror_study",
        "seeds",
        "mirror_probability",
        "visibility_probability",
    }
    if not isinstance(study, dict) or set(study) != fields:
        raise ValueError("Invalid visibility study configuration")
    if (
        study["schema_version"] != "visibility-study-1.0"
        or study["mirror_probability"] != 0.5
        or study["visibility_probability"] != 0.5
    ):
        raise ValueError("Use version 1.0 with fixed 50% mirror and visibility probabilities")
    reference = Path(study["reference_mirror_study"])
    prior = json.loads((reference / "summary.json").read_text())
    seeds = study["seeds"]
    if seeds != prior["seeds"] or len(seeds) != 3 or prior["smoke"]:
        raise ValueError("Use the same three seeds as the completed full mirror study")
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Choose a fresh output directory under reports/")
    preserved = {str(p): sha256(p) for p in reference.rglob("*") if p.is_file()}
    output.mkdir(parents=True)
    (output / "study.yaml").write_text(config_path.read_text())
    write_json(output / "preserved-reference.json", preserved)
    runs, audit = {}, {}
    seeds = seeds[:1] if smoke else seeds
    for seed in seeds:
        old = reference / f"seed-{seed}" / "mirrored"
        base = load_config(old / "config.yaml")
        original_provenance = json.loads((old / "provenance.json").read_text())
        if base["seed"] != seed or base["mirror_probability"] != 0.5:
            raise ValueError("Mismatched original mirror configuration")
        for name, digest in original_provenance["code"]["sha256"].items():
            if name.endswith(("training/data.py", "training/models.py")) and sha256(name) != digest:
                raise ValueError("Feature representation or architecture changed")
        runs[str(seed)], audit[str(seed)] = {}, {}
        for arm, probability in zip(ARMS, (0.0, 0.5), strict=True):
            config = {
                **base,
                "schema_version": "level1-experiment-1.2",
                "visibility_probability": probability,
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
                    if provenance[split][key] != original_provenance[split][key]:
                        raise ValueError("Prepared dataset or source quality policy changed")
                if (
                    not smoke
                    and provenance[split]["source_rows"]
                    != original_provenance[split]["source_rows"]
                ):
                    raise ValueError("Training/development candidate selections changed")
            logs = {
                name: json.loads((directory / name / "augmentation.json").read_text())
                for name in ("tcn", "tcn_mask_only")
            }
            if logs["tcn"] != logs["tcn_mask_only"]:
                raise ValueError("Full/mask-only augmentation decisions or inputs differ")
            mirror_schedule = [
                {k: e[k] for k in ("epoch", "seen", "mirrored_source_rows", "shuffle_sha256")}
                for e in logs["tcn"]
            ]
            events = [event for e in logs["tcn"] for event in e["visibility_events"]]
            audit[str(seed)][arm] = {
                "presentations": sum(e["seen"] for e in logs["tcn"]),
                "visibility_events": len(events),
                "no_op_events": sum(e["newly_hidden_usable_entries"] == 0 for e in events),
                "newly_hidden_usable_entries": sum(
                    e["newly_hidden_usable_entries"] for e in events
                ),
                "mirror_schedule": mirror_schedule,
            }
            if not smoke:
                old_log = json.loads((old / "tcn/augmentation.json").read_text())
                if mirror_schedule != [{k: e[k] for k in mirror_schedule[0]} for e in old_log]:
                    raise ValueError("Mirror decisions or data order changed from previous study")
        a, b = [audit[str(seed)][arm]["mirror_schedule"] for arm in ARMS]
        if a != b:
            raise ValueError("Paired mirror decisions/data order differ")
        for model in ("tcn", "tcn_mask_only"):
            baseline, augmented = [runs[str(seed)][arm][model] for arm in ARMS]
            if baseline["initial_model_sha256"] != augmented["initial_model_sha256"]:
                raise ValueError("Paired initialization differs")
            if not smoke:
                previous = json.loads((old / model / "metrics.json").read_text())
                for key in ("selected_epoch", "train", "development", "last_epoch"):
                    if baseline[key] != previous[key]:
                        raise ValueError("Mirror-only control failed to reproduce prior metrics")
    summary = aggregate(runs, seeds, ARMS)
    summary.update(
        schema_version="visibility-study-result-1.0",
        smoke=smoke,
        visibility_policy=VISIBILITY_POLICY,
        augmentation_audit=audit,
        config_sha256=sha256(config_path),
        mirror_only_reference_reproduced_exactly=not smoke,
        warning="V5 is development, repeatedly selected; seeds reuse the same "
        "recordings and are not independent boxer tests. No V9 access.",
    )
    write_json(output / "summary.json", summary)
    write_report(output, summary, runs)
    plot_summary(output, summary, ARMS)
    if any(sha256(path) != digest for path, digest in preserved.items()):
        raise ValueError("Prior mirror study artifacts changed")
    progress(f"Study complete: {output / 'summary.md'}")
    return summary


def write_report(output, summary, runs):
    lines = [
        "# Mirror + temporary visibility comparison",
        "",
        summary["warning"],
        "",
        f"Seeds: {summary['seeds']}. Smoke: {summary['smoke']}. "
        "Both arms retain training-only 50% mirroring; V5 always unaugmented.",
        "",
        "| Seed | Input | Arm | Epoch | V4 F1 | V5 accuracy | V5 F1 | Final V4 F1 | Final V5 F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in summary["seeds"]:
        for arm in ARMS:
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
        "## Per-class development recall (seed mean)",
        "",
        "| Class | Full mirror | Full mirror+visibility | Mask mirror | Mask mirror+visibility |",
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
        "Each seed/arm contains unchanged-source predictions, best/final checkpoints, "
        "loss/F1 curves and confusion matrices. Summary contains means, sample SDs, "
        "paired changes, mean confusion counts and augmentation events. Same V5 clips "
        "are repeated across seeds. No classes are discarded; original unknown footage "
        "remains unknown. No pre-clip context, pose filling or torso-anchor masking.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
