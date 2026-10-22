"""Matched V4/V5 velocity study; V6 excluded, V9 never opened."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import yaml

from ..vision.pipeline import sha256
from .data import CLASSES, PoseClips, exact_times
from .experiment import load_config, run_experiment, write_json
from .mirror_study import aggregate, plot_summary
from .velocity import VELOCITY_POLICY, VelocityClips

ARMS = ("xy", "xy_velocity")


def frozen_feature_body(path):
    tree = ast.parse(Path(path).read_text())
    return ast.dump(
        next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "features")
    )


def noise_report(dataset):
    """Descriptive apparent speeds, not calibrated physical speed or automatic exclusions."""
    values, largest, by_class = [], [], {c: [] for c in CLASSES}
    valid_count, possible = 0, 0
    for c in dataset.clips:
        v = c["x"][:, 41:].numpy().reshape(-1, 8, 3)
        t, j = np.nonzero(v[..., 2])
        speeds = np.linalg.norm(v[t, j, :2], axis=-1)
        values.extend(speeds.tolist())
        by_class[CLASSES[c["label"]]].extend(speeds.tolist())
        valid_count += len(speeds)
        possible += (len(v) - 1) * 8
        times = exact_times(c)
        for k in np.argsort(speeds)[-3:]:
            frame, joint = int(t[k]), int(j[k])
            largest.append(
                {
                    "source_row": c["source_row"],
                    "class": CLASSES[c["label"]],
                    "core_joint_index": joint,
                    "frame_index": int(c["raw"]["frame_index"][frame]),
                    "previous_frame_index": int(c["raw"]["frame_index"][frame - 1]),
                    "pts_seconds_exact": str(times[frame]),
                    "dt_seconds_exact": str(times[frame] - times[frame - 1]),
                    "speed_torso_per_second": float(speeds[k]),
                    "velocity_xy": v[frame, joint, :2].tolist(),
                    "normalized_xy_pair": c["x"][frame - 1 : frame + 1, :40]
                    .reshape(2, 8, 5)[:, joint, :2]
                    .tolist(),
                }
            )

    def describe(a):
        return (
            dict(
                zip(
                    ("median", "p90", "p99", "max"),
                    np.quantile(a, [0.5, 0.9, 0.99, 1]).tolist(),
                    strict=True,
                )
            )
            if len(a)
            else None
        )

    return {
        "policy": dataset.provenance["velocity_policy"],
        "valid_pairs": valid_count,
        "possible_within_clip_joint_pairs": possible,
        "valid_pair_fraction": valid_count / possible,
        "speed": describe(values),
        "by_class": {k: describe(v) for k, v in by_class.items()},
        "largest_pairs": sorted(largest, key=lambda r: -r["speed_torso_per_second"])[:12],
        "limitations": "Apparent normalized image-plane speeds, not metres/s. High speeds can "
        "reflect pose jitter, normalization changes or true movement. No clipping/smoothing, "
        "automatic correction or independently validated physical outlier cutoff.",
    }


def run_study(config_path, output, smoke=False, progress=print):
    config_path, output = Path(config_path), Path(output)
    study = yaml.safe_load(config_path.read_text())
    fields = {
        "schema_version",
        "base_config",
        "reference_mirror_study",
        "seeds",
        "mirror_probability",
        "visibility_probability",
        "context_seconds",
    }
    if not isinstance(study, dict) or set(study) != fields:
        raise ValueError("Invalid velocity study configuration")
    if (
        study["schema_version"] != "velocity-study-1.0"
        or study["mirror_probability"] != 0.5
        or study["visibility_probability"] != 0
        or study["context_seconds"] != 0
        or study["seeds"] != [20261021, 20261022, 20261023]
    ):
        raise ValueError("Frozen three seeds, mirror 0.5, visibility/context disabled required")
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Choose a fresh output directory below reports/")
    base = load_config(Path(study["base_config"]))
    reference = Path(study["reference_mirror_study"])
    for seed in study["seeds"]:
        old_config = yaml.safe_load((reference / f"seed-{seed}/mirrored/config.yaml").read_text())
        expected = {
            **base,
            "schema_version": "level1-experiment-1.1",
            "seed": seed,
            "mirror_probability": 0.5,
            "runs": ["tcn", "tcn_mask_only"],
        }
        if old_config != expected:
            raise ValueError("Training budget/data differs from preserved mirror reference")
        source = reference / f"seed-{seed}/mirrored/source/src/boxing_motion_ai/training"
        if sha256(source / "models.py") != sha256(Path(__file__).with_name("models.py")):
            raise ValueError("TCN architecture changed")
        if frozen_feature_body(source / "data.py") != frozen_feature_body(
            Path(__file__).with_name("data.py")
        ):
            raise ValueError("Existing XY normalization changed")
    # Explicit prior-artifact manifest: no discovery or reads under reserved V9 inputs.
    prior = json.loads(Path("reports/boxingvi/v6-full-v1/preserved-inputs.json").read_text())
    preserved = {p: d for p, d in prior.items() if not p.startswith("src/")}
    for root in (reference, Path("reports/boxingvi/v6-full-v1")):
        preserved.update({str(p): sha256(p) for p in root.rglob("*") if p.is_file()})
    if any("/V9/" in p or "/V9." in p for p in preserved):
        raise ValueError("Reserved recording is outside this study")
    if any(sha256(p) != digest for p, digest in preserved.items()):
        raise ValueError("Preserved artifacts changed")
    output.mkdir(parents=True)
    (output / "study.yaml").write_text(config_path.read_text())
    write_json(output / "preserved-inputs.json", preserved)
    seeds = study["seeds"][:1] if smoke else study["seeds"]
    runs, audit, diagnostics = {}, {}, {}
    for seed in seeds:
        runs[str(seed)], audit[str(seed)] = {}, {}
        for arm in ARMS:
            cfg = {
                **base,
                "schema_version": "level1-experiment-1.5",
                "seed": seed,
                "mirror_probability": 0.5,
                "feature_representation": arm,
                "runs": ["tcn", "tcn_mask_only"],
            }
            if smoke:
                cfg.update(epochs=2, limit_per_class=2)
            path = output / f"seed-{seed}-{arm}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            directory = output / f"seed-{seed}/{arm}"
            progress(f"seed={seed} representation={arm} smoke={smoke}")
            runs[str(seed)][arm] = run_experiment(path, directory, progress)
            provenance = json.loads((directory / "provenance.json").read_text())
            old = json.loads((reference / f"seed-{seed}/mirrored/provenance.json").read_text())
            for split in ("train", "development"):
                if provenance[split]["hashes"] != old[split]["hashes"]:
                    raise ValueError("Dataset differs from original mirror reference")
            logs = [
                json.loads((directory / name / "augmentation.json").read_text())
                for name in ("tcn", "tcn_mask_only")
            ]
            if logs[0] != logs[1]:
                raise ValueError("Full/mask control augmentation differs")
            audit[str(seed)][arm] = logs[0]
            if not smoke and arm == "xy":
                old_results = json.loads(
                    (reference / f"seed-{seed}/mirrored/results.json").read_text()
                )
                if any(
                    runs[str(seed)][arm][m][k] != old_results[m][k]
                    for m in ("tcn", "tcn_mask_only")
                    for k in ("selected_epoch", "train", "development", "last_epoch")
                ):
                    raise ValueError("XY reference metrics did not reproduce exactly")
        if audit[str(seed)][ARMS[0]] != audit[str(seed)][ARMS[1]]:
            raise ValueError("Paired data order or augmentation differs")
        for model in ("tcn", "tcn_mask_only"):
            if (
                runs[str(seed)][ARMS[0]][model]["shared_initial_model_sha256"]
                != runs[str(seed)][ARMS[1]][model]["shared_initial_model_sha256"]
            ):
                raise ValueError("Shared parameter initialization differs")
    for video, split in (("V4", "train"), ("V5", "development")):
        ds = VelocityClips(PoseClips(base[split], video, 2 if smoke else None))
        diagnostics[video] = noise_report(ds)
    summary = aggregate(runs, seeds, ARMS)
    summary.update(
        schema_version="velocity-study-result-1.0",
        smoke=smoke,
        velocity_policy=VELOCITY_POLICY,
        xy_reference_reproduced_exactly=not smoke,
        warning="V4 training / repeatedly selected V5 development, not unbiased test. "
        "Three seeds reuse the same recordings; V6 excluded, V9 untouched.",
    )
    write_json(output / "runs.json", runs)
    write_json(output / "summary.json", summary)
    write_json(output / "velocity-diagnostics.json", diagnostics)
    write_json(output / "augmentation-audit.json", audit)
    write_report(output, summary, runs)
    plot_summary(output, summary, ARMS)
    if any(sha256(p) != digest for p, digest in preserved.items()):
        raise ValueError("Preserved artifacts changed")
    return summary


def write_report(output, summary, runs):
    lines = [
        "# XY versus XY + velocity",
        "",
        summary["warning"],
        "",
        "| Seed | Representation | Input | Epoch | V4 F1 | V5 F1 | V5 accuracy | "
        "Final V4 / V5 F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for seed in summary["seeds"]:
        for arm in ARMS:
            for model in ("tcn", "tcn_mask_only"):
                r = runs[str(seed)][arm][model]
                lines.append(
                    f"| {seed} | {arm} | {model} | {r['selected_epoch']} | "
                    f"{r['train']['macro_f1']:.4f} | {r['development']['macro_f1']:.4f} | "
                    f"{r['development']['accuracy']:.4f} | "
                    f"{r['last_epoch']['train']['macro_f1']:.4f} / "
                    f"{r['last_epoch']['development']['macro_f1']:.4f} |"
                )
    lines += [
        "",
        "| Class | XY recall | XY + velocity recall | Paired change |",
        "|---|---:|---:|---:|",
    ]
    d = summary["by_model"]["tcn"]
    for c in CLASSES:
        lines.append(
            f"| {c} | {d['xy']['per_class_recall'][c]['mean']:.4f} | "
            f"{d['xy_velocity']['per_class_recall'][c]['mean']:.4f} | "
            f"{d['paired_recall_gains'][c]['mean']:+.4f} |"
        )
    lines += [
        "",
        "No visibility augmentation, pre-clip context, resampling, smoothing or velocity "
        "clipping. New input columns start at zero; shared weights, RNG, augmentation and "
        "data order match. All labels/bounds remain provisional. See summary.json for "
        "per-seed/aggregate metrics, runs.json for full precision/recall/confusions, and "
        "velocity-diagnostics.json for tails/validity and source-indexed large derivatives.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_study(args.config, args.output, args.smoke)
