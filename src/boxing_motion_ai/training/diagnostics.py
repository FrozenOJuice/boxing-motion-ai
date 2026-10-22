"""Frozen-checkpoint V4/V5 diagnosis. No optimization or source-data writes."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.pose_quality import CORE, mutual_information
from ..vision.pipeline import sha256
from .data import CLASSES, PoseClips, collate
from .experiment import load_config, metrics, seed_all, write_json
from .models import CausalTCN, StaticBaseline

SWAP = (1, 0, 3, 2, 5, 4, 7, 6)


def quantiles(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return (
        dict(
            zip(
                ("min", "p10", "median", "p90", "max"),
                np.quantile(a, [0, 0.1, 0.5, 0.9, 1]).tolist(),
                strict=True,
            )
        )
        if len(a)
        else None
    )


def clip_statistics(c):
    raw = c["raw"]
    xy = c["x"].numpy()[:, :40].reshape(-1, 8, 5)[..., :2]
    valid = raw["valid_mask"][:, CORE]
    dt = np.diff(c["timestamp_seconds"])
    # Directed shoulder span is a projected orientation proxy, not a stance label.
    shoulder = xy[:, 1] - xy[:, 0]
    raw_xy = raw["coordinates"][..., :2].copy()
    raw_xy[..., 0] *= 16 / 9  # Source dimensions are checked below, both recordings are 16:9.
    scale = np.linalg.norm(raw_xy[:, [11, 12]].mean(1) - raw_xy[:, [23, 24]].mean(1), axis=-1)
    pair = valid[:-1] & valid[1:]
    velocity = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / dt[:, None]
    return {
        "source_row": c["source_row"],
        "class": CLASSES[c["label"]],
        "torso_scale_image_height_units": quantiles(scale),
        "torso_scale_coefficient_of_variation": float(scale.std() / scale.mean()),
        "frames": len(xy),
        "center_span_seconds": float(np.diff(c["timestamp_seconds"]).sum()),
        "median_dt": float(np.median(dt)),
        "full_receptive_field_frame_fraction": max(0, len(xy) - 14) / len(xy),
        "mask_coverage": valid.mean(0).tolist(),
        "right_minus_left_shoulder_x": float(np.median(shoulder[:, 0])),
        "shoulder_width_over_torso": float(np.median(np.linalg.norm(shoulder, axis=-1))),
        "left_wrist_x": quantiles(xy[valid[:, 4], 4, 0]),
        "right_wrist_x": quantiles(xy[valid[:, 5], 5, 0]),
        "left_wrist_speed_torso_per_second": quantiles(velocity[pair[:, 4], 4]),
        "right_wrist_speed_torso_per_second": quantiles(velocity[pair[:, 5], 5]),
    }


def distribution(dataset):
    if (
        dataset.provenance["source_dimensions"][0] / dataset.provenance["source_dimensions"][1]
        != 16 / 9
    ):
        raise ValueError("This diagnosis expects verified 16:9 source dimensions")
    rows = [clip_statistics(c) for c in dataset.clips]
    labels = np.array([c["label"] for c in dataset.clips])
    coverage = np.array([r["mask_coverage"] for r in rows])
    lengths = np.array([r["frames"] for r in rows])
    result = {"clips": rows, "by_class": {}}
    for name in CLASSES:
        group = [r for r in rows if r["class"] == name]
        result["by_class"][name] = {
            "count": len(group),
            "fraction": len(group) / len(rows),
            "frames": quantiles([r["frames"] for r in group]),
            "center_span_seconds": quantiles([r["center_span_seconds"] for r in group]),
            "valid_fraction_per_joint": np.mean(
                [r["mask_coverage"] for r in group], axis=0
            ).tolist(),
            "right_minus_left_shoulder_x": quantiles(
                [r["right_minus_left_shoulder_x"] for r in group]
            ),
            "shoulder_width_over_torso": quantiles([r["shoulder_width_over_torso"] for r in group]),
        }
        for field in (
            "left_wrist_x",
            "right_wrist_x",
            "left_wrist_speed_torso_per_second",
            "right_wrist_speed_torso_per_second",
            "torso_scale_image_height_units",
        ):
            result["by_class"][name][field] = quantiles(
                [r[field]["median"] for r in group if r[field] is not None]
            )
        source = [c for c in dataset.clips if CLASSES[c["label"]] == name]
        conf = np.concatenate([c["raw"]["confidence"][:, CORE] for c in source])
        result["by_class"][name]["frame_weighted_visibility_below_0_5"] = (
            (conf[..., 0] < 0.5).mean(0).tolist()
        )
        result["by_class"][name]["frame_weighted_presence_below_0_5"] = (
            (conf[..., 1] < 0.5).mean(0).tolist()
        )
    result["all"] = {
        "torso_scale_image_height_units": quantiles(
            [r["torso_scale_image_height_units"]["median"] for r in rows]
        ),
        "scale_coefficient_of_variation": quantiles(
            [r["torso_scale_coefficient_of_variation"] for r in rows]
        ),
        "frames": quantiles(lengths),
        "center_span_seconds": quantiles([r["center_span_seconds"] for r in rows]),
        "median_dt": quantiles([r["median_dt"] for r in rows]),
        "fraction_clips_shorter_than_15_frames": float((lengths < 15).mean()),
        "mean_full_receptive_field_frame_fraction": float(
            np.mean([r["full_receptive_field_frame_fraction"] for r in rows])
        ),
        "right_minus_left_shoulder_x": quantiles([r["right_minus_left_shoulder_x"] for r in rows]),
        "shoulder_width_over_torso": quantiles([r["shoulder_width_over_torso"] for r in rows]),
        "mask_coverage": coverage.mean(0).tolist(),
        "clip_length_class_mi_bits": mutual_information(lengths, labels),
        "mask_coverage_class_mi_bits": [
            mutual_information(np.minimum((coverage[:, j] * 4).astype(int), 3), labels)
            for j in range(8)
        ],
        "mi_note": "Descriptive plug-in MI; length uses exact counts, masks fixed quartile bins; "
        "dependent clips, differing cardinalities; no significance/causal claim.",
    }
    return result


def perturb(x, kind, mask_only):
    """Diagnostic copies only. Reflection/swapping do not assert semantic label invariance."""
    x = x.clone()
    if kind == "reverse":
        return x.flip(0)
    if kind == "shuffle":
        generator = torch.Generator().manual_seed(1942 + len(x))
        return x[torch.randperm(len(x), generator=generator)]
    if kind == "constant_validity":
        if not mask_only:
            raise ValueError("Constant-validity control is for mask-only models")
        return torch.ones_like(x)
    if kind == "repeat_first":
        return x[:1].expand_as(x).clone()
    if kind in ("reflect_x", "swap_sides", "reflect_swap"):
        if mask_only:
            if kind in ("swap_sides", "reflect_swap"):
                x = x[:, SWAP]
        else:
            joints = x[:, :40].reshape(-1, 8, 5)
            if kind in ("swap_sides", "reflect_swap"):
                joints = joints[:, SWAP].clone()
            if kind in ("reflect_x", "reflect_swap"):
                joints[..., 0] *= -1
            x[:, :40] = joints.reshape(-1, 40)
    elif kind != "original":
        raise ValueError(f"Unknown diagnostic: {kind}")
    return x


def predictions(model, dataset, mask_only=False, kind="original", batch_size=64):
    key = "mask_x" if mask_only else "x"
    clones = [{**c, key: perturb(c[key], kind, mask_only)} for c in dataset.clips]
    probabilities = []
    with torch.no_grad():
        for batch in DataLoader(clones, batch_size=batch_size, collate_fn=collate):
            probabilities.extend(model(batch[key], batch["time_mask"]).softmax(-1).tolist())
    return np.asarray(probabilities)


def load_model(experiment, name, checkpoint):
    state = torch.load(experiment / name / f"{checkpoint}.pt", weights_only=True)
    config = state["config"]
    mask_only = name.endswith("mask_only")
    cls = StaticBaseline if name.startswith("static") else CausalTCN
    model = cls(8 if mask_only else 41, config["hidden"], config["dropout"])
    model.load_state_dict(state["model_state"])
    return model.eval(), mask_only, state["epoch"]


def length_control(model, channels):
    logits = []
    with torch.no_grad():
        for n in range(3, 33):
            logits.append(
                model(torch.ones(1, n, channels), torch.ones(1, n, dtype=torch.bool))[0].numpy()
            )
    values = np.asarray(logits)
    return {
        "input": "identical all-one tokens; only real clip length changes, no right padding",
        "lengths": list(range(3, 33)),
        "logits": values.tolist(),
        "predicted_class": [CLASSES[i] for i in values.argmax(1)],
        "max_absolute_logit_difference": float(np.abs(values - values[0]).max()),
    }


def runtime_checks(model, dataset):
    c = dataset[0]
    x = c["x"][None]
    n = x.shape[1]
    with torch.no_grad():
        original = model(x, torch.ones(1, n, dtype=torch.bool))
        extended = torch.cat((x, torch.full((1, 25, x.shape[-1]), 100.0)), 1)
        padded = model(extended, torch.arange(n + 25)[None] < n)
        changed = x.clone()
        pivot = max(1, n // 2)
        changed[:, pivot:] += 99
        causal_error = (model.encode(x)[:, :pivot] - model.encode(changed)[:, :pivot]).abs().max()
    return {
        "right_padding_logit_max_error": float((original - padded).abs().max()),
        "future_perturbation_prefix_max_error": float(causal_error),
    }


def contract_checks(dataset):
    errors = {
        "hip_center_max_error": 0.0,
        "unit_torso_max_error": 0.0,
        "invalid_feature_max_absolute_value": 0.0,
        "label_mapping_mismatches": 0,
        "raw_interval_mismatches": 0,
    }
    manifest = json.loads((Path(dataset.provenance["dataset"]) / "preparation.json").read_text())
    for c in dataset.clips:
        a = c["annotation"]
        start, stop = a["array_slice_half_open"]
        errors["raw_interval_mismatches"] += int(
            a["raw_start"] != start
            or a["raw_end"] + 1 != stop
            or a["candidate_start_index"] != start
            or a["candidate_end_index"] != stop - 1
        )
        errors["label_mapping_mismatches"] += int(
            manifest["class_map"][a["source_class"]] != CLASSES[c["label"]]
        )
        values = c["x"].numpy()[:, :40].reshape(-1, 8, 5)
        hip = values[:, [6, 7], :2].mean(1)
        torso = values[:, [0, 1], :2].mean(1) - hip
        errors["hip_center_max_error"] = max(
            errors["hip_center_max_error"], float(np.abs(hip).max())
        )
        errors["unit_torso_max_error"] = max(
            errors["unit_torso_max_error"], float(np.abs(np.linalg.norm(torso, axis=-1) - 1).max())
        )
        invalid = ~c["raw"]["valid_mask"][:, CORE]
        if invalid.any():
            errors["invalid_feature_max_absolute_value"] = max(
                errors["invalid_feature_max_absolute_value"], float(np.abs(values[invalid]).max())
            )
    with np.load(Path(dataset.provenance["dataset"]) / "poses.npz", allow_pickle=False) as a:
        times = a["pts"] * a["time_base_num"] / a["time_base_den"]
        step = float((times[-1] - times[0]) / (len(times) - 1))
    errors["source_mean_dt_seconds"] = step
    errors["source_effective_fps"] = 1 / step
    errors["15_sample_receptive_field_center_span_seconds"] = 14 * step
    return errors


def scene_metrics(model, dataset):
    p = predictions(model, dataset)
    result = {}
    for scene in sorted({int(c["raw"]["scene_id"][0]) for c in dataset.clips}):
        selected = [i for i, c in enumerate(dataset.clips) if c["raw"]["scene_id"][0] == scene]
        m = metrics([dataset.clips[i]["label"] for i in selected], p[selected].argmax(1))
        m["clip_count"] = len(selected)
        result[str(scene)] = m
    return result


def select_examples(datasets, experiment):
    cases = []
    for video, split in (("V4", "train"), ("V5", "development")):
        details = json.loads((experiment / "tcn" / f"{split}_predictions.json").read_text())
        by_row = {d["source_row"]: d for d in details}
        ds = datasets[video]
        for label, name in enumerate(CLASSES):
            for correct in [True] if video == "V4" else [False, True]:
                group = [
                    c
                    for c in ds.clips
                    if c["label"] == label
                    and (by_row[c["source_row"]]["prediction"] == name) == correct
                ]
                if not group:
                    continue
                c = group[len(group) // 2]
                cases.append(
                    {
                        "recording": video,
                        "source_row": c["source_row"],
                        "target": name,
                        "prediction": by_row[c["source_row"]]["prediction"],
                        "checkpoint": "tcn/best.pt",
                        "correct": correct,
                        "interval": c["annotation"],
                        "frame_indices": c["raw"]["frame_index"].tolist(),
                        "pts": c["raw"]["pts"].tolist(),
                    }
                )
    late = [c for c in datasets["V5"].clips if c["label"] == 1 and int(c["raw"]["scene_id"][0]) > 0]
    if late:
        c = late[len(late) // 2]
        detail = next(
            d
            for d in json.loads((experiment / "tcn/development_predictions.json").read_text())
            if d["source_row"] == c["source_row"]
        )
        cases.append(
            {
                "recording": "V5",
                "source_row": c["source_row"],
                "target": "cross",
                "prediction": detail["prediction"],
                "checkpoint": "tcn/best.pt",
                "correct": detail["prediction"] == "cross",
                "interval": c["annotation"],
                "frame_indices": c["raw"]["frame_index"].tolist(),
                "pts": c["raw"]["pts"].tolist(),
            }
        )
    # The selected V4 checkpoint has no correct rear uppercuts: keep that failure explicit.
    group = [c for c in datasets["V4"].clips if c["label"] == 5]
    c = group[len(group) // 2]
    detail = next(
        d
        for d in json.loads((experiment / "tcn/train_predictions.json").read_text())
        if d["source_row"] == c["source_row"]
    )
    cases.append(
        {
            "recording": "V4",
            "source_row": c["source_row"],
            "target": "rear_uppercut",
            "prediction": detail["prediction"],
            "checkpoint": "tcn/best.pt",
            "correct": False,
            "interval": c["annotation"],
            "frame_indices": c["raw"]["frame_index"].tolist(),
            "pts": c["raw"]["pts"].tolist(),
        }
    )
    return cases


def diagnose(experiment, output, media_root=None, progress=print):
    experiment, output = Path(experiment), Path(output)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Choose a new output directory below reports/")
    config = load_config(experiment / "config.yaml")
    provenance = json.loads((experiment / "provenance.json").read_text())
    for name, digest in provenance["code"]["sha256"].items():
        if name.endswith(("training/data.py", "training/models.py")) and sha256(name) != digest:
            raise ValueError("Model or feature code differs from experiment snapshot")
    datasets = {
        v: PoseClips(config[split], v) for v, split in (("V4", "train"), ("V5", "development"))
    }
    for v, split in (("V4", "train"), ("V5", "development")):
        if datasets[v].provenance["hashes"] != provenance[split]["hashes"]:
            raise ValueError("Prepared data differs from experiment")
    seed_all(config["seed"], config["threads"])
    preserved = {
        str(p): sha256(p)
        for p in experiment.rglob("*")
        if p.is_file() and p.suffix in (".pt", ".json", ".csv", ".yaml", ".png", ".md")
    }
    output.mkdir(parents=True)
    result = {
        "schema_version": "level1-diagnosis-1.0",
        "experiment": str(experiment),
        "experiment_provenance_sha256": sha256(experiment / "provenance.json"),
        "diagnostic_code_sha256": sha256(__file__),
        "preserved_experiment_hashes": preserved,
        "no_training": True,
        "reserved_evaluation": "V9 not opened",
        "distributions": {},
        "frozen_interventions": {},
        "checks": {},
        "length_controls": {},
        "warning": "Frozen-model counterfactuals are often out-of-distribution, "
        "not improved model results; side swapping is NOT a verified stance mapping.",
    }
    for v, ds in datasets.items():
        result["distributions"][v] = distribution(ds)
        result["checks"][v + "_data_contract"] = contract_checks(ds)
    for name in config["runs"]:
        result["frozen_interventions"][name] = {}
        for checkpoint in ("best", "last"):
            model, mask_only, epoch = load_model(experiment, name, checkpoint)
            variants = ["original", "reverse", "shuffle", "repeat_first", "swap_sides"]
            variants += ["constant_validity"] if mask_only else ["reflect_x", "reflect_swap"]
            result["frozen_interventions"][name][checkpoint] = {"epoch": epoch}
            for v, ds in datasets.items():
                progress(f"{name}/{checkpoint}: frozen diagnostics on {v}")
                records = {}
                original = predictions(model, ds, mask_only)
                target = [c["label"] for c in ds.clips]
                for kind in variants:
                    p = original if kind == "original" else predictions(model, ds, mask_only, kind)
                    records[kind] = metrics(target, p.argmax(1))
                    records[kind]["changed_prediction_fraction"] = float(
                        (p.argmax(1) != original.argmax(1)).mean()
                    )
                    records[kind]["max_probability_change"] = float(np.abs(p - original).max())
                result["frozen_interventions"][name][checkpoint][v] = records
                if checkpoint == "best":
                    split = "train" if v == "V4" else "development"
                    saved = json.loads(
                        (experiment / name / f"{split}_predictions.json").read_text()
                    )
                    error = float(
                        np.abs(original - np.array([s["probabilities"] for s in saved])).max()
                    )
                    result["checks"][f"{name}_{v}_saved_probabilities_max_error"] = error
                    if error > 1e-5:
                        raise ValueError("Could not reproduce saved predictions")
                if name == "tcn":
                    result["checks"][f"tcn_{checkpoint}_{v}"] = runtime_checks(model, ds)
                    result["checks"][f"tcn_{checkpoint}_{v}_by_scene"] = scene_metrics(model, ds)
            if mask_only:
                result["length_controls"][f"{name}_{checkpoint}"] = length_control(model, 8)
    cases = select_examples(datasets, experiment)
    result["visual_cases"] = cases
    write_json(output / "diagnosis.json", result)
    plots(datasets, result, output)
    render_pose_cases(cases, datasets, output)
    if media_root:
        render_media_cases(cases, datasets, Path(media_root), output, progress)
    write_summary(result, output)
    for ds in datasets.values():
        for name, digest in ds.provenance["hashes"].items():
            if sha256(Path(ds.provenance["dataset"]) / name) != digest:
                raise ValueError("Input changed during read-only diagnosis")
    if any(sha256(path) != digest for path, digest in preserved.items()):
        raise ValueError("Existing experiment artifact changed during diagnosis")
    return result


def plots(datasets, result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for video, color in (("V4", "tab:blue"), ("V5", "tab:orange")):
        d = result["distributions"][video]
        axes[0, 0].hist(
            [r["right_minus_left_shoulder_x"] for r in d["clips"]],
            bins=30,
            alpha=0.55,
            label=video,
            color=color,
            density=True,
        )
        axes[0, 1].hist(
            [r["frames"] for r in d["clips"]],
            bins=np.arange(2.5, 26),
            alpha=0.55,
            label=video,
            color=color,
            density=True,
        )
        medians = [d["by_class"][c]["center_span_seconds"]["median"] for c in CLASSES]
        axes[1, 0].plot(range(6), medians, marker="o", label=video, color=color)
    axes[0, 0].set(title="Projected shoulder direction / torso", xlabel="Right minus left X")
    axes[0, 1].set(title="Clip length (native frames)", xlabel="Frames")
    axes[1, 0].set(
        title="Median first-to-last frame time (not verified punch duration)",
        xticks=range(6),
        xticklabels=CLASSES,
        ylabel="Seconds",
    )
    plt.setp(axes[1, 0].get_xticklabels(), rotation=35, ha="right")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.legend()
    for video in ("V4", "V5"):
        d = result["distributions"][video]["all"]
        axes[1, 1].plot(range(8), d["mask_coverage"], marker="o", label=video)
    axes[1, 1].set(
        title="Mean per-clip joint validity",
        ylim=(0, 1.05),
        xticks=range(8),
        xticklabels=["L sh", "R sh", "L el", "R el", "L wr", "R wr", "L hip", "R hip"],
    )
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(output / "distributions.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, video in zip(axes, ("V4", "V5"), strict=True):
        values = np.array(
            [
                result["distributions"][video]["by_class"][c]["valid_fraction_per_joint"]
                for c in CLASSES
            ]
        )
        ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set(
            title=f"{video}: validity per class (0 dark / 1 bright)",
            yticks=range(6),
            yticklabels=CLASSES,
            xticks=range(8),
            xticklabels=["L sh", "R sh", "L el", "R el", "L wr", "R wr", "L hip", "R hip"],
        )
    fig.tight_layout()
    fig.savefig(output / "class_visibility.png")
    plt.close(fig)


def render_pose_cases(cases, datasets, output):
    import matplotlib.pyplot as plt

    directory = output / "review"
    directory.mkdir()
    for case in cases:
        c = next(
            c for c in datasets[case["recording"]].clips if c["source_row"] == case["source_row"]
        )
        values = c["x"].numpy()[:, :40].reshape(-1, 8, 5)
        indices = np.unique(np.linspace(0, len(values) - 1, 5, dtype=int))
        fig, axes = plt.subplots(1, len(indices), figsize=(15, 4), squeeze=False)
        for ax, i in zip(axes[0], indices, strict=True):
            for a, b in ((0, 1), (0, 6), (1, 7), (6, 7), (0, 2), (2, 4), (1, 3), (3, 5)):
                if values[i, a, 4] and values[i, b, 4]:
                    color = "tab:blue" if a in (0, 2, 6) else "tab:orange"
                    ax.plot(values[i, [a, b], 0], values[i, [a, b], 1], color=color)
            visible = values[i, :, 4].astype(bool)
            ax.scatter(values[i, visible, 0], values[i, visible, 1], color="black", s=10)
            ax.set(
                xlim=(-2, 2),
                ylim=(1, -2),
                aspect="equal",
                title=f"n={c['raw']['frame_index'][i]}\nPTS={c['timestamp_seconds'][i]:.3f}s",
            )
        fig.suptitle(
            f"{case['recording']} row {case['source_row']}: provisional {case['target']} "
            f"-> {case['prediction']}; best checkpoint\n"
            "Blue=anatomical left, orange=right; invalid joints omitted; fixed axes"
        )
        fig.tight_layout()
        fig.savefig(directory / f"{case['recording']}-row{case['source_row']}-poses.png")
        plt.close(fig)


def decode_exact_frames(path, requested):
    """Seek only as acceleration; accept frames solely on exact stored PTS/time base."""
    import av
    from PIL import Image

    images = {}
    previous_target, decoder = None, None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 4
        for target, index in sorted(requested.items()):
            # stream-relative seek units, always round down and decode forwards.
            if previous_target is None or target - previous_target > 1:
                container.seek(int(target / stream.time_base), stream=stream, backward=True)
                decoder = container.decode(stream)
            for frame in decoder:
                if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                    raise ValueError("Missing or corrupt source timing")
                time = Fraction(frame.pts) * frame.time_base
                if time < target:
                    continue
                if time != target:
                    raise ValueError(f"Exact source PTS not found: {index} at {target}")
                resized = frame.reformat(width=384, height=216)
                images[index] = Image.fromarray(resized.to_ndarray(format="rgb24"))
                previous_target = target
                break
            else:
                raise ValueError(f"EOF before frame {index}")
    return images


def render_media_cases(cases, datasets, media_root, output, progress):
    from PIL import Image, ImageDraw, ImageFont

    for video, ds in datasets.items():
        path = media_root / f"{video}.mkv"
        manifest = json.loads((Path(ds.provenance["dataset"]) / "preparation.json").read_text())
        if sha256(path) != manifest["inputs"]["video_sha256"]:
            raise ValueError("Video does not match prepared poses")
        with np.load(Path(ds.provenance["dataset"]) / "poses.npz", allow_pickle=False) as a:
            arrays = {k: a[k] for k in a.files}
        requests, case_frames = {}, {}
        for case in [c for c in cases if c["recording"] == video]:
            start, stop = case["interval"]["array_slice_half_open"]
            frames = sorted(
                set(
                    [
                        max(0, start - 3),
                        start,
                        (start + stop - 1) // 2,
                        stop - 1,
                        min(len(arrays["pts"]) - 1, stop + 2),
                    ]
                )
            )
            case_frames[case["source_row"]] = frames
            for index in frames:
                time = Fraction(
                    int(arrays["pts"][index]) * int(arrays["time_base_num"][index]),
                    int(arrays["time_base_den"][index]),
                )
                requests[time] = index
        progress(f"Exact-PTS visual decoding: {video}, {len(requests)} frames")
        images = decode_exact_frames(path, requests)
        for case in [c for c in cases if c["recording"] == video]:
            frames = case_frames[case["source_row"]]
            sheet = Image.new("RGB", (384 * len(frames), 320), "#101827")
            draw = ImageDraw.Draw(sheet)
            font = ImageFont.load_default(size=14)
            start, stop = case["interval"]["array_slice_half_open"]
            for col, index in enumerate(frames):
                xoff = col * 384
                sheet.paste(images[index], (xoff, 104))
                raw = arrays["coordinates"][index]
                valid = arrays["valid_mask"][index]

                def point(j, xoff=xoff, raw=raw):
                    return xoff + raw[j, 0] * 384, 104 + raw[j, 1] * 216

                for a, b in (
                    (11, 12),
                    (11, 23),
                    (12, 24),
                    (23, 24),
                    (11, 13),
                    (13, 15),
                    (12, 14),
                    (14, 16),
                ):
                    if valid[a] and valid[b]:
                        color = "#00d6ff" if a in (11, 13, 23) else "#ffad33"
                        draw.line([point(a), point(b)], fill=color, width=3)
                for j in CORE:
                    if (
                        np.isfinite(raw[j, :2]).all()
                        and (raw[j, :2] >= 0).all()
                        and (raw[j, :2] <= 1).all()
                    ):
                        x, y = point(j)
                        draw.ellipse(
                            (x - 2, y - 2, x + 2, y + 2), fill="white" if valid[j] else "#ff3040"
                        )
                seconds = (
                    arrays["pts"][index]
                    * arrays["time_base_num"][index]
                    / arrays["time_base_den"][index]
                )
                text = (
                    f"{video} row {case['source_row']} | best TCN\n"
                    f"Expected (provisional): {case['target']}\n"
                    f"Predicted clip: {case['prediction']}\n"
                    f"n={index} PTS={seconds:.3f}s | "
                    f"{'ANNOTATED' if start <= index < stop else 'CONTEXT/UNKNOWN'}\n"
                    f"raw interval {start}..{stop - 1} inclusive?"
                )
                draw.multiline_text((xoff + 4, 3), text, fill="white", font=font, spacing=2)
            sheet.save(output / "review" / f"{video}-row{case['source_row']}-video.jpg", quality=92)
            case["review_frame_indices"] = frames
    write_json(output / "visual-cases.json", cases)


def write_summary(result, output):
    lines = [
        "# Frozen-checkpoint Level 1 diagnosis",
        "",
        result["warning"],
        "",
        "No training; V4/V5 only. Original experiment and inputs preserved.",
        "",
        "| Model/checkpoint | V4 original F1 | V5 original F1 | V5 reversed F1 | V5 side-swap F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, checkpoints in result["frozen_interventions"].items():
        for checkpoint, r in checkpoints.items():
            lines.append(
                f"| {model}/{checkpoint} | {r['V4']['original']['macro_f1']:.4f} | "
                f"{r['V5']['original']['macro_f1']:.4f} | "
                f"{r['V5']['reverse']['macro_f1']:.4f} | "
                f"{r['V5']['swap_sides']['macro_f1']:.4f} |"
            )
    lines += [
        "",
        "## Compact visual review",
        "",
        "Selection: middle source row among correct/incorrect predictions per class, "
        "not the most confident example. V4 best has no correct rear uppercut: "
        "its middle-row failure is shown explicitly. No V5 correct cross/rear-hook/"
        "rear-uppercut exists at best checkpoint.",
        "",
        "Blue/cyan = anatomical left; orange = anatomical right. Red video points are "
        "invalid estimates (no edges); normalized plots omit them. Context frames outside "
        "annotation interval remain UNKNOWN. All annotation labels/boundaries provisional.",
        "",
        "| Video/row | Source label | Best TCN prediction | Poses | Source frames |",
        "|---|---|---|---|---|",
    ]
    for case in result["visual_cases"]:
        stem = f"{case['recording']}-row{case['source_row']}"
        lines.append(
            f"| {stem} | {case['target']} | {case['prediction']} | "
            f"[poses](review/{stem}-poses.png) | "
            + (
                f"[video](review/{stem}-video.jpg) |"
                if (output / "review" / f"{stem}-video.jpg").exists()
                else "not requested |"
            )
        )
    (output / "diagnosis.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-root", type=Path)
    args = parser.parse_args()
    diagnose(args.experiment, args.output, args.media_root)


if __name__ == "__main__":
    main()
