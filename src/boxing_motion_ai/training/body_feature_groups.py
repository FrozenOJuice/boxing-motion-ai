"""Single predeclared group ablation, two training-source holdouts, then frozen V5 replay."""

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..analysis.common import sha
from ..services.dashboard import require
from ..vision.full_body import CHANNELS
from ..vision.observation_study import read
from .data import CLASSES, collate
from .experiment import seed_all, write_json
from .fullbody_classification import initialize as reference_initialize
from .fullbody_classification import score, state_hash
from .fullbody_classification_data import clip_identity, development_views, training_views
from .multi_recording import sample_probabilities
from .velocity import expand_input

CONFIG = Path("configs/body_feature_groups_v1.json")
GROUPS = dict(
    upper=list(range(48, 96)) + list(range(176, 188)),
    lower=list(range(112, 176)) + list(range(188, 192)),
    torso=list(range(192, 208)),
)


class GroupView(Dataset):
    def __init__(self, clips, mode):
        require(mode in ("core65", *GROUPS), "unknown_body_group")
        extra = GROUPS.get(mode, [])
        self.clips = []
        for c in clips:
            x = torch.cat((c["x"], c["body209"][:, extra]), 1)
            mirror = torch.cat((c["core65_reflected"], c["body209_reflected"][:, extra]), 1)
            self.clips.append(c | dict(x=x, reflected_x=mirror))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return self.clips[i]


def initialize(mode, cfg):
    require(mode in ("core65", *GROUPS), "unknown_body_group")
    model = reference_initialize("core65", cfg)
    return expand_input(model, 65 + len(GROUPS[mode])) if mode != "core65" else model


def probabilities(clips):
    require(
        bool(clips) and {c["recording_id"] for c in clips} <= {"V4", "V6"}, "group_training_role"
    )
    counts = Counter(c["label"] for c in clips)
    if len(counts) == 6:
        return sample_probabilities(clips)
    require(len({c["recording_id"] for c in clips}) == 1, "missing_class_pooled_sampler_forbidden")
    return torch.tensor([1 / len(counts) / counts[c["label"]] for c in clips], dtype=torch.double)


def fit(clips, mode, seed, cfg):
    seed_all(seed, cfg["threads"])
    model = initialize(mode, cfg)
    view = GroupView(clips, mode)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    sample = torch.Generator().manual_seed(seed + 3000003)
    mirror = torch.Generator().manual_seed(seed + 1000003)
    audit = hashlib.sha256()
    curves = []
    weights = probabilities(clips)
    for epoch in range(cfg["epochs"]):
        indices = torch.multinomial(weights, len(clips), True, generator=sample)
        coins = torch.rand(len(clips), generator=mirror) < cfg["mirror_probability"]
        audit.update(indices.numpy().tobytes())
        audit.update(coins.numpy().tobytes())
        examples = [
            view[int(i)] | {"x": view[int(i)]["reflected_x"] if m else view[int(i)]["x"]}
            for i, m in zip(indices, coins, strict=True)
        ]
        batches = DataLoader(
            examples,
            batch_size=cfg["batch_size"],
            collate_fn=collate,
            generator=torch.Generator().manual_seed(seed + 4000003),
        )
        model.train()
        total = 0.0
        for b in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(b["x"], b["time_mask"]), b["labels"])
            require(bool(torch.isfinite(loss)), "body_group_nonfinite_loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            optimizer.step()
            total += loss.item() * len(b["labels"])
        curves.append(dict(epoch=epoch + 1, cross_entropy=total / len(clips)))
    metrics, rows = score(model, view, cfg)
    return (
        model,
        dict(
            curves=curves,
            sampling_mirror_sha256=audit.hexdigest(),
            state_sha256=state_hash(model),
            training=metrics,
        ),
        rows,
    )


def supported_score(model, clips, mode, train_labels, cfg):
    selected = [c for c in clips if c["label"] in train_labels]
    require(bool(selected), "no_supported_comparison_classes")
    result, rows = score(model, GroupView(selected, mode), cfg)
    supported = [name for name, v in result["per_class"].items() if v["support"] > 0]
    result["supported_labels"] = supported
    result["supported_macro_f1"] = float(np.mean([result["per_class"][n]["f1"] for n in supported]))
    result["unsupported_reference_count"] = len(clips) - len(selected)
    return result, rows


def feature_audit(clips):
    """Descriptive training-source evidence only. Never fits an input transformation."""
    out = {}
    for v in ("V4", "V6"):
        selected = [c for c in clips if c["recording_id"] == v]
        x = torch.cat([c["body209"] for c in selected]).numpy()
        per = {}
        for group, indices in GROUPS.items():
            data = x[:, indices]
            masks = [i for i in indices if CHANNELS[i].endswith("valid")]
            per[group] = dict(
                channels=[CHANNELS[i] for i in indices],
                absolute_p50_p95_p99_max=np.quantile(np.abs(data), [0.5, 0.95, 0.99, 1]).tolist(),
                validity={CHANNELS[i]: float(x[:, i].mean()) for i in masks},
                per_class_validity={
                    CLASSES[label]: {
                        CHANNELS[i]: float(
                            torch.cat([c["body209"] for c in selected if c["label"] == label])[
                                :, i
                            ].mean()
                        )
                        for i in masks
                    }
                    for label in sorted({c["label"] for c in selected})
                },
            )
        out[v] = per
    return out


def bindings():
    paths = [
        CONFIG,
        Path("configs/fullbody_classification_v1.json"),
        Path(__file__),
        Path("src/boxing_motion_ai/vision/full_body.py"),
        Path("src/boxing_motion_ai/training/fullbody_classification.py"),
        Path("src/boxing_motion_ai/training/fullbody_classification_data.py"),
        Path("src/boxing_motion_ai/training/data.py"),
        Path("src/boxing_motion_ai/training/velocity.py"),
        Path("src/boxing_motion_ai/training/augmentation.py"),
        Path("src/boxing_motion_ai/training/models.py"),
        Path("src/boxing_motion_ai/training/multi_recording.py"),
        Path("src/boxing_motion_ai/training/experiment.py"),
    ]
    return {str(p): sha(p) for p in paths}


def train(output):
    protocol = read(CONFIG)
    cfg = read(protocol["base_protocol"])
    require(not output.exists(), "body_group_output_exists")
    output.mkdir(parents=True)
    clips, audit = training_views()
    write_json(
        output / "protocol.json", dict(protocol=protocol, training=cfg, code_sha256=bindings())
    )
    write_json(output / "inputs.json", audit)
    write_json(output / "training_clips.json", [clip_identity(c) for c in clips])
    write_json(output / "feature_audit.json", feature_audit(clips))
    runs = []
    for fold in protocol["folds"]:
        training = [c for c in clips if c["recording_id"] in fold["training"]]
        comparison = [c for c in clips if c["recording_id"] in fold["comparison"]]
        require(not set(fold["training"]) & set(fold["comparison"]), "body_group_source_overlap")
        for seed in cfg["seeds"]:
            sampling_hash = None
            for mode in protocol["modes"]:
                start = perf_counter()
                model, report, rows = fit(training, mode, seed, cfg)
                repeat, repeated, repeated_rows = fit(training, mode, seed, cfg)
                require(
                    report == repeated
                    and rows == repeated_rows
                    and state_hash(model) == state_hash(repeat),
                    "body_group_repeat_mismatch",
                )
                if sampling_hash is None:
                    sampling_hash = report["sampling_mirror_sha256"]
                require(
                    sampling_hash == report["sampling_mirror_sha256"],
                    "body_group_sampling_mismatch",
                )
                dest = output / fold["name"] / f"{mode}-{seed}"
                dest.mkdir(parents=True)
                torch.save(
                    dict(
                        state_dict=model.state_dict(),
                        mode=mode,
                        seed=seed,
                        epoch=cfg["epochs"],
                        config=cfg,
                        fold=fold,
                        feature_indices=GROUPS.get(mode, []),
                    ),
                    dest / "final.pt",
                )
                loaded = initialize(mode, cfg)
                loaded.load_state_dict(
                    torch.load(dest / "final.pt", weights_only=True)["state_dict"]
                )
                require(
                    score(loaded, GroupView(training, mode), cfg)[1] == rows,
                    "body_group_reload_mismatch",
                )
                report.update(
                    exact_repeat=True,
                    reload_equal=True,
                    elapsed_seconds_including_repeat=perf_counter() - start,
                    parameters=sum(p.numel() for p in model.parameters()),
                )
                if comparison:
                    result, pred = supported_score(
                        model, comparison, mode, {c["label"] for c in training}, cfg
                    )
                    report["comparison"] = result
                    write_json(dest / "comparison_predictions.json", pred)
                write_json(dest / "training.json", report)
                write_json(dest / "training_predictions.json", rows)
                runs.append(
                    dict(
                        fold=fold["name"],
                        mode=mode,
                        seed=seed,
                        checkpoint=str(dest / "final.pt"),
                        sha256=sha(dest / "final.pt"),
                        comparison=report.get("comparison"),
                    )
                )
                print(
                    fold["name"],
                    mode,
                    seed,
                    report.get("comparison", {}).get("supported_macro_f1"),
                    flush=True,
                )
    write_json(output / "source_comparison.json", runs)
    frozen = {str(p): sha(p) for p in output.rglob("*") if p.is_file()}
    write_json(
        output / "training-freeze.json",
        dict(
            runs=runs,
            files_sha256=frozen,
            code_sha256=bindings(),
            input_sha256=audit["input_sha256"],
            development_not_loaded=True,
        ),
    )


def development(output):
    frozen = read(output / "training-freeze.json")
    require(not (output / "development.json").exists(), "body_group_development_already_run")
    for key in ("files_sha256", "code_sha256", "input_sha256"):
        for p, h in frozen[key].items():
            require(sha(Path(p)) == h, "body_group_frozen_input_changed")
    protocol = read(output / "protocol.json")
    cfg = protocol["training"]
    seed_all(cfg["seeds"][0], cfg["threads"])
    clips, provenance = development_views()
    runs = []
    for run in frozen["runs"]:
        if run["fold"] != "pooled":
            continue
        state = torch.load(run["checkpoint"], weights_only=True)
        model = initialize(run["mode"], cfg)
        model.load_state_dict(state["state_dict"])
        result, rows = score(model, GroupView(clips, run["mode"]), cfg)
        dest = Path(run["checkpoint"]).parent
        write_json(dest / "development_metrics.json", result)
        write_json(dest / "development_predictions.json", rows)
        runs.append(dict(mode=run["mode"], seed=run["seed"], metrics=result))
        print(run["mode"], run["seed"], result["macro_f1"], flush=True)
    write_json(
        output / "development.json",
        dict(
            runs=runs,
            provenance=provenance,
            frozen_sha256=sha(output / "training-freeze.json"),
            no_model_selection=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["train", "development"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (train if args.phase == "train" else development)(args.output)
