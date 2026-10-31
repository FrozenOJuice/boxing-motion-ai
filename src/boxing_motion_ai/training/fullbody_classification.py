"""Fixed-final-epoch paired conditional classification, never production promotion."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..analysis.common import sha
from ..services.dashboard import require
from .data import CLASSES, collate
from .experiment import evaluate, metrics, seed_all, write_json
from .fullbody_classification_data import (
    MAP65,
    View,
    clip_identity,
    development_views,
    training_views,
)
from .models import CausalTCN
from .multi_recording import sample_probabilities
from .velocity import expand_input

CONFIG = Path("configs/fullbody_classification_v1.json")


def initialize(mode, config):
    core = expand_input(CausalTCN(41, config["hidden"], config["dropout"]), 65)
    if mode == "core65":
        return core
    with torch.random.fork_rng():
        model = CausalTCN(209, config["hidden"], config["dropout"])
    state = model.state_dict()
    for key, value in core.state_dict().items():
        if key != "input.weight":
            state[key] = value.clone()
    state["input.weight"].zero_()
    state["input.weight"][:, MAP65] = core.input.weight.detach()
    model.load_state_dict(state)
    return model


def state_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def loader(clips, config):
    return DataLoader(
        clips,
        batch_size=config["batch_size"],
        collate_fn=collate,
        generator=torch.Generator().manual_seed(5000003),
    )


def score(model, clips, config):
    result, rows = evaluate(model, loader(clips, config), "x", torch.ones(6))
    for row, c in zip(rows, clips.clips, strict=True):
        row["softmax_scores"] = row.pop("probabilities")
        row["source"] = clip_identity(c)
    result["per_recording"] = {}
    for v in sorted({r["recording_id"] for r in rows}):
        subset = [r for r in rows if r["recording_id"] == v]
        result["per_recording"][v] = metrics(
            [CLASSES.index(r["target"]) for r in subset],
            [CLASSES.index(r["prediction"]) for r in subset],
        )
    return result, rows


def fit(clips, mode, seed, config):
    seed_all(seed, config["threads"])
    model = initialize(mode, config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    view = View(clips, mode)
    probabilities = torch.tensor(sample_probabilities(clips), dtype=torch.double)
    sampling = torch.Generator().manual_seed(seed + 3000003)
    mirror = torch.Generator().manual_seed(seed + 1000003)
    audit = hashlib.sha256()
    curves = []
    for epoch in range(config["epochs"]):
        indices = torch.multinomial(probabilities, len(clips), True, generator=sampling)
        coins = torch.rand(len(clips), generator=mirror) < config["mirror_probability"]
        audit.update(indices.numpy().tobytes())
        audit.update(coins.numpy().tobytes())
        examples = [
            view[int(i)] | {"x": view[int(i)]["reflected_x"] if m else view[int(i)]["x"]}
            for i, m in zip(indices, coins, strict=True)
        ]
        batches = DataLoader(
            examples,
            batch_size=config["batch_size"],
            collate_fn=collate,
            generator=torch.Generator().manual_seed(seed + 4000003),
        )
        model.train()
        total = 0.0
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(batch["x"], batch["time_mask"]), batch["labels"]
            )
            require(bool(torch.isfinite(loss)), "classifier_nonfinite_loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            total += loss.item() * len(batch["labels"])
        curves.append(dict(epoch=epoch + 1, cross_entropy=total / len(clips)))
    result, rows = score(model, view, config)
    return (
        model,
        dict(
            curves=curves,
            sampling_mirror_sha256=audit.hexdigest(),
            state_sha256=state_hash(model),
            training=result,
        ),
        rows,
    )


def code_hashes():
    paths = [
        CONFIG,
        *Path("src/boxing_motion_ai/training").glob("*.py"),
        Path("src/boxing_motion_ai/vision/full_body.py"),
    ]
    return {str(p): sha(p) for p in paths}


def train(output):
    config = json.loads(CONFIG.read_text())
    require(not output.exists(), "classifier_output_exists")
    output.mkdir(parents=True)
    clips, audit = training_views()
    audit["counts"] = dict(Counter(f"{c['recording_id']}:{CLASSES[c['label']]}" for c in clips))
    write_json(output / "inputs.json", audit)
    write_json(output / "protocol.json", config)
    write_json(output / "training_clips.json", [clip_identity(c) for c in clips])
    frozen = []
    for seed in config["seeds"]:
        for mode in config["modes"]:
            start = time.perf_counter()
            model, report, rows = fit(clips, mode, seed, config)
            repeat, repeated, repeated_rows = fit(clips, mode, seed, config)
            require(
                report == repeated
                and rows == repeated_rows
                and state_hash(model) == state_hash(repeat),
                "classifier_repeat_mismatch",
            )
            dest = output / f"{mode}-{seed}"
            dest.mkdir()
            torch.save(
                dict(
                    state_dict=model.state_dict(),
                    mode=mode,
                    seed=seed,
                    config=config,
                    input_audit=audit,
                    epoch=config["epochs"],
                ),
                dest / "final.pt",
            )
            restored = initialize(mode, config)
            restored.load_state_dict(torch.load(dest / "final.pt", weights_only=True)["state_dict"])
            require(
                score(restored, View(clips, mode), config)[1] == rows, "classifier_reload_mismatch"
            )
            report.update(
                exact_repeat=True,
                reload_equal=True,
                elapsed_seconds_including_repeat=time.perf_counter() - start,
                parameters=sum(p.numel() for p in model.parameters()),
            )
            write_json(dest / "training.json", report)
            write_json(dest / "training_predictions.json", rows)
            frozen.append(
                dict(
                    mode=mode,
                    seed=seed,
                    checkpoint=str(dest / "final.pt"),
                    sha256=sha(dest / "final.pt"),
                )
            )
            print(mode, seed, report["training"]["macro_f1"], flush=True)
    files = {str(p): sha(p) for p in output.rglob("*") if p.is_file()}
    write_json(
        output / "training-freeze.json",
        dict(
            config_sha256=sha(CONFIG),
            code_sha256=code_hashes(),
            training_files_sha256=files,
            runs=frozen,
            development_not_loaded=True,
            input_sha256=audit["input_sha256"],
        ),
    )


def development(output):
    frozen = json.loads((output / "training-freeze.json").read_text())
    require(not (output / "development.json").exists(), "classifier_development_already_run")
    for group in ("code_sha256", "training_files_sha256", "input_sha256"):
        for p, h in frozen[group].items():
            require(sha(Path(p)) == h, "classifier_frozen_artifact_changed")
    config = json.loads(CONFIG.read_text())
    seed_all(config["seeds"][0], config["threads"])
    clips, provenance = development_views()
    summary = dict(
        provenance=provenance,
        reference_status="provisional_source_labels_not_human_ground_truth",
        freeze_sha256=sha(output / "training-freeze.json"),
        runs=[],
    )
    for run in frozen["runs"]:
        state = torch.load(run["checkpoint"], weights_only=True)
        model = initialize(run["mode"], config)
        model.load_state_dict(state["state_dict"])
        result, rows = score(model, View(clips, run["mode"]), config)
        path = Path(run["checkpoint"]).parent
        write_json(path / "development_predictions.json", rows)
        write_json(path / "development_metrics.json", result)
        summary["runs"].append(dict(mode=run["mode"], seed=run["seed"], metrics=result))
        print(run["mode"], run["seed"], result["macro_f1"], flush=True)
    summary["aggregate"] = {}
    for mode in config["modes"]:
        values = [r["metrics"]["macro_f1"] for r in summary["runs"] if r["mode"] == mode]
        summary["aggregate"][mode] = dict(
            macro_f1_mean=float(np.mean(values)), macro_f1_sd=float(np.std(values))
        )
    write_json(output / "development.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["train", "development"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (train if args.phase == "train" else development)(args.output)
