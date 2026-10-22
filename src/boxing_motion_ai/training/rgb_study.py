"""One-seed frozen RGB/pose feasibility pilot under the accepted grouped protocol."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..vision.pipeline import sha256
from .data import collate
from .experiment import evaluate, plot, seed_all, write_json
from .models import CausalTCN
from .multi_recording import (
    ALL,
    TRAIN,
    Pool,
    folds,
    load_pool,
    sample_probabilities,
    settings,
    validate_fold,
)
from .multi_study import evaluate_record, prediction_metrics, snapshot, verify_manifest
from .rgb import POLICY, VisualClips, cache_features
from .velocity import expand_input
from .velocity_transfer import supported_metrics

PRIOR = Path("reports/boxingvi/level1-multi-recording-v1")
SEED = 20261021


def visual_collate(clips):
    batch = collate(clips)
    batch["reflected_x"] = torch.nn.utils.rnn.pad_sequence(
        [c["reflected_x"] for c in clips], batch_first=True
    )
    return batch


def augment_visual(batch, rng):
    selected = torch.rand(len(batch["labels"]), generator=rng) < 0.5
    x = batch["x"].clone()
    x[selected] = batch["reflected_x"][selected]
    return {**batch, "x": x}, selected


def fit(train, dev, base, output):
    """Held-out recordings cannot enter optimization or checkpoint selection."""
    validate_fold(sorted({c["recording_id"] for c in train.clips}), ["V5"], [])
    if {c["recording_id"] for c in dev.clips} != {"V5"}:
        raise ValueError("Development must be V5 only")
    seed_all(SEED, base["threads"])
    model = expand_input(CausalTCN(41, base["hidden"], base["dropout"]), train[0]["x"].shape[-1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base["learning_rate"], weight_decay=base["weight_decay"]
    )
    sample_rng = torch.Generator().manual_seed(SEED + 3000003)
    mirror_rng = torch.Generator().manual_seed(SEED + 1000003)
    loader_rng = torch.Generator().manual_seed(SEED + 4000003)
    eval_rng = torch.Generator().manual_seed(SEED + 5000003)
    train_eval = DataLoader(
        train, batch_size=base["batch_size"], collate_fn=collate, generator=eval_rng
    )
    dev_eval = DataLoader(
        dev, batch_size=base["batch_size"], collate_fn=collate, generator=eval_rng
    )
    probabilities = sample_probabilities(train.clips)
    output.mkdir(parents=True)
    write_json(
        output / "provenance.json",
        dict(
            seed=SEED,
            settings=base,
            policy=POLICY,
            selection=["V5"],
            training_examples=[[c["recording_id"], c["source_row"]] for c in train.clips],
            input_channels=train[0]["x"].shape[-1],
            background_control="first cache-frame background repeated for every source clip",
            initialization="same 41-column reference initialization; additional columns zero",
        ),
    )
    curves, audit, best, best_epoch = [], [], -1.0, None
    for epoch in range(1, base["epochs"] + 1):
        indices = torch.multinomial(
            probabilities, len(train), replacement=True, generator=sample_rng
        ).tolist()
        loader = DataLoader(
            train,
            batch_size=base["batch_size"],
            sampler=indices,
            collate_fn=visual_collate,
            generator=loader_rng,
        )
        model.train()
        coins = []
        for b in loader:
            b, selected = augment_visual(b, mirror_rng)
            coins.extend(selected.tolist())
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(b["x"], b["time_mask"]), b["labels"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite visual training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        tr, _ = evaluate(model, train_eval, "x", torch.ones(6))
        dv, _ = evaluate(model, dev_eval, "x", torch.ones(6))
        curves.append(
            dict(
                epoch=epoch,
                train_loss=tr["weighted_loss"],
                dev_loss=dv["weighted_loss"],
                train_accuracy=tr["accuracy"],
                dev_accuracy=dv["accuracy"],
                train_macro_f1=tr["macro_f1"],
                dev_macro_f1=dv["macro_f1"],
            )
        )
        audit.append(
            dict(epoch=epoch, indices=indices, mirrored=coins, unique_examples=len(set(indices)))
        )
        state = dict(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            seed=SEED,
            settings=base,
            selection_recordings=["V5"],
            input_channels=train[0]["x"].shape[-1],
            torch_rng_state=torch.get_rng_state(),
            sample_rng_state=sample_rng.get_state(),
            mirror_rng_state=mirror_rng.get_state(),
        )
        if dv["macro_f1"] > best:
            best, best_epoch = dv["macro_f1"], epoch
            torch.save(state, output / "best.pt")
        if epoch == base["epochs"]:
            torch.save(state, output / "last.pt")
            last = dict(train=supported_metrics(tr), development=supported_metrics(dv))
        if epoch == 1 or epoch % 10 == 0:
            print(f"{output}: {epoch} trainF1={tr['macro_f1']:.3f} V5F1={best:.3f}", flush=True)
    model.load_state_dict(torch.load(output / "best.pt", weights_only=True)["model_state"])
    model.eval()
    tr, _ = evaluate(model, train_eval, "x", torch.ones(6))
    dv, _ = evaluate(model, dev_eval, "x", torch.ones(6))
    result = dict(
        selected_epoch=best_epoch,
        parameters=sum(p.numel() for p in model.parameters()),
        train=supported_metrics(tr),
        development=supported_metrics(dv),
        last_epoch=last,
        checkpoint_sha256=sha256(output / "best.pt"),
    )
    write_json(output / "metrics.json", result)
    write_json(output / "sampling-audit.json", audit)
    with (output / "curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    plot(output, curves, dv["confusion_matrix"])
    return model, audit


def load_settings(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Pilot configuration must be a mapping")
    if config != dict(
        schema_version="rgb-feasibility-1.0",
        pose_protocol="configs/level1_multi_recording.yaml",
        seed=SEED,
        encoder="mobilenet_v3_small.IMAGENET1K_V1",
        encoder_frozen=True,
        encoder_device=config.get("encoder_device"),
        sampling_interval_seconds=0.1,
        modes=["rgb", "hybrid", "background"],
        selection="V5_only",
        review="configs/level1_rgb_review.yaml",
    ):
        raise ValueError("Frozen one-seed RGB pilot protocol required")
    if config["encoder_device"] not in ("cpu", "mps"):
        raise ValueError("Invalid encoder device")
    review = yaml.safe_load(Path(config["review"]).read_text())
    if (
        review["status"] != "provisional_pilot_permitted"
        or review["new_exclusions"]
        or review["label_or_timing_corrections"]
    ):
        raise ValueError("Review gate or approved pool changed")
    return config


def run(config_path, output, cache_path, smoke=False, cache_only=False):
    config = load_settings(config_path)
    cfg, base, specs = settings(config["pose_protocol"])
    output, cache_path = Path(output), Path(cache_path)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a fresh ignored report output")
    seed_all(SEED, base["threads"])
    pose = load_pool(specs)
    approved = json.loads(
        Path("reports/boxingvi/level1-limb-readiness-v2/summary.json").read_text()
    )
    for v in ALL:
        actual = pose[v]["xy_velocity"].provenance
        prior = approved["recordings"][v]["provenance"]
        for key in ("hashes", "policy_sha256", "pose_source_hashes", "source_rows"):
            if actual[key] != prior[key]:
                raise ValueError("Approved provisional examples or quality policy changed")
    preserved = json.loads((PRIOR / "preserved-inputs.json").read_text())
    if any("/V9/" in p or "/V9." in p for p in preserved):
        raise ValueError("V9 outside the RGB pilot scope")
    preserved.update({str(p): sha256(p) for p in PRIOR.rglob("*") if p.is_file()})
    for v in ALL:
        source = json.loads(
            Path(f"data/processed/boxingvi/{v}/pose-full-image-v1/manifest.json").read_text()
        )
        for kind, directory, suffix in (
            ("video", "RGB_videos", "mkv"),
            ("workbook", "Annotation_files", "xlsx"),
        ):
            item = source["inputs"][kind]
            if item["filename"] != f"{v}.{suffix}":
                raise ValueError("Unexpected source filename; V9 remains reserved")
            preserved[str(Path("data/raw/boxingvi") / directory / item["filename"])] = item[
                "sha256"
            ]
    verify_manifest(preserved)
    cache_summary = cache_features(pose, cache_path, device=config["encoder_device"])
    if cache_only:
        return cache_summary
    seed_all(SEED, base["threads"])  # Restore the frozen CPU evaluation/training backend.
    visual = {
        v: {m: VisualClips(pose[v]["xy_velocity"], cache_path / v, m) for m in config["modes"]}
        for v in ALL
    }
    if smoke:
        for v in ALL:
            selected = []
            for label in range(6):
                selected.extend(
                    i for i, c in enumerate(pose[v]["xy_velocity"].clips) if c["label"] == label
                )
            selected = [
                i
                for label in range(6)
                for i in [j for j in selected if pose[v]["xy_velocity"][j]["label"] == label][:2]
            ]
            for ds in [pose[v]["xy_velocity"], *visual[v].values()]:
                ds.clips = [ds.clips[i] for i in selected]
    output.mkdir(parents=True)
    snapshot(output, Path(config_path))
    shutil.copyfile(config_path, output / "study.yaml")
    shutil.copyfile(config["review"], output / "review.yaml")
    write_json(output / "preserved-inputs.json", preserved)
    write_json(output / "cache-manifests.json", cache_summary)
    review = yaml.safe_load(Path(config["review"]).read_text())
    review_path = Path(review["review_artifact"])
    write_json(
        output / "review-evidence.json",
        dict(
            path=str(review_path),
            sha256=sha256(review_path),
            reviewed_sheets=review["reviewed_sheets"],
            preprocessing_policy=POLICY,
            source_specific_masks=True,
        ),
    )
    result = dict(smoke=smoke, seed=SEED, cases={})
    cases = [dict(name="combined", train=list(TRAIN), development=["V5"], heldout=[]), *folds()]
    for case in cases[:1] if smoke else cases:
        validate_fold(case["train"], case["development"], case["heldout"])
        root = output / case["name"]
        root.mkdir()
        write_json(root / "fold.json", case)
        runs = {}
        prior = PRIOR / case["name"] / f"seed-{SEED}" / "xy_velocity"
        for mode in ("pose", "mask", *config["modes"]):
            if mode in ("pose", "mask"):
                key = "mask_x" if mode == "mask" else "x"
                ds = {v: pose[v]["xy_velocity"] for v in ALL}
                src = prior / ("tcn_mask_only" if mode == "mask" else "tcn")
                state = torch.load(src / "best.pt", weights_only=True)
                model = CausalTCN(16 if mode == "mask" else 65, base["hidden"], base["dropout"])
                model.load_state_dict(state["model_state"])
                model.eval()
                dest = root / mode
                dest.mkdir()
                write_json(
                    dest / "reused-checkpoint.json",
                    dict(path=str(src / "best.pt"), sha256=sha256(src / "best.pt")),
                )
            else:
                key, ds = "x", {v: visual[v][mode] for v in ALL}
                dest = root / mode
                model, audit = fit(
                    Pool([ds[v] for v in case["train"]]),
                    ds["V5"],
                    base if not smoke else {**base, "epochs": 2},
                    dest,
                )
                if not smoke and audit != json.loads(
                    (prior / "tcn/sampling-audit.json").read_text()
                ):
                    raise ValueError("Paired sampling/mirror decisions changed")
            runs[mode] = {}
            for v in ["V5", *case["heldout"], *case["train"]]:
                metric, detail = evaluate_record(model, ds[v], key, dest / f"{v}-predictions.json")
                runs[mode][v] = metric
                if not smoke and mode in ("pose", "mask"):
                    if detail != json.loads((src / f"{v}-predictions.json").read_text()):
                        raise ValueError("Preserved pose predictions changed")
            if case["heldout"]:
                rows = [
                    r
                    for v in case["heldout"]
                    for r in json.loads((dest / f"{v}-predictions.json").read_text())
                ]
                runs[mode]["heldout_group"] = prediction_metrics(rows)
                write_json(dest / "heldout-group-predictions.json", rows)
        previous = json.loads((PRIOR / case["name"] / "results.json").read_text())
        result["cases"][case["name"]] = dict(
            fold=case, runs=runs, duration_control=previous["duration_control"]
        )
        write_json(root / "results.json", result["cases"][case["name"]])
        write_json(output / "summary.json", result)
    verify_manifest(preserved)
    write_json(output / "preservation.json", dict(verified_prior_files=len(preserved), changed=[]))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/level1_rgb.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    a = p.parse_args()
    run(a.config, a.output, a.cache, a.smoke, a.cache_only)
