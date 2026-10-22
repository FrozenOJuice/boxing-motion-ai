"""Controlled multi-recording TCN study and group-held-out pilot; no V9 access."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..vision.pipeline import sha256
from .augmentation import augment_batch
from .data import CLASSES, collate
from .experiment import evaluate, metrics, plot, seed_all, write_json
from .mirror_study import summarize
from .models import CausalTCN
from .multi_recording import (
    ALL,
    ARMS,
    GROUPS,
    GUARDS,
    MODELS,
    SEEDS,
    TRAIN,
    Pool,
    audit_pool,
    duration_control,
    duration_key,
    folds,
    load_pool,
    sample_probabilities,
    sampling_summary,
    settings,
    validate_fold,
)
from .velocity import expand_input
from .velocity_transfer import ast_item, supported_metrics


def tensor_digest(model):
    return hashlib.sha256(
        b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())
    ).hexdigest()


def fit(train, development, probabilities, base, seed, arm, name, output, progress=print):
    """Only training and V5 are arguments: a fold's held-out clips cannot select epochs."""
    train_ids = list(dict.fromkeys(c["recording_id"] for c in train.clips))
    dev_ids = list(dict.fromkeys(c["recording_id"] for c in development.clips))
    validate_fold(train_ids, dev_ids, [])
    if not torch.equal(probabilities, sample_probabilities(train.clips)):
        raise ValueError("Sampler must be computed from this training pool only")
    seed_all(seed, base["threads"])
    key = "mask_x" if name.endswith("mask_only") else "x"
    model = CausalTCN(8 if key == "mask_x" else 41, base["hidden"], base["dropout"])
    initial_shared = tensor_digest(model)
    model = expand_input(model, train[0][key].shape[-1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base["learning_rate"], weight_decay=base["weight_decay"]
    )
    sample_rng = torch.Generator().manual_seed(seed + 3000003)
    mirror_rng = torch.Generator().manual_seed(seed + 1000003)
    loader_rng = torch.Generator().manual_seed(seed + 4000003)
    eval_rng = torch.Generator().manual_seed(seed + 5000003)
    weights = torch.ones(6)  # Sampler already balances labels; no double weighting.
    output.mkdir(parents=True)
    provenance = dict(
        seed=seed,
        representation=arm,
        run=name,
        base_settings=base,
        training_examples=[[c["recording_id"], c["source_row"]] for c in train.clips],
        training_recordings=train_ids,
        development_recordings=dev_ids,
        training_sources=[ds.provenance for ds in train.datasets],
        development_source=getattr(development, "provenance", {}),
        selection_recordings=["V5"],
        selection="best V5 macro-F1; earliest tie",
        sampling=sampling_summary(train.clips, probabilities),
        feature_key=key,
        initial_shared_sha256=initial_shared,
        initial_model_sha256=tensor_digest(model),
    )
    write_json(output / "provenance.json", provenance)
    train_eval = DataLoader(
        train, batch_size=base["batch_size"], collate_fn=collate, generator=eval_rng
    )
    dev_eval = DataLoader(
        development, batch_size=base["batch_size"], collate_fn=collate, generator=eval_rng
    )
    curves, audit, best = [], [], -1.0
    for epoch in range(1, base["epochs"] + 1):
        indices = torch.multinomial(
            probabilities, len(train), replacement=True, generator=sample_rng
        ).tolist()
        loader = DataLoader(
            train,
            batch_size=base["batch_size"],
            sampler=indices,
            collate_fn=collate,
            generator=loader_rng,
            num_workers=0,
        )
        model.train()
        mirrored = []
        for batch in loader:
            b, selected = augment_batch(batch, mirror_rng, 0.5)
            mirrored.extend(selected.tolist())
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(b[key], b["time_mask"]), b["labels"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        tr, _ = evaluate(model, train_eval, key, weights)
        dev, _ = evaluate(model, dev_eval, key, weights)
        curves.append(
            dict(
                epoch=epoch,
                train_loss=tr["weighted_loss"],
                dev_loss=dev["weighted_loss"],
                train_accuracy=tr["accuracy"],
                dev_accuracy=dev["accuracy"],
                train_macro_f1=tr["macro_f1"],
                dev_macro_f1=dev["macro_f1"],
            )
        )
        audit.append(
            dict(epoch=epoch, indices=indices, mirrored=mirrored, unique_examples=len(set(indices)))
        )
        state = dict(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            seed=seed,
            representation=arm,
            run=name,
            base_settings=base,
            selection_recordings=["V5"],
            training_recordings=train_ids,
            development_recordings=dev_ids,
            provenance_sha256=sha256(output / "provenance.json"),
            torch_rng_state=torch.get_rng_state(),
            sample_rng_state=sample_rng.get_state(),
            mirror_rng_state=mirror_rng.get_state(),
            loader_rng_state=loader_rng.get_state(),
            eval_rng_state=eval_rng.get_state(),
        )
        if dev["macro_f1"] > best:
            best, best_epoch = dev["macro_f1"], epoch
            torch.save(state, output / "best.pt")
        if epoch == base["epochs"]:
            torch.save(state, output / "last.pt")
            last = dict(train=tr, development=dev)
        if epoch == 1 or epoch % 10 == 0 or epoch == base["epochs"]:
            progress(
                f"{seed}/{arm}/{name} {epoch}: train F1={tr['macro_f1']:.3f}; "
                f"V5 F1={dev['macro_f1']:.3f}"
            )
    model.load_state_dict(torch.load(output / "best.pt", weights_only=True)["model_state"])
    model.eval()
    tr, tr_predictions = evaluate(model, train_eval, key, weights)
    dev, dev_predictions = evaluate(model, dev_eval, key, weights)
    result = dict(
        selected_epoch=best_epoch,
        train=supported_metrics(tr),
        development=supported_metrics(dev),
        last_epoch=last,
        parameters=sum(p.numel() for p in model.parameters()),
        initial_shared_sha256=initial_shared,
        checkpoint_sha256=sha256(output / "best.pt"),
    )
    write_json(output / "metrics.json", result)
    write_json(output / "sampling-audit.json", audit)
    write_json(output / "train-predictions.json", tr_predictions)
    write_json(output / "V5-predictions.json", dev_predictions)
    with (output / "curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    plot(output, curves, dev["confusion_matrix"])
    return model, result, audit


def prediction_metrics(detail):
    return supported_metrics(
        metrics(
            [CLASSES.index(r["target"]) for r in detail],
            [CLASSES.index(r["prediction"]) for r in detail],
        )
    )


def evaluate_record(model, ds, key, output):
    metric, detail = evaluate(
        model, DataLoader(ds, batch_size=32, collate_fn=collate), key, torch.ones(6)
    )
    write_json(output, detail)
    return supported_metrics(metric), detail


def error_by_duration(clips, detail, cfg):
    lookup = {(r["recording_id"], r["source_row"]): r for r in detail}
    if set(lookup) != {(c["recording_id"], c["source_row"]) for c in clips}:
        raise ValueError("Prediction identity mismatch")
    result = {}
    for label in ("all", *CLASSES):
        bins = {}
        for c in clips:
            if label != "all" and CLASSES[c["label"]] != label:
                continue
            k = str(duration_key(c, cfg))
            item = bins.setdefault(k, dict(n=0, errors=0))
            p = lookup[c["recording_id"], c["source_row"]]
            item["n"] += 1
            item["errors"] += int(p["target"] != p["prediction"])
        result[label] = bins
    return result


def preserve_inputs():
    previous = Path("reports/boxingvi/level1-limb-study-v1")
    preserved = json.loads((previous / "preserved-inputs.json").read_text())
    preserved.update({str(p): sha256(p) for p in previous.rglob("*") if p.is_file()})
    if any("/V9/" in p or "/V9." in p for p in preserved):
        raise ValueError("V9 outside scope")
    verify_manifest(preserved)
    source = previous / "seed-20261021/xy_velocity_limb/source/src/boxing_motion_ai/training"
    current = Path(__file__).parent
    if sha256(source / "models.py") != sha256(current / "models.py"):
        raise ValueError("Architecture changed")
    for filename, names in {
        "data.py": ["features"],
        "velocity.py": ["joint_velocities", "source_gap_limit", "VELOCITY_POLICY"],
        "limbs.py": ["limb_features", "LIMB_POLICY"],
        "augmentation.py": ["mirror_features", "augment_batch"],
    }.items():
        if any(ast_item(source / filename, n) != ast_item(current / filename, n) for n in names):
            raise ValueError("Accepted representation/augmentation changed")
    return preserved


def verify_manifest(manifest):
    if any(sha256(p) != d for p, d in manifest.items()):
        raise ValueError("Preserved input changed")


def snapshot(output, config_path):
    paths = [
        *Path("src/boxing_motion_ai").rglob("*.py"),
        config_path,
        Path("configs/level1_initial.yaml"),
        Path("configs/level1_velocity_transfer.yaml"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    hashes = {}
    for p in paths:
        resolved = p.resolve()
        relative = (
            resolved.relative_to(Path.cwd().resolve())
            if resolved.is_relative_to(Path.cwd().resolve())
            else Path("submitted-config") / p.name
        )
        dest = output / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        hashes[relative.as_posix()] = sha256(p)
    write_json(output / "implementation.json", hashes)


def summarize_case(runs, seeds):
    result = {}
    for recording in runs[str(seeds[0])][ARMS[0]]:
        result[recording] = {}
        for name in MODELS:
            a = {}
            for arm in ARMS:
                rs = [runs[str(s)][arm][recording][name] for s in seeds]
                a[arm] = {
                    k: summarize([m[k] for m in rs])
                    for k in ("macro_f1", "supported_macro_f1", "accuracy")
                }
                a[arm]["per_class"] = {
                    c: {
                        k: summarize([m["per_class"][c][k] for m in rs])
                        if rs[0]["per_class"][c]["support"] or k == "precision"
                        else None
                        for k in ("precision", "recall")
                    }
                    for c in CLASSES
                }
                a[arm]["mean_confusion_counts"] = np.mean(
                    [m["confusion_matrix"] for m in rs], axis=0
                ).tolist()
            for measure in ("macro_f1", "supported_macro_f1", "accuracy"):
                a["paired_" + measure + "_gain"] = summarize(
                    [
                        runs[str(s)][ARMS[1]][recording][name][measure]
                        - runs[str(s)][ARMS[0]][recording][name][measure]
                        for s in seeds
                    ]
                )
            result[recording][name] = a
    return result


def run(config_path, output, smoke=False, audit_only=False, progress=print):
    config_path, output = Path(config_path), Path(output)
    cfg, base, specs = settings(config_path)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Choose a fresh output below ignored reports/")
    preserved = preserve_inputs()
    seed_all(0, base["threads"])
    cache = load_pool(specs, 2 if smoke else None)
    audit = audit_pool(cache)
    approved = json.loads(
        Path("reports/boxingvi/level1-limb-readiness-v2/summary.json").read_text()
    )
    for v in ALL:
        actual = cache[v][ARMS[0]].provenance
        prior = approved["recordings"][v]["provenance"]
        if any(actual[k] != prior[k] for k in ("hashes", "policy_sha256", "pose_source_hashes")):
            raise ValueError("Inputs differ from approved provisional pool")
        if not smoke and actual["source_rows"] != prior["source_rows"]:
            raise ValueError("Approved exclusions or selected examples changed")
    output.mkdir(parents=True)
    shutil.copyfile(config_path, output / "study.yaml")
    snapshot(output, config_path)
    write_json(output / "preserved-inputs.json", preserved)
    write_json(output / "dataset-audit.json", audit)
    cases = [dict(name="combined", train=list(TRAIN), development=["V5"], heldout=[]), *folds()]
    if audit_only:
        for case in cases:
            pool = Pool([cache[v][ARMS[0]] for v in case["train"]])
            p = sample_probabilities(pool.clips)
            write_json(output / f"{case['name']}-sampling.json", sampling_summary(pool.clips, p))
        return audit
    result = dict(
        schema_version="multi-recording-result-1.0",
        smoke=smoke,
        warning="V5 development-selected; LOGO pilot one seed; identities unknown. "
        "V7/V8 and V1/V6 guards retained. V9 unopened.",
        cases={},
    )
    for case in cases:
        validate_fold(case["train"], case["development"], case["heldout"])
        case_root = output / case["name"]
        case_root.mkdir()
        seeds = list(SEEDS[:1] if smoke or case["heldout"] else SEEDS)
        probabilities = sample_probabilities(Pool([cache[v][ARMS[0]] for v in case["train"]]).clips)
        write_json(
            case_root / "fold.json",
            {
                **case,
                "selection_recordings": ["V5"],
                "training_groups": sorted({GROUPS[v] for v in case["train"]}),
                "heldout_groups": sorted({GROUPS[v] for v in case["heldout"]}),
                "guards": GUARDS,
            },
        )
        train_meta = Pool([cache[v][ARMS[0]] for v in case["train"]])
        write_json(case_root / "sampling.json", sampling_summary(train_meta.clips, probabilities))
        duration = {}
        for v in ["V5", *case["heldout"]]:
            detail, table = duration_control(
                train_meta.clips, probabilities, cache[v][ARMS[0]].clips, cfg
            )
            duration[v] = prediction_metrics(detail)
            write_json(case_root / f"{v}-duration-predictions.json", detail)
        write_json(case_root / "duration-training-counts.json", table)
        runs = {}
        training = {}
        models = {}
        audits = {}
        # Held-out data are never passed into fit; all selectors are V5 only.
        for seed in seeds:
            runs[str(seed)] = {}
            training[str(seed)] = {}
            for arm in ARMS:
                pool = Pool([cache[v][arm] for v in case["train"]])
                training[str(seed)][arm] = {}
                runs[str(seed)][arm] = {}
                for name in MODELS:
                    dest = case_root / f"seed-{seed}" / arm / name
                    model, m, log = fit(
                        pool,
                        cache["V5"][arm],
                        probabilities,
                        base if not smoke else {**base, "epochs": 2, "limit_per_class": 2},
                        seed,
                        arm,
                        name,
                        dest,
                        progress,
                    )
                    models[seed, arm, name] = model
                    audits[seed, arm, name] = log
                    training[str(seed)][arm][name] = m
            for name in MODELS:
                if audits[seed, ARMS[0], name] != audits[seed, ARMS[1], name]:
                    raise ValueError("A/B sampling or mirror decisions differ")
                if (
                    training[str(seed)][ARMS[0]][name]["initial_shared_sha256"]
                    != training[str(seed)][ARMS[1]][name]["initial_shared_sha256"]
                ):
                    raise ValueError("A/B shared initialization differs")
            if audits[seed, ARMS[0], MODELS[0]] != audits[seed, ARMS[0], MODELS[1]]:
                raise ValueError("Full/mask sampling or mirror decisions differ")
        # Selections for every seed/arm/control in this case are now frozen.
        for seed in seeds:
            for arm in ARMS:
                for v in ["V5", *case["heldout"], *case["train"]]:
                    runs[str(seed)][arm][v] = {}
                    for name in MODELS:
                        key = "mask_x" if name.endswith("mask_only") else "x"
                        dest = case_root / f"seed-{seed}" / arm / name
                        m, detail = evaluate_record(
                            models[seed, arm, name],
                            cache[v][arm],
                            key,
                            dest / f"{v}-predictions.json",
                        )
                        runs[str(seed)][arm][v][name] = m
                        if v in ["V5", *case["heldout"]]:
                            write_json(
                                dest / f"{v}-duration-associations.json",
                                error_by_duration(cache[v][arm].clips, detail, cfg),
                            )
        if case["heldout"]:
            held_clips = Pool([cache[v][ARMS[0]] for v in case["heldout"]]).clips
            detail, _ = duration_control(train_meta.clips, probabilities, held_clips, cfg)
            duration["heldout_group"] = prediction_metrics(detail)
            write_json(case_root / "heldout-group-duration-predictions.json", detail)
            for seed in seeds:
                for arm in ARMS:
                    runs[str(seed)][arm]["heldout_group"] = {}
                    for name in MODELS:
                        dest = case_root / f"seed-{seed}" / arm / name
                        details = [
                            row
                            for v in case["heldout"]
                            for row in json.loads((dest / f"{v}-predictions.json").read_text())
                        ]
                        runs[str(seed)][arm]["heldout_group"][name] = prediction_metrics(details)
                        write_json(dest / "heldout-group-predictions.json", details)
        data = dict(
            fold=case,
            seeds=seeds,
            runs=runs,
            training=training,
            duration_control=duration,
            aggregate=summarize_case(runs, seeds),
        )
        result["cases"][case["name"]] = data
        write_json(case_root / "results.json", data)
        write_json(output / "summary.json", result)
        progress(f"Completed {case['name']}; checkpoints selected only on V5")
    verify_manifest(preserved)
    # Independently check every prepared selection, policy and source pose journal remains intact.
    for v in ALL:
        p = cache[v][ARMS[0]].provenance
        for name, digest in p["hashes"].items():
            if sha256(Path(p["dataset"]) / name) != digest:
                raise ValueError("Dataset changed")
        if sha256(p["policy_path"]) != p["policy_sha256"]:
            raise ValueError("Policy changed")
    write_json(output / "preservation.json", dict(verified_prior_files=len(preserved), changed=[]))
    write_report(output, result, audit)
    return result


def write_report(output, result, audit):
    lines = [
        "# Multi-recording Level 1 study",
        "",
        result["warning"],
        "",
        "Six-class F1 uses the full taxonomy. Supported F1 excludes absent ground-truth classes "
        "only from its averaging; predictions into absent classes remain errors.",
        "",
        "| Case | Evaluation | Input | A accuracy → B | A F1 six → B | A F1 supported → B |",
        "|---|---|---|---:|---:|---:|",
    ]
    for case, d in result["cases"].items():
        targets = ["V5", *d["fold"]["heldout"]]
        if d["fold"]["heldout"]:
            targets.append("heldout_group")
        for v in targets:
            for name in MODELS:
                a = d["aggregate"][v][name]
                x, y = [a[arm] for arm in ARMS]
                lines.append(
                    f"| {case} | {v} | {name} | {x['accuracy']['mean']:.4f} → "
                    f"{y['accuracy']['mean']:.4f} | {x['macro_f1']['mean']:.4f} → "
                    f"{y['macro_f1']['mean']:.4f} | {x['supported_macro_f1']['mean']:.4f} → "
                    f"{y['supported_macro_f1']['mean']:.4f} |"
                )
    for case, d in result["cases"].items():
        for v in ["V5", *d["fold"]["heldout"]]:
            lines += [
                "",
                f"## {case}: {v}",
                "",
                "| Class | Eligible | Source status | A recall → B |",
                "|---|---:|---|---:|",
            ]
            for c, s in audit[v]["source_support"].items():
                rec = [d["aggregate"][v]["tcn"][arm]["per_class"][c]["recall"] for arm in ARMS]
                cell = (
                    "unsupported"
                    if rec[0] is None
                    else f"{rec[0]['mean']:.4f} → {rec[1]['mean']:.4f}"
                )
                lines.append(f"| {c} | {s['retained']} | {s['status']} | {cell} |")
            for name in MODELS:
                for arm in ARMS:
                    lines += [
                        "",
                        f"### {name} {arm} mean confusion counts",
                        "",
                        "Rows true / columns predicted: " + ", ".join(CLASSES),
                        "",
                        "```json",
                        json.dumps(d["aggregate"][v][name][arm]["mean_confusion_counts"]),
                        "```",
                    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, args.smoke, args.audit_only)
