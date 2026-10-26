"""Matched original/curated XY+velocity study, with unchanged evaluation membership."""

import argparse
from copy import copy
from pathlib import Path

import numpy as np

from ..data.annotation_review import file_hash
from ..data.curated_subset import prepare
from .data import CLASSES
from .experiment import write_json
from .mirror_study import summarize
from .multi_recording import (
    SEEDS,
    TRAIN,
    Pool,
    duration_control,
    folds,
    load_pool,
    sample_probabilities,
    sampling_summary,
    settings,
    validate_fold,
)
from .multi_study import (
    evaluate_record,
    fit,
    prediction_metrics,
    preserve_inputs,
    snapshot,
    verify_manifest,
)


def subset(dataset, rows):
    """Reuse exact feature tensors and metadata, filtering by validated source IDs only."""
    wanted = set(rows)
    if len(wanted) != len(rows) or not wanted:
        raise ValueError("Empty or duplicate subset")
    ds = copy(dataset)
    ds.clips = [c for c in dataset.clips if c["source_row"] in wanted]
    if {c["source_row"] for c in ds.clips} != wanted:
        raise ValueError("Subset references unavailable examples")
    ds.provenance = {**dataset.provenance, "curated_source_rows": sorted(wanted)}
    return ds


def paired_metrics(original, curated):
    def keys(rows):
        return [(r["recording_id"], r["source_row"], r["target"]) for r in rows]

    if keys(original) != keys(curated):
        raise ValueError("Paired evaluation membership or labels differ")
    return prediction_metrics(original), prediction_metrics(curated)


def aggregate(runs):
    result = {}
    for target in next(iter(runs.values()))["original"]["evaluation"]:
        result[target] = {}
        for arm in ("original", "curated"):
            ms = [r[arm]["evaluation"][target] for r in runs.values()]
            result[target][arm] = {
                k: summarize([m[k] for m in ms])
                for k in ("macro_f1", "supported_macro_f1", "accuracy")
            }
            result[target][arm]["per_class"] = {
                c: {
                    k: summarize([m["per_class"][c][k] for m in ms])
                    for k in ("precision", "recall", "support")
                }
                for c in CLASSES
            }
            result[target][arm]["mean_confusion_counts"] = np.mean(
                [m["confusion_matrix"] for m in ms], axis=0
            ).tolist()
        result[target]["paired_gains"] = {
            k: summarize(
                [
                    r["curated"]["evaluation"][target][k] - r["original"]["evaluation"][target][k]
                    for r in runs.values()
                ]
            )
            for k in ("macro_f1", "supported_macro_f1", "accuracy")
        }
    return result


def run(output, smoke=False):
    output = Path(output)
    cfg_path = Path("configs/level1_curated_subset_v1.json")
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Choose a fresh output under reports/")
    cfg, base, specs = settings("configs/level1_multi_recording.yaml")
    selection = prepare(cfg_path, output / "dataset")
    preserved = preserve_inputs()
    preserved.update(selection["protected"])
    # Preserve all previously trained multi-recording reference artifacts as well.
    prior = Path("reports/boxingvi/level1-multi-recording-v1")
    for p in prior.rglob("*"):
        if p.is_file():
            preserved[str(p)] = file_hash(p)
    verify_manifest(preserved)
    snapshot(output, cfg_path)
    write_json(output / "preserved-inputs.json", preserved)
    cache = {v: arms["xy_velocity"] for v, arms in load_pool(specs).items()}
    for v in TRAIN:
        if sorted(c["source_row"] for c in cache[v].clips) != selection["baseline"][v]:
            raise ValueError("Baseline differs from pinned approved candidates")
    if sorted(c["source_row"] for c in cache["V5"].clips) != selection["development_rows"]:
        raise ValueError("Development membership changed")
    curated = {v: subset(cache[v], selection["selected"][v]) for v in TRAIN}
    cases = [dict(name="combined", train=list(TRAIN), development=["V5"], heldout=[]), *folds()]
    result = dict(
        schema_version="curation-study-1.0",
        smoke=smoke,
        cases={},
        warning="V5 selects checkpoints; group holdouts are diagnostic, not V9 test performance",
    )
    for case in cases:
        validate_fold(case["train"], case["development"], case["heldout"])
        dest = output / case["name"]
        dest.mkdir()
        write_json(dest / "fold.json", case)
        pools = {
            "original": Pool([cache[v] for v in case["train"]]),
            "curated": Pool([curated[v] for v in case["train"]]),
        }
        probs = {a: sample_probabilities(p.clips) for a, p in pools.items()}
        for arm in pools:
            write_json(
                dest / f"{arm}-sampling.json", sampling_summary(pools[arm].clips, probs[arm])
            )
        runs = {}
        for seed in SEEDS[:1] if smoke else SEEDS:
            runs[str(seed)] = {}
            paired = {}
            for arm, pool in pools.items():
                run_root = dest / f"seed-{seed}" / arm
                print(f"START {case['name']}/{seed}/{arm} n={len(pool)}", flush=True)
                model, m, _ = fit(
                    pool,
                    cache["V5"],
                    probs[arm],
                    base if not smoke else {**base, "epochs": 2},
                    seed,
                    "xy_velocity",
                    "tcn",
                    run_root,
                    progress=lambda message: print(message, flush=True),
                )
                evaluation, details = {}, {}
                for v in ["V5", *case["heldout"]]:
                    evaluation[v], details[v] = evaluate_record(
                        model, cache[v], "x", run_root / f"{v}-evaluation.json"
                    )
                if case["heldout"]:
                    details["heldout_group"] = [r for v in case["heldout"] for r in details[v]]
                    evaluation["heldout_group"] = prediction_metrics(details["heldout_group"])
                paired[arm] = details
                duration = {}
                for v in ["V5", *case["heldout"]]:
                    predictions, _ = duration_control(pool.clips, probs[arm], cache[v].clips, cfg)
                    duration[v] = prediction_metrics(predictions)
                runs[str(seed)][arm] = dict(training=m, evaluation=evaluation, duration=duration)
                write_json(run_root / "evaluation.json", runs[str(seed)][arm])
            for target in paired["original"]:
                paired_metrics(paired["original"][target], paired["curated"][target])
            if (
                runs[str(seed)]["original"]["training"]["initial_shared_sha256"]
                != runs[str(seed)]["curated"]["training"]["initial_shared_sha256"]
            ):
                raise ValueError("Matched initialization differs")
            data = dict(fold=case, runs=runs, aggregate=aggregate(runs))
            result["cases"][case["name"]] = data
            write_json(dest / "results.json", data)
            write_json(output / "summary.json", result)
    verify_manifest(preserved)
    write_json(output / "preservation.json", dict(verified_files=len(preserved), changed=[]))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.output, args.smoke)
