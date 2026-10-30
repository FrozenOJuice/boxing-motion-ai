"""One frozen three-arm blue-reference experiment; no production promotion or model selection."""

import argparse
import platform
import resource
from fractions import Fraction as Q
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..analysis.common import sha, write_json
from ..data.blue_localization import PIN, validate
from ..services.dashboard import require
from ..vision.observation_study import BASE, arrays, fresh, read
from . import instances_v2
from .localization_blue_features import CONFIG, VERSION, load
from .localization_decode import decode
from .localization_model import MODES, Localizer, fit, model_observable, predict
from .localization_smoke import comparison, state_hash

OLD_RUN = Path("reports/boxingvi/punch-localization-smoke-v1")


def config():
    cfg = read(CONFIG)
    original = read("configs/punch_localization_smoke_v1.json")
    training = {
        k: v
        for k, v in original.items()
        if k not in ("schema_version", "dataset_sha256", "purpose")
    }
    require(
        cfg["schema_version"] == "blue-punch-localization-experiment-1.0"
        and cfg["training"] == training
        and cfg["dataset"] == read(PIN),
        "blue_frozen_training_budget_changed",
    )
    return cfg


def evaluate(result, case):
    p = case["provenance"]
    if p["kind"] != "reviewed_source":
        return None
    if p.get("exhaustive") is False:
        # Full-instance IoU. Never clip a broad prediction to a sparse positive span.
        adapted = case | dict(
            provenance=p
            | dict(
                review_start=p["source_start"],
                review_stop=p["source_start"] + len(case["video"]["frames"]),
            )
        )
        c = comparison(result, adapted)
        c["coverage_kind"] = "partial_positive_references_only"
        c["unmatched_unlabeled_candidates"] = c.pop("candidate_without_reference_overlap")
        c["false_positive_count"] = None
        c["precision"] = c["full_event_recall"] = None
        c["conflicted_selected_references_unscored"] = p["selected_events"] - p["scorable_events"]
    else:
        c = comparison(result, case)
        c["coverage_kind"] = "original_exhaustive_core"
    return c


def masks_time(video, mask):
    times = [Q(f["timestamp_exact"]) for f in video["frames"]] + [Q(video["stop_timestamp_exact"])]
    return str(sum((times[i + 1] - times[i] for i in np.flatnonzero(mask)), Q()))


def record(name, result, case):
    n = len(case["video"]["frames"])
    unresolved = np.zeros(n, bool)
    for r in result.get("unresolved", result.get("unresolved_activity", [])):
        unresolved[r["start_frame"] : r["stop_frame_exclusive"]] = True
    unknown = ~model_observable(case["arrays"])
    return dict(
        case=name,
        provenance=case["provenance"],
        result=result,
        comparison=evaluate(result, case),
        unresolved_frames=int(unresolved.sum()),
        unresolved_seconds=masks_time(case["video"], unresolved),
        unobservable_frames=int(unknown.sum()),
        unobservable_seconds=masks_time(case["video"], unknown),
    )


def summarize(records):
    groups = {
        "v4_original_training": lambda r: r["case"].startswith("V4-bg-"),
        "v4_sparse_training": lambda r: r["case"].startswith("V4-anchor-"),
        "v6_blue_training": lambda r: r["case"].startswith("V6-motion-10"),
        "v5_development": lambda r: r["case"] == "V5-motion-101",
        "verified_negative_training": lambda r: (
            r["case"] in {f"{v}-motion-{i}" for v in ("V4", "V6") for i in range(1, 7)}
        ),
    }
    output = {}
    for name, predicate in groups.items():
        subset = [r for r in records if predicate(r)]
        cs = [r["comparison"] for r in subset]
        errors = [e for c in cs for e in c["matched_boundary_errors"]]
        partial = any(c["coverage_kind"] != "original_exhaustive_core" for c in cs)
        output[name] = dict(
            cases=len(subset),
            candidates=sum(c["reviewed_candidate_count"] for c in cs),
            references=sum(c["reviewed_reference_count"] for c in cs),
            matched=sum(len(c["matches_iou_half"]) for c in cs),
            missed=sum(len(c["missed_references"]) for c in cs),
            unmatched_candidates=sum(len(c["unmatched_candidates"]) for c in cs),
            no_reference_overlap=sum(
                len(
                    c.get(
                        "unmatched_unlabeled_candidates",
                        c.get("candidate_without_reference_overlap", []),
                    )
                )
                for c in cs
            ),
            extra_interpretation="unlabeled proposals, not false positives"
            if partial
            else "exhaustive-core background proposals where no overlap",
            merge_risks=sum(len(c["merge_risks"]) for c in cs),
            split_risks=sum(len(c["split_risks"]) for c in cs),
            unscored_conflicted_references=sum(
                c.get("conflicted_selected_references_unscored", 0) for c in cs
            ),
            median_absolute_onset_error_frames=float(
                np.median([abs(e["start_error_frames"]) for e in errors])
            )
            if errors
            else None,
            median_absolute_stop_error_frames=float(
                np.median([abs(e["stop_error_frames"]) for e in errors])
            )
            if errors
            else None,
            maximum_absolute_onset_error_frames=max(
                (abs(e["start_error_frames"]) for e in errors), default=None
            ),
            maximum_absolute_stop_error_frames=max(
                (abs(e["stop_error_frames"]) for e in errors), default=None
            ),
            unresolved_ranges=sum(
                len(r["result"].get("unresolved", r["result"].get("unresolved_activity", [])))
                for r in subset
            ),
            unresolved_frames=sum(r["unresolved_frames"] for r in subset),
            unresolved_seconds=str(sum((Q(r["unresolved_seconds"]) for r in subset), Q())),
            unobservable_frames=sum(r["unobservable_frames"] for r in subset),
            unobservable_seconds=str(sum((Q(r["unobservable_seconds"]) for r in subset), Q())),
        )
    output["uploads"] = {
        r["case"]: dict(
            candidates=len(r["result"]["candidates"]),
            unresolved_frames=r["unresolved_frames"],
            unobservable_frames=r["unobservable_frames"],
        )
        for r in records
        if r["provenance"]["kind"] != "reviewed_source"
    }
    return output


def run(cache, output):
    cfg = config()
    torch.set_num_threads(cfg["training"]["threads"])
    torch.use_deterministic_algorithms(True)
    dataset = validate(Path.cwd())
    cases, manifest = load(cache)
    out = fresh(output)
    write_json(out / "config.json", cfg)
    code = [
        Path(__file__),
        Path("src/boxing_motion_ai/detection/localization_model.py"),
        Path("src/boxing_motion_ai/detection/localization_decode.py"),
        Path("src/boxing_motion_ai/detection/instance_diagnostics.py"),
        Path("src/boxing_motion_ai/detection/instances_v2.py"),
    ]
    write_json(
        out / "provenance.json",
        dict(
            dataset=read(PIN),
            feature_version=VERSION,
            cache=dict(path=str(cache), sha256=sha(Path(cache) / "manifest.json")),
            config_sha256=sha(CONFIG),
            code_sha256={str(p): sha(p) for p in code},
            model="unchanged offline-punch-localizer-1.0 / 14819 parameters",
            decoder="unchanged offline-boundary-pair-decoder-1.0",
            runtime=dict(
                python=platform.python_version(),
                torch=str(torch.__version__),
                numpy=np.__version__,
                device="cpu",
            ),
            authorization=cfg["purpose"],
            selection="fixed final epoch, no architecture/threshold/search/model promotion",
            training_cases=[n for n, c in cases.items() if c["receipt"]["train_eligible"]],
            development_cases=[
                n for n, c in cases.items() if c["provenance"].get("recording") == "V5"
            ],
            old_run_sha256=sha(OLD_RUN / "summary.json"),
            frozen_v2_sha256=sha(Path("configs/punch_instances_v2.json")),
        ),
    )
    for p in code:
        target = out / "source-snapshot" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    policy = instances_v2.Policy.model_validate(read("configs/punch_instances_v2.json"))
    baselines = []
    for name, case in cases.items():
        raw = arrays(
            Path(cache) / name / "poses.npz"
            if name in manifest["cases"]
            else BASE / name / "poses.npz"
        )
        activity = read(
            Path(cache) / name / "activity.json"
            if name in manifest["cases"]
            else BASE / name / "activity.json"
        )
        result = instances_v2.segment(raw, case["video"], activity, policy)
        baselines.append(record(name, result, case))
    write_json(out / "v2-records.json", baselines)
    summaries = {"v2": summarize(baselines)}
    # Previously trained smoke weights get the same new clips/references, without optimization.
    for mode in MODES:
        old = Localizer(cfg["training"]["hidden"], cfg["training"]["dropout"])
        old.load_state_dict(
            torch.load(OLD_RUN / mode / "checkpoint.pt", weights_only=True, map_location="cpu")[
                "state_dict"
            ]
        )
        old.eval()
        old_records = []
        for name, case in cases.items():
            scores = predict(old, case, mode)
            previous_scores = OLD_RUN / mode / f"{name}-scores.npz"
            if previous_scores.exists():
                require(
                    np.array_equal(scores, arrays(previous_scores)["scores"], equal_nan=True),
                    "old_smoke_predictions_changed",
                )
            result = decode(
                scores,
                model_observable(case["arrays"]),
                case["arrays"]["reset"],
                case["video"],
                cfg["training"]["decoder"],
            )
            old_records.append(record(name, result, case))
        write_json(out / f"previous-{mode}-records.json", old_records)
        summaries[f"previous_{mode}"] = summarize(old_records)
    for mode in MODES:
        dest = out / mode
        dest.mkdir()
        began = perf_counter()
        model, diagnostic = fit(cases, mode, cfg["training"])
        seconds = perf_counter() - began
        torch.save(
            dict(
                state_dict=model.state_dict(),
                mode=mode,
                config=cfg,
                dataset=read(PIN),
                feature_version=VERSION,
                model_version="offline-punch-localizer-1.0",
            ),
            dest / "checkpoint.pt",
        )
        write_json(dest / "fit.json", diagnostic)
        repeated, repeat_diagnostic = fit(cases, mode, cfg["training"])
        require(
            state_hash(model) == state_hash(repeated) and diagnostic == repeat_diagnostic,
            "blue_training_not_deterministic",
        )
        write_json(dest / "repeat-fit.json", repeat_diagnostic)
        restored = Localizer(cfg["training"]["hidden"], cfg["training"]["dropout"])
        restored.load_state_dict(
            torch.load(dest / "checkpoint.pt", weights_only=True, map_location="cpu")["state_dict"]
        )
        restored.eval()
        records = []
        inference = decoding = 0.0
        for name, case in cases.items():
            began = perf_counter()
            scores = predict(model, case, mode)
            inference += perf_counter() - began
            require(
                np.array_equal(scores, predict(repeated, case, mode), equal_nan=True)
                and np.array_equal(scores, predict(restored, case, mode), equal_nan=True),
                "blue_reload_predictions_changed",
            )
            began = perf_counter()
            result = decode(
                scores,
                model_observable(case["arrays"]),
                case["arrays"]["reset"],
                case["video"],
                cfg["training"]["decoder"],
            )
            decoding += perf_counter() - began
            np.savez_compressed(dest / f"{name}-scores.npz", scores=scores)
            records.append(record(name, result, case))
        write_json(dest / "records.json", records)
        summaries[mode] = summarize(records)
        write_json(
            dest / "verification.json",
            dict(
                deterministic_fit=True,
                reload_exact=True,
                training_seconds=seconds,
                inference_seconds=inference,
                decoding_seconds=decoding,
                parameters=sum(p.numel() for p in model.parameters()),
                parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
                state_sha256=state_hash(model),
                checkpoint_sha256=sha(dest / "checkpoint.pt"),
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if platform.system() == "Darwin" else 1024),
            ),
        )
        print(mode, summaries[mode], flush=True)
    write_json(out / "summary.json", summaries)
    require(validate(Path.cwd()) == dataset, "blue_dataset_changed_during_fit")
    protected = read(Path(cache) / "protected-before.json")
    changed = [p for p, h in protected.items() if sha(Path(p)) != h]
    write_json(
        out / "preservation.json",
        dict(checked_files=len(protected), changed=changed, v9_accessed=False),
    )
    require(not changed, "blue_protected_artifacts_changed")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.output)
