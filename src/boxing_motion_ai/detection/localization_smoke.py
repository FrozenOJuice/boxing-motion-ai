"""One fixed-budget, three-input engineering experiment; never model promotion."""

import argparse
import hashlib
import importlib.metadata
import platform
import resource
from fractions import Fraction as Q
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from ..analysis.common import sha, write_json
from ..data.continuous_dataset import validate
from ..vision.observation_study import BASE, arrays, fresh, read
from . import instances_v2
from .instance_diagnostics import compare
from .localization_data import POLICY, authorized, load_cache
from .localization_data import VERSION as FEATURE_VERSION
from .localization_decode import VERSION as DECODER_VERSION
from .localization_decode import decode
from .localization_model import MODES, VERSION, Localizer, fit, model_observable, predict


class DecoderPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    activity_threshold: Literal[0.5]
    boundary_threshold: Literal[0.5]
    peak_radius_seconds: Literal["2/25"]
    minimum_span_seconds: Literal["2/25"]
    maximum_span_seconds: Literal["2"]
    minimum_activity_fraction: Literal[0.5]


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["punch-localization-smoke-1.0"]
    dataset_sha256: Literal["c7993318ee8f8a16e2cd89f04ad0acaf7ce95c0cad8aabad0c70079b683b2d59"]
    seed: Literal[20261029]
    epochs: Literal[120]
    hidden: Literal[32]
    dropout: Literal[0.1]
    learning_rate: Literal[0.003]
    weight_decay: Literal[0.0001]
    gradient_clip: Literal[1.0]
    threads: Literal[2]
    boundary_radius_seconds: Literal["1/12"]
    maximum_positive_weight: Literal[40.0]
    decoder: DecoderPolicy
    purpose: Literal["user_authorized_v4_v6_training_source_engineering_only_no_selection"]


def state_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def pose_origins(cases):
    asset = read("configs/assisted_level1_v1.json")["pose_model"]
    if sha(Path(asset["path"])) != asset["sha256"]:
        raise ValueError("Original pose asset changed")
    receipts = {}
    for recording in ("V4", "V6"):
        path = Path(f"data/processed/boxingvi/{recording}/pose-full-image-v1/manifest.json")
        original = read(path)
        if original["inputs"]["model"]["sha256"] != asset["sha256"]:
            raise ValueError("Unexpected native pose estimator")
        receipts[str(path)] = sha(path)
    for case in cases.values():
        source = case["receipt"]["source"]["source"]
        if source["kind"] != "prepared_upload":
            continue
        path = Path(source["directory"]) / "pose.json"
        original = read(path)
        if (
            original["pose_model_sha256"] != asset["sha256"]
            or original["poses_sha256"] != source["poses_sha256"]
        ):
            raise ValueError("Unexpected uploaded pose estimator/source")
        receipts[str(path)] = sha(path)
    return dict(
        asset=asset,
        metadata_sha256=receipts,
        extraction="original Full MediaPipe IMAGE, confidence mask 0.5; no pose changes",
    )


def comparison(result, case):
    video, prov = case["video"], case["provenance"]
    if prov["kind"] != "reviewed_source":
        return None
    times = [Q(r["timestamp_exact"]) for r in video["frames"]]
    c = compare(
        result["candidates"],
        case["references"],
        times,
        Q(video["stop_timestamp_exact"]),
        (prov["review_start"] - prov["source_start"], prov["review_stop"] - prov["source_start"]),
    )
    c["matched_boundary_errors"] = []
    for match in c["matches_iou_half"]:
        ref = next(r for r in case["references"] if r["event_id"] == match["reference_id"])
        p = result["candidates"][match["candidate_index"]]

        def clock(i):
            return times[i] if i < len(times) else Q(video["stop_timestamp_exact"])

        c["matched_boundary_errors"].append(
            dict(
                event_id=ref["event_id"],
                type=ref["type"],
                predicted_frames=[
                    p["start_frame"] + prov["source_start"],
                    p["stop_frame_exclusive"] + prov["source_start"],
                ],
                reference_frames=[
                    ref["start_frame"] + prov["source_start"],
                    ref["stop_frame_exclusive"] + prov["source_start"],
                ],
                start_error_frames=p["start_frame"] - ref["start_frame"],
                stop_error_frames=p["stop_frame_exclusive"] - ref["stop_frame_exclusive"],
                start_error_seconds_exact=str(clock(p["start_frame"]) - clock(ref["start_frame"])),
                stop_error_seconds_exact=str(
                    clock(p["stop_frame_exclusive"]) - clock(ref["stop_frame_exclusive"])
                ),
                iou_exact=match["iou_exact"],
            )
        )
    return c


def summarize(records):
    positive = [r for r in records if r["case"].startswith("V4-bg-")]
    negative = [r for r in records if "-motion-" in r["case"]]
    return dict(
        native_candidates=sum(r["comparison"]["reviewed_candidate_count"] for r in positive),
        matches=sum(len(r["comparison"]["matches_iou_half"]) for r in positive),
        missed=sum(len(r["comparison"]["missed_references"]) for r in positive),
        background_proposals=sum(
            len(r["comparison"]["candidate_without_reference_overlap"]) for r in positive
        ),
        merge_risks=sum(len(r["comparison"]["merge_risks"]) for r in positive),
        split_risks=sum(len(r["comparison"]["split_risks"]) for r in positive),
        negative_candidates=sum(len(r["result"]["candidates"]) for r in negative),
        unresolved_ranges=sum(
            len(r["result"].get("unresolved", r["result"].get("unresolved_activity", [])))
            for r in records
        ),
        upload_candidates=[
            len(next(r for r in records if r["case"] == n)["result"]["candidates"])
            for n in ("V6-upload", "V4-8s-upload", "V4-29s-upload")
        ],
    )


def run(cache, output):
    config_path = Path("configs/punch_localization_smoke_v1.json")
    config = Experiment.model_validate(read(config_path)).model_dump()
    manifest = validate(Path.cwd())
    authorized(manifest)
    cases, feature_manifest = load_cache(cache, config)
    out = fresh(output)
    write_json(out / "config.json", config)
    source_paths = [
        *Path(__file__).parent.glob("localization*.py"),
        Path("src/boxing_motion_ai/training/models.py"),
        Path("src/boxing_motion_ai/detection/instances_v2.py"),
        Path("src/boxing_motion_ai/detection/instance_diagnostics.py"),
        Path("src/boxing_motion_ai/analysis/automatic.py"),
    ]
    provenance = dict(
        config_sha256=sha(config_path),
        cache_manifest_sha256=sha(Path(cache) / "manifest.json"),
        pretrain_checks={
            name: sha(Path(cache) / name)
            for name in ("verification.json", "qa-receipt.json")
            if (Path(cache) / name).exists()
        },
        dataset=feature_manifest["dataset"],
        features=POLICY,
        code_sha256={str(p): sha(p) for p in source_paths},
        model_version=VERSION,
        decoder_version=DECODER_VERSION,
        feature_version=FEATURE_VERSION,
        frozen_v2_policy=read("configs/punch_instances_v2.json"),
        frozen_v2_policy_sha256=sha(Path("configs/punch_instances_v2.json")),
        pose_origins=pose_origins(cases),
        retained_production_pins={
            p: read(p)
            for p in ("configs/assisted_level1_v1.json", "configs/uploaded_activity_v1.json")
        },
        training_source_only=True,
        runtime=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            numpy=np.__version__,
            pillow=importlib.metadata.version("Pillow"),
            device="cpu",
            threads=config["threads"],
        ),
        context="offline centered RF15, seven native frames each direction; "
        "source resets/abstentions split sequences",
        selection="last fixed epoch only; no validation/threshold/model selection",
    )
    write_json(out / "provenance.json", provenance)
    snapshot = out / "source-snapshot"
    for path in source_paths:
        dest = snapshot / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    policy2 = instances_v2.Policy.model_validate(read("configs/punch_instances_v2.json"))
    baseline = []
    for name, case in cases.items():
        raw = arrays(BASE / name / "poses.npz")
        result = instances_v2.segment(
            raw, case["video"], read(BASE / name / "activity.json"), policy2
        )
        baseline.append(dict(case=name, result=result, comparison=comparison(result, case)))
    write_json(out / "v2-comparison.json", baseline)
    summaries = {"v2": summarize(baseline)}
    for mode in MODES:
        dest = out / mode
        dest.mkdir()
        began = perf_counter()
        model, diagnostic = fit(cases, mode, config)
        train_seconds = perf_counter() - began
        torch.save(
            dict(
                model_version=VERSION,
                mode=mode,
                config=config,
                state_dict=model.state_dict(),
                feature_version=FEATURE_VERSION,
                cache_manifest_sha256=provenance["cache_manifest_sha256"],
            ),
            dest / "checkpoint.pt",
        )
        write_json(dest / "fit.json", diagnostic)
        # Exact same seed/budget repeated only for determinism verification, not another setting.
        began = perf_counter()
        repeated, repeat_diagnostic = fit(cases, mode, config)
        repeat_seconds = perf_counter() - began
        identical = state_hash(model) == state_hash(repeated) and diagnostic == repeat_diagnostic
        write_json(dest / "repeat-fit.json", repeat_diagnostic)
        torch.save(
            dict(
                model_version=VERSION,
                mode=mode,
                config=config,
                state_dict=repeated.state_dict(),
                feature_version=FEATURE_VERSION,
            ),
            dest / "determinism-repeat.pt",
        )
        if not identical:
            raise ValueError("Determinism verification failed; preserve experiment, do not tune")
        restored = Localizer(config["hidden"], config["dropout"])
        restored.load_state_dict(
            torch.load(dest / "checkpoint.pt", weights_only=True, map_location="cpu")["state_dict"]
        )
        restored.eval()
        records = []
        inference_seconds = decode_seconds = 0.0
        ablations = []
        for name, case in cases.items():
            began = perf_counter()
            p = predict(model, case, mode)
            inference_seconds += perf_counter() - began
            if not np.array_equal(p, predict(restored, case, mode), equal_nan=True):
                raise ValueError("Checkpoint reload changed source-frame predictions")
            if not np.array_equal(p, predict(repeated, case, mode), equal_nan=True):
                raise ValueError("Repeated training changed source-frame predictions")
            began = perf_counter()
            result = decode(
                p,
                model_observable(case["arrays"]),
                case["arrays"]["reset"],
                case["video"],
                config["decoder"],
            )
            decode_seconds += perf_counter() - began
            np.savez_compressed(
                dest / f"{name}-scores.npz", scores=p, observable=model_observable(case["arrays"])
            )
            records.append(
                dict(
                    case=name,
                    source=case["provenance"],
                    result=result,
                    comparison=comparison(result, case),
                )
            )
            # Inference-only controls: remove values while preserving availability bits.
            controls = {}
            for what in ("zero_motion", "pose_masks_only"):
                altered = {**case, "arrays": {k: v.copy() for k, v in case["arrays"].items()}}
                if what == "zero_motion":
                    altered["arrays"]["visual"][:, :96] = 0
                else:
                    for j in range(8):
                        altered["arrays"]["pose"][:, j * 5 : j * 5 + 4] = 0
                        altered["arrays"]["pose"][:, 41 + j * 3 : 43 + j * 3] = 0
                cp = predict(model, altered, mode)
                controls[what] = dict(
                    mean_absolute_score_change=float(np.nanmean(abs(cp - p)))
                    if np.isfinite(p).any()
                    else None,
                    candidates=len(
                        decode(
                            cp,
                            model_observable(case["arrays"]),
                            case["arrays"]["reset"],
                            case["video"],
                            config["decoder"],
                        )["candidates"]
                    ),
                )
            ablations.append(dict(case=name, **controls))
        write_json(dest / "records.json", records)
        write_json(dest / "input-removal-controls.json", ablations)
        summary = dict(
            **summarize(records),
            parameters=sum(p.numel() for p in model.parameters()),
            parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
            checkpoint_sha256=sha(dest / "checkpoint.pt"),
            state_sha256=state_hash(model),
            training_seconds=train_seconds,
            determinism_repeat_seconds=repeat_seconds,
            inference_seconds=inference_seconds,
            decoding_seconds=decode_seconds,
            deterministic_repeat=True,
            reload_exact=True,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
            initial_loss=diagnostic["initial_loss"],
            final_loss=diagnostic["final_loss"],
            interpretation="training-source engineering fit only; "
            "no winner selected or six-class predictions",
        )
        write_json(dest / "summary.json", summary)
        summaries[mode] = summary
        print(mode, summary, flush=True)
    write_json(out / "summary.json", summaries)
    # Verify old artifacts after every expensive operation; never inspect V9.
    protected = read(Path(cache) / "protected-before.json")
    changed = []
    for path, digest in protected.items():
        if "v9" in path.lower() or sha(Path(path)) != digest:
            changed.append(path)
    write_json(
        out / "preservation.json",
        dict(checked_files=len(protected), changed=changed, v9_accessed=False),
    )
    if changed:
        raise ValueError("Protected artifacts changed")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.output)
