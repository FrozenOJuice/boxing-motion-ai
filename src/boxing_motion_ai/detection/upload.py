"""Full prepared-upload inference. No registered-source, workbook or human-label access."""

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from ..analysis.activity_contract import VERSION, inputs
from ..analysis.common import regular, write_json
from ..training.data import exact_times
from .activity_bouts import ActivityDecoder
from .data import VERSION as FEATURE_VERSION
from .data import encode_source
from .decoder import DecoderConfig, Observation, decode_batch
from .model import VERSION as MODEL_VERSION
from .model import BinaryDetector


def load_model(root, binding):
    saved = torch.load(
        regular(root / binding["checkpoint"]["path"]), map_location="cpu", weights_only=True
    )
    frozen = json.loads(regular(root / "configs/detector_smoke_v1.json").read_text())
    if not (
        saved["version"] == MODEL_VERSION
        and saved["feature_version"] == FEATURE_VERSION
        and saved["config"] == frozen
        and saved["dataset_sha256"] == frozen["dataset"]["sha256"] == binding["dataset_sha256"]
        and frozen["training_recording"] == "V4"
        and frozen["epochs"] == 30
        and frozen["seed"] == 20261021
        and frozen["hidden"] == 32
        and set(frozen["held_intervals"]) == {"V4-bg-4", "V4-bg-5"}
    ):
        raise ValueError("Incompatible frozen binary activity checkpoint provenance.")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = BinaryDetector()
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval()


def validate_timing(raw, video):
    if (
        len(raw["frame_index"]) != video["frame_count"]
        or len(video["frames"]) != video["frame_count"]
    ):
        raise ValueError("Prepared frame/pose inventory mismatch.")
    for i, row in enumerate(video["frames"]):
        for name in ("frame_index", "pts", "time_base_num", "time_base_den"):
            if type(row[name]) is not int or row[name] != int(raw[name][i]):
                raise ValueError("Cached pose timing does not match the prepared source frames.")
        if Fraction(row["timestamp_exact"]) != Fraction(
            row["pts"] * row["time_base_num"], row["time_base_den"]
        ):
            raise ValueError("Prepared exact timestamp mismatch.")
    times = exact_times({"raw": raw})
    end = Fraction(video["stop_timestamp_exact"])
    if end <= times[-1] or video["width"] <= 0 or video["height"] <= 0:
        raise ValueError("Invalid prepared video coverage/dimensions.")
    if (
        any(
            abs(row["media_time"] - float(t - times[0])) > 1e-9
            for row, t in zip(video["frames"], times, strict=True)
        )
        or abs(video["duration_seconds"] - float(end - times[0])) > 1e-9
    ):
        raise ValueError("Prepared player clock does not match native source timestamps.")
    return times, end


@torch.no_grad()
def predict_arrays(model, raw, video, key, policy, *, chunk_size=64):
    """Prepared full-source clock is fixed metadata, never future pose/probability context."""
    if chunk_size < 1:
        raise ValueError("Positive inference chunk size required")
    times, end = validate_timing(raw, video)
    cuts = video.get("confirmed_scene_cuts", [])
    if any(type(c) is not int or not 0 <= c < len(times) for c in cuts):
        raise ValueError("Invalid declared discontinuity")
    x, observable, epoch, gap = encode_source(raw, video["width"] / video["height"], cuts)
    maximum_gap = min(gap, Fraction(policy.maximum_timestamp_gap_seconds))
    rows, interruptions, unknown = [], [], []
    model_state = None
    decoder = ActivityDecoder(policy)
    probabilities = []
    run_start = None
    runs = []
    for offset in range(0, len(x), chunk_size):
        for i in range(offset, min(offset + chunk_size, len(x))):
            timegap = i > 0 and times[i] - times[i - 1] > maximum_gap
            reset = i == 0 or epoch[i] == i or timegap or not observable[i]
            if reset:
                if run_start is not None:
                    runs.append((run_start, i))
                    run_start = None
                model_state = None
                x[i, 41:] = 0  # No derivative context across any model/decoder reset.
                reason = (
                    "coverage_start"
                    if i == 0
                    else (
                        "timestamp_gap"
                        if timegap
                        else "unobservable_pose"
                        if not observable[i]
                        else "pose_recovery_or_declared_discontinuity"
                    )
                )
                interruptions.append(
                    dict(frame_index=i, timestamp_exact=str(times[i]), reason=reason)
                )
            if timegap:
                unknown.append(
                    dict(
                        reason="timestamp_gap",
                        start_frame=i,
                        stop_frame_exclusive=i,
                        start_timestamp_exact=str(times[i - 1]),
                        stop_timestamp_exact=str(times[i]),
                        time_support="open interval after last observed time, before resumed frame",
                    )
                )
            if observable[i]:
                if run_start is None:
                    run_start = i
                p, model_state = model.step(x[i], model_state)
                probability = float(p)
            else:
                probability = None
            probabilities.append(probability)
            row = dict(
                recording=f"upload:{key}",
                interval_id="prepared-full-video",
                frame_index=i,
                pts=int(raw["pts"][i]),
                time_base_num=int(raw["time_base_num"][i]),
                time_base_den=int(raw["time_base_den"][i]),
                timestamp_exact=str(times[i]),
                probability_punch=probability,
                observable=bool(observable[i]),
                scene_id=int(raw["scene_id"][i]),
                segment_id=int(raw["segment_id"][i]),
                discontinuity_before=bool(reset),
            )
            # The frozen decoder's legacy `reviewed` gate denotes eligible input scope here,
            # NOT human review. That flag is never exported as an upload/review attribute.
            decoder.push(Observation.from_row({**row, "reviewed": True}))
            rows.append(row)
    if run_start is not None:
        runs.append((run_start, len(x)))
    decoded = decoder.finish(len(x), end)
    replay = decode_batch(
        [{**r, "reviewed": True} for r in rows],
        {(f"upload:{key}", "prepared-full-video"): (len(x), end)},
        policy,
    )
    if (
        decoded["decoder_events"] != replay["events"]
        or decoded["decoder_updates"] != replay["updates"]
    ):
        raise ValueError("Frozen decoder streaming/batch disagreement.")
    error = 0.0
    for a, b in runs:
        batch = model(x[a:b][None])[0].sigmoid()
        error = max(error, float((batch - torch.tensor(probabilities[a:b])).abs().max()))
    if error > 1e-5:
        raise ValueError("Activity model streaming/batch disagreement.")
    i = 0
    while i < len(x):
        if observable[i]:
            i += 1
            continue
        a = i
        while i < len(x) and not observable[i]:
            i += 1
        unknown.append(
            dict(
                reason="missing_or_unreliable_pose",
                start_frame=a,
                stop_frame_exclusive=i,
                start_timestamp_exact=str(times[a]),
                stop_timestamp_exact=str(times[i] if i < len(x) else end),
                time_support="source-frame interval, unobservable rather than background",
            )
        )
    if end - times[-1] > maximum_gap:
        unknown.append(
            dict(
                reason="unobserved_tail",
                start_frame=len(x),
                stop_frame_exclusive=len(x),
                start_timestamp_exact=str(times[-1]),
                stop_timestamp_exact=str(end),
                time_support="open interval after last observed time, before coverage-end control",
            )
        )
    origin = times[0]
    for bout in decoded["bouts"]:
        bout["start_frame"] = bout["estimated_start"]["frame_index"]
        bout["start_media_time"] = float(
            Fraction(bout["estimated_start"]["timestamp_exact"]) - origin
        )
        stop = bout["stop_timestamp_exact"]
        bout["end_boundary_known"] = stop is not None
        bout["end_media_time"] = float(
            Fraction(stop or bout["estimated_end_inclusive"]["timestamp_exact"]) - origin
        )
        bout["first_emission_media_time"] = float(
            Fraction(bout["first_emission"]["timestamp_exact"]) - origin
        )
    for region in unknown:
        region["start_media_time"] = float(Fraction(region["start_timestamp_exact"]) - origin)
        region["end_media_time"] = float(Fraction(region["stop_timestamp_exact"]) - origin)
    return dict(
        **decoded,
        frames=rows,
        unobservable_intervals=unknown,
        resets=interruptions,
        observable_frames=int(observable.sum()),
        total_frames=len(x),
        feature_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
        feature_version=FEATURE_VERSION,
        feature_channels=65,
        source_velocity_gap_seconds_exact=str(gap),
        effective_reset_gap_seconds_exact=str(maximum_gap),
        timing_policy="full prepared source PTS clock fixes the original median-gap metadata "
        "policy; "
        "normalization is per-frame, derivatives backward-only, no future pose/probability inputs",
        model_streaming_batch_max_error=error,
        decoder_streaming_batch_equal=True,
    )


def scan(root, directory, key, expected):
    fingerprint, identity, video, poses, binding = inputs(root, directory, key)
    if fingerprint != expected:
        raise ValueError("Activity scan inputs changed while queued. Retry scanning.")
    policy = DecoderConfig.model_validate(
        json.loads((root / "configs/decoder_v1.json").read_text())
    )
    protocol = json.loads((root / "configs/activity_eval_protocol_v1.json").read_text())
    if protocol["reference"]["short_pause_seconds"] != "1/4":
        raise ValueError("Incompatible frozen activity grouping policy.")
    with np.load(regular(directory / "poses.npz"), allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    result = predict_arrays(load_model(root, binding), raw, video, key, policy)
    record = dict(
        schema_version=VERSION,
        video_id=key,
        input_fingerprint=fingerprint,
        source="model_prediction",
        human_reviewed=False,
        interpretation="experimental possible punching activity; training-only smoke model; "
        "not individual punch counts, six-class identification or validated detection",
        first_emission_clock="actual source observation when emitted, "
        "not estimated onset or wall latency",
        provenance=identity,
        checkpoint=binding["checkpoint"],
        training_dataset_sha256=binding["dataset_sha256"],
        decoder=policy.model_dump(mode="json"),
        grouping=dict(version="causal-activity-bouts-1.0", short_pause_seconds_exact="1/4"),
        pose_model_sha256=poses["pose_model_sha256"],
        pose_extraction_repeated=False,
        source_start_timestamp_exact=video["frames"][0]["timestamp_exact"],
        source_stop_timestamp_exact=video["stop_timestamp_exact"],
        **result,
    )
    # A separate deterministic record; operational wall time lives in activity-state.json.
    write_json(directory / f"activity-{fingerprint}.json", record)
    return record


if __name__ == "__main__":
    root, directory, key, expected = sys.argv[1:]
    scan(Path(root), Path(directory), key, expected)
