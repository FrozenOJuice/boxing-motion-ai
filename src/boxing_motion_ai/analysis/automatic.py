"""Offline automatic motion proposals + existing six-class classifier; no simulated path."""

import sys
from pathlib import Path

import numpy as np

from ..detection import instances, instances_v2
from .automatic_contract import VERSION, activity_record, inputs
from .common import regular, write_json
from .inference import Classifier


def bounds(row, video):
    a, z = row["start_frame"], row["stop_frame_exclusive"]
    frames = video["frames"]
    return dict(
        start_timestamp_exact=frames[a]["timestamp_exact"],
        last_timestamp_exact=frames[z - 1]["timestamp_exact"],
        stop_timestamp_exact=frames[z]["timestamp_exact"]
        if z < len(frames)
        else video["stop_timestamp_exact"],
        start_media_time=frames[a]["media_time"],
        end_media_time=frames[z]["media_time"] if z < len(frames) else video["duration_seconds"],
        source_start=frames[a],
        source_last_included=frames[z - 1],
    )


def classify_candidates(segmentation, classifier, raw, video):
    # The model is never used to choose boundaries, decide which movement is a punch,
    # resolve overlaps or pick among alternate segmentations.
    for number, candidate in enumerate(segmentation["candidates"], 1):
        candidate.update(
            candidate_id=f"candidate-{number}",
            verified=False,
            source=candidate.get("source", "pose_arm_cycle_heuristic"),
            **bounds(candidate, video),
        )
        candidate["motion_evidence_through"] = video["frames"][candidate["evidence_through_frame"]]
        if not candidate.get("classification_eligible", True):
            candidate["classification"] = dict(
                status="unavailable", reason=candidate["classification_unavailable_reason"]
            )
            continue
        try:
            result = classifier.classify(
                raw,
                video["width"] / video["height"],
                candidate["start_frame"],
                candidate["stop_frame_exclusive"],
            )
            result.pop(
                "inference_seconds", None
            )  # Operational timing is separate from deterministic output.
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            result = dict(
                status="unavailable",
                reason="Classifier failed for this candidate; run automatic analysis to retry.",
                retryable=True,
                error_type=type(exc).__name__,
            )
        candidate["classification"] = result
    for row in segmentation["unresolved_activity"] + segmentation["rejected_proposals"]:
        row.update(bounds(row, video))
    return segmentation


def analyze(root, directory, key, expected, activity_sha256):
    fingerprint, identity, video, _, policy = inputs(root, directory, key)
    if expected != fingerprint:
        raise ValueError("Automatic analysis inputs changed while queued. Retry.")
    activity_fp = identity["activity_input_fingerprint"]
    activity = activity_record(directory, key, activity_fp, activity_sha256)
    with np.load(regular(directory / "poses.npz"), allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    schema = policy.get("schema_version")
    if schema == "pose-reach-phase-policy-2.0":
        implementation = instances_v2
    elif schema == "pose-arm-cycle-policy-1.0":
        implementation = instances
    else:
        raise ValueError("Unsupported punch-candidate segmentation policy")
    segmentation = implementation.segment(
        raw, video, activity, implementation.Policy.model_validate(policy)
    )
    model = Classifier(
        root
    )  # Pinned model integrity required, even if no candidates are available.
    result = classify_candidates(segmentation, model, raw, video)
    record = dict(
        schema_version=VERSION,
        video_id=key,
        input_fingerprint=fingerprint,
        human_reviewed=False,
        verified=False,
        provenance=identity,
        activity_input_fingerprint=activity_fp,
        activity_result_sha256=activity_sha256,
        segmentation_version=implementation.VERSION,
        segmentation_policy=policy,
        classifier_checkpoint=model.pin["checkpoint"],
        classifier_feature_version=model.pin["feature_version"],
        classifier_classes=model.pin["classes"],
        score_semantics="uncalibrated_not_punch_evidence",
        activity_bouts=activity["bouts"],
        unobservable_intervals=activity["unobservable_intervals"],
        **result,
    )
    # Revalidate files after inference; never publish a result under a stale input identity.
    if inputs(root, directory, key)[0] != fingerprint:
        raise ValueError("Automatic analysis inputs changed during inference. Retry.")
    activity_record(directory, key, activity_fp, activity_sha256)
    write_json(directory / f"automatic-{fingerprint}.json", record)
    return record


if __name__ == "__main__":
    root, directory, key, expected, activity_sha256 = sys.argv[1:]
    analyze(Path(root), Path(directory), key, expected, activity_sha256)
