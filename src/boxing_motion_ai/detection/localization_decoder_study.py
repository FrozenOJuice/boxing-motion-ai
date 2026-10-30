"""Saved-score diagnosis/replay; a hash freeze gates V5 access. Never trains or predicts models."""

import argparse
from collections import Counter
from datetime import UTC, datetime
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

from ..analysis.common import sha, write_json
from ..services.dashboard import require
from ..vision.observation_study import BASE, arrays, fresh, read
from .localization_blue_experiment import record, summarize
from .localization_blue_features import OLD_CACHE
from .localization_data import load_cache, source_resets
from .localization_decode import decode as decode_v1
from .localization_decode_v2 import Policy, pair_evidence
from .localization_decode_v2 import decode as decode_v2
from .localization_model import MODES, model_observable

RUN = Path("reports/boxingvi/punch-localization-blue-v1")
CACHE = Path("reports/boxingvi/blue-localization-features-v1-redacted")
POLICY = Path("configs/localization_decoder_v2.json")
CODE = [
    Path(__file__),
    Path("src/boxing_motion_ai/detection/localization_decode_v2.py"),
    Path("tests/test_localization_decode_v2.py"),
    POLICY,
]


def binding():
    return {str(p): sha(p) for p in CODE}


def cases(phase):
    require(phase in ("training", "development"), "unknown_decoder_study_phase")
    result = {}
    if phase == "training":
        result, _ = load_cache(OLD_CACHE, read("configs/punch_localization_smoke_v1.json"))
    manifest = read(CACHE / "manifest.json")
    for name, h in manifest["cases"].items():
        # Do not open V5 metadata/features/references/scores during policy development.
        if phase == "training" and not name.startswith(("V4-", "V6-")):
            continue
        if phase == "development" and name != "V5-motion-101":
            continue
        d = CACHE / name
        r = read(d / "receipt.json")
        require(
            sha(d / "receipt.json") == h and sha(d / "features.npz") == r["features_sha256"],
            "decoder_feature_changed",
        )
        for s, digest in r["metadata_sha256"].items():
            require(sha(d / f"{s}.json") == digest, "decoder_metadata_changed")
        with np.load(d / "features.npz", allow_pickle=False) as z:
            a = dict(z)
        p = read(d / "provenance.json")
        require(
            p["recording"] in (("V4", "V6") if phase == "training" else ("V5",)),
            "decoder_role_leakage",
        )
        result[name] = dict(
            arrays=a,
            video=read(d / "video.json"),
            references=read(d / "references.json"),
            provenance=p,
            receipt=r,
        )
    for name, c in result.items():
        require(
            (not name.startswith("V5")) if phase == "training" else name.startswith("V5"),
            "decoder_phase_leakage",
        )
        a, v = c["arrays"], c["video"]
        raw_path = (
            CACHE / name / "poses.npz" if name in manifest["cases"] else BASE / name / "poses.npz"
        )
        if name in manifest["cases"]:
            require(sha(raw_path) == c["receipt"]["poses_sha256"], "decoder_pose_bytes_changed")
        raw = arrays(raw_path)
        for key in ("frame_index", "pts", "time_base_num", "time_base_den"):
            require(np.array_equal(a[key], raw[key]), "decoder_feature_clock_changed")
        _, reset = source_resets(raw, v)
        require(np.array_equal(reset, a["reset"]), "decoder_source_reset_changed")
        if "source_frame_index" in a:
            start = c["provenance"]["source_start"]
            require(
                np.array_equal(a["source_frame_index"], np.arange(start, start + len(a["pts"]))),
                "decoder_native_frame_map_changed",
            )
    return result


def iou(a, z, u, w, times, stop, core):
    a, z, u, w = max(a, core[0]), min(z, core[1]), max(u, core[0]), min(w, core[1])

    def clock(i):
        return times[i] if i < len(times) else stop

    intersection = (
        max(Q(), min(clock(z), clock(w)) - max(clock(a), clock(u))) if a < z and u < w else Q()
    )
    return (
        intersection / (max(clock(z), clock(w)) - min(clock(a), clock(u))) if intersection else Q()
    )


def diagnose(scores, case, old, new, policy):
    times = [Q(f["timestamp_exact"]) for f in case["video"]["frames"]]
    stop = Q(case["video"]["stop_timestamp_exact"])

    def clock(i):
        return times[i] if i < len(times) else stop

    observable = model_observable(case["arrays"])
    prov = case["provenance"]
    core = (
        (0, len(times))
        if prov.get("exhaustive") is False
        else (
            prov.get("review_start", prov.get("source_start", 0)) - prov.get("source_start", 0),
            prov.get("review_stop", len(times) + prov.get("source_start", 0))
            - prov.get("source_start", 0),
        )
    )
    v1_matches = {m["reference_id"] for m in (old["comparison"] or {}).get("matches_iou_half", [])}
    v2_matches = {m["reference_id"] for m in (new["comparison"] or {}).get("matches_iou_half", [])}
    viable = []
    duration_pairs = []
    for run in old["result"]["boundary_peaks"]:
        for a in run["onsets"]:
            for b in run["ends"]:
                if b <= a:
                    continue
                fraction, reason = pair_evidence(
                    a,
                    b,
                    run["start_frame"],
                    run["stop_frame_exclusive"],
                    scores,
                    times,
                    clock,
                    policy,
                )
                edge = dict(start_frame=a, stop_frame_exclusive=b + 1, activity_fraction=fraction)
                if reason is None:
                    viable.append(edge)
                if reason in (None, "boundary_pair_without_sufficient_activity"):
                    duration_pairs.append(edge)
    output = []
    onsets = [i for r in old["result"]["boundary_peaks"] for i in r["onsets"]]
    ends = [i for r in old["result"]["boundary_peaks"] for i in r["ends"]]
    radius = Q(1, 12)  # Existing supervised boundary support, not a tuned diagnostic tolerance.
    for ref in case["references"]:
        a, z = ref["start_frame"], ref["stop_frame_exclusive"]
        near = {}
        weak = []
        for channel, point, peaks, label in ((1, a, onsets, "onset"), (2, z - 1, ends, "end")):
            neighborhood = [
                i for i, t in enumerate(times) if abs(t - times[point]) <= radius and observable[i]
            ]
            maximum = max((float(scores[i, channel]) for i in neighborhood), default=None)
            nearby = [i for i in peaks if abs(times[i] - times[point]) <= radius]
            near[label] = dict(
                boundary_observable=bool(observable[point]),
                maximum_raw_score=maximum,
                nearby_peak_frames=nearby,
            )
            if not observable[point]:
                weak.append(f"{label}_unobservable")
            elif maximum is None or maximum < policy["boundary_threshold"]:
                weak.append(f"weak_{label}_score")
            elif not nearby:
                weak.append(f"{label}_peak_displaced_or_suppressed")
        compatible = []
        for edge in viable:
            overlap = iou(
                edge["start_frame"], edge["stop_frame_exclusive"], a, z, times, stop, core
            )
            if overlap >= Q(1, 2):
                compatible.append(edge | dict(iou_exact=str(overlap)))
        without_activity = any(
            iou(e["start_frame"], e["stop_frame_exclusive"], a, z, times, stop, core) >= Q(1, 2)
            for e in duration_pairs
        )
        output.append(
            dict(
                event_id=ref["event_id"],
                source_reference_frames=[
                    a + prov.get("source_start", 0),
                    z + prov.get("source_start", 0),
                ],
                v1_matched=ref["event_id"] in v1_matches,
                v2_matched=ref["event_id"] in v2_matches,
                boundary_evidence=near,
                evidence_limitations=weak,
                supported_pairs_iou_half=compatible,
                local_supported_pair_exists=bool(compatible),
                pair_exists_without_activity_gate=without_activity,
                interpretation=(
                    "label-assisted local feasibility diagnosis; "
                    "not a joint oracle or decoder input"
                ),
            )
        )
    stale = []
    for row in old["result"]["unresolved"]:
        if row["reason"] != "multiple_onsets_before_end":
            continue
        a, z = row["start_frame"], row["stop_frame_exclusive"]
        quiet = any(
            a < q["start_frame"] and q["stop_frame_exclusive"] < z
            for q in new["result"]["predicted_inactivity_resets"]
        )
        expired = times[z - 1] - times[a] > Q(policy["maximum_span_seconds"])
        if quiet or expired:
            stale.append(
                dict(
                    start_frame=a,
                    stop_frame_exclusive=z,
                    crosses_predicted_quiet=quiet,
                    exceeds_maximum_span=expired,
                )
            )
    return dict(
        references=output,
        stale_v1_pending_spans=stale,
        recovered_reference_ids=sorted(v2_matches - v1_matches),
        lost_reference_ids=sorted(v1_matches - v2_matches),
        v1_reasons=dict(Counter(r["reason"] for r in old["result"]["unresolved"])),
        v2_reasons=dict(Counter(r["reason"] for r in new["result"]["unresolved"])),
    )


def run(phase, output, freeze=None):
    if phase == "development":
        require(freeze is not None, "freeze_required_before_v5")
        frozen = read(freeze)
        require(frozen["code_and_policy"] == binding(), "decoder_policy_changed_after_freeze")
        for p, h in frozen["training_artifacts"].items():
            require(sha(Path(p)) == h, "frozen_training_audit_changed")
    policy = Policy.model_validate(read(POLICY)).model_dump()
    data = cases(phase)
    out = fresh(output)
    summaries = {}
    inputs = {}
    diagnostics = {}
    for mode in MODES:
        checkpoint = RUN / mode / "checkpoint.pt"
        require(
            sha(checkpoint) == read(RUN / mode / "verification.json")["checkpoint_sha256"],
            "historical_checkpoint_changed",
        )
        rows_old = []
        rows_new = []
        diagnostics[mode] = {}
        for name, case in data.items():
            score_file = RUN / mode / f"{name}-scores.npz"
            inputs[str(score_file)] = sha(score_file)
            with np.load(score_file, allow_pickle=False) as z:
                scores = z["scores"].copy()
            before = scores.copy()
            a = case["arrays"]
            video = case["video"]
            original = decode_v1(scores, model_observable(a), a["reset"], video, policy)
            updated = decode_v2(scores, model_observable(a), a["reset"], video, policy)
            require(np.array_equal(before, scores, equal_nan=True), "decoder_mutated_scores")
            require(
                original["boundary_peaks"] == updated["boundary_peaks"], "decoder_changed_peaks"
            )
            require(
                updated == decode_v2(scores, model_observable(a), a["reset"], video, policy),
                "decoder_not_deterministic",
            )
            old = record(name, original, case)
            new = record(name, updated, case)
            rows_old.append(old)
            rows_new.append(new)
            diagnostics[mode][name] = diagnose(scores, case, old, new, policy)
        write_json(out / f"{mode}-v1-records.json", rows_old)
        write_json(out / f"{mode}-v2-records.json", rows_new)
        summaries[mode] = dict(v1=summarize(rows_old), v2=summarize(rows_new))
        print(
            mode,
            phase,
            {
                g: (summaries[mode]["v1"][g]["matched"], summaries[mode]["v2"][g]["matched"])
                for g in [
                    "v4_original_training",
                    "v4_sparse_training",
                    "v6_blue_training",
                    "v5_development",
                ]
            },
            flush=True,
        )
    write_json(out / "summary.json", summaries)
    write_json(out / "diagnostics.json", diagnostics)
    write_json(
        out / "provenance.json",
        dict(
            phase=phase,
            created_at=datetime.now(UTC).isoformat(),
            code_and_policy=binding(),
            policy=policy,
            score_sha256=inputs,
            cases=list(data),
            freeze=({"path": str(freeze), "sha256": sha(freeze)} if freeze else None),
            new_training=False,
            new_model_inference=False,
            production_changed=False,
            restriction=(
                "V5 excluded before freeze; V1/V6 and V7/V8/V10 restrictions unchanged; no V9"
            ),
        ),
    )
    return out


def freeze(training, output):
    training = Path(training)
    provenance = read(training / "provenance.json")
    require(
        provenance["phase"] == "training"
        and not any(n.startswith("V5") for n in provenance["cases"]),
        "invalid_freeze_training_scope",
    )
    require(provenance["code_and_policy"] == binding(), "training_policy_changed")
    require(not Path(output).exists(), "freeze_already_exists")
    write_json(
        output,
        dict(
            schema_version="offline-decoder-v2-freeze-1.0",
            created_at=datetime.now(UTC).isoformat(),
            code_and_policy=binding(),
            training_artifacts={str(p): sha(p) for p in training.glob("*.json")},
            policy_origin=(
                "synthetic contracts and permitted V4/V6 only; single fixed policy, no V5 tuning"
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=["training", "development"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--freeze-training", type=Path)
    args = parser.parse_args()
    if args.freeze_training:
        freeze(args.freeze_training, args.output)
    else:
        run(args.phase, args.output, args.freeze)
