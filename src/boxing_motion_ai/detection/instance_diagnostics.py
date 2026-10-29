"""Read-only V4/V6 diagnostics. References enter comparison only, never segmentation.

Run after extracting the explicitly permitted, versioned case inventory with --prepare.
All generated arrays, figures and model outputs live in a new ignored reports directory.
"""

import argparse
import json
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

from ..analysis.activity_contract import activity_pin
from ..analysis.automatic import classify_candidates
from ..analysis.common import regular, sha, write_json
from ..analysis.inference import Classifier
from . import instances, instances_v2
from .decoder import DecoderConfig
from .upload import load_model, predict_arrays

UPLOADS = {
    "V6-upload": ".analysis-local/1cd7910e74964f13a095f1e1f1b9c7b7",
    "V4-8s-upload": "reports/boxingvi/automatic-punch-verification-v1/v4-eight-second-upload",
    "V4-29s-upload": ".analysis-local/cd0eb8c64502448090090ca16f1fcac2",
    "no-person": ".analysis-local/0ce167f8eb3a4874be6f67997c08e24c",
}


def arrays(path):
    with np.load(regular(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prepare(root, output):
    """Explicit diagnostic sources only. No source selection from prediction success."""
    root, output = Path(root), Path(output)
    if output.exists() or not output.resolve().is_relative_to(root.resolve() / "reports"):
        raise ValueError("Use a new ignored reports directory")
    manifest_path = root / "reports/boxingvi/continuous-dataset-v1/manifest.json"
    binding = json.loads((root / "configs/continuous_dataset_v1.json").read_text())
    if sha(manifest_path) != binding["sha256"]:
        raise ValueError("Reviewed-source manifest changed")
    manifest = json.loads(manifest_path.read_text())
    expected = {f"{r}-motion-{i}" for r in ("V4", "V6") for i in range(1, 7)} | {
        f"V4-bg-{i}" for i in range(1, 4)
    }
    if {x["id"] for x in manifest["intervals"]} != expected:
        raise ValueError("Unexpected diagnostic source pool; held intervals remain excluded")
    output.mkdir(parents=True)
    cases = []
    for name, path in UPLOADS.items():
        directory = root / path
        video = json.loads(regular(directory / "video.json").read_text())
        pose = json.loads(regular(directory / "pose.json").read_text())
        if sha(directory / "poses.npz") != pose["poses_sha256"]:
            raise ValueError("Cached diagnostic pose provenance changed")
        state = json.loads((directory / "activity-state.json").read_text())
        activity_path = directory / f"activity-{state['input_fingerprint']}.json"
        if sha(activity_path) != state["result_sha256"]:
            raise ValueError("Changed activity artifact")
        cases.append(
            (
                name,
                arrays(directory / "poses.npz"),
                video,
                json.loads(activity_path.read_text()),
                [],
                dict(
                    kind="prepared_upload",
                    directory=path,
                    poses_sha256=pose["poses_sha256"],
                    source_sha256=video["source_sha256"],
                    activity_sha256=state["result_sha256"],
                ),
            )
        )
    model = load_model(root, activity_pin(root))
    decoder = DecoderConfig.model_validate(
        json.loads((root / "configs/decoder_v1.json").read_text())
    )
    for recording, width, height in [("V4", 3840, 2160), ("V6", 1920, 1080)]:
        directory = root / f"data/processed/boxingvi/{recording}/candidates-full-v1"
        path = directory / "poses.npz"
        full = arrays(path)
        digest = sha(path)
        for interval in manifest["intervals"]:
            source = interval["provenance"]
            if source["recording"] != recording:
                continue
            if digest != source["source_inputs"]["poses.npz"]:
                raise ValueError("Native diagnostic pose source changed")
            start = source["context_start"] if interval["events"] else source["start"]
            stop = source["context_end"] if interval["events"] else source["stop"]
            raw = {k: v[start:stop].copy() for k, v in full.items()}
            raw["frame_index"] = np.arange(stop - start, dtype=np.int64)
            frames = [
                dict(
                    frame_index=i,
                    pts=int(raw["pts"][i]),
                    time_base_num=int(raw["time_base_num"][i]),
                    time_base_den=int(raw["time_base_den"][i]),
                    timestamp_exact=str(
                        Q(
                            int(raw["pts"][i] * raw["time_base_num"][i]),
                            int(raw["time_base_den"][i]),
                        )
                    ),
                )
                for i in range(stop - start)
            ]
            origin = Q(frames[0]["timestamp_exact"])
            end = Q(
                int(full["pts"][stop] * full["time_base_num"][stop]),
                int(full["time_base_den"][stop]),
            )
            for f in frames:
                f["media_time"] = float(Q(f["timestamp_exact"]) - origin)
            video = dict(
                frame_count=len(frames),
                frames=frames,
                width=width,
                height=height,
                stop_timestamp_exact=str(end),
                duration_seconds=float(end - origin),
            )
            # No reference boundaries/types enter predict_arrays or either segmenter.
            activity = predict_arrays(model, raw, video, interval["id"], decoder)
            refs = [
                dict(
                    start_frame=e["start"] - start,
                    stop_frame_exclusive=e["stop"] - start,
                    type=e["punch_type"],
                    event_id=e["event_id"],
                )
                for e in interval["events"]
            ]
            cases.append(
                (
                    interval["id"],
                    raw,
                    video,
                    activity,
                    refs,
                    dict(
                        kind="reviewed_source",
                        recording=recording,
                        source_start=start,
                        review_start=source["start"],
                        review_stop=source["stop"],
                        poses_sha256=digest,
                        source_video_sha256=source["video_sha256"],
                        manifest_sha256=binding["sha256"],
                    ),
                )
            )
    for name, raw, video, activity, refs, provenance in cases:
        directory = output / name
        directory.mkdir()
        np.savez_compressed(directory / "poses.npz", **raw)
        for stem, value in [
            ("video", video),
            ("activity", activity),
            ("references", refs),
            ("provenance", provenance),
        ]:
            write_json(directory / f"{stem}.json", value)
    write_json(output / "cases.json", [c[0] for c in cases])


def v1_audit(data, result, policy):
    """All eligible quiet-boundary pairs, with independent gates (not just first failure).

    This census is NOT a set of model predictions and does not alter v1's ordered algorithm.
    """
    times = data["times"]
    windows = []
    arm_stats = {}
    for arm, trace in data["arms"].items():
        valid = trace["wrist_valid"] & trace["elbow_valid"]
        quiet = [q for q in result["quiet_boundaries"] if q["arm"] == arm]
        for i, before in enumerate(quiet):
            for after in quiet[i + 1 :]:
                a, z = before["last_frame"], after["start_frame"] + 1
                if times[z - 1] - times[a] > Q(policy.maximum_cycle_seconds):
                    break
                if not valid[a:z].all() or data["reset"][a + 1 : z].any():
                    continue
                wrist = trace["wrist"][a:z]
                elbow = trace["elbow"][a:z]
                reach = np.linalg.norm(wrist - wrist[0], axis=1)
                peak = int(reach.argmax())
                exc = float(reach[peak])
                path = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
                ideal = 2 * exc - float(reach[-1])
                variation = float(np.abs(np.diff(reach)).sum())
                length = np.linalg.norm(elbow, axis=1) + np.linalg.norm(wrist - elbow, axis=1)
                ratios = np.linalg.norm(wrist, axis=1) / np.maximum(length, 1e-6)
                outward = float(path[:peak].sum())
                inward = float(path[peak:].sum())
                efficiencies = [
                    exc / outward if outward else 0,
                    (exc - float(reach[-1])) / inward if inward else 0,
                ]
                failures = dict(
                    compact_guard=max(ratios[0], ratios[-1]) > policy.maximum_guard_extension_ratio,
                    excursion=exc < policy.minimum_excursion_torso,
                    duration=not Q(policy.minimum_cycle_seconds)
                    <= times[z - 1] - times[a]
                    <= Q(policy.maximum_cycle_seconds),
                    return_distance=float(reach[-1]) > policy.maximum_return_distance_torso,
                    radial_variation=variation > policy.maximum_radial_variation_ratio * ideal,
                    phase_efficiency=min(efficiencies) < policy.minimum_phase_path_efficiency,
                )
                failures = {key: bool(value) for key, value in failures.items()}
                gated = any(
                    b["start_frame"] <= a + peak < b["stop_frame_exclusive"] for b in data["bouts"]
                )
                windows.append(
                    dict(
                        arm=arm,
                        start_frame=a,
                        stop_frame_exclusive=z,
                        peak_frame=a + peak,
                        activity_peak_gated=gated,
                        failures=failures,
                        excursion=exc,
                        return_distance=float(reach[-1]),
                        phase_efficiencies=efficiencies,
                        radial_variation_ratio=variation / ideal if ideal else None,
                        boundary_extension_ratios=[float(ratios[0]), float(ratios[-1])],
                    )
                )
        complete_runs = list(instances_v2.runs(valid, data["reset"]))
        wrist_runs = list(instances_v2.runs(trace["wrist_valid"], data["reset"]))
        arm_stats[arm] = dict(
            wrist_frames=int(trace["wrist_valid"].sum()),
            elbow_frames=int(trace["elbow_valid"].sum()),
            complete_arm_frames=int(valid.sum()),
            quiet_runs=len(quiet),
            longest_complete_run_seconds=max(
                (float(times[z - 1] - times[a]) for a, z in complete_runs), default=0
            ),
            longest_wrist_run_seconds=max(
                (float(times[z - 1] - times[a]) for a, z in wrist_runs), default=0
            ),
        )
    gated = [w for w in windows if w["activity_peak_gated"]]
    return dict(
        arms=arm_stats,
        quiet_pair_census=len(windows),
        activity_gated_pairs=len(gated),
        independent_failed_conditions={
            k: sum(w["failures"][k] for w in gated)
            for k in [
                "compact_guard",
                "excursion",
                "duration",
                "return_distance",
                "radial_variation",
                "phase_efficiency",
            ]
        },
        actual_first_rejections=dict(Counter(p["reason"] for p in result["rejected_proposals"])),
        windows=windows,
    )


def compare(candidates, references, times, stop_time, core):
    """Exact-time overlap + max-cardinality/IoU assignment on explicitly reviewed core only.

    No emission timestamps are invented to call the causal-latency metric implementation.
    """

    def clock(i):
        return times[i] if i < len(times) else stop_time

    a, z = core
    if not 0 <= a < z <= len(times) or len(references) > 16:
        raise ValueError("Bounded reviewed diagnostic exposure required")
    for row in list(references) + list(candidates):
        if not 0 <= row["start_frame"] < row["stop_frame_exclusive"] <= len(times):
            raise ValueError("Diagnostic interval outside source inventory")
    refs = [
        {**r, "a": max(a, r["start_frame"]), "z": min(z, r["stop_frame_exclusive"])}
        for r in references
    ]
    refs = [r for r in refs if r["a"] < r["z"]]
    preds = [
        {**p, "a": max(a, p["start_frame"]), "z": min(z, p["stop_frame_exclusive"]), "index": i}
        for i, p in enumerate(candidates)
    ]
    preds = [p for p in preds if p["a"] < p["z"]]
    pairs = []
    for p in preds:
        for r in refs:
            intersection = max(
                Q(0), min(clock(p["z"]), clock(r["z"])) - max(clock(p["a"]), clock(r["a"]))
            )
            if intersection:
                iou = intersection / (
                    max(clock(p["z"]), clock(r["z"])) - min(clock(p["a"]), clock(r["a"]))
                )
                pairs.append(
                    dict(
                        candidate_index=p["index"],
                        reference_id=r["event_id"],
                        reference_type=r["type"],
                        reference_fraction=float(intersection / (clock(r["z"]) - clock(r["a"]))),
                        iou_exact=str(iou),
                        iou=float(iou),
                        peak_inside_reference=r["a"] <= p["peak_frame"] < r["z"],
                    )
                )
    # At most four reviewed events per approved diagnostic block; exact DP avoids greedy mismatches.
    states = {0: (Q(0), [])}
    for p in preds:
        updated = dict(states)
        for mask, (score, matched) in states.items():
            for j, r in enumerate(refs):
                edge = next(
                    (
                        e
                        for e in pairs
                        if e["candidate_index"] == p["index"]
                        and e["reference_id"] == r["event_id"]
                        and Q(e["iou_exact"]) >= Q(1, 2)
                    ),
                    None,
                )
                if edge is None or mask & (1 << j):
                    continue
                new = mask | (1 << j)
                choice = (score + Q(edge["iou_exact"]), matched + [edge])
                if new not in updated or choice[0] > updated[new][0]:
                    updated[new] = choice
        states = updated
    best = max(states, key=lambda mask: (mask.bit_count(), states[mask][0]))
    matched = states[best][1]
    return dict(
        reviewed_reference_count=len(refs),
        reviewed_candidate_count=len(preds),
        pairs=pairs,
        matches_iou_half=matched,
        missed_references=[
            r["event_id"]
            for r in refs
            if not any(e["reference_id"] == r["event_id"] for e in matched)
        ],
        unmatched_candidates=[
            p["index"]
            for p in preds
            if not any(e["candidate_index"] == p["index"] for e in matched)
        ],
        candidate_without_reference_overlap=[
            p["index"] for p in preds if not any(e["candidate_index"] == p["index"] for e in pairs)
        ],
        merge_risks=[
            p["index"]
            for p in preds
            if sum(
                e["candidate_index"] == p["index"] and e["reference_fraction"] >= 0.25
                for e in pairs
            )
            > 1
        ],
        split_risks=[
            r["event_id"]
            for r in refs
            if sum(e["reference_id"] == r["event_id"] and e["peak_inside_reference"] for e in pairs)
            > 1
        ],
    )


def evaluate(root, directory):
    root, directory = Path(root), Path(directory)
    p1 = instances.Policy.model_validate(
        json.loads((root / "configs/punch_instances_v1.json").read_text())
    )
    p2 = instances_v2.Policy.model_validate(
        json.loads((root / "configs/punch_instances_v2.json").read_text())
    )
    allowed = (
        set(UPLOADS)
        | {f"{r}-motion-{i}" for r in ("V4", "V6") for i in range(1, 7)}
        | {f"V4-bg-{i}" for i in range(1, 4)}
    )
    names = json.loads(regular(directory / "cases.json").read_text())
    if (
        not directory.resolve().is_relative_to(root.resolve() / "reports")
        or set(names) != allowed
        or len(names) != len(allowed)
    ):
        raise ValueError("Use the explicit prepared V4/V6 diagnostic inventory under reports")
    model = Classifier(root)
    summaries = []
    for name in names:
        case = directory / name
        raw = arrays(case / "poses.npz")
        video = json.loads((case / "video.json").read_text())
        activity = json.loads((case / "activity.json").read_text())
        data = instances_v2.motion(raw, video, activity)
        v1 = instances.segment(raw, video, activity, p1)
        v2 = instances_v2.segment_motion(data, p2)
        # References are read only AFTER both annotation-free segmenters have returned.
        refs = json.loads((case / "references.json").read_text())
        prov = json.loads((case / "provenance.json").read_text())
        core = (
            (
                prov["review_start"] - prov["source_start"],
                prov["review_stop"] - prov["source_start"],
            )
            if prov["kind"] == "reviewed_source"
            else None
        )
        comparisons = {}
        if core is not None:
            comparisons = {
                k: compare(r["candidates"], refs, data["times"], data["stop_time"], core)
                for k, r in [("v1", v1), ("v2", v2)]
            }
        classified = classify_candidates(v2, model, raw, video)
        audit = v1_audit(data, v1, p1)
        events = []
        for ref in refs:
            a, z = ref["start_frame"], min(len(data["times"]), ref["stop_frame_exclusive"])
            events.append(
                {
                    **ref,
                    "arms": {
                        arm: dict(
                            wrist_fraction=float(t["wrist_valid"][a:z].mean()),
                            elbow_fraction=float(t["elbow_valid"][a:z].mean()),
                            complete_fraction=float(
                                (t["wrist_valid"][a:z] & t["elbow_valid"][a:z]).mean()
                            ),
                        )
                        for arm, t in data["arms"].items()
                    },
                }
            )
        summary = dict(
            case=name,
            provenance=prov,
            frames=len(data["times"]),
            duration_seconds=video["duration_seconds"],
            activity_bouts=len(activity["bouts"]),
            v1_candidates=len(v1["candidates"]),
            v2_candidates=len(v2["candidates"]),
            classified=sum(
                c["classification"]["status"] == "available" for c in classified["candidates"]
            ),
            v1_audit=audit,
            v2_rejections=dict(Counter(p["reason"] for p in v2["rejected_proposals"])),
            v2_ungated_supported=sum(
                p["reason"] == "no_sustained_activity_overlap" for p in v2["rejected_proposals"]
            ),
            comparisons=comparisons,
            reviewed_event_visibility=events,
        )
        for file, result in [
            ("v1-result.json", v1),
            ("v2-result.json", classified),
            ("diagnostics.json", summary),
        ]:
            write_json(case / file, result)
        summaries.append(summary)
    write_json(
        directory / "comparison.json",
        dict(
            schema_version="punch-instance-comparison-2.0",
            scope="V4 development/training-source + permitted V6 diagnostics; "
            "not generalization accuracy",
            binary_checkpoint=activity_pin(root)["checkpoint"],
            classifier_checkpoint=model.pin["checkpoint"],
            decoder_sha256=sha(root / "configs/decoder_v1.json"),
            implementation_sha256={
                name: sha(root / name)
                for name in (
                    "src/boxing_motion_ai/detection/instances.py",
                    "src/boxing_motion_ai/detection/instances_v2.py",
                    "src/boxing_motion_ai/detection/instance_diagnostics.py",
                    "src/boxing_motion_ai/analysis/inference.py",
                )
            },
            case_inputs_sha256={
                name: {
                    file: sha(directory / name / file)
                    for file in (
                        "poses.npz",
                        "video.json",
                        "activity.json",
                        "references.json",
                        "provenance.json",
                    )
                }
                for name in names
            },
            policy_v1_sha256=sha(root / "configs/punch_instances_v1.json"),
            policy_v2_sha256=sha(root / "configs/punch_instances_v2.json"),
            cases=summaries,
        ),
    )
    return summaries


def export_fixture(root, directory, target):
    """Small actual masked trajectories for CI, with labels kept comparison-only."""
    root, directory, target = Path(root), Path(directory), Path(target)
    if target.exists():
        raise ValueError("Use a new fixture path; existing real fixtures are preserved")
    policy = instances_v2.Policy.model_validate(
        json.loads((root / "configs/punch_instances_v2.json").read_text())
    )
    cases = []
    for name in ("V6-upload", "V4-bg-1", "V4-bg-2", "V4-bg-3", "V6-motion-6"):
        case = directory / name
        raw = arrays(case / "poses.npz")
        video = json.loads(regular(case / "video.json").read_text())
        activity = json.loads(regular(case / "activity.json").read_text())
        data = instances_v2.motion(raw, video, activity)
        arms = {}
        for arm, trace in data["arms"].items():
            arms[arm] = {
                joint: [
                    [round(float(x), 8), round(float(y), 8)] if valid else None
                    for (x, y), valid in zip(trace[joint], trace[joint + "_valid"], strict=True)
                ]
                for joint in ("wrist", "elbow")
            }
        result = instances_v2.segment_motion(data, policy)
        cases.append(
            dict(
                name=name,
                provenance=json.loads((case / "provenance.json").read_text()),
                derived_poses_sha256=sha(case / "poses.npz"),
                activity_sha256=sha(case / "activity.json"),
                times_exact=[str(t) for t in data["times"]],
                stop_time_exact=str(data["stop_time"]),
                resets=np.flatnonzero(data["reset"]).tolist(),
                arms=arms,
                bouts=[
                    {k: b[k] for k in ("bout_id", "start_frame", "stop_frame_exclusive")}
                    for b in activity["bouts"]
                ],
                expected_candidates=[
                    {k: p[k] for k in ("arm", "start_frame", "peak_frame", "stop_frame_exclusive")}
                    for p in result["candidates"]
                ],
                references_for_comparison_only=json.loads((case / "references.json").read_text()),
            )
        )
    value = dict(
        schema_version="real-punch-motion-fixture-2.0",
        semantics="Actual permitted V4/V6 shoulder-relative torso-scaled XY, "
        "rounded to 8 decimals. "
        "Null means masked, never filled. References are comparison-only, not inference input.",
        policy_sha256=sha(root / "configs/punch_instances_v2.json"),
        cases=cases,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    # One complete case per line keeps a compact numeric fixture reviewable by case.
    target.write_text(
        '{"schema_version":'
        + json.dumps(value["schema_version"])
        + ',\n"semantics":'
        + json.dumps(value["semantics"])
        + ',\n"policy_sha256":'
        + json.dumps(value["policy_sha256"])
        + ',\n"cases":[\n'
        + ",\n".join(json.dumps(c, separators=(",", ":")) for c in cases)
        + "\n]}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--fixture-out", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root, args.output)
    evaluate(args.root, args.output)
    if args.fixture_out:
        export_fixture(args.root, args.output, args.fixture_out)
