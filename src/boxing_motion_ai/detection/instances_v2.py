"""Offline per-arm extension/recovery phases. No annotation or classifier-score inputs."""

from fractions import Fraction
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..training.data import features, validate_arrays
from .upload import validate_timing

VERSION = "pose-reach-phase-candidates-2.0"
ARMS = {"left": (0, 2, 4), "right": (1, 3, 5)}


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["pose-reach-phase-policy-2.0"]
    smoothing_radius_seconds: str
    minimum_prominence_torso: float = Field(gt=0, allow_inf_nan=False)
    onset_fraction: float = Field(gt=0, lt=0.5, allow_inf_nan=False)
    recovery_fraction: float = Field(gt=0.5, le=1, allow_inf_nan=False)
    boundary_evidence_seconds: str
    minimum_phase_seconds: str
    minimum_cycle_seconds: str
    maximum_cycle_seconds: str
    maximum_turn_cosine: float = Field(ge=-1, lt=1, allow_inf_nan=False)
    maximum_wrist_speed_torso_per_second: float = Field(gt=0, allow_inf_nan=False)
    minimum_activity_overlap_seconds: str
    minimum_classification_arm_coverage: float = Field(gt=0, le=1, allow_inf_nan=False)
    overlap_peak_separation_seconds: str
    scope: str

    @model_validator(mode="after")
    def durations(self):
        for name in (
            "smoothing_radius_seconds",
            "boundary_evidence_seconds",
            "minimum_phase_seconds",
            "minimum_cycle_seconds",
            "maximum_cycle_seconds",
            "minimum_activity_overlap_seconds",
            "overlap_peak_separation_seconds",
        ):
            if Fraction(getattr(self, name)) <= 0:
                raise ValueError("Positive exact durations required")
        if not Fraction(self.minimum_cycle_seconds) < Fraction(self.maximum_cycle_seconds) <= 3:
            raise ValueError("Invalid phase duration range")
        return self


def motion(raw, video, activity):
    """Extract masked shoulder-relative trajectories. No filling; elbow need not hide wrist."""
    validate_arrays(raw)
    times, coverage = validate_timing(raw, video)
    n = len(times)
    if len(activity["frames"]) != n or any(
        r["frame_index"] != i or Fraction(r["timestamp_exact"]) != times[i]
        for i, r in enumerate(activity["frames"])
    ):
        raise ValueError("Activity observations do not match prepared source timing")
    last = 0
    for b in activity["bouts"]:
        a, z = b["start_frame"], b["stop_frame_exclusive"]
        if type(a) is not int or type(z) is not int or not last <= a < z <= n:
            raise ValueError("Invalid activity bounds")
        last = z
    x, _, normal = features(
        raw["coordinates"], raw["confidence"], raw["valid_mask"], video["width"] / video["height"]
    )
    j = x[:, :40].reshape(n, 8, 5)
    observable = (
        normal & (raw["pose_status"] != 2) & np.array([r["observable"] for r in activity["frames"]])
    )
    reset = np.array([r["discontinuity_before"] for r in activity["frames"]], dtype=bool)
    gap = Fraction(activity["effective_reset_gap_seconds_exact"])
    if gap <= 0:
        raise ValueError("Invalid observation gap")
    cuts = set(video.get("confirmed_scene_cuts", []))
    reset[0] = True
    for i in range(1, n):
        reset[i] |= (
            times[i] - times[i - 1] > gap
            or i in cuts
            or raw["scene_id"][i] != raw["scene_id"][i - 1]
            or raw["segment_id"][i] != raw["segment_id"][i - 1]
        )
    arms = {}
    for arm, (s, e, w) in ARMS.items():
        arms[arm] = dict(
            wrist=j[:, w, :2] - j[:, s, :2],
            elbow=j[:, e, :2] - j[:, s, :2],
            wrist_valid=observable & (j[:, s, 4] == 1) & (j[:, w, 4] == 1),
            elbow_valid=observable & (j[:, s, 4] == 1) & (j[:, e, 4] == 1),
        )
    return dict(times=times, stop_time=coverage, reset=reset, arms=arms, bouts=activity["bouts"])


def turns(reach, prominence):
    """Alternating extrema confirmed by a full prominence reversal, not every noise peak."""
    low = high = 0
    mode = 0
    points = []
    for i, value in enumerate(reach):
        if mode == 0:
            if value < reach[low]:
                low = i
            if value > reach[high]:
                high = i
            if value - reach[low] >= prominence:
                points.append(("min", low))
                high = i
                mode = 1
            elif reach[high] - value >= prominence:
                points.append(("max", high))
                low = i
                mode = -1
        elif mode == 1:
            if value > reach[high]:
                high = i
            if reach[high] - value >= prominence:
                points.append(("max", high))
                low = i
                mode = -1
        else:
            if value < reach[low]:
                low = i
            if value - reach[low] >= prominence:
                points.append(("min", low))
                high = i
                mode = 1
    if mode == -1:
        points.append(("min", low))  # Terminal minimum still needs boundary evidence.
    elif mode == 1:
        points.append(("max", high))
    return points


def runs(valid, reset):
    i = 0
    while i < len(valid):
        if not valid[i]:
            i += 1
            continue
        a = i
        i += 1
        while i < len(valid) and valid[i] and not reset[i]:
            i += 1
        yield a, i


def segment_motion(data, policy):
    times, arms = data["times"], data["arms"]
    n = len(times)
    proposed, rejected, arm_reports = [], [], {}
    for arm, trace in arms.items():
        wrist, valid = trace["wrist"], trace["wrist_valid"]
        reset = data["reset"].copy()
        jumps = []
        for i in range(1, n):
            if valid[i] and valid[i - 1] and not reset[i]:
                speed = float(np.linalg.norm(wrist[i] - wrist[i - 1])) / float(
                    times[i] - times[i - 1]
                )
                if speed > policy.maximum_wrist_speed_torso_per_second:
                    reset[i] = True
                    jumps.append(i)
        observed_runs = list(runs(valid, reset))
        extremum_log = []
        for first, stop in observed_runs:
            pos = wrist[first:stop].copy()
            t = np.array([float(x - times[first]) for x in times[first:stop]])
            radius = float(Fraction(policy.smoothing_radius_seconds))
            smooth = np.empty_like(pos)
            for i in range(len(pos)):
                weights = np.maximum(0, 1 - np.abs(t - t[i]) / radius)
                smooth[i] = (pos * weights[:, None]).sum(0) / weights.sum()
            reach = np.linalg.norm(smooth, axis=1)
            points = turns(reach, policy.minimum_prominence_torso)
            extremum_log.extend(
                dict(kind=k, frame_index=first + i, reach_torso=float(reach[i])) for k, i in points
            )
            for k in range(1, len(points) - 1):
                if [p[0] for p in points[k - 1 : k + 2]] != ["min", "max", "min"]:
                    continue
                lo, peak, hi = [p[1] for p in points[k - 1 : k + 2]]
                gain = float(reach[peak] - reach[lo])
                recovery = float(reach[peak] - reach[hi])
                onset_level = reach[lo] + policy.onset_fraction * gain
                start = lo
                for i in range(lo, peak):
                    if reach[i] <= onset_level:
                        start = i
                recovery_level = reach[peak] - policy.recovery_fraction * recovery
                end = next(i for i in range(peak + 1, hi + 1) if reach[i] <= recovery_level)
                a, p, z = first + start, first + peak, first + end + 1
                terminal = k + 1 == len(points) - 1
                after_end = np.array(reach[end:])
                terminal_settled = (
                    times[stop - 1] - times[z - 1] >= Fraction(policy.boundary_evidence_seconds)
                    and float(after_end.max() - after_end.min())
                    <= policy.minimum_prominence_torso / 2
                )
                left_censored = times[a] - times[first] < Fraction(policy.boundary_evidence_seconds)
                right_censored = terminal and not terminal_settled
                outgoing = smooth[peak] - smooth[start]
                incoming = smooth[end] - smooth[peak]
                denom = float(np.linalg.norm(outgoing) * np.linalg.norm(incoming))
                cosine = float(np.dot(outgoing, incoming) / denom) if denom else 1.0
                overlaps = []
                for b in data["bouts"]:
                    u, v = max(a, b["start_frame"]), min(z, b["stop_frame_exclusive"])
                    overlap = (
                        (times[v] if v < n else data["stop_time"]) - times[u]
                        if u < v
                        else Fraction(0)
                    )
                    if overlap >= Fraction(policy.minimum_activity_overlap_seconds):
                        overlaps.append(b["bout_id"])
                coverage = float(trace["elbow_valid"][a:z].mean())
                row = dict(
                    source="pose_reach_phase_heuristic_v2",
                    arm=arm,
                    start_frame=a,
                    stop_frame_exclusive=z,
                    peak_frame=p,
                    evidence_through_frame=stop - 1,
                    activity_bout_id=overlaps[0] if overlaps else None,
                    activity_bout_ids=overlaps,
                    extension_torso=gain,
                    recovery_torso=recovery,
                    turn_cosine=cosine,
                    elbow_coverage=coverage,
                    wrist_coverage=1.0,
                    left_censored=left_censored,
                    right_censored=right_censored,
                    classification_eligible=True,
                )
                reason = None
                if (
                    gain < policy.minimum_prominence_torso
                    or recovery < policy.minimum_prominence_torso
                ):
                    reason = "insufficient_reach_prominence"
                elif (
                    not Fraction(policy.minimum_cycle_seconds)
                    <= times[z - 1] - times[a]
                    <= Fraction(policy.maximum_cycle_seconds)
                ):
                    reason = "phase_duration_outside_policy"
                elif min(times[p] - times[a], times[z - 1] - times[p]) < Fraction(
                    policy.minimum_phase_seconds
                ):
                    reason = "insufficient_phase_observations"
                elif left_censored or right_censored:
                    reason = "phase_boundary_censored"
                elif cosine > policy.maximum_turn_cosine:
                    reason = "trajectory_reversal_not_supported"
                elif not overlaps:
                    reason = "no_sustained_activity_overlap"
                if reason:
                    rejected.append({**row, "reason": reason})
                else:
                    if len(overlaps) > 1:
                        row.update(
                            classification_eligible=False,
                            classification_unavailable_reason="Multiple activity bouts "
                            "intersect this "
                            "phase; individual context is unresolved.",
                        )
                    if coverage < policy.minimum_classification_arm_coverage:
                        row.update(
                            classification_eligible=False,
                            classification_unavailable_reason="Candidate arm elbow coverage is "
                            "insufficient for six-class interpretation.",
                        )
                    proposed.append(row)
        arm_reports[arm] = dict(
            wrist_observable_frames=int(valid.sum()),
            elbow_observable_frames=int(trace["elbow_valid"].sum()),
            observable_runs=[dict(start_frame=a, stop_frame_exclusive=z) for a, z in observed_runs],
            tracking_jump_resets=jumps,
            reach_extrema=extremum_log,
        )
    # Retain independently supported arm phases, but never force overlapping clips into six classes.
    candidates = sorted(
        {(p["arm"], p["peak_frame"]): p for p in proposed}.values(),
        key=lambda p: (p["start_frame"], p["arm"]),
    )
    for p in candidates:
        peers = [
            q
            for q in candidates
            if q is not p
            and (
                p["start_frame"] <= q["peak_frame"] < p["stop_frame_exclusive"]
                or abs(times[p["peak_frame"]] - times[q["peak_frame"]])
                < Fraction(policy.overlap_peak_separation_seconds)
            )
        ]
        if peers:
            p.update(
                classification_eligible=False,
                classification_unavailable_reason="Overlapping arm phases cannot be "
                "isolated for six-class interpretation.",
                overlapping_peaks=[dict(arm=q["arm"], frame_index=q["peak_frame"]) for q in peers],
            )
    unresolved = []
    for b in data["bouts"]:
        cursor = b["start_frame"]
        stop = b["stop_frame_exclusive"]
        for p in candidates:
            if (
                not p["classification_eligible"]
                or p["stop_frame_exclusive"] <= cursor
                or p["start_frame"] >= stop
            ):
                continue
            if cursor < p["start_frame"]:
                unresolved.append(
                    dict(
                        start_frame=cursor,
                        stop_frame_exclusive=min(stop, p["start_frame"]),
                        activity_bout_id=b["bout_id"],
                        reason="activity_without_isolated_supported_phase",
                    )
                )
            cursor = max(cursor, min(stop, p["stop_frame_exclusive"]))
        if cursor < stop:
            unresolved.append(
                dict(
                    start_frame=cursor,
                    stop_frame_exclusive=stop,
                    activity_bout_id=b["bout_id"],
                    reason="activity_without_isolated_supported_phase",
                )
            )
    return dict(
        candidates=candidates,
        rejected_proposals=rejected,
        unresolved_activity=unresolved,
        arm_motion=arm_reports,
        processing_mode="offline_centered_motion_extrema_with_future_recovery_evidence",
        interpretation="experimental per-arm extension/recovery candidates, not verified punches",
    )


def segment(raw, video, activity, policy):
    return segment_motion(motion(raw, video, activity), policy)
