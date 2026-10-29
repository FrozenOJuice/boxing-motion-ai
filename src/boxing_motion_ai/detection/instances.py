"""Offline conservative arm-cycle proposals. Never equates activity with a punch."""

from fractions import Fraction
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..training.data import features, validate_arrays
from .upload import validate_timing

VERSION = "pose-arm-cycle-candidates-1.0"


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["pose-arm-cycle-policy-1.0"]
    quiet_speed_torso_per_second: float = Field(gt=0, allow_inf_nan=False)
    quiet_seconds: str
    maximum_guard_extension_ratio: float = Field(gt=0, lt=1, allow_inf_nan=False)
    minimum_excursion_torso: float = Field(gt=0, allow_inf_nan=False)
    maximum_return_distance_torso: float = Field(gt=0, allow_inf_nan=False)
    minimum_cycle_seconds: str
    maximum_cycle_seconds: str
    maximum_radial_variation_ratio: float = Field(ge=1, allow_inf_nan=False)
    minimum_phase_path_efficiency: float = Field(gt=0, le=1, allow_inf_nan=False)
    overlap_policy: Literal["reject_entire_overlapping_cycle_cluster_without_trimming"]
    activity_gate: Literal["peak_frame_inside_observed_activity_bout"]
    scope: str

    @model_validator(mode="after")
    def ordered(self):
        if not (
            0 < Fraction(self.quiet_seconds) < Fraction(self.maximum_cycle_seconds)
            and 0 < Fraction(self.minimum_cycle_seconds) < Fraction(self.maximum_cycle_seconds) <= 3
            and self.maximum_return_distance_torso < self.minimum_excursion_torso
        ):
            raise ValueError("Invalid arm-cycle phase policy")
        return self


def disjoint(proposals):
    """Deduplicate; reject every member of an overlap cluster, including chained overlaps."""
    unique = {
        (p["start_frame"], p["stop_frame_exclusive"], p["arm"], p["peak_frame"]): p
        for p in proposals
    }
    rows = sorted(
        unique.values(), key=lambda p: (p["start_frame"], p["stop_frame_exclusive"], p["arm"])
    )
    accepted, rejected, cluster = [], [], []
    stop = -1
    for p in rows:
        if cluster and p["start_frame"] >= stop:
            (accepted if len(cluster) == 1 else rejected).extend(cluster)
            cluster = []
            stop = -1
        cluster.append(p)
        stop = max(stop, p["stop_frame_exclusive"])
    (accepted if len(cluster) == 1 else rejected).extend(cluster)
    return accepted, [{**p, "reason": "overlapping_arm_cycles_not_isolatable"} for p in rejected]


def segment(raw, video, activity, policy):
    """Use whole-video motion context; candidate peaks must lie inside an activity bout.

    Stable pre/post boundaries and full return are required, so this is explicitly offline.
    Exact indices are selected from observed frames; no interpolation or inferred punch count.
    """
    validate_arrays(raw)
    times, _ = validate_timing(raw, video)
    n = len(times)
    if len(activity["frames"]) != n or any(
        r["frame_index"] != i or Fraction(r["timestamp_exact"]) != times[i]
        for i, r in enumerate(activity["frames"])
    ):
        raise ValueError("Activity observations do not match prepared source timing")
    bouts = activity["bouts"]
    previous = 0
    for b in bouts:
        a, z = b["start_frame"], b["stop_frame_exclusive"]
        if not (type(a) is int and type(z) is int and previous <= a < z <= n):
            raise ValueError("Invalid or overlapping activity bounds")
        previous = z
    x, _, normal = features(
        raw["coordinates"], raw["confidence"], raw["valid_mask"], video["width"] / video["height"]
    )
    joints = x[:, :40].reshape(n, 8, 5)
    observed = np.array([r["observable"] for r in activity["frames"]], dtype=bool) & normal
    reset = np.array([r["discontinuity_before"] for r in activity["frames"]], dtype=bool)
    gap = Fraction(activity["effective_reset_gap_seconds_exact"])
    cuts = set(video.get("confirmed_scene_cuts", []))
    for i in range(1, n):
        reset[i] |= (
            times[i] - times[i - 1] > gap
            or i in cuts
            or raw["scene_id"][i] != raw["scene_id"][i - 1]
            or raw["segment_id"][i] != raw["segment_id"][i - 1]
        )
    proposals, rejected, quiet_evidence = [], [], []
    for arm, indices in [("left", (0, 2, 4)), ("right", (1, 3, 5))]:
        shoulder, elbow, wrist = indices
        usable = observed & (raw["pose_status"] != 2) & (joints[:, indices, 4] == 1).all(axis=1)
        position = joints[:, wrist, :2] - joints[:, shoulder, :2]
        arm_length = np.linalg.norm(
            joints[:, elbow, :2] - joints[:, shoulder, :2], axis=1
        ) + np.linalg.norm(joints[:, wrist, :2] - joints[:, elbow, :2], axis=1)
        guard_ratio = np.linalg.norm(position, axis=1) / np.maximum(arm_length, 1e-6)
        cursor = 0
        while cursor < n:
            if not usable[cursor]:
                cursor += 1
                continue
            end = cursor + 1
            while end < n and usable[end] and not reset[end]:
                end += 1
            # Each quiet run is a sequence of genuinely observed low-speed edges, not filled poses.
            quiet = []
            i = cursor + 1
            while i < end:
                speed = float(np.linalg.norm(position[i] - position[i - 1])) / float(
                    times[i] - times[i - 1]
                )
                if speed > policy.quiet_speed_torso_per_second:
                    i += 1
                    continue
                a = i - 1
                while i < end:
                    speed = float(np.linalg.norm(position[i] - position[i - 1])) / float(
                        times[i] - times[i - 1]
                    )
                    if speed > policy.quiet_speed_torso_per_second:
                        break
                    i += 1
                z = i - 1
                if times[z] - times[a] >= Fraction(policy.quiet_seconds):
                    quiet.append((a, z))
                    quiet_evidence.append(dict(arm=arm, start_frame=a, last_frame=z))
            q = 0
            while q + 1 < len(quiet):
                found = False
                for r in range(q + 1, len(quiet)):
                    before, after = quiet[q], quiet[r]
                    if times[after[0]] - times[before[1]] > Fraction(policy.maximum_cycle_seconds):
                        break
                    a, last = before[1], after[0]
                    radius = np.linalg.norm(position[a : last + 1] - position[a], axis=1)
                    k = int(radius.argmax())
                    peak = a + k
                    parents = [
                        b for b in bouts if b["start_frame"] <= peak < b["stop_frame_exclusive"]
                    ]
                    excursion = float(radius[k])
                    row = dict(
                        arm=arm,
                        start_frame=a,
                        stop_frame_exclusive=last + 1,
                        peak_frame=peak,
                        evidence_through_frame=after[1],
                        activity_bout_id=parents[0]["bout_id"] if parents else None,
                        excursion_torso=excursion,
                        return_distance_torso=float(radius[-1]),
                    )
                    elapsed = times[last] - times[a]
                    variation = float(np.abs(np.diff(radius)).sum())
                    ideal = 2 * excursion - float(radius[-1])
                    path = np.linalg.norm(np.diff(position[a : last + 1], axis=0), axis=1)
                    outbound = float(path[:k].sum())
                    inbound = float(path[k:].sum())
                    efficiencies = [
                        excursion / outbound if outbound else 0,
                        (excursion - float(radius[-1])) / inbound if inbound else 0,
                    ]
                    reason = None
                    row["boundary_extension_ratios"] = [
                        float(guard_ratio[a]),
                        float(guard_ratio[last]),
                    ]
                    if max(row["boundary_extension_ratios"]) > policy.maximum_guard_extension_ratio:
                        reason = "compact_guard_boundaries_not_established"
                    elif excursion < policy.minimum_excursion_torso:
                        reason = "insufficient_arm_excursion"
                    elif (
                        not Fraction(policy.minimum_cycle_seconds)
                        <= elapsed
                        <= Fraction(policy.maximum_cycle_seconds)
                    ):
                        reason = "cycle_duration_outside_policy"
                    elif radius[-1] > policy.maximum_return_distance_torso:
                        reason = "guard_return_not_established"
                    elif variation > policy.maximum_radial_variation_ratio * ideal:
                        reason = "multiple_or_irregular_motion_phases"
                    elif min(efficiencies) < policy.minimum_phase_path_efficiency:
                        reason = "ambiguous_outward_return_path"
                    row.update(
                        radial_variation_ratio=variation / ideal if ideal > 0 else None,
                        phase_path_efficiency=efficiencies,
                    )
                    if parents:
                        if reason is None and any(
                            b["bout_id"] != parents[0]["bout_id"]
                            and b["start_frame"] < last + 1
                            and b["stop_frame_exclusive"] > a
                            for b in bouts
                        ):
                            reason = "cycle_spans_multiple_activity_bouts"
                        (rejected if reason else proposals).append(
                            {**row, **(dict(reason=reason) if reason else {})}
                        )
                    if reason is None:
                        q = r
                        found = True
                        break
                if not found:
                    q += 1
            cursor = end
    candidates, overlaps = disjoint(proposals)
    rejected.extend(overlaps)
    unresolved = []
    for bout in bouts:
        cursor, stop = bout["start_frame"], bout["stop_frame_exclusive"]
        inside = [p for p in candidates if p["activity_bout_id"] == bout["bout_id"]]
        for p in inside:
            if cursor < p["start_frame"]:
                unresolved.append(
                    dict(
                        activity_bout_id=bout["bout_id"],
                        start_frame=cursor,
                        stop_frame_exclusive=min(p["start_frame"], stop),
                        reason="activity_without_isolated_complete_arm_cycle",
                    )
                )
            cursor = max(cursor, p["stop_frame_exclusive"])
        if cursor < stop:
            unresolved.append(
                dict(
                    activity_bout_id=bout["bout_id"],
                    start_frame=cursor,
                    stop_frame_exclusive=stop,
                    reason="activity_without_isolated_complete_arm_cycle",
                )
            )
    return dict(
        candidates=candidates,
        rejected_proposals=rejected,
        unresolved_activity=unresolved,
        quiet_boundaries=quiet_evidence,
        interpretation="heuristic motion candidates, not verified punches; "
        "incomplete or overlapping cycles remain unresolved",
        processing_mode="offline_full_video_with_post_return_evidence",
    )
