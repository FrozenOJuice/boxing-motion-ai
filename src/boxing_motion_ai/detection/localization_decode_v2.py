"""Conservative offline pairing of unchanged boundary peaks; no new model or threshold search."""

from fractions import Fraction as Q
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..analysis.automatic import bounds
from .instances_v2 import runs
from .localization_decode import decode as original_decode

VERSION = "offline-boundary-pair-decoder-2.0"


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["offline-boundary-pair-decoder-2.0"]
    activity_threshold: Literal[0.5]
    boundary_threshold: Literal[0.5]
    peak_radius_seconds: Literal["2/25"]
    minimum_span_seconds: Literal["2/25"]
    maximum_span_seconds: Literal["2"]
    minimum_activity_fraction: Literal[0.5]
    inactivity_reset_seconds: Literal["1/4"]
    assignment: Literal["first_compatible_ending_phase_unique_components"]
    coincident_boundaries: Literal["unresolved"]
    overlapping_pairs: Literal["unresolved_without_arm_assignment"]


def pair_evidence(a, b, first, stop, probabilities, times, clock, policy):
    """Same per-pair gates as v1. No references or scores used to invent endpoints."""
    if b <= a:
        return None, "end_not_after_onset"
    if a == first or b == stop - 1:
        return None, "boundary_censored_at_observation_edge"
    duration = times[b] - times[a]
    if not Q(policy["minimum_span_seconds"]) <= duration <= Q(policy["maximum_span_seconds"]):
        return None, "boundary_span_outside_policy"
    support = sum(
        (
            clock(i + 1) - times[i]
            for i in range(a, b + 1)
            if probabilities[i, 0] >= policy["activity_threshold"]
        ),
        Q(),
    )
    fraction = float(support / (clock(b + 1) - times[a]))
    if fraction < policy["minimum_activity_fraction"]:
        return None, "boundary_pair_without_sufficient_activity"
    return fraction, None


def decode(probabilities, observable, reset, video, policy):
    policy = Policy.model_validate(policy).model_dump()
    times = [Q(r["timestamp_exact"]) for r in video["frames"]]
    n = len(times)
    if (
        not n
        or Q(video["stop_timestamp_exact"]) <= times[-1]
        or any(r["frame_index"] != i for i, r in enumerate(video["frames"]))
    ):
        raise ValueError("Invalid native source frame/stop clock")
    base = original_decode(probabilities, observable, reset, video, policy)
    clock = lambda i: times[i] if i < n else Q(video["stop_timestamp_exact"])  # noqa: E731
    candidates, unresolved, audit, quiet_resets = [], [], [], []
    for run in base["boundary_peaks"]:
        first, stop = run["start_frame"], run["stop_frame_exclusive"]
        low = np.zeros(n, bool)
        low[first:stop] = probabilities[first:stop, 0] < policy["activity_threshold"]
        quiet = [
            (a, z)
            for a, z in runs(low, reset)
            if clock(z) - times[a] >= Q(policy["inactivity_reset_seconds"])
        ]
        epochs, cursor = [], first
        for a, z in quiet:
            if cursor < a:
                epochs.append((cursor, a))
            quiet_resets.append(
                dict(
                    start_frame=a,
                    stop_frame_exclusive=z,
                    reason="predicted_inactivity_reset_not_verified_background",
                )
            )
            cursor = z
        if cursor < stop:
            epochs.append((cursor, stop))
        for kind, values in (("onset", run["onsets"]), ("end", run["ends"])):
            for i in values:
                if any(a <= i < z for a, z in quiet):
                    unresolved.append(
                        dict(
                            start_frame=i,
                            stop_frame_exclusive=i + 1,
                            reason=f"{kind}_inside_predicted_inactivity",
                        )
                    )
        for left, right in epochs:
            starts = [a for a in run["onsets"] if left <= a < right]
            ends = [b for b in run["ends"] if left <= b < right]
            coincident = set(starts) & set(ends)
            for i in sorted(coincident):
                unresolved.append(
                    dict(
                        start_frame=i,
                        stop_frame_exclusive=i + 1,
                        reason="coincident_onset_end_ambiguous",
                        overlap_ambiguous=True,
                    )
                )
            starts = [i for i in starts if i not in coincident]
            ends = [i for i in ends if i not in coincident]
            # Ending phases are separated only by actual onset peaks, never a midpoint/time grid.
            phases, group, last_kind = {}, -1, None
            for i in sorted(set(starts + ends)):
                if i in starts:
                    last_kind = "onset"
                else:
                    if last_kind != "end":
                        group += 1
                    phases[i] = group
                    last_kind = "end"
            edges, rejected = [], []
            for a in starts:
                viable = []
                for b in ends:
                    if b <= a:
                        continue
                    if any(a < i < b for i in coincident):
                        rejected.append(
                            dict(onset=a, end=b, reason="crosses_coincident_boundary_ambiguity")
                        )
                        continue
                    fraction, reason = pair_evidence(
                        a, b, first, stop, probabilities, times, clock, policy
                    )
                    if reason:
                        rejected.append(dict(onset=a, end=b, reason=reason))
                    else:
                        viable.append(
                            dict(onset=a, end=b, activity_fraction=fraction, ending_phase=phases[b])
                        )
                if viable:
                    phase = min(e["ending_phase"] for e in viable)
                    edges.extend(e for e in viable if e["ending_phase"] == phase)
            # Only isolated one-onset/one-end components; no count/score optimization.
            remaining = list(edges)
            associated_starts, associated_ends = set(), set()
            epoch_candidates = []
            while remaining:
                component = [remaining.pop(0)]
                a_set, b_set = {component[0]["onset"]}, {component[0]["end"]}
                changed = True
                while changed:
                    changed = False
                    for edge in list(remaining):
                        if edge["onset"] in a_set or edge["end"] in b_set:
                            component.append(edge)
                            remaining.remove(edge)
                            a_set.add(edge["onset"])
                            b_set.add(edge["end"])
                            changed = True
                associated_starts |= a_set
                associated_ends |= b_set
                if len(a_set) != 1 or len(b_set) != 1:
                    unresolved.append(
                        dict(
                            start_frame=min(a_set),
                            stop_frame_exclusive=max(b_set) + 1,
                            reason="competing_boundary_assignments",
                            overlap_ambiguous=True,
                            onset_frames=sorted(a_set),
                            end_frames=sorted(b_set),
                        )
                    )
                else:
                    edge = component[0]
                    a, b = edge["onset"], edge["end"]
                    epoch_candidates.append(
                        dict(
                            start_frame=a,
                            stop_frame_exclusive=b + 1,
                            onset_frame=a,
                            end_frame=b,
                            activity_fraction=edge["activity_fraction"],
                            left_censored=False,
                            right_censored=False,
                            verified=False,
                            source="experimental_unique_phase_boundary_pair",
                            onset_score=float(probabilities[a, 1]),
                            end_score=float(probabilities[b, 2]),
                            peak_frame=int(a + np.argmax(probabilities[a : b + 1, 0])),
                            peak_interpretation="activity-score maximum, not impact",
                        )
                    )
            overlapping = set()
            for i, a in enumerate(epoch_candidates):
                for j, b in enumerate(epoch_candidates[i + 1 :], i + 1):
                    if (
                        a["start_frame"] < b["stop_frame_exclusive"]
                        and b["start_frame"] < a["stop_frame_exclusive"]
                    ):
                        overlapping |= {i, j}
            for i, c in enumerate(epoch_candidates):
                if i in overlapping:
                    unresolved.append(
                        c
                        | dict(
                            reason="overlapping_pairs_without_arm_assignment",
                            overlap_ambiguous=True,
                        )
                    )
                else:
                    candidates.append(c)
            for a in starts:
                if a in associated_starts:
                    continue
                expiry = next(
                    (
                        i
                        for i in range(a + 1, right)
                        if times[i] - times[a] > Q(policy["maximum_span_seconds"])
                    ),
                    None,
                )
                reason = (
                    "boundary_censored_at_observation_edge"
                    if a == first
                    else "onset_expired_at_maximum_span"
                    if expiry is not None
                    else "onset_interrupted_by_predicted_inactivity"
                    if right < stop
                    else "onset_without_compatible_end"
                )
                unresolved.append(
                    dict(
                        start_frame=a,
                        stop_frame_exclusive=expiry or right,
                        reason=reason,
                        left_censored=a == first,
                        right_censored=right == stop,
                        interrupted_by_predicted_inactivity=right < stop,
                    )
                )
            for b in ends:
                if b not in associated_ends:
                    unresolved.append(
                        dict(
                            start_frame=b,
                            stop_frame_exclusive=b + 1,
                            reason="end_without_compatible_onset",
                            possible_left_censoring=True,
                            right_censored=b == stop - 1,
                        )
                    )
            audit.append(
                dict(
                    start_frame=left,
                    stop_frame_exclusive=right,
                    edges=edges,
                    rejected_pairs=rejected,
                )
            )
    candidates.sort(key=lambda r: (r["start_frame"], r["stop_frame_exclusive"]))
    for i, c in enumerate(candidates):
        c["candidate_id"] = f"candidate-{i + 1}"
    for a in base["possible_activity"]:
        first, stop = a["start_frame"], a["stop_frame_exclusive"]
        covered = np.zeros(stop - first, bool)
        for c in candidates:
            u, z = max(first, c["start_frame"]), min(stop, c["stop_frame_exclusive"])
            if u < z:
                covered[u - first : z - first] = True
        for u, z in runs(~covered, np.zeros(stop - first, bool)):
            unresolved.append(
                dict(
                    start_frame=first + u,
                    stop_frame_exclusive=first + z,
                    reason="activity_without_supported_boundary_pair",
                )
            )
    for row in [*candidates, *unresolved, *quiet_resets]:
        row.update(bounds(row, video))
    return base | dict(
        version=VERSION,
        candidates=candidates,
        unresolved=unresolved,
        pairing_audit=audit,
        predicted_inactivity_resets=quiet_resets,
        processing=(
            "offline unchanged centered model/peaks; future ending-phase evidence; "
            "no emission claim"
        ),
    )
