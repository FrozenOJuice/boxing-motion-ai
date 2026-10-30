"""Deterministic offline boundary pairing, never arbitrary activity-bout splitting."""

from fractions import Fraction as Q

import numpy as np

from ..analysis.automatic import bounds
from .instances_v2 import runs

VERSION = "offline-boundary-pair-decoder-1.0"


def peaks(scores, times, first, stop, threshold, radius):
    selected = []
    proposals = []
    i = first
    while i < stop:
        z = i + 1
        while z < stop and scores[z] == scores[i]:
            z += 1
        if scores[i] >= threshold and all(
            scores[i] >= scores[j]
            for j in range(first, stop)
            if times[i] - radius <= times[j] <= times[z - 1] + radius
        ):
            proposals.append(i)  # One earliest anchor per plateau, never periodic pseudo-peaks.
        i = z
    # Deterministic plateau/NMS: best score then earliest source frame, same head only.
    for i in sorted(proposals, key=lambda j: (-scores[j], j)):
        if not any(abs(times[i] - times[j]) <= radius for j in selected):
            selected.append(i)
    return sorted(selected)


def decode(probabilities, observable, reset, video, policy):
    times = [Q(r["timestamp_exact"]) for r in video["frames"]]
    n = len(times)
    if (
        probabilities.shape != (n, 3)
        or len(observable) != n
        or len(reset) != n
        or any(b <= a for a, b in zip(times, times[1:], strict=False))
        or not np.isfinite(probabilities[observable]).all()
        or ((probabilities[observable] < 0) | (probabilities[observable] > 1)).any()
    ):
        raise ValueError("Invalid localization source probabilities/clock")
    stop_time = Q(video["stop_timestamp_exact"])
    clock = lambda i: times[i] if i < n else stop_time  # noqa: E731
    candidates, unresolved, activity, boundary_peaks = [], [], [], []
    for first, stop in runs(observable, reset):
        starts = peaks(
            probabilities[:, 1],
            times,
            first,
            stop,
            policy["boundary_threshold"],
            Q(policy["peak_radius_seconds"]),
        )
        ends = peaks(
            probabilities[:, 2],
            times,
            first,
            stop,
            policy["boundary_threshold"],
            Q(policy["peak_radius_seconds"]),
        )
        boundary_peaks.append(
            dict(start_frame=first, stop_frame_exclusive=stop, onsets=starts, ends=ends)
        )
        pending = []
        for i in sorted(set(starts + ends)):
            # Same-frame end and next onset are ambiguous, never repaired into a midpoint split.
            if i in starts:
                pending.append(i)
            if i not in ends:
                continue
            if not pending:
                unresolved.append(
                    dict(
                        start_frame=i,
                        stop_frame_exclusive=i + 1,
                        reason="end_without_observed_onset",
                        left_censored=True,
                    )
                )
                continue
            a, z = pending[0], i + 1
            reason = "multiple_onsets_before_end" if len(pending) > 1 else None
            if reason is None and (a == first or i == stop - 1):
                reason = "boundary_censored_at_observation_edge"
            duration = times[i] - times[a]
            support = sum(
                (clock(k + 1) - times[k])
                for k in range(a, z)
                if probabilities[k, 0] >= policy["activity_threshold"]
            )
            fraction = float(support / (clock(z) - times[a]))
            if reason is None and not Q(policy["minimum_span_seconds"]) <= duration <= Q(
                policy["maximum_span_seconds"]
            ):
                reason = "boundary_span_outside_policy"
            if reason is None and fraction < policy["minimum_activity_fraction"]:
                reason = "boundary_pair_without_sufficient_activity"
            row = dict(
                start_frame=a,
                stop_frame_exclusive=z,
                onset_frame=a,
                end_frame=i,
                activity_fraction=fraction,
                left_censored=a == first,
                right_censored=i == stop - 1,
            )
            if reason:
                unresolved.append(dict(**row, reason=reason, overlap_ambiguous=len(pending) > 1))
            else:
                row.update(
                    candidate_id=f"candidate-{len(candidates) + 1}",
                    verified=False,
                    source="experimental_learned_boundary_pair",
                    peak_frame=int(a + np.argmax(probabilities[a:z, 0])),
                    peak_interpretation="activity-score maximum, not impact",
                    onset_score=float(probabilities[a, 1]),
                    end_score=float(probabilities[i, 2]),
                )
                candidates.append(row)
            pending = []
        if pending:
            unresolved.append(
                dict(
                    start_frame=pending[0],
                    stop_frame_exclusive=stop,
                    reason="onset_without_observed_end",
                    right_censored=True,
                    overlap_ambiguous=len(pending) > 1,
                )
            )
        active = np.zeros(n, bool)
        active[first:stop] = probabilities[first:stop, 0] >= policy["activity_threshold"]
        for a, z in runs(active, reset):
            activity.append(
                dict(
                    start_frame=a,
                    stop_frame_exclusive=z,
                    left_censored=a == first,
                    right_censored=z == stop,
                )
            )
            covered = np.zeros(z - a, bool)
            for c in candidates:
                u, v = max(a, c["start_frame"]), min(z, c["stop_frame_exclusive"])
                if u < v:
                    covered[u - a : v - a] = True
            for u, v in runs(~covered, np.zeros(z - a, bool)):
                unresolved.append(
                    dict(
                        start_frame=a + u,
                        stop_frame_exclusive=a + v,
                        reason="activity_without_supported_boundary_pair",
                    )
                )
    unknown = [
        dict(
            start_frame=a,
            stop_frame_exclusive=z,
            reason="insufficient_common_person_or_motion_observation_not_background",
        )
        for a, z in runs(~observable, reset)
    ]
    for row in [*candidates, *activity, *unresolved, *unknown]:
        row.update(bounds(row, video))
    return dict(
        version=VERSION,
        candidates=candidates,
        possible_activity=activity,
        unresolved=unresolved,
        unobservable=unknown,
        boundary_peaks=boundary_peaks,
        interruptions=[
            dict(
                previous_frame=i - 1,
                next_frame=i,
                previous_timestamp_exact=str(times[i - 1]),
                next_timestamp_exact=str(times[i]),
                reason="source_reset_or_discontinuity_not_background",
            )
            for i in range(1, n)
            if reset[i]
        ],
        processing="offline centered 15-frame model plus future peak evidence; "
        "no causal emission claim",
        interpretation="uncalibrated activity/boundary scores, not verified punches or classes",
    )
