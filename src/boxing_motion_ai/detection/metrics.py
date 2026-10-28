"""Bounded event metrics on declared exhaustive exposure; never infer unreviewed minutes."""

from dataclasses import dataclass
from fractions import Fraction
from statistics import median

import numpy as np

VERSION = "binary-event-metrics-1.0"


@dataclass(frozen=True)
class Event:
    event_id: str
    recording: str
    start: Fraction
    stop: Fraction
    emitted_at: Fraction | None = None


@dataclass(frozen=True)
class Exposure:
    recording: str
    start: Fraction
    stop: Fraction


def event_metrics(references, predictions, exposure, iou_threshold=Fraction(1, 2)):
    """Maximum-cardinality then maximum-total-IoU matching, deterministic stable ties.

    All endpoints must be exact Fraction seconds. Edge/unknown-region events must
    be handled explicitly by a future scoring protocol, not silently clipped here.
    """
    refs = sorted(references, key=lambda e: (e.recording, e.start, e.event_id))
    preds = sorted(predictions, key=lambda e: (e.recording, e.start, e.event_id))
    coverage = sorted(exposure, key=lambda e: (e.recording, e.start))
    if not isinstance(iou_threshold, Fraction) or not 0 < iou_threshold <= 1:
        raise ValueError("Exact IoU threshold in (0,1] required")
    if not coverage or len(refs) + len(preds) > 500:
        raise ValueError("Nonempty bounded reviewed exposure required")
    for group in (refs, preds):
        if len({e.event_id for e in group}) != len(group):
            raise ValueError("Duplicate event ID")
    for group in (refs, preds, coverage):
        for e in group:
            if (
                not isinstance(e.start, Fraction)
                or not isinstance(e.stop, Fraction)
                or e.stop <= e.start
            ):
                raise ValueError("Exact nonempty source-time intervals required")
    for a, b in zip(coverage, coverage[1:], strict=False):
        if a.recording == b.recording and b.start < a.stop:
            raise ValueError("Overlapping exposure double-counts minutes")
    for e in refs + preds:
        if not any(
            c.recording == e.recording and c.start <= e.start < e.stop <= c.stop for c in coverage
        ):
            raise ValueError(
                "Event crosses scored boundary or unknown exposure; explicit edge policy required"
            )
    if any(not isinstance(p.emitted_at, Fraction) for p in preds):
        raise ValueError("Predictions require true first-emission times, not backdated onset")
    # Unit-capacity residual network; exact costs avoid float tie instability.
    n, m = len(preds), len(refs)
    source, sink = n + m, n + m + 1
    graph = [[] for _ in range(n + m + 2)]

    def edge(a, b, cost):
        forward = [b, len(graph[b]), 1, cost]
        reverse = [a, len(graph[a]), 0, -cost]
        graph[a].append(forward)
        graph[b].append(reverse)
        return forward

    for i in range(n):
        edge(source, i, Fraction())
    for j in range(m):
        edge(n + j, sink, Fraction())
    pairs = []
    for i, p in enumerate(preds):
        for j, r in enumerate(refs):
            if p.recording != r.recording:
                continue
            intersection = max(Fraction(), min(p.stop, r.stop) - max(p.start, r.start))
            overlap = intersection / (max(p.stop, r.stop) - min(p.start, r.start))
            if overlap >= iou_threshold:
                pairs.append((i, j, overlap, edge(i, n + j, -overlap)))
    while True:
        distance = [None] * len(graph)
        distance[source] = Fraction()
        previous = [None] * len(graph)
        for _ in range(len(graph) - 1):
            changed = False
            for a, edges in enumerate(graph):
                if distance[a] is None:
                    continue
                for k, e in enumerate(edges):
                    b, _, capacity, cost = e
                    candidate = distance[a] + cost
                    if capacity and (distance[b] is None or candidate < distance[b]):
                        distance[b] = candidate
                        previous[b] = (a, k)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        b = sink
        while b != source:
            a, k = previous[b]
            e = graph[a][k]
            e[2] -= 1
            graph[b][e[1]][2] += 1
            b = a
    matched = [(i, j, overlap) for i, j, overlap, e in pairs if e[2] == 0]
    tp = len(matched)
    fp = n - tp
    fn = m - tp
    seconds = sum((c.stop - c.start for c in coverage), Fraction())
    delays = [float(preds[i].emitted_at - refs[j].start) for i, j, _ in matched]
    return dict(
        version=VERSION,
        iou_threshold=str(iou_threshold),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=tp / n if n else None,
        recall=tp / m if m else None,
        f1=2 * tp / (n + m) if n + m else None,
        scored_seconds_exact=str(seconds),
        false_positives_per_minute=float(fp * 60 / seconds),
        latency_seconds=dict(
            count=len(delays),
            median=median(delays) if delays else None,
            p90=float(np.quantile(delays, 0.9)) if delays else None,
        ),
        matches=[
            dict(
                prediction=preds[i].event_id,
                reference=refs[j].event_id,
                iou=str(overlap),
                latency_seconds=float(preds[i].emitted_at - refs[j].start),
            )
            for i, j, overlap in matched
        ],
    )
