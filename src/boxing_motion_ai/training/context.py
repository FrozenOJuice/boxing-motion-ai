"""Exact-time, boundary-safe real context; supervision/pooling stays on the punch."""

import copy
import json
from bisect import bisect_left
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..vision.pipeline import sha256
from .data import collate, features

CONTEXT_VERSION = "real-preclip-context-1.0"
CONTEXT_POLICY = {
    "version": CONTEXT_VERSION,
    "maximum_contiguous_gap_seconds": 0.1,
    "window": "[punch_start_time - seconds, punch_start_time), exact rational PTS",
    "pool": "original annotated frames only; context unknown, no per-frame class labels",
    "resets": [
        "recording start",
        "confirmed scene cut",
        "scene_id change",
        "segment_id change",
        "nonconsecutive frame index",
        "time-base change",
        "nonpositive or >100ms time step",
    ],
    "missing_detection": "keep original invalid masks/NaNs; not an extraction boundary",
    "missing_extraction": "incomplete prepared inventories rejected by source loader",
    "features": "unchanged frame-local normalization; no resampling, filling or smoothing",
}


def exact_timeline(arrays):
    return [
        Fraction(int(p) * int(n), int(d))
        for p, n, d in zip(
            arrays["pts"], arrays["time_base_num"], arrays["time_base_den"], strict=True
        )
    ]


def run_starts(arrays, times, confirmed_cuts=()):
    """Latest hard boundary for every array position, independently of annotations."""
    starts = np.zeros(len(times), dtype=np.int64)
    boundaries = {0: ["recording start"]}
    cuts = set(confirmed_cuts)
    for i in range(1, len(times)):
        reasons = []
        if int(arrays["frame_index"][i]) in cuts:
            reasons.append("confirmed scene cut")
        for key in ("scene_id", "segment_id"):
            if arrays[key][i] != arrays[key][i - 1]:
                reasons.append(key + " change")
        if arrays["frame_index"][i] != arrays["frame_index"][i - 1] + 1:
            reasons.append("nonconsecutive frame index")
        if any(arrays[k][i] != arrays[k][i - 1] for k in ("time_base_num", "time_base_den")):
            reasons.append("time-base change")
        if not 0 < times[i] - times[i - 1] <= Fraction(1, 10):
            reasons.append("nonpositive or >100ms time step")
        if reasons:
            boundaries[i] = reasons
        starts[i] = i if reasons else starts[i - 1]
    return starts, boundaries


def context_start(times, start, run_start, seconds):
    window = Fraction(str(seconds))
    if window < 0:
        raise ValueError("Context duration must be nonnegative")
    return bisect_left(times, times[start] - window, lo=int(run_start), hi=start)


class ContextClips(Dataset):
    """Extend an approved source dataset without changing/excluding any target example."""

    def __init__(self, base, seconds):
        if not 0 < seconds <= 2:
            raise ValueError("Context duration must be >0 and <=2 seconds")
        root = Path(base.provenance["dataset"])
        if any(sha256(root / name) != digest for name, digest in base.provenance["hashes"].items()):
            raise ValueError("Prepared source changed before context loading")
        manifest = json.loads((root / "preparation.json").read_text())
        with np.load(root / "poses.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        times = exact_timeline(arrays)
        starts, boundaries = run_starts(arrays, times, manifest["confirmed_scene_cuts"])
        candidates = [
            json.loads(line) for line in (root / "candidates.jsonl").read_text().splitlines()
        ]
        aspect = base.provenance["source_dimensions"][0] / base.provenance["source_dimensions"][1]
        clips, records = [], []
        for original in base.clips:
            start, stop = original["annotation"]["array_slice_half_open"]
            # A newly catalogued discontinuity inside the punch must fail, never be crossed.
            if starts[stop - 1] > start:
                raise ValueError(
                    f"Punch crosses a continuity boundary: row {original['source_row']}"
                )
            lo = context_start(times, start, starts[start], seconds)
            a = {key: values[lo:stop].copy() for key, values in arrays.items()}
            x, masks, normal = features(a["coordinates"], a["confidence"], a["valid_mask"], aspect)
            offset = start - lo
            if not torch.equal(torch.from_numpy(x[offset:]), original["x"]) or not torch.equal(
                torch.from_numpy(masks[offset:]), original["mask_x"]
            ):
                raise ValueError("Context changed original punch features")
            pool_mask = torch.arange(stop - lo) >= offset
            overlaps, excluded = [], []
            for c in candidates:
                bounds = c.get("array_slice_half_open")
                if (
                    not bounds
                    or None in bounds
                    or c["source_row"] == original["source_row"]
                    or not 0 <= bounds[0] < bounds[1] <= len(times)
                ):
                    continue
                if lo < start and bounds[0] < start and bounds[1] > lo:
                    overlaps.append(c["source_row"])
                    if not c["annotation_eligible"]:
                        excluded.append(c["source_row"])
            record = {
                "source_row": original["source_row"],
                "context_frames": offset,
                "context_start_frame": int(a["frame_index"][0]),
                "punch_start_frame": int(arrays["frame_index"][start]),
                "punch_end_frame": int(arrays["frame_index"][stop - 1]),
                "context_start_pts": int(a["pts"][0]),
                "punch_start_pts": int(arrays["pts"][start]),
                "time_base": str(Fraction(int(a["time_base_num"][0]), int(a["time_base_den"][0]))),
                "actual_lookback_seconds": float(times[start] - times[lo]),
                "boundary_limited": bool(
                    lo == starts[start] and times[lo] > times[start] - Fraction(str(seconds))
                ),
                "last_boundary_frame": int(arrays["frame_index"][starts[start]]),
                "last_boundary_reasons": boundaries[int(starts[start])],
                "overlapping_other_annotation_rows": overlaps,
                "overlapping_excluded_annotation_rows": excluded,
            }
            clips.append(
                {
                    **original,
                    "x": torch.from_numpy(x),
                    "mask_x": torch.from_numpy(masks),
                    "raw": a,
                    "punch_raw": original["raw"],
                    "pool_mask": pool_mask,
                    "context_length": offset,
                    "punch_length": len(original["x"]),
                    "context_record": record,
                    "normalization_valid": normal,
                    "timestamp_seconds": a["pts"] * a["time_base_num"] / a["time_base_den"],
                }
            )
            records.append(record)
        self.clips = clips
        self.provenance = copy.deepcopy(base.provenance)
        reason_counts = Counter(reason for reasons in boundaries.values() for reason in reasons)
        counts = [r["context_frames"] for r in records]
        self.provenance["context"] = {
            **CONTEXT_POLICY,
            "seconds": seconds,
            "records": records,
            "boundary_reason_counts": dict(reason_counts),
            "context_frame_range": [min(counts), max(counts)],
            "boundary_limited_examples": sum(r["boundary_limited"] for r in records),
            "examples_with_less_than_14_history_frames": sum(n < 14 for n in counts),
            "examples_overlapping_other_annotations": sum(
                bool(r["overlapping_other_annotation_rows"]) for r in records
            ),
            "examples_overlapping_excluded_annotations": sum(
                bool(r["overlapping_excluded_annotation_rows"]) for r in records
            ),
            "total_input_frames": sum(len(c["x"]) for c in clips),
            "missing_pose_context_frames": int(
                sum((c["raw"]["pose_status"][: c["context_length"]] == 2).sum() for c in clips)
            ),
            "invalid_normalization_context_frames": int(
                sum((~c["normalization_valid"][: c["context_length"]]).sum() for c in clips)
            ),
            "note": (
                "Source quality/counts apply only to original punches; context never relabeled."
            ),
        }

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return self.clips[i]


def context_collate(clips):
    result = collate(clips)
    result["pool_mask"] = torch.nn.utils.rnn.pad_sequence(
        [c.get("pool_mask", torch.ones(len(c["x"]), dtype=torch.bool)) for c in clips],
        batch_first=True,
    )
    if (result["pool_mask"] & ~result["time_mask"]).any() or not result["pool_mask"].any(1).all():
        raise ValueError("Pooling mask must select nonempty real punch frames only")
    return result


def pooling_mask(batch):
    return batch.get("pool_mask", batch["time_mask"])
