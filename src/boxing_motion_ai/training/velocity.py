"""Optional causal, validity-gated image-plane joint velocities; no pose imputation."""

from fractions import Fraction
from pathlib import Path
from statistics import median

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .data import FEATURE_VERSION, exact_times

VELOCITY_VERSION = "torso-core-2d-velocity-1.0"
VELOCITY_POLICY = {
    "version": VELOCITY_VERSION,
    "base_version": FEATURE_VERSION,
    "layout": "41 original channels + 8 anatomical joints * [vx,vy,velocity_valid]",
    "units": "normalized image-plane torso lengths per second; not metres per second",
    "time": "backward difference / exact rational PTS difference, converted to float once",
    "validity": "both usable joints/anchors, no missing pose, consecutive indices, "
    "same scene/segment",
    "timestamp_gap": "at most twice the median positive adjacent source-frame PTS difference",
    "gap_clock": "source metadata only, never annotation FPS; no quality-policy changes",
    "first_frame": "velocity invalid; no pre-clip context",
    "invalid": "zero sentinel and false velocity mask; never a filled joint estimate",
    "clipping_smoothing_resampling": "none",
    "mask_control": "8 original core masks + 8 derivative validity bits; no vector values",
}


def joint_velocities(x, raw, max_gap):
    """Return [T,8,3]. Inputs untouched; future samples cannot affect past values.

    x is the existing normalized feature tensor, including its usable-joint bits.
    Metadata discontinuities invalidate the pair even if both endpoint poses exist.
    Non-increasing timestamps are contract errors, not repaired data.
    """
    if (
        x.ndim != 2
        or not len(x)
        or x.shape[1] != 41
        or not isinstance(max_gap, Fraction)
        or max_gap <= 0
    ):
        raise ValueError("Expected original 41-channel XY and positive exact gap limit")
    integer_fields = (
        "pts",
        "time_base_num",
        "time_base_den",
        "frame_index",
        "scene_id",
        "segment_id",
        "pose_status",
    )
    if any(np.asarray(raw[k]).dtype.kind not in "iu" for k in integer_fields) or any(
        np.any(np.asarray(raw[k]) <= 0) for k in ("time_base_num", "time_base_den")
    ):
        raise ValueError("Exact integer timing/metadata and positive time bases required")
    if any(
        len(raw[k]) != len(x)
        for k in (
            "pts",
            "time_base_num",
            "time_base_den",
            "frame_index",
            "scene_id",
            "segment_id",
            "pose_status",
        )
    ):
        raise ValueError("Velocity metadata length mismatch")
    times = exact_times({"raw": raw})
    dt = [b - a for a, b in zip(times, times[1:], strict=False)]
    if any(t <= 0 for t in dt):
        raise ValueError("Velocity requires strictly increasing exact timestamps")
    joints = x[:, :40].reshape(-1, 8, 5)
    usable = (joints[..., 4] == 1) & (x[:, 40, None] == 1)
    continuous = torch.tensor(
        [
            raw["frame_index"][i] == raw["frame_index"][i - 1] + 1
            and raw["scene_id"][i] == raw["scene_id"][i - 1]
            and raw["segment_id"][i] == raw["segment_id"][i - 1]
            and raw["pose_status"][i] != 2
            and raw["pose_status"][i - 1] != 2
            and dt[i - 1] <= max_gap
            for i in range(1, len(x))
        ],
        dtype=torch.bool,
        device=x.device,
    )
    valid = usable[1:] & usable[:-1] & continuous[:, None]
    elapsed = torch.tensor([float(t) for t in dt], dtype=torch.float64, device=x.device)
    delta = joints[1:, :, :2].double() - joints[:-1, :, :2].double()
    values = torch.where(valid[..., None], delta / elapsed[:, None, None], 0)
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite reliable velocity; do not silently clip or fill")
    result = x.new_zeros((len(x), 8, 3))
    result[1:, :, :2] = values.to(x.dtype)
    result[1:, :, 2] = valid.to(x.dtype)
    return result


def source_gap_limit(path):
    """Frozen continuity rule from full source timing, without labels or tuned thresholds."""
    with np.load(Path(path) / "poses.npz", allow_pickle=False) as z:
        a = {
            k: z[k]
            for k in (
                "pts",
                "time_base_num",
                "time_base_den",
                "frame_index",
                "scene_id",
                "segment_id",
            )
        }
    times = exact_times({"raw": a})
    diffs = [
        times[i] - times[i - 1]
        for i in range(1, len(times))
        if a["frame_index"][i] == a["frame_index"][i - 1] + 1
        and a["scene_id"][i] == a["scene_id"][i - 1]
        and a["segment_id"][i] == a["segment_id"][i - 1]
    ]
    if not diffs or any(t <= 0 for t in diffs):
        raise ValueError("Invalid source continuity clock")
    return 2 * median(diffs)


class VelocityClips(Dataset):
    """Wrap immutable PoseClips; never refilter examples, alter labels or read other sources."""

    def __init__(self, dataset, *, diagnostic=False, multi_recording=False):
        if diagnostic and multi_recording:
            raise ValueError("Separate velocity access gates required")
        allowed = ("V6", "V7", "V8", "V10") if diagnostic else ("V4", "V5")
        if multi_recording:
            allowed = ("V4", "V5", "V6", "V7", "V8", "V10")
        if dataset.provenance["recording_id"] not in allowed:
            raise ValueError("Velocity experiment permits V4/V5 only; V9 reserved")
        gap = source_gap_limit(dataset.provenance["dataset"])
        self.clips = []
        for c in dataset.clips:
            v = joint_velocities(c["x"], c["raw"], gap)
            self.clips.append(
                {
                    **c,
                    "x": torch.cat((c["x"], v.reshape(len(v), 24)), dim=1),
                    "mask_x": torch.cat((c["mask_x"], v[..., 2]), dim=1),
                }
            )
        self.provenance = {
            **dataset.provenance,
            "velocity_policy": {**VELOCITY_POLICY, "max_gap_seconds_exact": str(gap)},
            "feature_channels": 65,
            "mask_channels": 16,
        }

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return self.clips[i]


def expand_input(model, channels):
    """Copy reference initialization exactly, append zero weights, preserve global RNG.

    The necessary input-width adjustment is the only architecture difference. At
    initialization both models implement exactly the same function; shared weights
    and subsequent dropout RNG match. New columns can learn immediately.
    """
    original = model.input
    if channels <= original.in_features:
        raise ValueError("Input expansion must add channels")
    with torch.random.fork_rng(devices=[]):
        expanded = nn.Linear(channels, original.out_features)
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, : original.in_features].copy_(original.weight)
        expanded.bias.copy_(original.bias)
    model.input = expanded
    return model
