"""Exact-source windows. Unknown labels and warm-up never contribute to loss."""

from statistics import median

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.continuous_dataset import validate
from ..data.verified_background import workspace_at
from ..services.dashboard import require
from ..training.data import exact_times, features, validate_arrays
from ..training.velocity import joint_velocities
from ..vision.contracts import JOINTS

VERSION = "continuous-xy-velocity-1.0"
WARMUP = 14  # Existing TCN's 15-frame receptive field.
HELD = {"V4-bg-4", "V4-bg-5"}


def encode_source(raw, aspect, cuts=()):
    """No temporal filling. A missing torso/pose resets temporal state, not the clock."""
    validate_arrays(raw)
    times = exact_times({"raw": raw})
    steps = [b - a for a, b in zip(times, times[1:], strict=False)]
    gap = 2 * median(steps)
    x, _, normal = features(raw["coordinates"], raw["confidence"], raw["valid_mask"], aspect)
    x = torch.from_numpy(x)
    velocity = joint_velocities(x, raw, gap)
    observable = normal & (raw["pose_status"] != 2)
    epoch = np.full(len(x), -1, dtype=np.int64)
    start = 0
    for i in range(len(x)):
        discontinuity = (
            i == 0
            or not observable[i - 1]
            or (
                raw["frame_index"][i] != raw["frame_index"][i - 1] + 1
                or raw["scene_id"][i] != raw["scene_id"][i - 1]
                or raw["segment_id"][i] != raw["segment_id"][i - 1]
                or times[i] - times[i - 1] > gap
                or i in cuts
            )
        )
        if discontinuity:
            start = i
            velocity[i] = 0  # Also respect explicit source cuts with unchanged scene IDs.
        if observable[i]:
            epoch[i] = start
    return torch.cat((x, velocity.reshape(len(x), 24)), dim=1), observable, epoch, gap


def make_windows(x, observable, epoch, interval, core_frames=32):
    p = interval["provenance"]
    start, stop = p["start"], p["stop"]
    if not (0 <= start < stop <= len(x)) or core_frames < 1:
        raise ValueError("Invalid reviewed window")
    labels = torch.full((len(x),), -1, dtype=torch.long)
    for run in interval["supervision_runs"]:
        a, b, target = run["start"], run["stop"], run["activity"]
        if not start <= a < b <= stop or target not in (0, 1) or (labels[a:b] != -1).any():
            raise ValueError("Invalid/overlapping supervision runs")
        labels[a:b] = target
    if (labels[start:stop] < 0).any():
        raise ValueError("Incomplete exhaustive interval")
    windows = []
    cursor = start
    while cursor < stop:
        if not observable[cursor]:
            cursor += 1
            continue
        end = cursor + 1
        while end < stop and observable[end] and epoch[end] == epoch[cursor]:
            end += 1
        for core in range(cursor, end, core_frames):
            right = min(core + core_frames, end)
            left = max(int(epoch[core]), core - WARMUP)
            mask = torch.zeros(right - left, dtype=torch.bool)
            mask[core - left :] = True
            windows.append(
                dict(
                    x=x[left:right].clone(),
                    labels=labels[left:right].clone(),
                    loss_mask=mask,
                    frames=np.arange(left, right),
                    interval_id=interval["id"],
                    recording=p["recording"],
                )
            )
        cursor = end
    return windows, labels[start:stop].numpy()


class ReviewedWindows(Dataset):
    def __init__(self, root, recording="V4", core_frames=32):
        require(recording in ("V4", "V6"), "detector_recording_not_in_verified_pool")
        self.manifest = validate(root)
        require(
            not any(i["id"] in HELD for i in self.manifest["intervals"]),
            "held_interval_in_detector",
        )
        self.intervals = [
            i for i in self.manifest["intervals"] if i["provenance"]["recording"] == recording
        ]
        require(bool(self.intervals), "no_reviewed_detector_intervals")
        require(
            all(
                i["exhaustive"] and i["provenance"]["role"] == "train_candidate"
                for i in self.intervals
            ),
            "detector_split_leakage",
        )
        ws = workspace_at(root, "original")
        spec = ws.base._recording(recording)
        prep = ws.base._preparation(recording, spec)
        require(
            prep["pose_contract"]["confidence_order"] == ["visibility", "presence"]
            and prep["pose_contract"]["joint_order"] == list(JOINTS),
            "unsupported_detector_confidence",
        )
        with ws.base.store.open_regular(f"{spec.dataset}/poses.npz") as stream:
            ws.frames._verify(stream, spec.inputs["poses.npz"])
            with np.load(stream, allow_pickle=False) as z:
                self.raw = {k: z[k] for k in z.files}
        import av

        with ws.base.store.open_regular(f"data/raw/boxingvi/RGB_videos/{recording}.mkv") as stream:
            ws.frames._verify(stream, prep["inputs"]["video_sha256"])
            with av.open(stream) as video:
                track = video.streams.video[0]
                width, height = track.width, track.height
        require(width > 0 and height > 0, "detector_missing_source_dimensions")
        self.x, self.observable, self.epoch, self.gap = encode_source(
            self.raw, width / height, prep.get("confirmed_scene_cuts", [])
        )
        self.windows = []
        self.labels = {}
        for interval in self.intervals:
            windows, labels = make_windows(
                self.x, self.observable, self.epoch, interval, core_frames
            )
            self.windows.extend(windows)
            self.labels[interval["id"]] = labels
        counts = torch.bincount(
            torch.cat([w["labels"][w["loss_mask"]] for w in self.windows]), minlength=2
        )
        self.diagnostics = dict(
            recording=recording,
            source_group_guard=self.intervals[0]["provenance"]["source_group_guard"],
            reviewed_intervals=[i["id"] for i in self.intervals],
            windows=len(self.windows),
            supervised_observable_frames=counts.tolist(),
            reviewed_frames=sum(len(y) for y in self.labels.values()),
            unreliable_reviewed_frames=sum(
                int((~self.observable[i["provenance"]["start"] : i["provenance"]["stop"]]).sum())
                for i in self.intervals
            ),
            source_inputs=spec.inputs,
            video_sha256=prep["inputs"]["video_sha256"],
            source_dimensions=[width, height],
            feature_version=VERSION,
            channels=65,
            max_velocity_gap_seconds_exact=str(self.gap),
            warmup_frames=WARMUP,
            unknown_or_warmup_loss=False,
            temporal_reset="cut, scene/segment change, gap, missing pose or unreliable torso",
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]


def collate(windows):
    length = max(len(w["x"]) for w in windows)
    x = torch.zeros(len(windows), length, 65)
    labels = torch.zeros(len(windows), length)
    mask = torch.zeros(len(windows), length, dtype=torch.bool)
    for i, w in enumerate(windows):
        n = len(w["x"])
        x[i, :n] = w["x"]
        labels[i, :n] = w["labels"].clamp_min(0)
        mask[i, :n] = w["loss_mask"]
    return x, labels, mask
