"""Paired known-clip views, immutable weak selection and V5-only development access."""

from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..analysis.common import sha
from ..services.dashboard import require
from ..vision.full_body import CHANNELS, encode
from ..vision.observation_study import arrays, read
from .augmentation import mirror_features
from .data import CLASSES, PoseClips
from .velocity import VelocityClips

PIN = Path("configs/video_foundation_snapshot_v1.json")
CORE = (0, 1, 2, 3, 4, 5, 12, 13)
MAP65 = (
    [j * 8 + k for j in CORE for k in range(5)]
    + [208]
    + [j * 8 + k for j in CORE for k in (5, 6, 7)]
)
MASKS = [i for i, n in enumerate(CHANNELS) if n.endswith("valid")]
DERIVATIVES = [
    i
    for i, n in enumerate(CHANNELS)
    if n.endswith(
        (
            ".vx",
            ".vy",
            ".velocity_valid",
            ".extension_rate",
            ".extension_rate_valid",
            ".wrist_foot_velocity_cos",
            ".coordination_valid",
        )
    )
]


def source_spec(recording):
    require(recording in ("V4", "V5", "V6"), "classifier_forbidden_recording")
    config = read("configs/level1_curated_subset_v1.json")
    ref = config["reviewed_config"]
    require(sha(Path(ref["path"])) == ref["sha256"], "classifier_review_config_changed")
    s = read(ref["path"])["sources"][recording]
    return dict(
        dataset=s["dataset"],
        policy=s["policy"]["path"],
        pose_run=f"data/processed/boxingvi/{recording}/pose-full-image-v1",
    )


def reflected_raw(raw):
    """Synthetic physical reflection on a copy; swap anatomical sides, preserve tactical labels."""
    result = {k: v.copy() for k, v in raw.items()}
    swap = np.arange(33)
    for a in range(11, 33, 2):
        swap[a], swap[a + 1] = a + 1, a
    for key in ("coordinates", "confidence", "valid_mask"):
        result[key] = result[key][:, swap].copy()
    result["coordinates"][:, :, 0] = 1 - result["coordinates"][:, :, 0]
    return result


def clip_body(clip, aspect, cuts):
    raw = clip["raw"]
    reset = np.isin(raw["frame_index"], cuts)
    x, normal, _ = encode(raw, aspect, reset)
    reflected, _, _ = encode(reflected_raw(raw), aspect, reset)
    return torch.from_numpy(x), torch.from_numpy(reflected), normal


def training_views():
    pin = read(PIN)
    for key in ("dataset", "eligibility"):
        r = pin[key]
        require(sha(Path(r["path"])) == r["sha256"], "classifier_snapshot_changed")
    manifest = read(pin["dataset"]["path"])
    overlay = read(pin["eligibility"]["path"])
    eligible = {r["id"] for r in overlay["rows"] if r["weak_training_eligible"]}
    require(
        len(eligible) == 812 and len(manifest["weak_clips"]) == 815, "classifier_812_scope_changed"
    )
    entries = {c["id"]: c for c in manifest["weak_clips"]}
    result = []
    bindings = {str(PIN): sha(PIN)}
    drift = []
    for v in ("V4", "V6"):
        spec = source_spec(v)
        dataset = VelocityClips(PoseClips(spec, v, multi_recording=True), multi_recording=True)
        meta = manifest["recordings"][v]
        for name, h in meta["provenance"]["hashes"].items():
            p = Path(spec["dataset"]) / name
            require(sha(p) == h, "classifier_preparation_changed")
            bindings[str(p)] = h
        f = Path(pin["dataset"]["path"]).parent / v / "fullbody.npz"
        require(sha(f) == meta["feature_sha256"], "classifier_fullbody_cache_changed")
        bindings[str(f)] = sha(f)
        cached = arrays(f)
        width, height = dataset.provenance["source_dimensions"]
        cuts = read(Path(spec["dataset"]) / "preparation.json").get("confirmed_scene_cuts", [])
        lookup = {int(i): j for j, i in enumerate(cached["frame_index"])}
        for c in dataset.clips:
            key = f"{v}:{c['source_row']}"
            if key not in eligible:
                continue
            e = entries[key]
            require(
                e["clip_class"] == CLASSES[c["label"]]
                and e["source_frames"] == c["annotation"]["array_slice_half_open"],
                "classifier_clip_membership_changed",
            )
            x, mirror, normal = clip_body(c, width / height, cuts)
            view = cached["features"][[lookup[int(i)] for i in c["raw"]["frame_index"]]].copy()
            view[0, DERIVATIVES] = 0
            require(np.array_equal(view, x.numpy()), "classifier_209_clip_view_differs_from_cache")
            core = c["x"]
            drift.append(float((core - x[:, MAP65]).abs().max()))
            result.append(
                dict(
                    **c,
                    body209=x,
                    body209_reflected=mirror,
                    core65_reflected=mirror_features(core),
                    body_normalization=normal,
                )
            )
    require(
        {f"{c['recording_id']}:{c['source_row']}" for c in result} == eligible,
        "classifier_missing_training_clip",
    )
    require(not any(c["recording_id"] == "V5" for c in result), "classifier_development_leakage")
    return result, dict(
        input_sha256=bindings,
        clips=len(result),
        common_core_max_absolute_drift=max(drift),
        clips_with_common_core_drift_gt_1e_4=sum(d > 1e-4 for d in drift),
        roles={"V4": "train", "V6": "train", "V5": "development"},
        per_clip_derivative_reset=True,
        feature_channels=CHANNELS,
        source_snapshot_sha256=pin["dataset"]["sha256"],
        overlay_sha256=pin["eligibility"]["sha256"],
    )


def development_views():
    """Caller must enforce frozen training before this function. No labels/statistics are fitted."""
    spec = source_spec("V5")
    ds = VelocityClips(PoseClips(spec, "V5"))
    require(len(ds.clips) == 583, "classifier_frozen_v5_membership_changed")
    width, height = ds.provenance["source_dimensions"]
    cuts = read(Path(spec["dataset"]) / "preparation.json").get("confirmed_scene_cuts", [])
    result = []
    for c in ds.clips:
        x, mirror, normal = clip_body(c, width / height, cuts)
        result.append(
            dict(
                **c,
                body209=x,
                body209_reflected=mirror,
                core65_reflected=mirror_features(c["x"]),
                body_normalization=normal,
            )
        )
    return result, ds.provenance


class View(Dataset):
    def __init__(self, clips, mode):
        require(mode in ("core65", "body209", "mask209"), "classifier_unknown_mode")
        self.clips = []
        for c in clips:
            if mode == "core65":
                x, ref = c["x"], c["core65_reflected"]
            else:
                x, ref = c["body209"], c["body209_reflected"]
                if mode == "mask209":
                    x, ref = x.clone(), ref.clone()
                    keep = torch.zeros(209, dtype=torch.bool)
                    keep[MASKS] = True
                    x[:, ~keep] = 0
                    ref[:, ~keep] = 0
            self.clips.append(c | dict(x=x, reflected_x=ref))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return self.clips[i]


def clip_identity(c):
    r = c["raw"]
    return dict(
        recording=c["recording_id"],
        row=c["source_row"],
        target=CLASSES[c["label"]],
        native_frames=[int(r["frame_index"][0]), int(r["frame_index"][-1]) + 1],
        first_pts_exact=str(
            Q(int(r["pts"][0]) * int(r["time_base_num"][0]), int(r["time_base_den"][0]))
        ),
        class_status="provisional",
        boundary_status="provisional_known_clip_not_detected",
    )
