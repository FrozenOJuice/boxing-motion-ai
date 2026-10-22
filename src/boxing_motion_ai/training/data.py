"""Read-only, hash-checked positive clips. Unknown footage never becomes background."""

import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.pose_quality import ANCHORS, ARMS, CORE, POLICIES, retain, score_validity
from ..vision.contracts import JOINTS
from ..vision.pipeline import sha256

CLASSES = ("jab", "cross", "lead_hook", "rear_hook", "lead_uppercut", "rear_uppercut")
FEATURE_VERSION = "torso-core-2d-1.0"
POLICY_VERSION = "recording-mask-policy-index-1.1"


def features(coordinates, confidence, valid, aspect):
    """Per-frame hip-centered XY / torso length, no temporal filling or future statistics.

    z is deliberately unused (estimator-relative depth). Invalid zeros are sentinels,
    disambiguated by masks, never estimates of missing landmark positions.
    """
    core_valid = valid[:, CORE]
    xy = np.where(core_valid[..., None], coordinates[:, CORE, :2], 0).copy()
    xy[..., 0] *= aspect
    anchors = valid[:, ANCHORS].all(axis=1)
    hip = (xy[:, 6] + xy[:, 7]) / 2
    shoulder = (xy[:, 0] + xy[:, 1]) / 2
    scale = np.linalg.norm(shoulder - hip, axis=1)
    normalization_valid = anchors & np.isfinite(scale) & (scale > 1e-6)
    usable = core_valid & normalization_valid[:, None]
    normalized = (xy - hip[:, None]) / np.where(normalization_valid, scale, 1)[:, None, None]
    normalized = np.where(usable[..., None], normalized, 0)
    conf = np.where(usable[..., None], confidence[:, CORE], 0)
    main = np.concatenate((normalized, conf, usable[..., None]), axis=-1).reshape(len(xy), -1)
    main = np.concatenate((main, normalization_valid[:, None]), axis=1).astype(np.float32)
    masks = core_valid.astype(np.float32)  # Only original masks; no coordinates/confidence.
    if not np.isfinite(main).all():
        raise ValueError("Nonfinite gated features")
    return main, masks, normalization_valid


def validate_arrays(a):
    n = len(a["frame_index"])
    shapes = {
        "frame_index": (n,),
        "pts": (n,),
        "time_base_num": (n,),
        "time_base_den": (n,),
        "coordinates": (n, 33, 3),
        "confidence": (n, 33, 2),
        "valid_mask": (n, 33),
        "scene_id": (n,),
        "segment_id": (n,),
        "pose_status": (n,),
    }
    if not n or any(a[k].shape != s for k, s in shapes.items()):
        raise ValueError("Unexpected pose array shapes")
    if a["valid_mask"].dtype != bool or not np.array_equal(
        a["valid_mask"], score_validity(a["coordinates"], a["confidence"], 0.5)
    ):
        raise ValueError("Source validity masks violate the fixed 0.5 contract")
    integer_keys = (
        "frame_index",
        "pts",
        "time_base_num",
        "time_base_den",
        "scene_id",
        "segment_id",
        "pose_status",
    )
    if any(a[k].dtype.kind not in "iu" for k in integer_keys):
        raise ValueError("Indices and exact timing must be integers")
    if not np.array_equal(a["frame_index"], np.arange(n)):
        raise ValueError("Expected complete, contiguous decoded frame inventory")
    if (a["time_base_num"] <= 0).any() or (a["time_base_den"] <= 0).any():
        raise ValueError("Invalid time base")
    times = a["pts"] * (a["time_base_num"] / a["time_base_den"])
    if not (np.diff(times) > 0).all() or not np.isin(a["pose_status"], [0, 1, 2]).all():
        raise ValueError("Nonmonotonic timing or invalid status")


class PoseClips(Dataset):
    def __init__(
        self, spec, expected_recording, limit_per_class=None, *, expansion=False, diagnostic=False
    ):
        if expansion and diagnostic:
            raise ValueError("Diagnostic evaluation and training expansion are separate gates")
        allowed = ("V4", "V5", "V7", "V8", "V10") if expansion else ("V4", "V5")
        if diagnostic:
            allowed = ("V6", "V7", "V8", "V10")
        if expected_recording not in allowed:
            raise ValueError(
                "Only V4/V5 by default; explicit expansion adds V7/V8 or V10; V9 reserved"
            )
        root, index_path, pose_run = map(Path, (spec["dataset"], spec["policy"], spec["pose_run"]))
        manifest = json.loads((root / "preparation.json").read_text())
        index = json.loads(index_path.read_text())
        if manifest["video_id"] != expected_recording or index["video_id"] != expected_recording:
            raise ValueError("Recording split mismatch")
        schema = manifest["schema_version"]
        if not (
            schema == "recording-preparation-1.1"
            and manifest["scope"] == "full_video"
            or schema == "v5-preparation-1.0"
            and expected_recording == "V5"
        ):
            raise ValueError("Unsupported preparation version/scope")
        contract = manifest["pose_contract"]
        if contract["joint_order"] != list(JOINTS) or contract["confidence_order"] != [
            "visibility",
            "presence",
        ]:
            raise ValueError("Unsupported pose joint/confidence contract")
        expected_policy = {
            "schema_version": POLICY_VERSION,
            "policy": "mask_aware70",
            "definition": POLICIES["mask_aware70"],
            "max_missing_frame_fraction": 0.1,
            "source_mask_threshold": 0.5,
            "joint_groups": {
                "core": list(CORE),
                "anchors": list(ANCHORS),
                "arms": [list(arm) for arm in ARMS],
            },
        }
        if any(index.get(k) != v for k, v in expected_policy.items()):
            raise ValueError("Unapproved quality policy")
        hashes = {
            name: sha256(root / name)
            for name in ("preparation.json", "poses.npz", "candidates.jsonl")
        }
        if hashes != index["inputs"] or hashes["poses.npz"] != manifest["npz_sha256"]:
            raise ValueError("Stale or changed dataset / policy hashes")
        pose_hashes = {name: sha256(pose_run / name) for name in ("manifest.json", "poses.jsonl")}
        if (
            pose_hashes["manifest.json"] != manifest["inputs"]["pose_manifest_sha256"]
            or pose_hashes["poses.jsonl"] != manifest["inputs"]["poses_sha256"]
        ):
            raise ValueError("Source pose provenance mismatch")
        with (pose_run / "poses.jsonl").open() as f:
            first = json.loads(next(f))
        width, height = first["source_width"], first["source_height"]
        if width <= 0 or height <= 0 or first["video_id"] != expected_recording:
            raise ValueError("Invalid source dimensions/identity")
        with np.load(root / "poses.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        validate_arrays(arrays)
        if len(arrays["frame_index"]) != manifest["source_frame_count"]:
            raise ValueError("Incomplete frame inventory")
        candidates = [
            json.loads(line) for line in (root / "candidates.jsonl").read_text().splitlines()
        ]
        if len({c["source_row"] for c in candidates}) != len(candidates):
            raise ValueError("Duplicate source rows")
        selected, clips = [], []
        for c in candidates:
            if not c["annotation_eligible"] or not c.get("quality_assessable", True):
                continue
            if c["exclusion_reasons"]:
                raise ValueError("Eligible row has unresolved exclusion")
            start, stop = c["array_slice_half_open"]
            if not 0 <= start < stop <= len(arrays["frame_index"]):
                raise ValueError("Invalid candidate bounds")
            a = {k: v[start:stop].copy() for k, v in arrays.items()}
            if not retain(a["valid_mask"], a["pose_status"] == 2, POLICIES["mask_aware70"]):
                continue
            if any(len(np.unique(a[k])) != 1 for k in ("scene_id", "segment_id")):
                raise ValueError("Candidate crosses a scene/segment boundary")
            if c["punch_type_provisional"] not in CLASSES:
                raise ValueError("Unrecognized eligible class")
            selected.append(c["source_row"])
            x, masks, normal = features(
                a["coordinates"], a["confidence"], a["valid_mask"], width / height
            )
            clips.append(
                {
                    "x": torch.from_numpy(x),
                    "mask_x": torch.from_numpy(masks),
                    "label": CLASSES.index(c["punch_type_provisional"]),
                    "source_row": c["source_row"],
                    "recording_id": expected_recording,
                    "boxer_id": manifest.get("boxer_id"),
                    "session_id": manifest.get("session_id"),
                    "annotation": c,
                    "raw": a,
                    "normalization_valid": normal,
                    "timestamp_seconds": a["pts"] * a["time_base_num"] / a["time_base_den"],
                }
            )
        if sorted(selected) != sorted(index["selected_source_rows"]):
            raise ValueError("Recomputed selection differs from versioned policy index")
        if limit_per_class:
            subset = []
            # Smoke subset only: deterministic evenly spaced rows in every class.
            for label in range(6):
                group = [c for c in clips if c["label"] == label]
                for i in np.linspace(
                    0, len(group) - 1, min(limit_per_class, len(group)), dtype=int
                ):
                    subset.append(group[i])
            clips = subset
        self.clips = clips
        counts = Counter(c["label"] for c in clips)
        if not clips or (
            not expansion and not diagnostic and any(counts[i] == 0 for i in range(6))
        ):
            raise ValueError("Every split must retain all six classes")
        self.provenance = {
            "recording_id": expected_recording,
            "boxer_id": manifest.get("boxer_id"),
            "session_id": manifest.get("session_id"),
            "preparation_version": schema,
            "dataset": str(root),
            "hashes": hashes,
            "policy_path": str(index_path),
            "policy_sha256": sha256(index_path),
            "policy": expected_policy,
            "pose_source_hashes": pose_hashes,
            "source_dimensions": [width, height],
            "mapping_uncertainty": manifest["mapping_uncertainty"],
            "source_input_hashes": manifest["inputs"],
            "excluded_candidates": [
                {
                    "source_row": c["source_row"],
                    "class": c.get("punch_type_provisional"),
                    "reasons": c.get("exclusion_reasons") or ["mask_aware70_rejection"],
                }
                for c in candidates
                if c["source_row"] not in selected
            ],
            "scene_inventory": manifest.get("scene_inventory"),
            "class_counts": {name: counts[i] for i, name in enumerate(CLASSES)},
            "source_rows": [c["source_row"] for c in clips],
            "normalization_invalid_frames": int(
                sum((~c["normalization_valid"]).sum() for c in clips)
            ),
            "total_clip_frames": sum(len(c["x"]) for c in clips),
            "clip_length_frames": [
                min(len(c["x"]) for c in clips),
                max(len(c["x"]) for c in clips),
            ],
        }
        if diagnostic:
            self.provenance["evaluation_scope"] = "diagnostic only; no training admission"

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return self.clips[i]


def collate(clips):
    lengths = torch.tensor([len(c["x"]) for c in clips])
    time_mask = torch.arange(int(lengths.max()))[None] < lengths[:, None]
    result = {
        "time_mask": time_mask,
        "lengths": lengths,
        "labels": torch.tensor([c["label"] for c in clips]),
        "source_rows": [c["source_row"] for c in clips],
        "clips": clips,
    }
    for key in ("x", "mask_x"):
        result[key] = torch.nn.utils.rnn.pad_sequence([c[key] for c in clips], batch_first=True)
    return result


def exact_times(clip):
    """Optional exact rational seconds; models never receive absolute source timing."""
    a = clip["raw"]
    return [
        Fraction(int(p) * int(n), int(d))
        for p, n, d in zip(a["pts"], a["time_base_num"], a["time_base_den"], strict=True)
    ]
