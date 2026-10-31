"""Frozen video feature engineering only; no training, event proposals or production imports."""

import argparse
import shutil
from fractions import Fraction as Q
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..analysis.common import sha, write_json
from ..data.video_foundation import CONFIG, TRAIN, admitted
from ..detection.localization_data import POLICY as MOTION_POLICY
from ..detection.localization_decoder_study import BASE, CACHE, cases
from ..training.rgb import GRAY
from ..training.rgb import POLICY as RGB_POLICY
from ..training.rgb_review import decode_requested
from .full_body import CHANNELS, JOINTS, VERSION, encode
from .observation_study import arrays, fresh, read

PIXEL_VERSION = "source-fixed-identifiers-fullbody-1.1"


def shirt_identifiers(raw, size):
    """One static chest-text union per context; never a per-frame moving artificial edge."""
    anchors = raw["coordinates"][:, [11, 12, 23, 24], :2]
    good = (
        raw["valid_mask"][:, [11, 12, 23, 24]].all(1)
        & np.isfinite(anchors).all((1, 2))
        & (raw["pose_status"] != 2)
    )
    if not good.any():
        return [(0, 0, *size)]  # Unknown person framing: no unredacted source identity fallback.
    points = anchors[good] * size
    s, h = points[:, :2].mean(1), points[:, 2:].mean(1)
    length = np.linalg.norm(h - s, axis=1)
    center = s + 0.45 * (h - s)
    lo = center - length[:, None] * [0.45, 0.25]
    hi = center + length[:, None] * [0.45, 0.25]
    return [
        (
            int(lo[:, 0].min()),
            int(lo[:, 1].min()),
            int(np.ceil(hi[:, 0].max())),
            int(np.ceil(hi[:, 1].max())),
        )
    ]


def windows(count, reset, width=16):
    """One native center per real frame. Missing edge context stays unavailable, never padded."""
    if count != len(reset) or width != 16:
        raise ValueError("Fixed native 16-frame contract required")
    return [(i, i - 7, i + 9) for i in range(7, count - 8) if not np.any(reset[i - 6 : i + 9])]


def full_body_crop(raw, size):
    w, h = size
    xy = raw["coordinates"][:, JOINTS, :2]
    valid = (
        raw["valid_mask"][:, JOINTS]
        & np.isfinite(xy).all(2)
        & ((xy >= 0) & (xy <= 1)).all(2)
        & (raw["pose_status"][:, None] != 2)
    )
    points = xy[valid] * [w, h]
    if not len(points):
        return (0, 0, w, h), True
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    side = max(32, float((hi - lo).max()) * 1.2)
    left, top = np.floor(center - side / 2).astype(int)
    return (int(left), int(top), int(np.ceil(left + side)), int(np.ceil(top + side))), False


def pixels(image, box, recording, identifiers=()):
    if not admitted(recording, "existing_localization"):
        raise ValueError("Only permitted training pixels may enter this experimental cache")
    rgb = image.convert("RGB").copy()
    w, h = rgb.size
    d = ImageDraw.Draw(rgb)
    # Retain torso/legs: only existing source-fixed identifiers/HUD are removed.
    rectangles = (
        RGB_POLICY["hud_rectangles_normalized"][recording]
        + MOTION_POLICY["source_identifier_rectangles"][recording]
    )
    for a, b, c, e in rectangles:
        d.rectangle(
            (
                int(max(0, a - 0.01) * w),
                int(max(0, b - 0.01) * h),
                int(min(1, c + 0.01) * w),
                int(min(1, e + 0.01) * h),
            ),
            fill=GRAY,
        )
    for rectangle in identifiers:
        d.rectangle(rectangle, fill=GRAY)
    left, top, right, bottom = box
    crop = Image.new("RGB", (right - left, bottom - top), GRAY)
    crop.paste(rgb, (-left, -top))
    # Fit ALL crop pixels into the model's central 112x112 view. Official transforms then
    # center-crop the surrounding letterbox; they cannot amputate wrists or feet to fill it.
    crop.thumbnail((112, 112), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (171, 128), GRAY)
    canvas.paste(crop, (int(round((171 - crop.width) / 2)), int(round((128 - crop.height) / 2))))
    return canvas


def encoder(config, device):
    from torchvision.models.video import R3D_18_Weights, r3d_18

    if device not in ("cpu", "mps") or device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested local encoder device unavailable")
    p = Path(config["path"])
    if not p.is_file() or sha(p) != config["sha256"]:
        raise ValueError("Pinned official video weights missing/changed; no implicit download")
    model = r3d_18(weights=None)
    model.load_state_dict(torch.load(p, weights_only=True, map_location="cpu"))
    model.fc = torch.nn.Identity()
    model.eval().requires_grad_(False)
    return model.to(device), R3D_18_Weights.KINETICS400_V1.transforms()


def guard_space(config):
    if shutil.disk_usage(".").free < config["minimum_free_bytes"] + config["max_new_feature_bytes"]:
        raise ValueError("Insufficient disk headroom; preserve caches and use another volume")


def prepare(output, device):
    cfg = read(CONFIG)
    guard_space(cfg)
    data = cases("training")  # No V5/reserved feature tensors.
    selected = cfg["feature_cases"]
    if selected != ["V4-bg-1", "V6-motion-101", "V6-motion-105"]:
        raise ValueError("Fixed engineering feature checks only; no outcome-selected clips")
    out = fresh(output)
    code = [Path(__file__), Path("src/boxing_motion_ai/vision/full_body.py"), CONFIG]
    write_json(
        out / "protocol.json",
        dict(
            config=cfg,
            code_sha256={str(p): sha(p) for p in code},
            cases=selected,
            device=device,
            pixel_version=PIXEL_VERSION,
            training=False,
        ),
    )
    torch.set_num_threads(2)
    model, transform = encoder(cfg["pretrained_encoder"], device)
    before = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
    results = {}
    for name in selected:
        began = perf_counter()
        case = data[name]
        v = case["provenance"]["recording"]
        if v not in TRAIN:
            raise ValueError("Forbidden feature source")
        rawpath = (
            CACHE / name / "poses.npz"
            if (CACHE / name / "poses.npz").exists()
            else BASE / name / "poses.npz"
        )
        raw = arrays(rawpath)
        video = case["video"]
        offset = case["provenance"]["source_start"]
        source = Path(f"data/raw/boxingvi/RGB_videos/{v}.mkv")
        expected = case["receipt"].get("source_video_sha256") or case["provenance"].get(
            "source_video_sha256"
        )
        if sha(source) != expected:
            raise ValueError("Video feature source changed")
        body, observable, reset = encode(
            raw, video["width"] / video["height"], case["arrays"]["reset"]
        )
        plan = windows(len(body), reset)
        box, fallback = full_body_crop(raw, (video["width"], video["height"]))
        identifiers = shirt_identifiers(raw, (video["width"], video["height"]))
        if not raw["valid_mask"][:, [11, 12, 23, 24]].all(1).any():
            plan = []  # Fully identity-redacted fallback has no usable visual evidence.
        request = {offset + i: Q(f["timestamp_exact"]) for i, f in enumerate(video["frames"])}
        images = [pixels(im, box, v, identifiers) for _, im, _ in decode_requested(source, request)]
        if len(images) != len(body):
            raise ValueError("Incomplete native pixels")
        feature = np.zeros((len(body), 512), np.float32)
        valid = np.zeros(len(body), bool)
        source_windows = np.full((len(body), 2), -1, np.int64)
        with torch.inference_mode():
            for j in range(0, len(plan), cfg["batch_size"]):
                group = plan[j : j + cfg["batch_size"]]
                tensors = []
                for _center, a, z in group:
                    clip = torch.from_numpy(np.stack([np.asarray(im) for im in images[a:z]]))
                    tensors.append(transform(clip.permute(0, 3, 1, 2)))
                x = torch.stack(tensors).to(device)
                y = model(x).cpu().numpy()
                repeat = model(x).cpu().numpy()
                if not np.allclose(y, repeat, atol=1e-6, rtol=1e-6):
                    raise ValueError("Frozen encoder repeat drift")
                for (center, a, z), row in zip(group, y, strict=True):
                    feature[center] = row
                    valid[center] = True
                    source_windows[center] = [offset + a, offset + z]
        if not np.isfinite(feature).all():
            raise ValueError("Invalid pretrained features")
        dest = out / name
        dest.mkdir()
        np.savez_compressed(
            dest / "features.npz",
            fullbody=body,
            video=feature,
            video_valid=valid,
            normalization_valid=observable,
            reset=reset,
            source_frame_index=np.arange(offset, offset + len(body)),
            video_source_windows=source_windows,
            **{k: raw[k] for k in ("pts", "time_base_num", "time_base_den")},
        )
        preview = Image.new("RGB", (171 * 3, 128), GRAY)
        for k, i in enumerate((0, len(images) // 2, len(images) - 1)):
            preview.paste(images[i], (k * 171, 0))
        preview.save(dest / "preview.png")
        results[name] = dict(
            recording=v,
            source_sha256=expected,
            poses_sha256=sha(rawpath),
            feature_sha256=sha(dest / "features.npz"),
            crop=box,
            crop_fallback=fallback,
            static_identifier_rectangles=identifiers,
            source_stop_timestamp_exact=video["stop_timestamp_exact"],
            frames=len(body),
            encoded_centers=int(valid.sum()),
            unavailable_context_frames=int((~valid).sum()),
            body_normalization_frames=int(observable.sum()),
            seconds=perf_counter() - began,
            references=case["references"],
            inputs=case["provenance"],
        )
        print(
            name, results[name]["encoded_centers"], round(results[name]["seconds"], 2), flush=True
        )
        guard_space(cfg)
    if not all(torch.equal(t.detach().cpu(), before[n]) for n, t in model.state_dict().items()):
        raise ValueError("Frozen encoder state changed")
    if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > cfg["max_new_feature_bytes"]:
        raise ValueError("Feature artifact budget exceeded")
    write_json(
        out / "manifest.json",
        dict(
            version="native-fullbody-r3d-feature-check-1.0",
            fullbody_version=VERSION,
            pixel_version=PIXEL_VERSION,
            channels=CHANNELS,
            cases=results,
            weights=cfg["pretrained_encoder"],
            device=device,
            frozen_state_unchanged=True,
            trained_models=False,
            training_ready=False,
            unavailable_new_source_or_boundary_evidence=True,
            torchvision=version("torchvision"),
            torch=str(torch.__version__),
        ),
    )
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    a = p.parse_args()
    prepare(a.output, a.device)
