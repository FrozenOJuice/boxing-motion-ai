"""Bounded frozen RGB cache: exact PTS, causal 10 Hz sampling, reviewed HUD redaction."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from ..vision.pipeline import sha256
from .augmentation import mirror_features
from .data import exact_times
from .experiment import write_json
from .multi_recording import ALL
from .rgb_review import decode_requested

POLICY = {
    "version": "rgb-feasibility-1.2",
    "encoder": "mobilenet_v3_small.IMAGENET1K_V1.features.avgpool",
    "channels": 576,
    "sample_interval_seconds": "1/10",
    "sampling": "first clip frame then >=0.1s; causal hold on native pose grid",
    "crop": "square 4.5 torso lengths centered on shoulder/hip midpoints; gray padding",
    "fallback": "full-source centered square when any torso anchor is unreliable",
    "hud_rectangles_normalized": {
        "V4": [[0, 0, 0.20, 0.35], [0.80, 0, 1, 0.35], [0, 0.86, 1, 1]],
        "V5": [[0.84, 0.70, 1, 1]],
        "V6": [[0.78, 0, 1, 0.25], [0, 0.80, 1, 1]],
        "V7": [[0, 0.85, 1, 1]],
        "V8": [],
        "V10": [[0, 0, 0.42, 0.20]],
    },
    "background_control": (
        "full source with boxer square enlarged 1.3x erased; mask may leak motion"
    ),
    "mirror": "encode physical flipped crop; same clip coin as anatomical pose reflection",
    "encoder_preprocessing": "weights.transforms: resize256 center224 ImageNet mean/std",
}
GRAY = (124, 116, 104)


def crop_box(coordinates, valid, size):
    """Only reliable torso anchors determine a crop; no fabricated missing joints."""
    w, h = size
    anchors = np.asarray(coordinates)[[11, 12, 23, 24], :2] * [w, h]
    good = np.asarray(valid)[[11, 12, 23, 24]].all() and np.isfinite(anchors).all()
    if good:
        shoulders, hips = anchors[:2].mean(0), anchors[2:].mean(0)
        length = float(np.linalg.norm(shoulders - hips))
        good = length >= 4
    if good:
        center = (shoulders + hips) / 2
        side = max(32, int(round(4.5 * length)))
    else:
        center, side = np.array([w / 2, h / 2]), max(w, h)
    left, top = np.round(center - side / 2).astype(int)
    return (int(left), int(top), int(left + side), int(top + side)), not good


def views(image, box, recording):
    """Never draw source row, timestamp, label or pose skeleton into encoder pixels."""
    if recording not in ALL:
        raise ValueError("Approved non-V9 recording required")
    redacted = image.convert("RGB").copy()
    d = ImageDraw.Draw(redacted)
    w, h = image.size
    for x1, y1, x2, y2 in POLICY["hud_rectangles_normalized"][recording]:
        d.rectangle((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)), fill=GRAY)
    a, b, c, e = box
    crop = Image.new("RGB", (c - a, e - b), GRAY)
    crop.paste(redacted, (-a, -b))
    background = redacted.copy()
    margin = int(round((c - a) * 0.15))
    ImageDraw.Draw(background).rectangle(
        (a - margin, b - margin, c + margin, e + margin), fill=GRAY
    )
    return crop, ImageOps.mirror(crop), background, ImageOps.mirror(background)


def sampling_plan(clip):
    times = exact_times(clip)
    if not times or any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("Strictly increasing source PTS required")
    selected, mapping = [], []
    for j, t in enumerate(times):
        boundary = j > 0 and any(
            clip["raw"][k][j] != clip["raw"][k][j - 1] for k in ("scene_id", "segment_id")
        )
        missing = j > 0 and clip["raw"]["frame_index"][j] != clip["raw"]["frame_index"][j - 1] + 1
        if not selected or boundary or missing or t - times[selected[-1]] >= Fraction(1, 10):
            selected.append(j)
        mapping.append(int(clip["raw"]["frame_index"][selected[-1]]))
    return selected, mapping


def encoder(device="cpu"):
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    if device not in ("cpu", "mps"):
        raise ValueError("Encoder device must be cpu or mps")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable; use CPU in the pilot config and a fresh cache")
    model = mobilenet_v3_small(weights=weights).eval().to(device)
    model.classifier = torch.nn.Identity()
    model.requires_grad_(False)
    path = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    return (
        model,
        weights.transforms(),
        dict(
            url=weights.url,
            sha256=sha256(path),
            device=device,
            torch_version=str(torch.__version__),
        ),
    )


def validate_cache(saved, inputs, feature_path):
    if saved["inputs"] != inputs or sha256(feature_path) != saved["feature_sha256"]:
        raise ValueError("Stale or changed RGB cache")


def cache_features(pose_cache, output, batch_size=64, device="cpu"):
    """Resume completed recording caches only when source, policy and encoder match."""
    output = Path(output)
    if not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("RGB artifacts must remain below ignored reports/")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model, transform, weight = encoder(device)
    summaries = {}
    for v in ALL:
        ds = pose_cache[v]["xy_velocity"]
        provenance = ds.provenance
        source = json.loads(
            Path(f"data/processed/boxingvi/{v}/pose-full-image-v1/manifest.json").read_text()
        )
        if source["inputs"]["video"]["filename"] != f"{v}.mkv":
            raise ValueError("Unexpected video filename; V9 remains reserved")
        video = Path("data/raw/boxingvi/RGB_videos") / source["inputs"]["video"]["filename"]
        video_hash = sha256(video)
        expected = json.loads((Path(provenance["dataset"]) / "preparation.json").read_text())
        if video_hash != expected["inputs"]["video_sha256"]:
            raise ValueError("Video hash differs from approved pose source")
        inputs = dict(
            policy=POLICY, encoder=weight, video_sha256=video_hash, pose_provenance=provenance
        )
        dest = output / v
        if (dest / "manifest.json").exists():
            saved = json.loads((dest / "manifest.json").read_text())
            validate_cache(saved, inputs, dest / "features.npz")
            summaries[v] = saved
            continue
        dest.mkdir(exist_ok=True)
        requests, raw = {}, {}
        for c in ds.clips:
            times = exact_times(c)
            selected, _ = sampling_plan(c)
            for j in selected:
                i = int(c["raw"]["frame_index"][j])
                requests[i] = times[j]
                raw[i] = (c["raw"]["coordinates"][j], c["raw"]["valid_mask"][j])
        indices, tensors, features, fallback, boxes = [], [], [], [], []
        preview = Image.new("RGB", (224 * 4, 224 * 3), GRAY)
        preview_count = 0
        with torch.inference_mode():
            for i, im, _ in decode_requested(video, requests):
                box, failed = crop_box(*raw[i], im.size)
                variants = views(im, box, v)
                if preview_count < 3:
                    for k, view in enumerate(variants):
                        preview.paste(view.resize((224, 224)), (k * 224, preview_count * 224))
                    preview_count += 1
                tensors.extend(transform(view) for view in variants)
                indices.append(i)
                if len(indices) % 500 == 0:
                    print(f"{v}: encoded/queued {len(indices)}/{len(requests)} frames", flush=True)
                fallback.append(failed)
                boxes.append(box)
                if len(tensors) >= batch_size:
                    features.append(model(torch.stack(tensors).to(device)).cpu().numpy())
                    tensors = []
            if tensors:
                features.append(model(torch.stack(tensors).to(device)).cpu().numpy())
        values = np.concatenate(features).reshape(len(indices), 4, 576)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite frozen features")
        np.savez_compressed(
            dest / "features.npz",
            frame_index=np.array(indices),
            features=values,
            boxes=np.array(boxes),
            fallback=np.array(fallback),
        )
        preview.save(dest / "preprocessing.jpg")
        saved = dict(
            inputs=inputs,
            frames=len(indices),
            fallback_frames=sum(fallback),
            exact_pts=[str(requests[i]) for i in indices],
            feature_sha256=sha256(dest / "features.npz"),
        )
        write_json(dest / "manifest.json", saved)
        summaries[v] = saved
        print(
            f"Cached {v}: {len(indices)} exact-PTS frames; {sum(fallback)} fallback crops",
            flush=True,
        )
    return summaries


class VisualClips:
    """Model receives pixels/pose only; identities and timestamps remain audit metadata."""

    def __init__(self, pose, cache, mode):
        if mode not in ("rgb", "hybrid", "background"):
            raise ValueError("Unknown RGB mode")
        if pose.provenance["recording_id"] not in ALL:
            raise ValueError("Approved non-V9 recording required")
        self.provenance, self.mode, self.clips = pose.provenance, mode, []
        with np.load(Path(cache) / "features.npz", allow_pickle=False) as z:
            lookup = {int(i): j for j, i in enumerate(z["frame_index"])}
            values = z["features"]
            for c in pose.clips:
                _, mapping = sampling_plan(c)
                # Static scene control: same first redacted/boxer-erased source image
                # for every clip in this recording. No moving erasure-box trajectory.
                positions = (
                    [0] * len(mapping) if mode == "background" else [lookup[i] for i in mapping]
                )
                features = torch.from_numpy(values[positions].copy())
                k = 2 if mode == "background" else 0
                x, reflected = features[:, k], features[:, k + 1]
                if mode == "hybrid":
                    x = torch.cat((c["x"], x), -1)
                    reflected = torch.cat((mirror_features(c["x"]), reflected), -1)
                self.clips.append({**c, "x": x, "reflected_x": reflected})

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return self.clips[i]
