"""Bounded source-pixel overlay audit, not another encoder or production preprocessing path."""

import argparse
from fractions import Fraction as Q
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..analysis.common import sha
from ..detection.localization_decoder_study import BASE, CACHE, cases
from ..services.dashboard import require
from ..training.data import CLASSES
from ..training.experiment import write_json
from ..training.fullbody_classification_data import training_views
from ..training.rgb import POLICY
from ..training.rgb_review import decode_requested
from .observation_study import arrays, fresh, read

CONFIG = Path("configs/rgb_overlay_integrity_v1.json")
LIMBS = {"left_arm": (13, 15), "right_arm": (14, 16), "feet": (27, 28, 29, 30, 31, 32)}


def discs(raw, i, size, joints):
    result = Image.new("L", size, 0)
    d = ImageDraw.Draw(result)
    r = max(3, min(size) * 0.012)
    for j in joints:
        xy = raw["coordinates"][i, j, :2]
        if raw["valid_mask"][i, j] and np.isfinite(xy).all() and ((xy >= 0) & (xy <= 1)).all():
            x, y = xy * np.array(size)
            d.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return np.asarray(result) > 0


def masks(image, probability, raw, i, v):
    require(v in ("V4", "V6"), "rgb_overlay_forbidden_source")
    require(probability.shape == (image.height, image.width), "rgb_overlay_mask_shape")
    person = (
        np.asarray(
            Image.fromarray((probability >= 0.5).astype("uint8") * 255).filter(
                ImageFilter.MaxFilter(11)
            )
        )
        > 0
    )
    rgb = np.asarray(image.convert("RGB"))
    w, h = image.size
    region = Image.new("L", image.size, 0)
    d = ImageDraw.Draw(region)
    footer = int(0.8 * h)
    if v == "V6":
        d.rectangle((0, footer, w, h), fill=255)
    else:
        anchors = raw["coordinates"][i, [11, 12, 23, 24], :2]
        if raw["valid_mask"][i, [11, 12, 23, 24]].all() and np.isfinite(anchors).all():
            p = anchors * np.array(image.size)
            length = np.linalg.norm(p[:2].mean(0) - p[2:].mean(0))
            d.rectangle(
                (
                    max(0, int(p[:, 0].min() - 0.1 * length)),
                    max(0, int(p[:2, 1].min())),
                    min(w, int(p[:, 0].max() + 0.1 * length)),
                    min(h, int(p[2:, 1].max() + 0.5 * length)),
                ),
                fill=255,
            )
    neutral = (rgb.min(2) >= 180) & ((rgb.max(2).astype(int) - rgb.min(2)) <= 55)
    ink = Image.fromarray((neutral & (np.asarray(region) > 0)).astype("uint8") * 255).filter(
        ImageFilter.MaxFilter(3)
    )
    targeted = person & ~(np.asarray(ink) > 0)
    full = person.copy()
    if v == "V6":
        full[footer:] = False
    else:
        full &= ~(np.asarray(region) > 0)
    if v == "V4":
        hud = Image.new("L", image.size, 0)
        d = ImageDraw.Draw(hud)
        for a, b, c, e in POLICY["hud_rectangles_normalized"][v]:
            d.rectangle((int(a * w), int(b * h), int(c * w), int(e * h)), fill=255)
        targeted &= np.asarray(hud) == 0
        full &= np.asarray(hud) == 0
    return dict(matte_only=person, targeted_bright_ink=targeted, whole_footer_removed=full)


def process(image, keep):
    rgb = np.asarray(image.convert("RGB")).copy()
    rgb[~keep] = 128
    return Image.fromarray(rgb)


def clock(raw, i):
    return Q(int(raw["pts"][i]) * int(raw["time_base_num"][i]), int(raw["time_base_den"][i]))


def run(output):
    import av
    import mediapipe as mp

    cfg = read(CONFIG)
    out = fresh(output)
    clips, audit = training_views()
    snapshot = read(read("configs/video_foundation_snapshot_v1.json")["dataset"]["path"])
    data = cases("training")
    jobs = []
    inventory = {}
    for name in cfg["diagnostic_cases"]:
        case = data[name]
        p = CACHE / name / "poses.npz"
        raw = arrays(p if p.exists() else BASE / name / "poses.npz")
        indices = sorted(
            {j for a in np.linspace(0, len(raw["pts"]) - 8, 4, dtype=int) for j in range(a, a + 8)}
        )
        jobs.append(
            (
                name,
                case["provenance"]["recording"],
                raw,
                indices,
                case["provenance"]["source_start"],
            )
        )
    for v in cfg["recordings"]:
        info = read(f"data/raw/boxingvi/RGB_videos/{v}.info.json")
        source = Path(f"data/raw/boxingvi/RGB_videos/{v}.mkv")
        with av.open(str(source)) as container:
            streams = [
                dict(type=s.type, index=s.index, codec=s.codec_context.name)
                for s in container.streams
            ]
        inventory[v] = dict(
            source_sha256=sha(source),
            original_url=info.get("webpage_url"),
            streams=streams,
            format=info.get("format_id"),
            local_same_prefix_assets=[str(p) for p in source.parent.glob(v + ".*")],
            clean_master_available_locally=False,
        )
        require(
            inventory[v]["source_sha256"] == snapshot["recordings"][v]["video_sha256"],
            "rgb_overlay_source_changed",
        )
        for label in range(6):
            selected = sorted(
                [c for c in clips if c["recording_id"] == v and c["label"] == label],
                key=lambda c: c["source_row"],
            )
            if not selected:
                continue
            for j in sorted({0, len(selected) // 2, len(selected) - 1}):
                c = selected[j]
                raw = c["raw"]
                i = len(raw["pts"]) // 2
                jobs.append((f"{v}-{CLASSES[label]}-row{c['source_row']}", v, raw, [i], 0))
    write_json(
        out / "protocol.json",
        dict(
            config=cfg,
            config_sha256=sha(CONFIG),
            code_sha256=sha(Path(__file__)),
            input_sha256=audit["input_sha256"],
            model_sha256=sha(Path(cfg["model"])),
            source_inventory=inventory,
        ),
    )
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=cfg["model"]),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        output_segmentation_masks=True,
    )
    summary = {}
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as model:
        for name, v, raw, indices, offset in jobs:
            request = {int(raw["frame_index"][i]) + offset: clock(raw, i) for i in indices}
            lookup = {int(raw["frame_index"][i]) + offset: i for i in indices}
            previous = None
            records = []
            start = perf_counter()
            for k, (source_frame, image, pts) in enumerate(
                decode_requested(Path(f"data/raw/boxingvi/RGB_videos/{v}.mkv"), request)
            ):
                i = lookup[source_frame]
                rgb = np.asarray(image.convert("RGB"))
                result = model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                probability = (
                    result.segmentation_masks[0].numpy_view().copy().squeeze(2)
                    if result.segmentation_masks
                    else np.zeros(rgb.shape[:2], np.float32)
                )
                keep = masks(image, probability, raw, i, v)
                geometry = {g: discs(raw, i, image.size, j) for g, j in LIMBS.items()}
                record = dict(
                    source_frame=source_frame,
                    pts_exact=str(pts),
                    segmentation_available=bool(result.segmentation_masks),
                    views={},
                )
                for mode, mask in keep.items():
                    stats = {}
                    for g, disc in geometry.items():
                        stats[g] = dict(
                            disc_pixels=int(disc.sum()), retained_pixels=int((disc & mask).sum())
                        )
                        if previous is not None and source_frame == previous[0] + 1:
                            delta = np.abs(rgb.astype(float) - previous[1]).mean(2)
                            stats[g].update(
                                motion_sum=float(delta[disc].sum()),
                                retained_motion_sum=float(
                                    delta[disc & mask & previous[2][mode]].sum()
                                ),
                            )
                    record["views"][mode] = stats
                records.append(record)
                previous = (source_frame, rgb, keep)
                if len(indices) == 1 or k in (0, 7, 15, 23, 31):
                    # Original plus all alternatives, fitted into equal-size previews without crop.
                    sheet = Image.new("RGB", (960, 540), (20, 20, 20))
                    for q, pic in enumerate(
                        [image, *[process(image, keep[m]) for m in cfg["comparison"]]]
                    ):
                        pic = pic.copy()
                        pic.thumbnail((480, 270))
                        sheet.paste(pic, ((q % 2) * 480, (q // 2) * 270))
                    sheet.save(out / f"{name}-frame{source_frame}-comparison.jpg")
                    if name in cfg["diagnostic_cases"] and k == 15:
                        image.save(out / f"{name}-frame{source_frame}-original.png")
                        process(image, keep["targeted_bright_ink"]).save(
                            out / f"{name}-frame{source_frame}-targeted.png"
                        )
            summary[name] = dict(
                recording=v,
                records=records,
                elapsed_seconds=perf_counter() - start,
                source_sha256=inventory[v]["source_sha256"],
            )
            write_json(out / "measurements.json", summary)
            print(name, len(records), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
