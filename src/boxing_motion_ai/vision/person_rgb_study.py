"""Experimental person matte and targeted OCR gate; never a production feature path."""

import argparse
import csv
import io
import subprocess
import tempfile
from fractions import Fraction as Q
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..analysis.common import sha, write_json
from ..detection.localization_decoder_study import BASE, CACHE, cases
from ..services.dashboard import require
from ..training.rgb_review import decode_requested
from .observation_study import arrays, fresh, read
from .video_foundation import full_body_crop, pixels, shirt_identifiers

CONFIG = Path("configs/person_preserving_rgb_v1.json")


def text_boxes(image):
    """Local words only; OCR failure is not proof that identifiers are absent."""
    with tempfile.TemporaryDirectory() as temp:
        p = Path(temp) / "frame.png"
        image.save(p)
        result = subprocess.run(
            ["/opt/homebrew/bin/tesseract", str(p), "stdout", "--psm", "11", "tsv"],
            check=True,
            capture_output=True,
            text=True,
        )
    rows = csv.DictReader(io.StringIO(result.stdout), delimiter="\t")
    return [
        (
            int(r["left"]),
            int(r["top"]),
            int(r["left"]) + int(r["width"]),
            int(r["top"]) + int(r["height"]),
            r["text"],
        )
        for r in rows
        if r.get("text", "").strip()
    ]


def preserve(image, probability, boxes, threshold=0.5, dilation=5):
    require(probability.shape == (image.height, image.width), "rgb_matte_shape")
    mask = Image.fromarray((probability >= threshold).astype("uint8") * 255)
    mask = mask.filter(ImageFilter.MaxFilter(2 * dilation + 1))
    text = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(text)
    for a, b, c, d, *_ in boxes:
        draw.rectangle((a - 2, b - 2, c + 2, d + 2), fill=255)
    retained = np.asarray(mask) > 0
    retained &= np.asarray(text) == 0
    result = np.asarray(image.convert("RGB")).copy()
    result[~retained] = 128
    return Image.fromarray(result), retained


def joint_discs(raw, i, size):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for j in (13, 14, 15, 16, 27, 28, 29, 30, 31, 32):
        xy = raw["coordinates"][i, j, :2]
        if raw["valid_mask"][i, j] and np.isfinite(xy).all() and ((xy >= 0) & (xy <= 1)).all():
            x, y = xy * np.array(size)
            r = max(3, min(size) * 0.012)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return np.asarray(mask) > 0


def run(output):
    import mediapipe as mp

    cfg = read(CONFIG)
    out = fresh(output)
    write_json(
        out / "protocol.json",
        dict(
            config=cfg,
            config_sha256=sha(CONFIG),
            code_sha256=sha(Path(__file__)),
            model_sha256=sha(Path(cfg["segmentation_model"])),
        ),
    )
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=cfg["segmentation_model"]),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        output_segmentation_masks=True,
    )
    data = cases("training")
    summary = {}
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as model:
        for name in cfg["cases"]:
            case = data[name]
            v = case["provenance"]["recording"]
            require(v in ("V4", "V6"), "rgb_role")
            rawpath = CACHE / name / "poses.npz"
            raw = arrays(rawpath if rawpath.exists() else BASE / name / "poses.npz")
            n = len(raw["pts"])
            starts = np.linspace(0, n - 8, 4, dtype=int)
            indices = sorted({i for a in starts for i in range(a, a + 8)})
            offset = case["provenance"]["source_start"]
            request = {
                offset + i: Q(
                    int(raw["pts"][i]) * int(raw["time_base_num"][i]), int(raw["time_base_den"][i])
                )
                for i in indices
            }
            source = Path(f"data/raw/boxingvi/RGB_videos/{v}.mkv")
            expected = case["receipt"].get("source_video_sha256") or case["provenance"].get(
                "source_video_sha256"
            )
            require(sha(source) == expected, "rgb_source_changed")
            frames = list(decode_requested(source, request))
            records = []
            previous = None
            crop, _ = full_body_crop(raw, frames[0][1].size)
            shirt = shirt_identifiers(raw, frames[0][1].size)
            sheet = Image.new("RGB", (4 * 320, 4 * 180 * 3), (20, 20, 20))
            start = perf_counter()
            for k, (global_i, image, pts) in enumerate(frames):
                i = global_i - offset
                rgb = np.asarray(image.convert("RGB"))
                t = perf_counter()
                result = model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                segmentation_seconds = perf_counter() - t
                probability = (
                    result.segmentation_masks[0].numpy_view().copy().squeeze(axis=2)
                    if result.segmentation_masks
                    else np.zeros(rgb.shape[:2], np.float32)
                )
                t = perf_counter()
                boxes = text_boxes(image)
                ocr_seconds = perf_counter() - t
                processed, retained = preserve(image, probability, boxes)
                discs = joint_discs(raw, i, image.size)
                record = dict(
                    source_frame=global_i,
                    pts_exact=str(pts),
                    observed_disc_pixels=int(discs.sum()),
                    retained_disc_pixels=int((discs & retained).sum()),
                    ocr_words=[b[4] for b in boxes],
                    segmentation_available=bool(result.segmentation_masks),
                    segmentation_seconds=segmentation_seconds,
                    ocr_seconds=ocr_seconds,
                )
                if previous is not None and i == previous[0] + 1:
                    delta = np.abs(rgb.astype(float) - previous[1]).mean(2)
                    record["motion_disc_sum"] = float(delta[discs].sum())
                    common = discs & retained & previous[2]
                    record["retained_motion_disc_sum"] = float(delta[common].sum())
                previous = (i, rgb, retained)
                records.append(record)
                if k % 2 == 0:
                    p = k // 2
                    x = (p % 4) * 320
                    y = (p // 4) * 540
                    sheet.paste(image.resize((320, 180)), (x, y))
                    old = pixels(image, crop, v, shirt)
                    sheet.paste(old.resize((320, 180)), (x, y + 180))
                    sheet.paste(processed.resize((320, 180)), (x, y + 360))
                if k in (0, 7, 15, 23, 31):
                    processed.save(out / f"{name}-frame{global_i}-person.png")
                    image.save(out / f"{name}-frame{global_i}-original.png")
            disc = sum(r["observed_disc_pixels"] for r in records)
            motion = sum(r.get("motion_disc_sum", 0) for r in records)
            summary[name] = dict(
                records=records,
                frames=len(records),
                elapsed_seconds=perf_counter() - start,
                valid_joint_disc_retention=sum(r["retained_disc_pixels"] for r in records) / disc,
                motion_disc_retention=sum(r.get("retained_motion_disc_sum", 0) for r in records)
                / motion
                if motion
                else None,
                source_sha256=expected,
                requires_visual_identifier_review=True,
            )
            sheet.save(out / f"{name}-comparison.jpg")
            write_json(out / "measurements.json", summary)
            print(
                name,
                summary[name]["valid_joint_disc_retention"],
                summary[name]["motion_disc_retention"],
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
