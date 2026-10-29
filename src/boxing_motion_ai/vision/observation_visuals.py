"""Exact-frame raw/estimator panels for the diagnostic observation study only."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .observation_study import BASE, CASES, MP, VARIANTS, arrays, read, study_manifest


def render(out, cases=None, frames=None):
    study_manifest(out)
    font = ImageFont.load_default(size=13)
    edges = [(0, 2), (2, 4), (1, 3), (3, 5), (0, 1), (0, 6), (1, 7), (6, 7)]
    for name in cases or [
        "V4-bg-1",
        "V4-bg-2",
        "V4-bg-3",
        "V6-upload",
        "V4-8s-upload",
        "V4-29s-upload",
    ]:
        if name not in CASES:
            raise ValueError("Unknown diagnostic case")
        source = BASE / name
        case = out / name
        raw = arrays(source / "poses.npz")
        video = read(source / "video.json")
        size = np.array([video["width"], video["height"]])
        variants = {"stored": (raw["coordinates"][:, MP, :2] * size, raw["valid_mask"][:, MP])}
        for v in VARIANTS:
            if v == "rtmpose-crop":
                a = arrays(case / v / "raw-rtmpose.npz")
                variants[v] = (a["xy"], np.ones(a["scores"].shape, bool))
            else:
                a = arrays(case / v / "poses.npz")
                variants[v] = (a["coordinates"][:, MP, :2] * size, a["valid_mask"][:, MP])
        refs = read(source / "references.json")
        idx = (
            sorted(
                set(
                    (r["start_frame"] + min(len(raw["pts"]), r["stop_frame_exclusive"]) - 1) // 2
                    for r in refs
                )
            )
            if refs
            else np.linspace(12, len(raw["pts"]) - 8, 6, dtype=int).tolist()
        )
        if frames is not None:
            if any(i < 0 or i >= len(raw["pts"]) for i in frames):
                raise ValueError("Frame outside diagnostic source")
            idx = frames
        # Display upper body union, same crop for every estimator; no pose input changes.
        xy = variants["stored"][0]
        mask = variants["stored"][1]
        pts = xy[mask]
        if len(pts):
            lo, hi = pts.min(0), pts.max(0)
            pad = (hi - lo) * 0.25
            box = np.r_[np.maximum(0, lo - pad), np.minimum(size, hi + pad)].astype(int)
        else:
            box = np.r_[0, 0, size]
        columns = 4 if len(idx) == 1 else 7
        w, h = (480, 480) if len(idx) == 1 else (320, 340)
        rows_per_frame = (7 + columns - 1) // columns
        scale = min(w / (box[2] - box[0]), (h - 40) / (box[3] - box[1]))
        canvas = Image.new("RGB", (w * columns, h * len(idx) * rows_per_frame), "#111827")
        d = ImageDraw.Draw(canvas)
        for row, i in enumerate(idx):
            image = Image.open(case / "frames" / f"{i:04d}.png").crop(tuple(box))
            image = image.resize((round(image.width * scale), round(image.height * scale)))
            for col, v in enumerate(["raw", *variants]):
                im = image.copy()
                pen = ImageDraw.Draw(im)
                if v != "raw":
                    p, m = variants[v]
                    p = (p[i] - box[:2]) * scale
                    m = m[i]
                    for a, b in edges:
                        if np.isfinite(p[[a, b]]).all():
                            pen.line(
                                [tuple(p[a]), tuple(p[b])],
                                fill=("#00e5ff" if a % 2 == 0 else "#ff51cb")
                                if m[a] and m[b]
                                else "#ff9500",
                                width=2,
                            )
                    for k, (x, y) in enumerate(p):
                        if np.isfinite([x, y]).all():
                            c = ("#00e5ff" if k % 2 == 0 else "#ff51cb") if m[k] else "#ff9500"
                            pen.ellipse((x - 3, y - 3, x + 3, y + 3), fill=c)
                            pen.text((x + 4, y), str(k), font=font, fill=c)
                x, y = (col % columns) * w, (row * rows_per_frame + col // columns) * h
                canvas.paste(im, (x, y + 35))
                label = "RTM raw, no validity claim" if v == "rtmpose-crop" else v
                d.text((x + 3, y + 2), f"{name} f{i} {label}", font=font, fill="white")
        suffix = f"-frame-{idx[0]}" if len(idx) == 1 else ""
        canvas.save(out / f"{name}{suffix}-visual.png")
        print(name, idx, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--frame", type=int, action="append")
    args = parser.parse_args()
    render(args.output, args.case, args.frame)
