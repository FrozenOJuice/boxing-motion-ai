"""Read-only source-frame rendering for measured pose-quality diagnostics."""

import json
from pathlib import Path

from ..vision.decoder import decode_video
from ..vision.pipeline import sha256, write_json


def render_quality_review(dataset, video, output):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    dataset, video, output = map(Path, (dataset, video, output))
    report = json.loads((output / "quality.json").read_text())
    manifest = json.loads((dataset / "preparation.json").read_text())
    if sha256(video) != manifest["inputs"]["video_sha256"]:
        raise ValueError("Video hash does not match the extracted source")
    folder = output / "visual-review"
    if folder.exists():
        raise ValueError("Visual review output already exists")
    folder.mkdir()
    selection = report["visual_review_selection"]
    wanted = {n for item in selection for n in item["frames"]}
    pts_observed = []
    with np.load(dataset / "poses.npz", allow_pickle=False) as arrays:
        xyz, conf, valid = arrays["coordinates"], arrays["confidence"], arrays["valid_mask"]
        frames = decode_video(video, 1280)
        try:
            for frame in frames:
                n = frame.index
                if n in wanted:
                    if frame.pts != int(arrays["pts"][n]):
                        raise ValueError("Diagnostic source frame timing mismatch")
                    if frame.time_base.numerator != int(
                        arrays["time_base_num"][n]
                    ) or frame.time_base.denominator != int(arrays["time_base_den"][n]):
                        raise ValueError("Diagnostic source timebase mismatch")
                    image = Image.fromarray(frame.rgb)
                    w, h = image.size
                    canvas = Image.new("RGB", (w, h + 112), "#101827")
                    canvas.paste(image, (0, 112))
                    draw = ImageDraw.Draw(canvas)
                    draw.font = ImageFont.load_default(size=17)
                    draw.text(
                        (12, 8),
                        f"V5 decoded n={n} / PTS {float(frame.time):.3f}s / raw estimates",
                        fill="white",
                    )
                    draw.text(
                        (12, 31),
                        "cyan/green valid; orange unreliable raw estimate; numbers are joint IDs.",
                        fill="white",
                    )
                    scores = " | ".join(
                        f"{name} {j}: v={conf[n, j, 0]:.3f} p={conf[n, j, 1]:.3f}"
                        for name, j in [("LE", 13), ("LW", 15), ("RE", 14), ("RW", 16)]
                    )
                    draw.text((12, 54), scores, fill="#f5d28b")
                    draw.text(
                        (12, 78),
                        "Model scores do not verify geometric accuracy or occlusion.",
                        fill="white",
                    )

                    def point(j, n=n, w=w, h=h):
                        return xyz[n, j, 0] * w, xyz[n, j, 1] * h + 112

                    for a, b in [
                        (11, 12),
                        (11, 13),
                        (13, 15),
                        (12, 14),
                        (14, 16),
                        (11, 23),
                        (12, 24),
                        (23, 24),
                    ]:
                        if (
                            np.isfinite(xyz[n, [a, b], :2]).all()
                            and ((xyz[n, [a, b], :2] >= 0) & (xyz[n, [a, b], :2] <= 1)).all()
                        ):
                            color = "#29ed97" if valid[n, a] and valid[n, b] else "#ff9e45"
                            draw.line([point(a), point(b)], fill=color, width=3)
                    for j in [11, 12, 13, 14, 15, 16, 23, 24]:
                        if (
                            np.isfinite(xyz[n, j, :2]).all()
                            and ((xyz[n, j, :2] >= 0) & (xyz[n, j, :2] <= 1)).all()
                        ):
                            x, y = point(j)
                            color = "#00d9ff" if valid[n, j] else "#ff9e45"
                            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
                            draw.text(
                                (x + 5, y + 3),
                                str(j),
                                fill=color,
                                stroke_width=2,
                                stroke_fill="black",
                            )
                    canvas.save(folder / f"frame_{n:05d}.jpg", quality=94)
                    pts_observed.append(
                        {"frame_index": n, "pts": frame.pts, "time_base": str(frame.time_base)}
                    )
                if n >= max(wanted):
                    break
        finally:
            frames.close()
    if {r["frame_index"] for r in pts_observed} != wanted:
        raise ValueError("Some requested source frames were not decoded")
    for label in report["by_class"]:
        items = [item for item in selection if item["class"] == label]
        sheet = Image.new("RGB", (1920, 3 * 490), "#101827")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=21)
        for row, item in enumerate(items):
            y = row * 490
            draw.text(
                (12, y + 8),
                f"Expected {label} / row {item['source_row']} / {item['rank']} core coverage",
                font=font,
                fill="white",
            )
            for col, n in enumerate(item["frames"]):
                with Image.open(folder / f"frame_{n:05d}.jpg") as im:
                    im.thumbnail((640, 442))
                    sheet.paste(im, (col * 640, y + 42))
        sheet.save(folder / f"{label}.jpg", quality=94)
    write_json(
        folder / "source-frame-manifest.json",
        {"video_sha256": sha256(video), "frames": pts_observed},
    )
    return len(wanted)
