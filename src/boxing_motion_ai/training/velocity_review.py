"""Exact-PTS visual inspection of largest observed derivatives; no training or corrections."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..data.pose_quality import CORE
from ..data.source_review import decode_frames
from ..vision.pipeline import sha256
from .data import exact_times
from .experiment import write_json


def render(experiment, output):
    from PIL import Image, ImageDraw, ImageFont

    experiment, output = Path(experiment), Path(output)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a fresh output below reports/")
    diagnostics = json.loads((experiment / "velocity-diagnostics.json").read_text())
    if set(diagnostics) != {"V4", "V5"}:
        raise ValueError("Only V4/V5 permitted; no other source reads")
    output.mkdir(parents=True)
    evidence = []
    for video in ("V4", "V5"):
        root = Path(f"data/processed/boxingvi/{video}/candidates-full-v1")
        manifest = json.loads((root / "preparation.json").read_text())
        media = Path(f"data/raw/boxingvi/RGB_videos/{video}.mkv")
        if sha256(media) != manifest["inputs"]["video_sha256"]:
            raise ValueError("Source video hash differs")
        if sha256(root / "poses.npz") != manifest["npz_sha256"]:
            raise ValueError("Pose source changed")
        with np.load(root / "poses.npz", allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        times = exact_times({"raw": a})
        selected, seen = [], set()
        for case in diagnostics[video]["largest_pairs"]:
            if case["source_row"] not in seen:
                selected.append(case)
                seen.add(case["source_row"])
            if len(selected) == 2:
                break
        indices = {
            i
            for c in selected
            for i in range(max(0, c["frame_index"] - 2), min(len(times), c["frame_index"] + 3))
        }
        folder = output / video
        decode_frames(media, times, indices, folder / "frames")
        for c in selected:
            targets = list(
                range(max(0, c["frame_index"] - 2), min(len(times), c["frame_index"] + 3))
            )
            sheet = Image.new("RGB", (384 * len(targets), 320), "#101827")
            draw = ImageDraw.Draw(sheet)
            font = ImageFont.load_default(size=12)
            draw.text(
                (8, 6),
                f"{video} row {c['source_row']} expected {c['class']} (provisional); "
                f"joint {c['core_joint_index']}; {c['speed_torso_per_second']:.2f} torso/s",
                font=font,
                fill="white",
            )
            for col, i in enumerate(targets):
                with Image.open(folder / "frames" / f"{i}.jpg") as im:
                    sheet.paste(im, (384 * col, 96))
                joint = CORE[c["core_joint_index"]]
                scores = a["confidence"][i, joint]
                draw.text(
                    (384 * col + 5, 35),
                    f"frame {i} PTS {float(times[i]):.3f}s",
                    font=font,
                    fill="white",
                )
                draw.text(
                    (384 * col + 5, 53),
                    f"v={scores[0]:.3f} p={scores[1]:.3f}; raw masks",
                    font=font,
                    fill="#f5d28b",
                )

                def point(j, a=a, i=i, col=col):
                    return 384 * col + float(a["coordinates"][i, j, 0]) * 384, 96 + float(
                        a["coordinates"][i, j, 1]
                    ) * 216

                for j, k in (
                    (11, 12),
                    (11, 13),
                    (13, 15),
                    (12, 14),
                    (14, 16),
                    (11, 23),
                    (12, 24),
                    (23, 24),
                ):
                    if np.isfinite(a["coordinates"][i, [j, k], :2]).all():
                        color = (
                            "#29ed97"
                            if a["valid_mask"][i, j] and a["valid_mask"][i, k]
                            else "#ff9e45"
                        )
                        draw.line([point(j), point(k)], fill=color, width=2)
                if np.isfinite(a["coordinates"][i, joint, :2]).all():
                    x, y = point(joint)
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="yellow")
            name = f"{video}-row-{c['source_row']}.jpg"
            sheet.save(output / name, quality=94)
            evidence.append(
                {
                    **c,
                    "video": video,
                    "sheet": name,
                    "review_indices": targets,
                    "source_video_sha256": sha256(media),
                }
            )
    write_json(
        output / "review.json",
        {
            "cases": evidence,
            "corrections": [],
            "note": "Future frames are visual context only, never derivative inputs.",
        },
    )
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.experiment, args.output)
