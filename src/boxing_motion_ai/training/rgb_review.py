"""Read-only exact-PTS rear-punch review before any RGB model implementation."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..vision.contracts import EDGES
from ..vision.pipeline import sha256
from .experiment import write_json

RECORDINGS = ("V4", "V6", "V7", "V8", "V10")


def decode_requested(video, requests):
    """Seek/decode exact integer-PTS frames; never infer a timestamp from nominal FPS."""
    import av

    previous, decoder = None, None
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        for index, target in sorted(requests.items()):
            if previous is None or target - previous > Fraction(3, 4):
                container.seek(int(target / stream.time_base), stream=stream, backward=True)
                decoder = container.decode(stream)
            for frame in decoder:
                if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                    raise ValueError("Missing/corrupt decoded frame timing")
                actual = frame.pts * frame.time_base
                if actual < target:
                    continue
                if actual != target:
                    raise ValueError(f"No exact source frame at {index}: {target}")
                yield index, frame.to_image(), actual
                previous = target
                break
            else:
                raise ValueError("EOF before requested frame")


def choose_cases(rows, selected):
    result = []
    for label in ("rear_hook", "rear_uppercut"):
        available = [
            r for r in rows if r["source_row"] in selected and r["punch_type_provisional"] == label
        ]
        eligible = bool(available)
        if not available:
            available = [
                r
                for r in rows
                if r["punch_type_provisional"] == label
                and r["candidate_start_index"] is not None
                and r["candidate_end_index"] >= r["candidate_start_index"]
            ]
        if available:
            for i in sorted(set((0, len(available) - 1))):
                result.append(
                    {
                        **available[i],
                        "review_selection": "approved" if eligible else "excluded_context_only",
                    }
                )
    return result


def run(output):
    from PIL import Image, ImageDraw, ImageFont

    output = Path(output)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a fresh ignored report directory")
    output.mkdir(parents=True)
    accepted = json.loads(
        Path("reports/boxingvi/level1-limb-readiness-v2/summary.json").read_text()
    )
    cases = []
    for v in RECORDINGS:
        root = Path(f"data/processed/boxingvi/{v}/candidates-full-v1")
        manifest = json.loads((root / "preparation.json").read_text())
        source = json.loads(
            Path(f"data/processed/boxingvi/{v}/pose-full-image-v1/manifest.json").read_text()
        )
        if source["inputs"]["video"]["filename"] != f"{v}.mkv":
            raise ValueError("Unexpected source filename; V9 remains reserved")
        video = Path("data/raw/boxingvi/RGB_videos") / f"{v}.mkv"
        if sha256(video) != manifest["inputs"]["video_sha256"]:
            raise ValueError("Source video changed")
        rows = [json.loads(s) for s in (root / "candidates.jsonl").read_text().splitlines()]
        chosen = choose_cases(rows, accepted["recordings"][v]["provenance"]["source_rows"])
        for row_id in [685] if v == "V6" else [9] if v == "V10" else []:
            item = next(r for r in rows if r["source_row"] == row_id)
            chosen.append({**item, "review_selection": "excluded_context_only"})
        with np.load(root / "poses.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        requests = {}
        indices = {}
        for c in chosen:
            a, b = c["candidate_start_index"], c["candidate_end_index"]
            idx = sorted(
                set(
                    np.linspace(
                        max(0, a - 2), min(len(arrays["pts"]) - 1, b + 2), 8, dtype=int
                    ).tolist()
                )
            )
            indices[c["source_row"]] = idx
            for i in idx:
                requests[i] = Fraction(
                    int(arrays["pts"][i]) * int(arrays["time_base_num"][i]),
                    int(arrays["time_base_den"][i]),
                )
        images = {i: im for i, im, _ in decode_requested(video, requests)}
        for c in chosen:
            idx = indices[c["source_row"]]
            w, h = 320, 180
            sheet = Image.new("RGB", (w * 4, (h * 2 + 35) * 2 + 65), "#101827")
            draw = ImageDraw.Draw(sheet)
            font = ImageFont.load_default(size=15)
            draw.text(
                (5, 5),
                f"{v} row {c['source_row']} {c['source_class']} "
                f"raw {c['raw_start']}..{c['raw_end']}",
                font=font,
            )
            draw.text(
                (5, 26),
                f"{c['review_selection']} flags: {c['flags']} | provisional labels, no correction",
                font=font,
                fill="#ffd97d",
            )
            for k, i in enumerate(idx):
                x = (k % 4) * w
                y = 65 + (k // 4) * (2 * h + 35)
                im = images[i].resize((w, h))
                sheet.paste(im, (x, y + 30))
                over = im.copy()
                d = ImageDraw.Draw(over)
                xy = arrays["coordinates"][i, :, :2]
                valid = arrays["valid_mask"][i]
                for a, b in EDGES:
                    if valid[a] and valid[b]:
                        d.line(
                            [(xy[a, 0] * w, xy[a, 1] * h), (xy[b, 0] * w, xy[b, 1] * h)],
                            fill="#00ff80",
                            width=2,
                        )
                for j in (11, 12, 13, 14, 15, 16):
                    if np.isfinite(xy[j]).all():
                        xx, yy = xy[j] * [w, h]
                        d.ellipse(
                            (xx - 3, yy - 3, xx + 3, yy + 3), fill="cyan" if valid[j] else "red"
                        )
                sheet.paste(over, (x, y + 30 + h))
                active = c["candidate_start_index"] <= i <= c["candidate_end_index"]
                draw.text(
                    (x + 3, y + 3),
                    f"f{i} PTS {requests[i]} {'IN' if active else 'context'}",
                    font=font,
                )
            name = f"{v}-row-{c['source_row']}.jpg"
            sheet.save(output / name, quality=92)
            cases.append(
                dict(
                    video=v,
                    source_row=c["source_row"],
                    source_class=c["source_class"],
                    label=c["punch_type_provisional"],
                    selection=c["review_selection"],
                    source_bounds=[c["raw_start"], c["raw_end"]],
                    flags=c["flags"],
                    exclusion_reasons=c["exclusion_reasons"],
                    indices=idx,
                    exact_pts=[str(requests[i]) for i in idx],
                    sheet=name,
                    video_sha256=sha256(video),
                )
            )
    write_json(
        output / "review.json",
        dict(
            status="requires_visual_review_before_training",
            cases=cases,
            dataset_modified=False,
            labels_corrected=False,
        ),
    )
    return cases


def preprocessing_review(output):
    """Per-class raw/crop/static-erasure previews; review labels never enter cached pixels."""
    from PIL import Image, ImageDraw

    from .data import CLASSES, exact_times
    from .multi_recording import ALL, load_pool, settings
    from .rgb import POLICY, crop_box, views

    output = Path(output)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a fresh ignored report directory")
    output.mkdir(parents=True)
    _, _, specs = settings("configs/level1_multi_recording.yaml")
    pool = load_pool(specs)
    selected_rows = {}
    for v in ALL:
        ds = pool[v]["xy_velocity"]
        chosen = [next((c for c in ds.clips if c["label"] == k), None) for k in range(6)]
        requests = {
            int(c["raw"]["frame_index"][len(c["x"]) // 2]): exact_times(c)[len(c["x"]) // 2]
            for c in chosen
            if c is not None
        }
        source = json.loads(
            Path(f"data/processed/boxingvi/{v}/pose-full-image-v1/manifest.json").read_text()
        )
        if source["inputs"]["video"]["filename"] != f"{v}.mkv":
            raise ValueError("Unexpected source video; V9 reserved")
        video = Path("data/raw/boxingvi/RGB_videos") / f"{v}.mkv"
        if sha256(video) != source["inputs"]["video"]["sha256"]:
            raise ValueError("Source video changed")
        images = {i: im for i, im, _ in decode_requested(video, requests)}
        sheet = Image.new("RGB", (224 * 3, 244 * 6), "#202020")
        draw = ImageDraw.Draw(sheet)
        selected_rows[v] = []
        for k, c in enumerate(chosen):
            row = c["source_row"] if c is not None else "ABSENT"
            draw.text((2, k * 244), f"{v} {CLASSES[k]} row {row}: raw / crop / background")
            if c is None:
                continue
            j = len(c["x"]) // 2
            i = int(c["raw"]["frame_index"][j])
            im = images[i]
            box, fallback = crop_box(c["raw"]["coordinates"][j], c["raw"]["valid_mask"][j], im.size)
            crop, _, background, _ = views(im, box, v)
            for col, image in enumerate((im, crop, background)):
                sheet.paste(image.resize((224, 224)), (224 * col, 244 * k + 20))
            selected_rows[v].append(
                dict(source_row=row, index=i, pts=str(requests[i]), fallback=fallback)
            )
        sheet.save(output / f"{v}-preprocessing.jpg")
    write_json(output / "review.json", dict(policy=POLICY, selected=selected_rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preprocessing", action="store_true")
    args = parser.parse_args()
    if args.preprocessing:
        preprocessing_review(args.output)
    else:
        run(args.output)
