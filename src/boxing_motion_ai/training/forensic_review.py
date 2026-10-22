"""Read-only stratified exact-PTS label/pose review; no fitting or label corrections."""

import argparse
import html
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..vision.pipeline import sha256
from .data import CLASSES, exact_times
from .experiment import write_json
from .multi_recording import ALL, GROUPS, load_pool, settings
from .multi_study import verify_manifest
from .rgb import crop_box
from .rgb_review import decode_requested

ROOT = Path("reports/boxingvi/level1-multi-recording-v1")
QUOTAS = {
    "V4": [1, 1, 1, 3, 3, 3],
    "V5": [1, 1, 1, 3, 3, 3],
    "V6": [0, 0, 2, 3, 3, 4],
    "V7": [1, 1, 4, 1, 5, 0],
    "V8": [1, 1, 3, 0, 3, 4],
    "V10": [1, 1, 1, 3, 3, 3],
}
HOLDS = {"V4": [], "V5": [], "V6": [321, 685], "V7": [5, 54, 132], "V8": [], "V10": [9, 130]}
FOLDS = {
    "V4": "hold-V4",
    "V5": "combined",
    "V6": "hold-V6",
    "V7": "hold-V7-V8",
    "V8": "hold-V7-V8",
    "V10": "hold-V10",
}
ARMS = {"L": (11, 13, 15), "R": (12, 14, 16)}


def select_cases(candidates, predictions, quotas):
    """Prespecified ordinal coverage, with correct/error alternation where available.

    This is a purposive diagnostic sample, not an error-rate estimator. Confidence
    ranks are not used, and labels/predictions are not correctness evidence.
    """
    lookup = {p["source_row"]: p for p in predictions}
    if len(lookup) != len(predictions):
        raise ValueError("Duplicate prediction row")
    if set(lookup) != {r["source_row"] for r in candidates}:
        raise ValueError("Predictions and eligible examples must match exactly")
    selected = []
    for label, quota in zip(CLASSES, quotas, strict=True):
        pool = sorted(
            [c for c in candidates if c["punch_type_provisional"] == label],
            key=lambda c: (c["candidate_start_index"], c["source_row"]),
        )
        if quota > len(pool):
            raise ValueError("Quota exceeds approved support")
        for c in pool:
            if lookup[c["source_row"]]["target"] != label:
                raise ValueError("Prediction target differs from source label")
        picked = []
        for k in range(quota):
            target = (k + 0.5) * len(pool) / quota - 0.5
            wanted_correct = k % 2 == 1
            options = [
                i
                for i in range(len(pool))
                if i not in picked
                and (lookup[pool[i]["source_row"]]["prediction"] == label) == wanted_correct
            ]
            if not options:
                options = [i for i in range(len(pool)) if i not in picked]
            index = min(options, key=lambda i: (abs(i - target), i))
            picked.append(index)
            selected.append(pool[index])
    return selected


def review_indices(arrays, start, end):
    if not 0 <= start <= end < len(arrays["frame_index"]):
        raise ValueError("Never reverse, clip or repair invalid source bounds")
    times = exact_times({"raw": arrays})
    if any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("Exact timestamps must increase; do not repair timing")
    lo, hi = start, end
    while lo > 0 and times[start] - times[lo - 1] <= Fraction(1, 5):
        if any(arrays[k][lo - 1] != arrays[k][start] for k in ("scene_id", "segment_id")):
            break
        if arrays["frame_index"][lo] != arrays["frame_index"][lo - 1] + 1:
            break
        lo -= 1
    while hi + 1 < len(times) and times[hi + 1] - times[end] <= Fraction(1, 5):
        if any(arrays[k][hi + 1] != arrays[k][end] for k in ("scene_id", "segment_id")):
            break
        if arrays["frame_index"][hi + 1] != arrays["frame_index"][hi] + 1:
            break
        hi += 1
    inside = np.linspace(start, end, 6, dtype=int).tolist()
    indices = sorted(set([lo, *inside, hi]))
    # Preserve every in-interval frame for very short annotations; context fills unused slots.
    extras = sorted(
        set(range(lo, hi + 1)) - set(indices), key=lambda i: min(abs(i - start), abs(i - end))
    )
    indices = sorted(indices + extras[: max(0, 8 - len(indices))])
    return indices


def finite_median(values):
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else None


def pose_stats(a, start, end, *, include_failures=False):
    result = {}
    for side, joints in ARMS.items():
        result[side] = {}
        for j in joints:
            valid = a["valid_mask"][start : end + 1, j]
            scores = a["confidence"][start : end + 1, j]
            result[side][str(j)] = dict(
                valid_frames=int(valid.sum()),
                frames=len(valid),
                median_visibility=finite_median(scores[:, 0]),
                median_presence=finite_median(scores[:, 1]),
                both_scores_ge_080=int(np.all(scores >= 0.8, axis=1).sum()),
            )
            if include_failures:
                result[side][str(j)].update(
                    visibility_below_050=int((scores[:, 0] < 0.5).sum()),
                    presence_below_050=int((scores[:, 1] < 0.5).sum()),
                    nonfinite_scores=int((~np.isfinite(scores).all(axis=1)).sum()),
                )
    return result


def render_sheet(v, row, arrays, indices, images, times, output):
    """Blind sheet: source ID and intervals visible; class/prediction shown only in index."""
    w, cell, top = 280, 470, 52
    sheet = Image.new("RGB", (w * 4, cell * 2 + top), "#111827")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    start, end = row["candidate_start_index"], row["candidate_end_index"]
    draw.text(
        (5, 5),
        f"{v}:{row['source_row']} raw {row['raw_start']}..{row['raw_end']} | "
        "class/prediction hidden for review",
        font=font,
    )
    draw.text(
        (5, 25),
        "L cyan / R magenta; invalid orange (diagnostic only); v/p = scores",
        font=font,
        fill="#ffd47a",
    )
    for col, i in enumerate(indices):
        x, y = (col % 4) * w, top + (col // 4) * cell
        im = images[i]
        raw = ImageOps.contain(im, (w, 158))
        sheet.paste(raw, (x, y + 35))
        box, fallback = crop_box(arrays["coordinates"][i], arrays["valid_mask"][i], im.size)
        # Gray padding preserves native aspect in the zoom, without altering source files.
        a, b, c, d = box
        zoom = Image.new("RGB", (c - a, d - b), "#333333")
        zoom.paste(im, (-a, -b))
        zoom = zoom.resize((w, w))
        pen = ImageDraw.Draw(zoom)
        xy = arrays["coordinates"][i, :, :2] * np.array(im.size)
        coords = (xy - np.array([a, b])) * w / (c - a)
        valid = arrays["valid_mask"][i]
        for side, joints in ARMS.items():
            color = "#00ffff" if side == "L" else "#ff66ee"
            for j, k in zip(joints, joints[1:], strict=False):
                if np.isfinite(coords[[j, k]]).all():
                    pen.line(
                        [tuple(coords[j]), tuple(coords[k])],
                        fill=color if valid[j] and valid[k] else "#ff9b32",
                        width=3,
                    )
            for j in joints:
                if np.isfinite(coords[j]).all():
                    xx, yy = coords[j]
                    pen.ellipse(
                        (xx - 3, yy - 3, xx + 3, yy + 3), fill=color if valid[j] else "#ff9b32"
                    )
                    pen.text((xx + 3, yy + 2), f"{side}{j}", fill=color, font=font)
        sheet.paste(zoom, (x, y + 190))
        draw.text(
            (x + 3, y + 2),
            f"f{arrays['frame_index'][i]} PTS {times[i]} "
            f"{'IN' if start <= i <= end else 'context'}",
            font=font,
        )
        for n, j in enumerate((13, 15, 14, 16)):
            scores = arrays["confidence"][i, j]
            draw.text(
                (x + 4 + (n % 2) * 140, y + 427 + (n // 2) * 19),
                f"{j} {scores[0]:.2f}/{scores[1]:.2f} {'+' if valid[j] else '-'}",
                font=font,
                fill="#00ffff" if j % 2 else "#ff66ee",
            )
    sheet.save(output, quality=94)


def run(output):
    output = Path(output)
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Use a fresh ignored report output")
    _, _, specs = settings("configs/level1_multi_recording.yaml")
    pool = load_pool(specs)
    protected = json.loads(
        Path("reports/boxingvi/level1-rgb-study-v1/preserved-inputs.json").read_text()
    )
    if any("/V9/" in p or "/V9." in p for p in protected):
        raise ValueError("V9 outside scope")
    for previous in (ROOT, Path("reports/boxingvi/level1-rgb-study-v1")):
        protected.update({str(p): sha256(p) for p in previous.rglob("*") if p.is_file()})
    verify_manifest(protected)
    output.mkdir(parents=True)
    write_json(output / "protected.json", protected)
    records = []
    for v in ("V4", "V5", "V6", "V7", "V8", "V10"):
        if v not in ALL:
            raise ValueError("Reserved recording")
        ds = pool[v]["xy_velocity"]
        root = Path(ds.provenance["dataset"])
        rows = [json.loads(s) for s in (root / "candidates.jsonl").read_text().splitlines()]
        pred_path = ROOT / FOLDS[v] / "seed-20261021/xy_velocity/tcn" / f"{v}-predictions.json"
        predictions = json.loads(pred_path.read_text())
        eligible = set(ds.provenance["source_rows"])
        chosen = select_cases(
            [r for r in rows if r["source_row"] in eligible], predictions, QUOTAS[v]
        )
        chosen += [next(r for r in rows if r["source_row"] == i) for i in HOLDS[v]]
        if len({c["source_row"] for c in chosen}) != len(chosen):
            raise ValueError("Duplicate review selection")
        with np.load(root / "poses.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        if not np.array_equal(arrays["frame_index"], np.arange(len(arrays["frame_index"]))):
            raise ValueError("Review requires the intact zero-indexed full-video frame contract")
        times = exact_times({"raw": arrays})
        requests, plans = {}, {}
        for c in chosen:
            indices = review_indices(arrays, c["candidate_start_index"], c["candidate_end_index"])
            plans[c["source_row"]] = indices
            requests.update({i: times[i] for i in indices})
        video = Path(f"data/raw/boxingvi/RGB_videos/{v}.mkv")
        if sha256(video) != ds.provenance["source_input_hashes"]["video_sha256"]:
            raise ValueError("Video changed")
        images = {i: im for i, im, _ in decode_requested(video, requests)}
        lookup = {p["source_row"]: p for p in predictions}
        for c in chosen:
            name = f"{v}-row-{c['source_row']}.jpg"
            idx = plans[c["source_row"]]
            render_sheet(v, c, arrays, idx, images, times, output / name)
            records.append(
                dict(
                    id=f"{v}:{c['source_row']}",
                    recording_id=v,
                    source_group=GROUPS[v],
                    source_row=c["source_row"],
                    source_class=c["source_class"],
                    expected=c["punch_type_provisional"],
                    raw_bounds=[c["raw_start"], c["raw_end"]],
                    approved_candidate=c["source_row"] in eligible,
                    exclusion_reasons=c["exclusion_reasons"],
                    flags=c["flags"],
                    prediction=lookup.get(c["source_row"]),
                    prediction_source=str(pred_path),
                    prediction_sha256=sha256(pred_path),
                    selection_fold=FOLDS[v],
                    selection="development-selected" if v == "V5" else "source-group held out",
                    frames=idx,
                    exact_pts=[str(times[i]) for i in idx],
                    sheet=name,
                    scores_by_frame={
                        str(i): {
                            str(j): dict(
                                visibility=float(arrays["confidence"][i, j, 0]),
                                presence=float(arrays["confidence"][i, j, 1]),
                                valid=bool(arrays["valid_mask"][i, j]),
                            )
                            for j in (13, 14, 15, 16)
                        }
                        for i in idx
                    },
                    pose_summary=pose_stats(
                        arrays, c["candidate_start_index"], c["candidate_end_index"]
                    ),
                    review={axis: "unreviewed" for axis in ("family", "side", "boundary", "pose")},
                    boxer_id=None,
                    session_id=None,
                    mirroring="unverified",
                )
            )
        write_json(output / "cases.json", records)
        print(f"Reviewed media rendered {v}: {len(chosen)} cases", flush=True)
    pages = [
        "<!doctype html><meta charset='utf-8'><title>Label and pose review</title>",
        "<style>body{font:16px sans-serif;max-width:1150px;margin:auto}img{width:100%}</style>",
        "<h1>Read-only review: 79 purposive examples</h1>",
        "<p>Inspect movement before revealing labels/predictions. They are hypotheses, not truth. "
        "Orange joints are invalid and are not trusted model coordinates. "
        "V5 is development-only; identities/mirroring unverified. No annotations changed.</p>",
    ]
    for c in records:
        pages += [
            f"<h2 id='{c['id']}'>{c['id']}</h2><img loading='lazy' src='{c['sheet']}'>",
            "<details><summary>Reveal source annotation and frozen prediction</summary><pre>"
            + html.escape(
                json.dumps(
                    {
                        k: c[k]
                        for k in (
                            "source_class",
                            "raw_bounds",
                            "prediction",
                            "approved_candidate",
                            "exclusion_reasons",
                            "selection",
                        )
                    },
                    indent=2,
                )
            )
            + "</pre></details>",
        ]
    (output / "index.html").write_text("\n".join(pages))
    verify_manifest(protected)
    write_json(output / "preservation.json", dict(files_verified=len(protected), changed=[]))
    write_json(
        output / "summary.json",
        dict(
            cases=len(records),
            counts=dict(Counter(c["recording_id"] for c in records)),
            class_counts=dict(Counter(c["expected"] for c in records)),
            selection="purposive stratified diagnostic, not representative error prevalence",
            training_performed=False,
            corrections=[],
            v9_accessed=False,
        ),
    )
    return records


def dense_review(review_path, output, case_ids):
    """All native frames around existing cases; no retiming or additional candidates."""
    output = Path(output)
    if (
        output.exists()
        or not output.resolve().is_relative_to(Path("reports").resolve())
        or not case_ids
        or len(set(case_ids)) != len(case_ids)
    ):
        raise ValueError("Use unique cases and a fresh ignored output")
    # Reject reserved/out-of-scope IDs before any source read.
    if any(c.split(":")[0] not in ALL for c in case_ids):
        raise ValueError("V9 reserved; only approved recordings")
    cases = json.loads(Path(review_path).read_text())
    chosen = [c for c in cases if c["id"] in case_ids]
    if len(chosen) != len(case_ids):
        raise ValueError("Dense review must reference already selected cases")
    protected = json.loads((Path(review_path).parent / "protected.json").read_text())
    if any("/V9/" in p or "/V9." in p for p in protected):
        raise ValueError("V9 outside scope")
    verify_manifest(protected)
    output.mkdir(parents=True)
    pages = []
    for v in ALL:
        rows = [c for c in chosen if c["recording_id"] == v]
        if not rows:
            continue
        root = Path(f"data/processed/boxingvi/{v}/candidates-full-v1")
        with np.load(root / "poses.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        if not np.array_equal(arrays["frame_index"], np.arange(len(arrays["frame_index"]))):
            raise ValueError("Review requires the intact zero-indexed full-video frame contract")
        times = exact_times({"raw": arrays})
        plans = {}
        for c in rows:
            start, end = c["raw_bounds"]
            endpoints = review_indices(arrays, int(start), int(end))
            plans[c["id"]] = list(range(min(endpoints), max(endpoints) + 1))
        requests = {i: times[i] for indices in plans.values() for i in indices}
        images = {
            i: ImageOps.contain(im, (1280, 720))
            for i, im, _ in decode_requested(
                Path(f"data/raw/boxingvi/RGB_videos/{v}.mkv"), requests
            )
        }
        for c in rows:
            start, end = c["raw_bounds"]
            row = dict(
                source_row=c["source_row"],
                raw_start=start,
                raw_end=end,
                candidate_start_index=int(start),
                candidate_end_index=int(end),
            )
            indices = plans[c["id"]]
            for j in range(0, len(indices), 8):
                selected = indices[j : j + 8]
                name = f"{v}-row-{c['source_row']}-dense-{j // 8 + 1}.jpg"
                render_sheet(v, row, arrays, selected, images, times, output / name)
                pages.append(
                    dict(
                        id=c["id"],
                        sheet=name,
                        frames=selected,
                        exact_pts=[str(times[i]) for i in selected],
                    )
                )
    write_json(output / "dense.json", pages)
    verify_manifest(protected)
    return pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense-cases", nargs="+")
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.dense_cases:
        if not args.review:
            parser.error("--review cases.json required for dense inspection")
        dense_review(args.review, args.output, args.dense_cases)
    else:
        run(args.output)
