"""Bounded exact-frame estimator comparison; diagnostic outputs never enter datasets."""

import argparse
import json
from fractions import Fraction
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..data.annotation_review import AXES, file_hash, verified_axes
from ..services.dashboard import DashboardService
from ..services.review_workspace import ReviewWorkspace

RECORDINGS = ("V4", "V5", "V6", "V7", "V8", "V10")
MP = [11, 12, 13, 14, 15, 16, 23, 24]
COCO = [5, 6, 7, 8, 9, 10, 11, 12]
EDGES = [(0, 2), (2, 4), (1, 3), (3, 5), (0, 1), (0, 6), (1, 7), (6, 7)]


def bounded_indices(a, start, end, context=2, maximum=100):
    n = len(a["frame_index"])
    if not 0 <= start <= end < n or end - start + 1 > maximum:
        raise ValueError("Invalid or unbounded diagnostic interval")
    if any(np.any(a[k][start : end + 1] != a[k][start]) for k in ("scene_id", "segment_id")):
        raise ValueError("Annotation crosses a discontinuity; no silent repair")
    lo, hi = start, end
    for _ in range(context):
        if (
            lo > 0
            and all(a[k][lo - 1] == a[k][start] for k in ("scene_id", "segment_id"))
            and a["frame_index"][lo] == a["frame_index"][lo - 1] + 1
        ):
            lo -= 1
        if (
            hi + 1 < n
            and all(a[k][hi + 1] == a[k][start] for k in ("scene_id", "segment_id"))
            and a["frame_index"][hi + 1] == a["frame_index"][hi] + 1
        ):
            hi += 1
    idx = list(range(lo, hi + 1))
    if not np.all(np.diff(a["frame_index"][idx]) == 1):
        raise ValueError("Missing extraction boundary")
    return idx


def common_mp(a, indices, size):
    xy = a["coordinates"][indices][:, MP, :2] * np.asarray(size)
    scores = a["confidence"][indices][:, MP]
    valid = a["valid_mask"][indices][:, MP] & np.isfinite(xy).all(axis=-1)
    return xy, scores, valid


def case_box(a, indices, size):
    coords = a["coordinates"][indices, :, :2]
    valid = a["valid_mask"][indices] & np.isfinite(coords).all(axis=-1)
    points = coords[valid] * np.asarray(size)
    if len(points) < 8:
        return [0.0, 0.0, float(size[0]), float(size[1])]
    low, high = np.quantile(points, [0, 1], axis=0)
    pad = (high - low) * 0.15
    low = np.maximum(low - pad, 0)
    high = np.minimum(high + pad, size)
    if np.any(high - low < 10):
        raise ValueError("Degenerate diagnostic crop")
    return [*low.tolist(), *high.tolist()]


def motion_stats(xy, valid, indices, times, scenes, segments, scale):
    """Raw projected speed; not an accuracy metric or an invitation to smooth punches."""
    dt = np.array([float(b - a) for a, b in zip(times, times[1:], strict=False)])
    continuity = (np.diff(indices) == 1) & (np.diff(scenes) == 0) & (np.diff(segments) == 0)
    continuity &= (dt > 0) & (dt <= 0.1)
    ok = valid[:-1] & valid[1:] & continuity[:, None]
    displacement = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / scale
    speeds = displacement / np.where(dt > 0, dt, 1)[:, None]
    values = speeds[ok]
    return dict(
        pairs=int(ok.sum()),
        speed_p50=float(np.median(values)) if len(values) else None,
        speed_p95=float(np.quantile(values, 0.95)) if len(values) else None,
        steps_over_half_torso=int(((displacement > 0.5) & ok).sum()),
    )


def sheet(case, indices, images, mp, rt, valid, scores, out):
    # Six columns of exact frames; raw / MediaPipe / RTMPose rows.
    chosen = np.unique(np.linspace(0, len(indices) - 1, 6, dtype=int))
    w, h = 300, 280
    canvas = Image.new("RGB", (w * len(chosen), h * 3 + 58), "#151b25")
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    d.text(
        (8, 5),
        f"{case['id']} | {case['expected']} | manual: {case['manual_status']} | cyan L / magenta R",
        font=font,
    )
    d.text(
        (8, 27),
        "Raw / MediaPipe (invalid orange) / RTMPose (scores not equivalent; all finite shown)",
        font=font,
    )
    box = np.asarray(case["box"])
    bx, by, ex, ey = box
    # Same fixed crop and scaling for all three views.
    factor = min(w / (ex - bx), (h - 32) / (ey - by))
    for col, j in enumerate(chosen):
        im = (
            images[indices[j]]
            .crop(tuple(box))
            .resize((max(1, int((ex - bx) * factor)), max(1, int((ey - by) * factor))))
        )
        for row, points in enumerate((None, mp[j], rt[j])):
            image = im.copy()
            pen = ImageDraw.Draw(image)
            if points is not None:
                xy = (points - box[:2]) * factor
                for a, b in EDGES:
                    if np.isfinite(xy[[a, b]]).all():
                        color = "#00ffff" if a % 2 == 0 else "#ff60e0"
                        if row == 1 and not (valid[j, a] and valid[j, b]):
                            color = "#ff9b32"
                        pen.line([tuple(xy[a]), tuple(xy[b])], fill=color, width=3)
                for k, (x, y) in enumerate(xy):
                    if np.isfinite([x, y]).all():
                        color = "#00ffff" if k % 2 == 0 else "#ff60e0"
                        pen.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            x, y = col * w, 58 + row * h
            canvas.paste(image, (x, y + 27))
            d.text((x + 3, y + 2), f"f{indices[j]} " + (["raw", "MP", "RTM"][row]), font=font)
    canvas.save(out)


def run(config, model, output):
    import cv2
    from rtmlib import RTMPose

    from ..training.rgb_review import decode_requested

    config, model, output = map(Path, (config, model, output))
    c = json.loads(config.read_text())
    if output.exists() or not output.resolve().is_relative_to(Path("reports").resolve()):
        raise ValueError("Fresh ignored report directory required")
    if not model.is_file():
        raise ValueError("Supply local downloaded model; no implicit download")
    cases_path = Path("reports/boxingvi/level1-label-pose-review-v1/cases.json")
    if file_hash(cases_path) != c["cases_sha256"]:
        raise ValueError("Diagnostic cases changed")
    cases = {x["id"]: x for x in json.loads(cases_path.read_text())}
    if len(c["case_ids"]) != len(set(c["case_ids"])) or len(c["case_ids"]) > 40:
        raise ValueError("Bounded unique cases required")
    if any(k not in cases or k.split(":")[0] not in RECORDINGS for k in c["case_ids"]):
        raise ValueError("Reserved/unknown recording")
    live = ReviewWorkspace(DashboardService(Path.cwd())).snapshot()
    if live.ledger_hash != c["review_ledger"]["sha256"]:
        raise ValueError("Review snapshot changed")
    protection = json.loads(
        Path("reports/boxingvi/level1-curation-study-v1/preserved-inputs.json").read_text()
    )
    protection.update(
        json.loads(
            Path(
                "reports/boxingvi/level1-curation-study-v1/raw-source-preservation.json"
            ).read_text()
        )
    )
    protection[str(config)] = file_hash(config)
    protection[str(model)] = file_hash(model)
    for path, h in protection.items():
        if file_hash(path) != h:
            raise ValueError(f"Input changed: {path}")
    cv2.setNumThreads(2)
    pose = RTMPose(str(model), model_input_size=(192, 256), backend="opencv", device="cpu")
    pose(np.zeros((256, 192, 3), np.uint8))  # Untimed warm-up, no dataset frame.
    output.mkdir(parents=True)
    results = []
    latencies = []
    for v in RECORDINGS:
        subset = [cases[k] for k in c["case_ids"] if k.startswith(v + ":")]
        root = Path(f"data/processed/boxingvi/{v}/candidates-full-v1")
        with np.load(root / "poses.npz", allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        for case in subset:
            start, end = map(int, case["raw_bounds"])
            idx = bounded_indices(a, start, end, c["context_frames"], c["max_frames_per_case"])
            times = [
                Fraction(int(a["pts"][i]) * int(a["time_base_num"][i]), int(a["time_base_den"][i]))
                for i in idx
            ]
            if any(y <= x for x, y in zip(times, times[1:], strict=False)):
                raise ValueError("Nonmonotonic PTS")
            images = {
                i: im
                for i, im, _ in decode_requested(
                    Path(f"data/raw/boxingvi/RGB_videos/{v}.mkv"),
                    dict(zip(idx, times, strict=True)),
                )
            }
            size = images[idx[0]].size
            box = case_box(a, idx, size)
            mp, conf, valid = common_mp(a, idx, size)
            rt = []
            rs = []
            for i in idx:
                # Official exported pipeline says to_rgb=true. Pillow already supplies RGB.
                begin = perf_counter()
                k, s = pose(np.asarray(images[i]), bboxes=[box])
                latencies.append(perf_counter() - begin)
                rt.append(k[0, COCO])
                rs.append(s[0, COCO])
            rt = np.asarray(rt)
            rs = np.asarray(rs)
            torso = mp[:, 0:2].mean(axis=1) - mp[:, 6:8].mean(axis=1)
            anchor = valid[:, [0, 1, 6, 7]].all(axis=1)
            scale = float(np.median(np.linalg.norm(torso[anchor], axis=-1)))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("No reliable torso scale")
            rev = live.ledger.cases[case["id"]].revisions[-1]
            status = (
                "fully_verified"
                if rev.actor_type == "human" and verified_axes(rev) == set(AXES)
                else "partial_hold"
                if rev.actor_type == "human"
                else "automated_only"
            )
            inside = np.array([(start <= i <= end) for i in idx])
            difference = np.linalg.norm(mp - rt, axis=-1) / scale
            rec = dict(
                id=case["id"],
                recording_id=v,
                expected=case["expected"],
                manual_status=status,
                manual_notes=rev.reasons if rev.actor_type == "human" else [],
                bounds=[start, end],
                box=box,
                frames=idx,
                timestamps=list(map(str, times)),
                scale_pixels=scale,
                mp_valid=valid[inside].sum(axis=0).tolist(),
                slots=int(inside.sum()),
                mp_high_conf=(conf[inside] >= 0.8).all(axis=-1).sum(axis=0).tolist(),
                disagreement_median=np.median(difference[inside], axis=0).tolist(),
                mp_motion=motion_stats(
                    mp, valid, np.array(idx), times, a["scene_id"][idx], a["segment_id"][idx], scale
                ),
                rtm_motion=motion_stats(
                    rt,
                    valid & np.isfinite(rt).all(axis=-1),
                    np.array(idx),
                    times,
                    a["scene_id"][idx],
                    a["segment_id"][idx],
                    scale,
                ),
                high_conf_disagreement=[
                    dict(
                        frame=idx[t],
                        joint=j,
                        distance_torso=float(difference[t, j]),
                        mp_scores=conf[t, j].tolist(),
                        rtm_score=float(rs[t, j]),
                    )
                    for t in range(len(idx))
                    for j in range(6)
                    if inside[t] and np.all(conf[t, j] >= 0.8) and difference[t, j] > 0.3
                ],
                sheet=case["id"].replace(":", "-") + ".jpg",
            )
            np.savez_compressed(
                output / (case["id"].replace(":", "-") + ".npz"),
                frame_index=idx,
                mp_xy=mp,
                mp_confidence=conf,
                mp_valid=valid,
                rtm_xy=rt,
                rtm_score=rs,
            )
            sheet(rec, idx, images, mp, rt, valid, rs, output / rec["sheet"])
            results.append(rec)
            print(v, case["id"], len(idx), "frames", flush=True)
        (output / "cases.json").write_text(json.dumps(results, indent=2) + "\n")
    for path, h in protection.items():
        if file_hash(path) != h:
            raise ValueError(f"Protected file changed: {path}")
    summary = dict(
        schema_version="pose-estimator-audit-result-1.0",
        cases=len(results),
        frames=len(latencies),
        model_sha256=file_hash(model),
        config_sha256=file_hash(config),
        opencv=cv2.__version__,
        rtm_inference_ms_p50=float(np.median(latencies) * 1000),
        rtm_inference_ms_p95=float(np.quantile(latencies, 0.95) * 1000),
        preservation_verified=len(protection),
        source="automated",
        no_ground_truth_keypoints=True,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "protected.json").write_text(json.dumps(protection, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.config, args.model, args.output)
