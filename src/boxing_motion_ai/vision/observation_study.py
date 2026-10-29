"""Bounded offline pose-observation study; never registered as production extraction.

Prepare pixels separately from inference to isolate PyAV and MediaPipe native libraries.
References are used only by evaluate(), after annotation-free extraction/segmentation.
"""

import argparse
import importlib.metadata
import json
import shutil
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from ..analysis.common import sha, write_json
from ..detection import instances_v2 as v2
from ..detection.instance_diagnostics import UPLOADS, arrays, compare
from ..training.data import validate_arrays
from .contracts import DecodedFrame, pose_record
from .estimator_audit import COCO, MP, case_box

VERSION = "pose-observation-study-1.0"
BASE = Path("reports/boxingvi/punch-instances-v2-repeat")
VARIANTS = ("image960", "image1920", "crop960", "crop960-video", "rtmpose-crop")
CASES = (
    tuple(UPLOADS)
    + tuple(f"{r}-motion-{i}" for r in ("V4", "V6") for i in range(1, 7))
    + tuple(f"V4-bg-{i}" for i in range(1, 4))
)


def fresh(path):
    path = Path(path)
    if path.exists() or not path.resolve().is_relative_to(Path.cwd() / "reports"):
        raise ValueError("Fresh ignored report path required")
    path.mkdir(parents=True)
    return path


def read(path):
    return json.loads(Path(path).read_text())


def study_manifest(output):
    if not Path(output).resolve().is_relative_to(Path.cwd() / "reports"):
        raise ValueError("Use the ignored study directory")
    manifest = read(Path(output) / "manifest.json")
    if (
        manifest["schema_version"] != VERSION
        or set(manifest["cases"]) != set(CASES)
        or tuple(manifest["variants"]) != VARIANTS
        or manifest["segmentation_code_sha256"] != sha(Path(v2.__file__))
        or manifest["segmentation_policy_sha256"] != sha(Path("configs/punch_instances_v2.json"))
    ):
        raise ValueError("Study inventory or frozen segmentation changed")
    return manifest


def prepare(output):
    from ..training.rgb_review import decode_requested

    out = fresh(output)
    names = read(BASE / "cases.json")
    if set(names) != set(CASES) or len(names) != len(CASES):
        raise ValueError("Unexpected diagnostic inventory")
    # Protect retained artifacts before reading source pixels; never enumerate V9.
    protected = read("reports/boxingvi/punch-instances-v2-diagnostics/protected-before.json")
    for path in BASE.rglob("*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    for name, digest in protected.items():
        if "v9" in name.lower() or sha(Path(name)) != digest:
            raise ValueError("Protected input changed or forbidden source")
    write_json(out / "protected-before.json", protected)
    inventory = {}
    for name in names:
        source = BASE / name
        raw = arrays(source / "poses.npz")
        video, prov = read(source / "video.json"), read(source / "provenance.json")
        dest = out / name
        (dest / "frames").mkdir(parents=True)
        if name in UPLOADS:
            upload = Path(UPLOADS[name])
            media = upload / "source.video"
            original = read(upload / "video.json")
            if original != video or sha(media) != prov["source_sha256"]:
                raise ValueError("Uploaded source changed")
            # Prepared oriented lossless RGB is exactly what production inference received.
            for i in range(len(raw["pts"])):
                shutil.copy2(upload / "frames" / f"{i:04d}.png", dest / "frames" / f"{i:04d}.png")
            frame_origin = "existing lossless oriented upload frames"
        else:
            recording = prov["recording"]
            if recording not in ("V4", "V6"):
                raise ValueError("Forbidden recording")
            media = Path(f"data/raw/boxingvi/RGB_videos/{recording}.mkv")
            if sha(media) != prov["source_video_sha256"]:
                raise ValueError("Native video changed")
            requests = {i: Q(r["timestamp_exact"]) for i, r in enumerate(video["frames"])}
            for i, image, _ in decode_requested(media, requests):
                if image.size != (video["width"], video["height"]):
                    raise ValueError("Native pixel dimensions changed")
                image.save(dest / "frames" / f"{i:04d}.png", compress_level=1)
            frame_origin = "exact native PTS decode, full resolution"
        # Fixed entire-case crop; uses baseline masks only, no event labels. Offline upper bound.
        box = [
            int(x)
            for x in case_box(raw, list(range(len(raw["pts"]))), (video["width"], video["height"]))
        ]
        identity = dict(
            source=prov,
            source_media=str(media),
            source_sha256=sha(media),
            frame_origin=frame_origin,
            crop=box,
            baseline_sha256={
                p: sha(source / p)
                for p in ("poses.npz", "video.json", "activity.json", "references.json")
            },
            pixel_sha256={p.name: sha(p) for p in sorted((dest / "frames").glob("*.png"))},
        )
        write_json(dest / "input.json", identity)
        inventory[name] = sha(dest / "input.json")
    manifest = dict(
        schema_version=VERSION,
        cases=inventory,
        variants=VARIANTS,
        segmentation_policy_sha256=sha(Path("configs/punch_instances_v2.json")),
        crop_policy="baseline valid-landmark case union plus 15 percent; offline",
        production_changed=False,
        code_sha256=sha(Path(__file__)),
        segmentation_code_sha256=sha(Path(v2.__file__)),
    )
    write_json(out / "manifest.json", manifest)


def map_landmarks(points, box, size):
    """Inverse crop transform; keep confidence and out-of-crop missingness explicit."""
    if points is None:
        return None
    bx, by, ex, ey = box
    width, height = size
    result = []
    for x, y, z, *scores in points:
        # In-crop coordinates required before mapping, even if outside crop is inside source.
        if not (0 <= x <= 1 and 0 <= y <= 1):
            scores = [0.0, 0.0]
        result.append(
            (
                (bx + x * (ex - bx)) / width,
                (by + y * (ey - by)) / height,
                z * (ex - bx) / width,
                *scores,
            )
        )
    return result


def reset_before(raw, video, activity):
    times = [Q(r["timestamp_exact"]) for r in video["frames"]]
    gap = Q(activity["effective_reset_gap_seconds_exact"])
    cuts = set(video.get("confirmed_scene_cuts", []))
    return [
        i == 0
        or i in cuts
        or activity["frames"][i]["discontinuity_before"]
        or (
            i > 0
            and (
                times[i] - times[i - 1] > gap
                or raw["scene_id"][i] != raw["scene_id"][i - 1]
                or raw["segment_id"][i] != raw["segment_id"][i - 1]
            )
        )
        for i in range(len(times))
    ]


def extract(output, variant):
    from .mediapipe_pose import MediaPipePose

    out = Path(output)
    manifest = study_manifest(out)
    if variant not in VARIANTS:
        raise ValueError("Unknown study variant/inventory")
    pin = read("configs/assisted_level1_v1.json")
    model = Path(pin["pose_model"]["path"])
    if sha(model) != pin["pose_model"]["sha256"]:
        raise ValueError("MediaPipe model changed")
    rt = variant == "rtmpose-crop"
    if rt:
        import cv2
        from rtmlib import RTMPose

        cv2.setNumThreads(2)
        model = Path("weights/pose-audit/rtmpose-m-body7.onnx")
        if sha(model) != "5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c":
            raise ValueError("RTMPose model changed")
        estimator = RTMPose(str(model), model_input_size=(192, 256), backend="opencv", device="cpu")
    for name, digest in manifest["cases"].items():
        case_began = perf_counter()
        case = out / name
        if sha(case / "input.json") != digest:
            raise ValueError("Input identity changed")
        identity = read(case / "input.json")
        source = BASE / name
        for p, h in identity["baseline_sha256"].items():
            if sha(source / p) != h:
                raise ValueError("Baseline changed")
        dest = fresh(case / variant)
        raw = arrays(source / "poses.npz")
        video, activity = read(source / "video.json"), read(source / "activity.json")
        size = (video["width"], video["height"])
        box = identity["crop"] if "crop" in variant else [0, 0, *size]
        resets = reset_before(raw, video, activity)
        backend = None
        poses, confidence, masks, statuses, timings, rt_xy, rt_scores = [], [], [], [], [], [], []
        original_landmarks = []
        origin = None
        try:
            for i, row in enumerate(video["frames"]):
                path = case / "frames" / f"{i:04d}.png"
                if sha(path) != identity["pixel_sha256"][path.name]:
                    raise ValueError("Pixels changed")
                with Image.open(path) as im:
                    image = im.convert("RGB")
                if rt:
                    began = perf_counter()
                    k, s = estimator(np.asarray(image), bboxes=[box])
                    timings.append(perf_counter() - began)
                    rt_xy.append(k[0, COCO])
                    rt_scores.append(s[0, COCO])
                    continue
                if resets[i]:
                    if backend is not None:
                        backend.close()
                    backend = MediaPipePose(
                        model, mode="video" if variant.endswith("video") else "image"
                    )
                    origin = Q(row["timestamp_exact"])
                image = image.crop(box)
                cap = 1920 if variant == "image1920" else 960
                if image.width > cap:
                    image = image.resize(
                        (cap, round(image.height * cap / image.width)), Image.Resampling.BILINEAR
                    )
                rgb = np.asarray(image)
                ms = round((Q(row["timestamp_exact"]) - origin) * 1000)
                began = perf_counter()
                points = backend.detect(rgb, ms)
                timings.append(perf_counter() - began)
                original_landmarks.append(
                    points if points is not None else np.full((33, 5), np.nan)
                )
                points = map_landmarks(points, box, size)
                f = DecodedFrame(
                    i, row["pts"], Q(row["time_base_num"], row["time_base_den"]), *size, rgb
                )
                record = pose_record(f, points, ms, 0.5)
                poses.append(
                    [
                        [p[k] if p and p[k] is not None else np.nan for k in ("x", "y", "z")]
                        for p in record["landmarks"]
                    ]
                )
                confidence.append(
                    [
                        [
                            p[k] if p and p[k] is not None else np.nan
                            for k in ("visibility", "presence")
                        ]
                        for p in record["landmarks"]
                    ]
                )
                masks.append(record["valid_mask"])
                statuses.append(2 if points is None else (0 if record["status"] == "ok" else 1))
        finally:
            if backend is not None:
                backend.close()
        if rt:
            np.savez_compressed(
                dest / "raw-rtmpose.npz", xy=np.array(rt_xy), scores=np.array(rt_scores)
            )
        else:
            raw.update(
                coordinates=np.array(poses, np.float32),
                confidence=np.array(confidence, np.float32),
                valid_mask=np.array(masks, bool),
                pose_status=np.array(statuses, np.int64),
            )
            validate_arrays(raw)
            np.savez_compressed(dest / "poses.npz", **raw)
            np.savez_compressed(dest / "raw-landmarks.npz", landmarks=np.array(original_landmarks))
        write_json(
            dest / "provenance.json",
            dict(
                schema_version=VERSION,
                variant=variant,
                model_sha256=sha(model),
                input_sha256=digest,
                crop=box,
                coordinate_system="original source normalized image XY; MP Z scaled by crop width",
                mode="video" if variant.endswith("video") else "image",
                timing_seconds=timings,
                reset_frames=np.flatnonzero(resets).tolist(),
                no_gap_filling=True,
                processing_seconds=perf_counter() - case_began,
                classifier_compatibility="unvalidated; not used for recognition",
                code_sha256=sha(Path(__file__)),
                versions={
                    **{k: importlib.metadata.version(k) for k in ("numpy", "Pillow", "mediapipe")},
                    **(
                        {"rtmlib": importlib.metadata.version("rtmlib"), "opencv": cv2.__version__}
                        if rt
                        else {}
                    ),
                },
                output_sha256=sha(dest / ("raw-rtmpose.npz" if rt else "poses.npz")),
                original_landmarks_sha256=sha(dest / "raw-landmarks.npz") if not rt else None,
                preprocessing="Pillow RGB bilinear width cap, no upscaling"
                if not rt
                else "existing RTMPose RGB affine preprocessing, 192x256, raw scores",
            ),
        )
        print(f"{variant} {name}: {len(timings)} frames {sum(timings):.2f}s", flush=True)


def mask_matched_rtmpose(raw, xy, size):
    """Coordinate-only sensitivity, NOT an RTMPose validity contract or production poses."""
    result = {k: v.copy() for k, v in raw.items()}
    normalized = xy / np.asarray(size)
    for j, mp in enumerate(MP):
        ok = raw["valid_mask"][:, mp] & np.isfinite(normalized[:, j]).all(axis=1)
        ok &= ((normalized[:, j] >= 0) & (normalized[:, j] <= 1)).all(axis=1)
        # Do not manufacture visibility/presence: preserve the original masks and confidence.
        result["coordinates"][ok, mp, :2] = normalized[ok, j]
    return result


def coverage(data, arm_motion=None):
    result = {}
    for arm, t in data["arms"].items():
        reset = data["reset"].copy()
        reset[(arm_motion or {}).get(arm, {}).get("tracking_jump_resets", [])] = True
        result[arm] = {}
        for label, mask in (
            ("wrist", t["wrist_valid"]),
            ("elbow", t["elbow_valid"]),
            ("complete", t["wrist_valid"] & t["elbow_valid"]),
        ):
            longest = max(
                (
                    float(
                        (data["times"][z] if z < len(mask) else data["stop_time"])
                        - data["times"][a]
                    )
                    for a, z in v2.runs(mask, reset)
                ),
                default=0,
            )
            result[arm][label] = dict(
                frames=int(mask.sum()), fraction=float(mask.mean()), longest_seconds=longest
            )
    return result


def evaluate(output):
    out = Path(output)
    manifest = study_manifest(out)
    policy = v2.Policy.model_validate(read("configs/punch_instances_v2.json"))
    results = []
    for name in manifest["cases"]:
        source, case = BASE / name, out / name
        if sha(case / "input.json") != manifest["cases"][name]:
            raise ValueError("Study input identity changed")
        identity = read(case / "input.json")
        for filename, digest in identity["baseline_sha256"].items():
            if sha(source / filename) != digest:
                raise ValueError("Study baseline changed")
        baseline = arrays(source / "poses.npz")
        video, activity = read(source / "video.json"), read(source / "activity.json")
        for variant in ("stored", *VARIANTS):
            if variant != "stored":
                provenance = read(case / variant / "provenance.json")
                filename = "raw-rtmpose.npz" if variant == "rtmpose-crop" else "poses.npz"
                if (
                    provenance["input_sha256"] != manifest["cases"][name]
                    or sha(case / variant / filename) != provenance["output_sha256"]
                ):
                    raise ValueError("Study output changed")
            if variant == "stored":
                raw = baseline
            elif variant == "rtmpose-crop":
                rt = arrays(case / variant / "raw-rtmpose.npz")
                raw = mask_matched_rtmpose(baseline, rt["xy"], (video["width"], video["height"]))
            else:
                raw = arrays(case / variant / "poses.npz")
            data = v2.motion(raw, video, activity)
            segmentation = v2.segment_motion(data, policy)
            # Only now may existing human event bounds enter the report.
            refs, prov = read(source / "references.json"), read(source / "provenance.json")
            core = (
                (
                    prov["review_start"] - prov["source_start"],
                    prov["review_stop"] - prov["source_start"],
                )
                if prov["kind"] == "reviewed_source"
                else None
            )
            events = []
            for ref in refs:
                a, z = ref["start_frame"], min(len(raw["pts"]), ref["stop_frame_exclusive"])
                events.append(
                    dict(
                        **ref,
                        coverage={
                            arm: {
                                k: float(t[k][a:z].mean()) for k in ("wrist_valid", "elbow_valid")
                            }
                            for arm, t in data["arms"].items()
                        },
                    )
                )
            report = dict(
                case=name,
                variant=variant,
                coverage=coverage(data, segmentation["arm_motion"]),
                arm_motion=segmentation["arm_motion"],
                events=events,
                candidates=segmentation["candidates"],
                rejections=dict(Counter(r["reason"] for r in segmentation["rejected_proposals"])),
                comparison=compare(
                    segmentation["candidates"], refs, data["times"], data["stop_time"], core
                )
                if core
                else None,
                interpretation="mask-matched coordinate sensitivity only"
                if variant == "rtmpose-crop"
                else "mask availability is not localization accuracy",
            )
            if report["comparison"]:
                errors = []
                for match in report["comparison"]["matches_iou_half"]:
                    ref = next(r for r in refs if r["event_id"] == match["reference_id"])
                    pred = segmentation["candidates"][match["candidate_index"]]
                    errors.append(
                        dict(
                            event_id=ref["event_id"],
                            start_error_frames=pred["start_frame"] - ref["start_frame"],
                            stop_error_frames=pred["stop_frame_exclusive"]
                            - ref["stop_frame_exclusive"],
                        )
                    )
                report["matched_full_boundary_errors"] = errors
            results.append(report)
            write_json(case / f"{variant}-comparison.json", report)
    write_json(out / "comparison.json", results)
    summaries = []
    for variant in ("stored", *VARIANTS):
        native = [r for r in results if r["variant"] == variant and r["case"].startswith("V4-bg-")]
        negative = [r for r in results if r["variant"] == variant and "-motion-" in r["case"]]
        timings, wall = [], 0
        if variant != "stored":
            for name in manifest["cases"]:
                prov = read(out / name / variant / "provenance.json")
                timings.extend(prov["timing_seconds"])
                wall += prov["processing_seconds"]
        summaries.append(
            dict(
                variant=variant,
                upload_candidates=[
                    len(
                        next(r for r in results if r["variant"] == variant and r["case"] == name)[
                            "candidates"
                        ]
                    )
                    for name in ("V6-upload", "V4-8s-upload", "V4-29s-upload")
                ],
                native_candidates=sum(r["comparison"]["reviewed_candidate_count"] for r in native),
                matches=sum(len(r["comparison"]["matches_iou_half"]) for r in native),
                missed=sum(len(r["comparison"]["missed_references"]) for r in native),
                background_proposals=sum(
                    len(r["comparison"]["candidate_without_reference_overlap"]) for r in native
                ),
                merge_risks=sum(len(r["comparison"]["merge_risks"]) for r in native),
                split_risks=sum(len(r["comparison"]["split_risks"]) for r in native),
                negative_candidates=sum(len(r["candidates"]) for r in negative),
                inference_ms_median=float(np.median(timings) * 1000) if timings else None,
                inference_ms_p95=float(np.quantile(timings, 0.95) * 1000) if timings else None,
                processing_seconds=wall if timings else None,
            )
        )
    write_json(out / "summary.json", summaries)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("stage", choices=("prepare", "extract", "evaluate"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=VARIANTS)
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args.output)
    elif args.stage == "extract":
        extract(args.output, args.variant)
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()
