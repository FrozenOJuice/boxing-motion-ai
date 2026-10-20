"""Reproducible local extraction with explicit partial/failure states and provenance."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from bisect import bisect_right
from collections import Counter
from pathlib import Path

from .annotations import read_v5_annotations
from .contracts import BODY_JOINTS, JOINTS, pose_record
from .decoder import decode_video
from .mediapipe_pose import MediaPipePose
from .overlay import contact_sheet, save_overlay


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_ranges(values):
    """Half-open *extraction* ranges; never an assertion about annotation ends."""
    ranges = []
    for value in values:
        try:
            start, stop = (int(v) for v in value.split(":"))
        except (ValueError, TypeError) as exc:
            raise ValueError("Ranges must be START:STOP integer decoded indices") from exc
        if start < 0 or stop <= start:
            raise ValueError("Ranges require 0 <= START < STOP (exclusive)")
        ranges.append((start, stop))
    ranges.sort()
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:], strict=False)):
        raise ValueError("Extraction ranges must not overlap")
    return ranges


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def code_provenance():
    root = Path(__file__).resolve().parents[3]
    files = sorted(Path(__file__).parent.glob("*.py"))
    files += [root / "pyproject.toml", root / "uv.lock", Path(__file__).parent.parent / "cli.py"]
    result = {"sha256": {p.relative_to(root).as_posix(): sha256(p) for p in files if p.exists()}}
    try:
        result["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        result["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            )
        )
    except (OSError, subprocess.SubprocessError):
        result["git_revision"] = None
    return result


def extract_v5(
    video,
    workbook,
    model,
    output,
    ranges=(),
    max_width=960,
    confidence=0.5,
    overlay_every=30,
    overlay_limit=18,
    decoder=decode_video,
    backend_factory=MediaPipePose,
    progress=lambda count: None,
    pose_mode="image",
    scene_cuts=(),
    overlay_frames=(),
):
    video, workbook, model, output = map(Path, (video, workbook, model, output))
    if not all(p.is_file() for p in (video, workbook, model)):
        raise ValueError("Video, V5 workbook and local pretrained model must all exist")
    if video.stem.casefold() != "v5" or workbook.stem.casefold() != "v5":
        raise ValueError(
            "This initial pipeline supports only explicitly named V5 video/workbook inputs"
        )
    if output.exists():
        raise ValueError("Output already exists; choose a new directory to preserve prior runs")
    if any(output.resolve().is_relative_to(p.resolve().parent) for p in (video, workbook, model)):
        raise ValueError("Output must be outside source video, workbook and model directories")
    if any(p.name == "raw" for p in output.resolve().parents):
        raise ValueError("Output must be outside raw dataset directories")
    if max_width < 64 or not 0 <= confidence <= 1 or overlay_every < 1 or overlay_limit < 0:
        raise ValueError("Invalid width, confidence, overlay interval or overlay limit")
    if pose_mode not in {"image", "video"}:
        raise ValueError("Pose mode must be image or video")
    cuts = list(scene_cuts)
    if any(type(n) is not int or n <= 0 for n in cuts) or cuts != sorted(set(cuts)):
        raise ValueError("Scene cuts must be unique, increasing positive decoded indices")
    if any(type(n) is not int or n < 0 for n in overlay_frames):
        raise ValueError("Overlay frames must be nonnegative decoded indices")
    ranges = frame_ranges([f"{a}:{b}" for a, b in ranges])
    annotations = read_v5_annotations(workbook)
    versions = {}
    for name in ("av", "mediapipe", "numpy", "pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = {
        "schema_version": "pose-run-1.1",
        "status": "running",
        "video_id": "V5",
        "inputs": {
            name: {"filename": path.name, "sha256": sha256(path)}
            for name, path in [("video", video), ("workbook", workbook), ("model", model)]
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "code": code_provenance(),
        "config": {
            "ranges_half_open": ranges,
            "pose_mode": pose_mode,
            "scene_cuts": cuts,
            "overlay_frames": list(overlay_frames),
            "max_width": max_width,
            "confidence": confidence,
            "overlay_every": overlay_every,
            "overlay_limit": overlay_limit,
            "delegate": "CPU",
            "num_poses": 1,
        },
        "decoder": "PyAV sequential software decode in an isolated local worker; no seek/resample",
        "pose_contract": {
            "schema_version": "pose-1.1",
            "joint_order": JOINTS,
            "coordinates": "MediaPipe normalized image x/y, raw hip-relative z in x-scale units",
            "normalization": "image width/height only; no body normalization or world coordinates",
            "confidence": "raw visibility and presence, not calibrated accuracy",
            "validity": "finite x/y/z/confidences; in-image x/y; both scores >= threshold and <=1",
            "missing": "33 null landmarks and 33 false mask entries; no interpolation or carry",
            "handedness": "anatomical left/right estimator output; lead/rear and stance unknown",
            "identity": "one pose maximum; boxer identity unknown, no identity tracking",
            "time": "exact integer pts * rational time_base; timestamp_seconds is convenience",
            "estimator_clock": "rounded relative PTS milliseconds, reset at each extraction range",
            "tracking": (
                "IMAGE: independent frames, no tracking/smoothing"
                if pose_mode == "image"
                else "VIDEO: current/past tracking, reset at range gaps and listed cuts"
            ),
            "scene_cuts": "First frame after each listed cut; inventory may be incomplete",
            "interpolation": "none",
        },
        "annotation_mapping": {k: v for k, v in annotations.items() if k != "rows"},
        "licensing_and_redistribution": "unverified; local review artifacts only",
        "processed_frame_count": 0,
        "source_frame_count": None,
    }
    output.mkdir(parents=True)
    (output / "overlays").mkdir()
    write_json(output / "annotations.provisional.json", annotations)
    write_json(output / "manifest.json", manifest)
    counts, joint_valid = Counter(), [0] * 33
    overlays, backend, segment, origin, previous_ms = [], None, None, None, None
    decoded_count, processed, last_time = 0, 0, None
    active_key, segment_number = None, -1
    frames = None
    try:
        frames = decoder(video, max_width)
        with (output / "poses.jsonl.partial").open("w") as stream:
            for frame in frames:
                if frame.index != decoded_count:
                    raise ValueError("Decoder returned non-sequential frame indices")
                decoded_count += 1
                if last_time is not None and frame.time <= last_time:
                    raise ValueError("Decoder returned non-increasing PTS")
                last_time = frame.time
                if ranges and frame.index >= ranges[-1][1]:
                    break
                current = next((i for i, (a, b) in enumerate(ranges) if a <= frame.index < b), None)
                if ranges and current is None:
                    continue
                current = current if ranges else 0
                scene_id = bisect_right(cuts, frame.index)
                key = (current, scene_id)
                if backend is None or key != active_key:
                    if backend is not None:
                        backend.close()
                        backend = None
                    backend = backend_factory(model, confidence, mode=pose_mode)
                    active_key = key
                    segment_number += 1
                    segment, origin, previous_ms = segment_number, frame.time, None
                ms = round((frame.time - origin) * 1000)
                if previous_ms is not None and ms <= previous_ms:
                    raise ValueError(
                        "PTS cannot be represented as increasing estimator milliseconds"
                    )
                previous_ms = ms
                landmarks = backend.detect(frame.rgb, ms)
                record = pose_record(frame, landmarks, ms, confidence)
                record.update(
                    schema_version="pose-1.1",
                    segment_id=segment,
                    scene_id=scene_id,
                    scene_cut_before=frame.index in cuts,
                )
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                counts[record["status"]] += 1
                for i, valid in enumerate(record["valid_mask"]):
                    joint_valid[i] += valid
                processed += 1
                if len(overlays) < overlay_limit and (
                    frame.index in overlay_frames
                    or (
                        not overlay_frames
                        and (
                            ms == 0
                            or (frame.index - (ranges[current][0] if ranges else 0)) % overlay_every
                            == 0
                        )
                    )
                ):
                    path = output / "overlays" / f"frame_{frame.index:06d}.jpg"
                    save_overlay(frame, record, path, annotations["rows"])
                    overlays.append(path)
                if processed % 100 == 0:
                    progress(processed)
            if not processed:
                raise ValueError("No requested frames decoded")
            if ranges and decoded_count < ranges[-1][1]:
                raise ValueError("Requested extraction range extends beyond decoded video")
        if cuts and cuts[-1] >= decoded_count and not ranges:
            raise ValueError("Scene cut exceeds decoded video bounds")
        backend.close()
        backend = None
        if not ranges:
            manifest["source_frame_count"] = decoded_count
            for row in annotations["rows"]:
                if any(
                    row[k] is not None and row[k] >= decoded_count
                    for k in ("candidate_start_index", "candidate_end_index")
                ):
                    row["flags"].append("out_of_video_bounds")
            write_json(output / "annotations.provisional.json", annotations)
        if overlays:
            contact_sheet(overlays, output / "skeleton-contact-sheet.jpg")
        (output / "poses.jsonl.partial").replace(output / "poses.jsonl")
        manifest["status"] = "complete"
        manifest["scope"] = "selected_windows" if ranges else "full_video"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for resource in (backend, frames):
            if resource is not None and hasattr(resource, "close"):
                try:
                    resource.close()
                except Exception as exc:
                    manifest["cleanup_error"] = str(exc)
        manifest["processed_frame_count"] = processed
        manifest["status_counts"] = dict(counts)
        manifest["valid_joint_fraction"] = {
            JOINTS[i]: joint_valid[i] / processed if processed else None for i in range(33)
        }
        manifest["valid_body_joint_fraction"] = (
            sum(joint_valid[i] for i in BODY_JOINTS) / (processed * 22) if processed else None
        )
        manifest["overlay_files"] = [p.relative_to(output).as_posix() for p in overlays]
        write_json(output / "manifest.json", manifest)
    return manifest
