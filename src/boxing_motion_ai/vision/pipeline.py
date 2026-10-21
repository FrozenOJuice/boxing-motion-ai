"""Reproducible local extraction with explicit partial/failure states and provenance."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from bisect import bisect_right
from collections import Counter
from pathlib import Path

from .annotations import read_annotations
from .contracts import BODY_JOINTS, JOINTS, pose_record
from .decoder import decode_video
from .mediapipe_pose import MediaPipePose
from .overlay import contact_sheet, save_overlay
from .recording import v5_recording


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


def _extract_recording(
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
    recording=None,
    resume=False,
):
    recording = recording or v5_recording()
    video, workbook, model, output = map(Path, (video, workbook, model, output))
    if not all(p.is_file() for p in (video, workbook, model)):
        raise ValueError("Video, V5 workbook and local pretrained model must all exist")
    if any(p.stem.casefold() != recording.video_id.casefold() for p in (video, workbook)):
        raise ValueError("Video/workbook filenames must match the configured recording ID")
    if output.exists() and not resume:
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
    annotations = read_annotations(workbook, recording)
    versions = {}
    for name in ("av", "mediapipe", "numpy", "pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = {
        "schema_version": "pose-run-1.1",
        "resume_unit": "complete newline JSONL record; IMAGE only; replay decode verifies timing",
        "status": "running",
        "video_id": recording.video_id,
        "recording_id": recording.video_id,
        "boxer_id": recording.boxer_id,
        "session_id": recording.session_id,
        "recording_config": recording.model_dump(),
        "inputs": {
            name: {"filename": path.name, "sha256": sha256(path)}
            for name, path in [("video", video), ("workbook", workbook), ("model", model)]
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "code": code_provenance(),
        "config": {
            "ranges_half_open": [list(pair) for pair in ranges],
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
    cached, counts, joint_valid = [], Counter(), [0] * 33
    overlays = []
    if output.exists():
        from .resume import compatible, recover_prefix

        previous = json.loads((output / "manifest.json").read_text())
        compatible(previous, manifest)
        if previous["status"] == "complete":
            if sha256(output / "poses.jsonl") != previous.get("poses_sha256"):
                raise ValueError("Completed pose file hash differs or predates resumable format")
            return previous
        partial = output / "poses.jsonl.partial"
        # Native exit between rename and manifest completion: recover the completed journal.
        if not partial.exists() and (output / "poses.jsonl").exists():
            (output / "poses.jsonl").replace(partial)
        if not partial.exists():
            raise ValueError("No resumable pose journal found")
        cached, counts, joint_valid = recover_prefix(partial, manifest)
        manifest["resume_history"] = previous.get("resume_history", []) + [
            {"reused_records": len(cached), "prior_status": previous["status"]}
        ]
        overlays = sorted((output / "overlays").glob("frame_*.jpg"))
    else:
        output.mkdir(parents=True)
        (output / "overlays").mkdir()
    write_json(output / "annotations.provisional.json", annotations)
    write_json(output / "manifest.json", manifest)
    backend, segment, origin, previous_ms = None, None, None, None
    decoded_count, processed, last_time = 0, len(cached), None
    active_key, segment_number, replayed = None, -1, 0
    frames = None
    try:
        frames = decoder(video, max_width)
        with (output / "poses.jsonl.partial").open("a") as stream:
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
                if key != active_key:
                    if backend is not None:
                        backend.close()
                        backend = None
                    active_key = key
                    segment_number += 1
                    segment, origin, previous_ms = segment_number, frame.time, None
                ms = round((frame.time - origin) * 1000)
                if previous_ms is not None and ms <= previous_ms:
                    raise ValueError(
                        "PTS cannot be represented as increasing estimator milliseconds"
                    )
                previous_ms = ms
                if replayed < len(cached):
                    expected = cached[replayed]
                    observed = {
                        "frame_index": frame.index,
                        "pts": frame.pts,
                        "time_base": str(frame.time_base),
                        "scene_id": scene_id,
                        "segment_id": segment,
                        "scene_cut_before": frame.index in cuts,
                        "estimator_timestamp_ms": ms,
                        "source_width": frame.source_width,
                        "source_height": frame.source_height,
                        "input_width": frame.rgb.shape[1],
                        "input_height": frame.rgb.shape[0],
                        "recording_id": recording.video_id,
                        "boxer_id": recording.boxer_id,
                        "session_id": recording.session_id,
                    }
                    if expected != observed:
                        raise ValueError(f"Resume replay mismatch at source frame {frame.index}")
                    replayed += 1
                    continue
                if backend is None:
                    backend = backend_factory(model, confidence, mode=pose_mode)
                landmarks = backend.detect(frame.rgb, ms)
                record = pose_record(frame, landmarks, ms, confidence)
                record.update(
                    schema_version="pose-1.1",
                    video_id=recording.video_id,
                    recording_id=recording.video_id,
                    boxer_id=recording.boxer_id,
                    session_id=recording.session_id,
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
                    stream.flush()
                    os.fsync(stream.fileno())
                    manifest["processed_frame_count"] = processed
                    write_json(output / "manifest.json", manifest)
                    progress(processed)
            if replayed != len(cached):
                raise ValueError("Decoder ended before the cached resume prefix")
            stream.flush()
            os.fsync(stream.fileno())
            if not processed:
                raise ValueError("No requested frames decoded")
            if ranges and decoded_count < ranges[-1][1]:
                raise ValueError("Requested extraction range extends beyond decoded video")
        if cuts and cuts[-1] >= decoded_count and not ranges:
            raise ValueError("Scene cut exceeds decoded video bounds")
        if backend is not None:
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
        manifest["poses_sha256"] = sha256(output / "poses.jsonl")
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


def extract_recording(video, workbook, model, output, *args, **kwargs):
    """Prevent two writers to one journal; OS releases the lock after native crashes."""
    import fcntl

    output = Path(output)
    sources = [Path(p) for p in (video, workbook, model)]
    if not all(p.is_file() for p in sources):
        raise ValueError("Video, workbook and local pretrained model must all exist")
    if any(output.resolve().is_relative_to(p.resolve().parent) for p in sources) or any(
        p.name == "raw" for p in output.resolve().parents
    ):
        raise ValueError("Output must be outside source directories and outside raw directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another extraction is already writing this output") from exc
        try:
            return _extract_recording(video, workbook, model, output, *args, **kwargs)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def extract_v5(*args, **kwargs):
    """Compatibility entry point for the original V5 prototype."""
    return extract_recording(*args, **kwargs, recording=v5_recording())
