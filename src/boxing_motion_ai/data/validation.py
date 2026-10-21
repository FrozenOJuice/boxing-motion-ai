"""Restartable orchestration of existing local sampling/extraction/preparation commands."""

import json
import subprocess
import sys
from pathlib import Path

from ..vision.config import load_config
from ..vision.pipeline import frame_ranges, sha256, write_json
from ..vision.recording import load_recording
from ..vision.sampling import plan_sample
from .pose_quality import assess_quality
from .preparation import prepare_recording


def completed_run_matches(run, video, workbook, model, recording, plan):
    """Completed immutable artifacts remain reusable after unrelated implementation edits."""
    path = run / "manifest.json"
    if not path.exists():
        return False
    manifest = json.loads(path.read_text())
    if manifest["status"] != "complete":
        return False
    settings = load_config(plan / "pose-config.json")
    expected = settings.model_dump()
    expected["ranges_half_open"] = [list(p) for p in frame_ranges(expected.pop("ranges"))]
    if manifest["recording_config"] != recording.model_dump() or any(
        manifest["config"].get(k) != v for k, v in expected.items()
    ):
        raise ValueError("Completed extraction settings differ from requested sample")
    if any(
        manifest["inputs"][name]["sha256"] != sha256(path)
        for name, path in [("video", video), ("workbook", workbook), ("model", model)]
    ):
        raise ValueError("Completed extraction input hashes differ")
    if sha256(run / "poses.jsonl") != manifest.get("poses_sha256"):
        raise ValueError("Completed extraction journal hash differs")
    return True


def validate_samples(root, processed, output, model, recordings, per_class=5, progress=print):
    root, processed, output, model = map(Path, (root, processed, output, model))
    if any(p.resolve().is_relative_to(root.resolve()) for p in (processed, output)):
        raise ValueError("Generated outputs must be outside raw dataset root")
    if not model.is_file():
        raise ValueError("Local pose model is missing; nothing is downloaded")
    output.mkdir(parents=True, exist_ok=True)
    (output / "jobs").mkdir(exist_ok=True)
    state = {"schema_version": "sample-validation-batch-1.0", "recordings": {}}
    for config_path in recordings:
        config_path = Path(config_path)
        recording = load_recording(config_path)
        video_id = recording.video_id
        entry = {
            "recording_id": video_id,
            "boxer_id": recording.boxer_id,
            "session_id": recording.session_id,
            "status": "running",
        }
        state["recordings"][video_id] = entry
        write_json(output / "batch-state.json", state)
        progress(f"{video_id}: validating local sample")
        try:
            workbook = root / "Annotation_files" / f"{video_id}.xlsx"
            video = root / "RGB_videos" / f"{video_id}.mkv"
            plan = output / "plans" / video_id
            if plan.exists():
                prior = json.loads((plan / "sample-plan.json").read_text())
                if (
                    prior["recording"] != recording.model_dump()
                    or prior["workbook_sha256"] != sha256(workbook)
                    or prior["per_class_requested"] != per_class
                ):
                    raise ValueError("Existing sample plan differs; choose a new output root")
            else:
                plan_sample(workbook, recording, plan, per_class)
            run = processed / video_id / "pose-sample-expansion-v1"
            dataset = processed / video_id / "candidates-sample-expansion-v1"
            quality = output / "quality" / video_id
            command = [
                sys.executable,
                "-c",
                "from boxing_motion_ai.cli import app; app()",
                "extract-poses",
                "--video",
                str(video),
                "--workbook",
                str(workbook),
                "--model",
                str(model),
                "--recording",
                str(config_path),
                "--config",
                str(plan / "pose-config.json"),
                "--output",
                str(run),
                "--resume",
            ]
            if not completed_run_matches(run, video, workbook, model, recording, plan):
                with (output / "jobs" / f"{video_id}.log").open("a") as log:
                    result = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=False
                    )
                if result.returncode:
                    raise ValueError(
                        f"Extractor exited {result.returncode}; see jobs/{video_id}.log"
                    )
            else:
                progress(f"{video_id}: reusing completed extraction without decoder/inference")
            if dataset.exists():
                prior = json.loads((dataset / "preparation.json").read_text())
                if (
                    prior["status"] != "complete"
                    or prior["inputs"]["poses_sha256"] != sha256(run / "poses.jsonl")
                    or prior["npz_sha256"] != sha256(dataset / "poses.npz")
                ):
                    raise ValueError("Existing preparation differs or is incomplete")
            else:
                prepare_recording(run, workbook, dataset, allow_sample=True)
            if quality.exists():
                prior = json.loads((quality / "quality.json").read_text())
                if any(prior["inputs"][name] != sha256(dataset / name) for name in prior["inputs"]):
                    raise ValueError("Existing quality report input hashes differ")
            else:
                assess_quality(dataset, quality)
            entry.update(
                status="complete",
                pose_run=run.as_posix(),
                dataset=dataset.as_posix(),
                quality_report=quality.as_posix(),
            )
            progress(f"{video_id}: extraction, preparation and fixed-policy assessment complete")
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            progress(f"{video_id}: {entry['error']}")
        write_json(output / "batch-state.json", state)
    return state
