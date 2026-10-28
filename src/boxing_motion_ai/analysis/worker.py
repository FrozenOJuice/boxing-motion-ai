"""Isolated pose/inference process; no PyAV import or BoxingVI workbook required."""

import json
import sys
from fractions import Fraction
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from ..data.pose_quality import policy_metrics
from ..vision.contracts import JOINTS, DecodedFrame, pose_record
from ..vision.mediapipe_pose import MediaPipePose
from .common import pin, regular, sha, write_json


def extract(root, directory):
    began = perf_counter()
    binding = pin(root, pose=True)
    video = json.loads(regular(directory / "video.json").read_text())
    backend = MediaPipePose(root / binding["pose_model"]["path"], confidence=0.5, mode="image")
    coordinates, confidence, valid, statuses = [], [], [], []
    try:
        for row in video["frames"]:
            with Image.open(
                regular(directory / "frames" / f"{row['frame_index']:04d}.png")
            ) as image:
                rgb = np.asarray(image.convert("RGB"))
            timebase = Fraction(row["time_base_num"], row["time_base_den"])
            ms = round(
                (Fraction(row["timestamp_exact"]) - Fraction(video["frames"][0]["timestamp_exact"]))
                * 1000
            )
            frame = DecodedFrame(
                row["frame_index"], row["pts"], timebase, video["width"], video["height"], rgb
            )
            landmarks = backend.detect(rgb, ms)
            record = pose_record(frame, landmarks, ms, 0.5)
            coordinates.append(
                [
                    [p[k] if p and p[k] is not None else np.nan for k in ("x", "y", "z")]
                    for p in record["landmarks"]
                ]
            )
            confidence.append(
                [
                    [p[k] if p and p[k] is not None else np.nan for k in ("visibility", "presence")]
                    for p in record["landmarks"]
                ]
            )
            valid.append(record["valid_mask"])
            statuses.append(2 if landmarks is None else (0 if record["status"] == "ok" else 1))
            if row["frame_index"] % 5 == 0:
                write_json(
                    directory / "progress.json",
                    dict(stage="extracting_poses", frames=len(valid), total=video["frame_count"]),
                )
    finally:
        backend.close()
    raw = {
        k: np.array([r[k] for r in video["frames"]], dtype=np.int64)
        for k in ("frame_index", "pts", "time_base_num", "time_base_den")
    }
    raw.update(
        coordinates=np.asarray(coordinates, dtype=np.float32),
        confidence=np.asarray(confidence, dtype=np.float32),
        valid_mask=np.asarray(valid, dtype=bool),
        pose_status=np.asarray(statuses, dtype=np.int64),
        scene_id=np.zeros(len(valid), dtype=np.int64),
        segment_id=np.zeros(len(valid), dtype=np.int64),
    )
    np.savez_compressed(directory / "poses.npz", **raw)
    value = dict(
        schema_version="assisted-poses-1.0",
        joint_order=JOINTS,
        source_sha256=video["source_sha256"],
        poses_sha256=sha(directory / "poses.npz"),
        pose_model_sha256=binding["pose_model"]["sha256"],
        threshold=0.5,
        mode="image",
        pose_seconds=perf_counter() - began,
        pose_coverage=policy_metrics(raw["valid_mask"], raw["pose_status"] == 2),
        continuity="user-supplied continuous footage; "
        "no automatic scene-cut or person-count verification",
    )
    write_json(directory / "pose.json", value)


def classify(root, directory, start, stop):
    from .inference import Classifier

    began = perf_counter()
    metadata = json.loads(regular(directory / "pose.json").read_text())
    if sha(regular(directory / "poses.npz")) != metadata["poses_sha256"]:
        raise ValueError("Cached poses have changed. Remove the upload and try again.")
    video = json.loads(regular(directory / "video.json").read_text())
    if video["source_sha256"] != metadata["source_sha256"] or metadata["joint_order"] != JOINTS:
        raise ValueError("Cached pose provenance is invalid.")
    model = Classifier(root)
    with np.load(directory / "poses.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    result = model.classify(arrays, video["width"] / video["height"], start, stop)
    result.update(start_frame=start, stop_frame_exclusive=stop)
    result["classification_wall_seconds"] = perf_counter() - began
    write_json(directory / f"prediction-{start}-{stop}.json", result)


if __name__ == "__main__":
    mode, root, directory, *bounds = sys.argv[1:]
    if mode == "pose":
        extract(Path(root), Path(directory))
    elif mode == "classify":
        classify(Path(root), Path(directory), *map(int, bounds))
    else:
        raise ValueError("Unknown analysis worker mode")
