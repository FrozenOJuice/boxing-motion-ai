"""Validated IMAGE-run recovery. Complete newline records are immutable resume units."""

import json
from collections import Counter
from fractions import Fraction

from .contracts import pose_record


def recover_prefix(path, manifest):
    """Reject interior corruption; retain a crash-torn tail separately before truncation."""
    records, counts, joints = [], Counter(), [0] * 33
    ranges = manifest["config"]["ranges_half_open"]
    expected = iter(n for a, b in ranges for n in range(a, b)) if ranges else None
    previous = None
    with path.open("rb+") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                backup = path.with_name(f"resume-torn-tail-{offset}.bin")
                if backup.exists() and backup.read_bytes() != line:
                    raise ValueError("Conflicting torn-tail recovery file")
                backup.write_bytes(line)
                stream.truncate(offset)
                break
            try:
                record = json.loads(line)
                index = next(expected) if expected is not None else len(records)
                if record["frame_index"] != index or record["video_id"] != manifest["video_id"]:
                    raise ValueError("Resume frame sequence/recording mismatch")
                base = Fraction(record["time_base"])
                time = record["pts"] * base
                if (
                    type(record["pts"]) is not int
                    or base <= 0
                    or (previous is not None and time <= previous)
                ):
                    raise ValueError("Resume timing is invalid")
                points = record["landmarks"]
                values = (
                    None
                    if record["status"] == "missing"
                    else [
                        tuple(p[k] for k in ("x", "y", "z", "visibility", "presence"))
                        for p in points
                    ]
                )
                # Reuse the exact validity contract, including finite coordinates/scores.
                from types import SimpleNamespace

                frame = SimpleNamespace(
                    index=index,
                    pts=record["pts"],
                    time_base=base,
                    time=time,
                    source_width=record["source_width"],
                    source_height=record["source_height"],
                    rgb=SimpleNamespace(shape=(record["input_height"], record["input_width"], 3)),
                )
                rebuilt = pose_record(
                    frame,
                    values,
                    record["estimator_timestamp_ms"],
                    manifest["config"]["confidence"],
                )
                if (
                    record["schema_version"] != "pose-1.1"
                    or rebuilt["valid_mask"] != record["valid_mask"]
                    or rebuilt["status"] != record["status"]
                    or rebuilt["valid_body_joint_count"] != record["valid_body_joint_count"]
                    or (values is None and points != [None] * 33)
                ):
                    raise ValueError("Resume pose contract mismatch")
                previous = time
                counts[record["status"]] += 1
                for j, valid in enumerate(record["valid_mask"]):
                    joints[j] += valid
                records.append(
                    {
                        k: record[k]
                        for k in (
                            "frame_index",
                            "pts",
                            "time_base",
                            "scene_id",
                            "segment_id",
                            "scene_cut_before",
                            "estimator_timestamp_ms",
                            "source_width",
                            "source_height",
                            "input_width",
                            "input_height",
                            "recording_id",
                            "boxer_id",
                            "session_id",
                        )
                    }
                )
            except (ValueError, KeyError, TypeError, StopIteration) as exc:
                raise ValueError(
                    f"Corrupt committed resume record at byte {offset}: {exc}"
                ) from exc
    return records, counts, joints


def compatible(previous, current):
    for key in (
        "schema_version",
        "video_id",
        "recording_config",
        "inputs",
        "config",
        "packages",
        "python",
    ):
        if previous.get(key) != current.get(key):
            raise ValueError(f"Resume refused: {key} differs from original run")
    # Pin pose-producing code; unrelated CLI/report changes do not affect raw poses.
    names = [
        "pipeline.py",
        "resume.py",
        "decoder.py",
        "contracts.py",
        "mediapipe_pose.py",
        "annotations.py",
        "recording.py",
        "overlay.py",
    ]
    for name in names:
        path = f"src/boxing_motion_ai/vision/{name}"
        if previous["code"]["sha256"].get(path) != current["code"]["sha256"].get(path):
            raise ValueError(f"Resume refused: pose-producing code changed ({name})")
    if current["config"]["pose_mode"] != "image":
        raise ValueError("Interrupted VIDEO-mode tracking cannot resume exactly; use IMAGE mode")
