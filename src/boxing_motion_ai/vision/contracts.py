"""Pose schema 1.0: MediaPipe order, raw image coordinates, explicit missing masks."""

import math
from dataclasses import dataclass
from fractions import Fraction

JOINTS = (
    "nose left_eye_inner left_eye left_eye_outer right_eye_inner right_eye right_eye_outer "
    "left_ear right_ear mouth_left mouth_right left_shoulder right_shoulder left_elbow "
    "right_elbow left_wrist right_wrist left_pinky right_pinky left_index right_index "
    "left_thumb right_thumb left_hip right_hip left_knee right_knee left_ankle right_ankle "
    "left_heel right_heel left_foot_index right_foot_index"
).split()
BODY_JOINTS = tuple(range(11, 33))
EDGES = (
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)


@dataclass
class DecodedFrame:
    index: int
    pts: int
    time_base: Fraction
    source_width: int
    source_height: int
    rgb: object

    @property
    def time(self) -> Fraction:
        return self.pts * self.time_base


def pose_record(frame, landmarks, estimator_ms, threshold):
    """Never impute, clip coordinates, or interpret confidence as calibrated accuracy."""
    points = [None] * 33
    valid = [False] * 33
    if landmarks is not None:
        if len(landmarks) != 33:
            raise ValueError(f"Expected 33 landmarks, received {len(landmarks)}")
        for i, point in enumerate(landmarks):
            if len(point) != 5:
                raise ValueError("Expected x, y, z, visibility, presence")
            clean = [float(v) if v is not None and math.isfinite(v) else None for v in point]
            points[i] = dict(zip(("x", "y", "z", "visibility", "presence"), clean, strict=True))
            x, y, z, visibility, presence = clean
            valid[i] = (
                all(v is not None for v in clean)
                and 0 <= x <= 1
                and 0 <= y <= 1
                and threshold <= visibility <= 1
                and threshold <= presence <= 1
            )
    count = sum(valid[i] for i in BODY_JOINTS)
    status = "missing" if landmarks is None else ("ok" if count == 22 else "unreliable")
    return {
        "schema_version": "pose-1.0",
        "video_id": "V5",
        "boxer_id": None,
        "frame_index": frame.index,
        "pts": frame.pts,
        "time_base": f"{frame.time_base.numerator}/{frame.time_base.denominator}",
        "timestamp_seconds": float(frame.time),
        "estimator_timestamp_ms": estimator_ms,
        "source_width": frame.source_width,
        "source_height": frame.source_height,
        "input_width": frame.rgb.shape[1],
        "input_height": frame.rgb.shape[0],
        "status": status,
        "valid_body_joint_count": count,
        "landmarks": points,
        "valid_mask": valid,
    }
