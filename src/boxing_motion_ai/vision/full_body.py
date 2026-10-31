"""Versioned masked 2D whole-body evidence; zeros are sentinels, never recovered joints."""

from fractions import Fraction as Q

import numpy as np

VERSION = "full-body-native-2d-1.1"
JOINTS = tuple(range(11, 33))
NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
CHANNELS = [
    f"{name}.{field}"
    for name in NAMES
    for field in ("x", "y", "visibility", "presence", "valid", "vx", "vy", "velocity_valid")
]
for side in ("left", "right"):
    CHANNELS += [
        f"{side}.{s}"
        for s in (
            "extension",
            "extension_valid",
            "elbow_cos",
            "elbow_valid",
            "extension_rate",
            "extension_rate_valid",
        )
    ]
CHANNELS += [f"{side}.knee_{s}" for side in ("left", "right") for s in ("cos", "valid")]
CHANNELS += [f"{part}.{s}" for part in ("shoulder_axis", "hip_axis") for s in ("x", "y", "valid")]
CHANNELS += [
    "axis_alignment.cos",
    "axis_alignment.sin",
    "axis_alignment.valid",
    "root.vx",
    "root.vy",
    "root.velocity_valid",
    "left.wrist_foot_velocity_cos",
    "left.coordination_valid",
    "right.wrist_foot_velocity_cos",
    "right.coordination_valid",
    "normalization_valid",
]


def clock_and_resets(raw):
    n = len(raw["frame_index"])
    times = [
        Q(int(p) * int(a), int(b))
        for p, a, b in zip(raw["pts"], raw["time_base_num"], raw["time_base_den"], strict=True)
    ]
    if not n or any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("Invalid full-body native clock")
    typical = (
        sorted(b - a for a, b in zip(times, times[1:], strict=False))[(n - 2) // 2]
        if n > 1
        else Q(1, 25)
    )
    reset = np.ones(n, bool)
    for i in range(1, n):
        reset[i] = (
            raw["frame_index"][i] != raw["frame_index"][i - 1] + 1
            or times[i] - times[i - 1] > 2 * typical
            or any(raw[k][i] != raw[k][i - 1] for k in ("scene_id", "segment_id"))
        )
    return times, reset


def encode(raw, aspect, explicit_reset=None):
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError("Positive source aspect required")
    times, reset = clock_and_resets(raw)
    n = len(times)
    if explicit_reset is not None:
        if np.asarray(explicit_reset).shape != (n,) or np.asarray(explicit_reset).dtype != bool:
            raise ValueError("Explicit source reset mask must be native-frame boolean")
        reset |= explicit_reset
    xy = np.asarray(raw["coordinates"][:, JOINTS, :2], dtype=np.float64).copy()
    conf = np.asarray(raw["confidence"][:, JOINTS], dtype=np.float64)
    if xy.shape != (n, 22, 2) or conf.shape != (n, 22, 2):
        raise ValueError("33-landmark source schema required")
    valid = (
        raw["valid_mask"][:, JOINTS]
        & np.isfinite(xy).all(2)
        & np.isfinite(conf).all(2)
        & (raw["pose_status"][:, None] != 2)
        & ((xy >= 0) & (xy <= 1)).all(2)
    )
    xy[..., 0] *= aspect
    xy = np.where(valid[..., None], xy, 0)
    # Local indices: shoulders 0/1, hips 12/13.
    root, shoulder = xy[:, [12, 13]].mean(1), xy[:, [0, 1]].mean(1)
    scale = np.linalg.norm(shoulder - root, axis=1)
    normal = valid[:, [0, 1, 12, 13]].all(1) & (scale > 1e-6)
    good = valid & normal[:, None]
    positions = np.where(
        good[..., None], (xy - root[:, None]) / np.where(normal, scale, 1)[:, None, None], 0
    )
    velocities, vgood = np.zeros_like(positions), np.zeros_like(good)
    root_velocity, root_good = np.zeros((n, 2)), np.zeros(n, bool)
    for i in range(1, n):
        if reset[i]:
            continue
        dt = float(times[i] - times[i - 1])
        vgood[i] = good[i] & good[i - 1]
        velocities[i] = np.where(vgood[i, :, None], (positions[i] - positions[i - 1]) / dt, 0)
        root_good[i] = normal[i] & normal[i - 1]
        if root_good[i]:
            root_velocity[i] = (root[i] - root[i - 1]) / scale[i - 1] / dt
    values = [
        np.concatenate(
            (
                positions,
                np.where(good[..., None], conf, 0),
                good[..., None],
                velocities,
                vgood[..., None],
            ),
            axis=2,
        ).reshape(n, -1)
    ]

    def cosine(a, b, c):
        u, v = positions[:, a] - positions[:, b], positions[:, c] - positions[:, b]
        den = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        ok = good[:, [a, b, c]].all(1) & (den > 1e-6)
        return np.where(ok, (u * v).sum(1) / np.maximum(den, 1e-6), 0), ok

    for s, e, w in ((0, 2, 4), (1, 3, 5)):
        ok = good[:, s] & good[:, w]
        ext = np.where(ok, np.linalg.norm(positions[:, w] - positions[:, s], axis=1), 0)
        angle, angle_good = cosine(s, e, w)
        rate, rate_good = np.zeros(n), np.zeros(n, bool)
        for i in range(1, n):
            rate_good[i] = ok[i] & ok[i - 1] & ~reset[i]
            if rate_good[i]:
                rate[i] = (ext[i] - ext[i - 1]) / float(times[i] - times[i - 1])
        values.append(np.column_stack((ext, ok, angle, angle_good, rate, rate_good)))
    for h, k, a in ((12, 14, 16), (13, 15, 17)):
        angle, ok = cosine(h, k, a)
        values.append(np.column_stack((angle, ok)))
    axes = []
    for a, b in ((0, 1), (12, 13)):
        ok = good[:, a] & good[:, b]
        axis = np.where(ok[:, None], positions[:, b] - positions[:, a], 0)
        axes.append(axis)
        values.append(np.column_stack((axis, ok)))
    a, b = axes
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = normal & (den > 1e-6)
    co = np.where(ok, (a * b).sum(1) / np.maximum(den, 1e-6), 0)
    si = np.where(ok, (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / np.maximum(den, 1e-6), 0)
    values += [np.column_stack((co, si, ok)), np.column_stack((root_velocity, root_good))]
    for w, f in ((4, 20), (5, 21)):
        a, b = velocities[:, w], velocities[:, f]
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        ok = vgood[:, w] & vgood[:, f] & (den > 1e-6)
        values.append(
            np.column_stack((np.where(ok, (a * b).sum(1) / np.maximum(den, 1e-6), 0), ok))
        )
    values.append(normal[:, None])
    output = np.concatenate(values, axis=1).astype(np.float32)
    if output.shape != (n, len(CHANNELS)) or not np.isfinite(output).all():
        raise ValueError("Nonfinite or wrong full-body feature schema")
    return output, normal, reset
