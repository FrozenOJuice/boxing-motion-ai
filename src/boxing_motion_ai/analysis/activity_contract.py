"""Lightweight hash binding for uploaded activity inference and its cache. No model import."""

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .common import regular, sha

VERSION = "uploaded-activity-detection-1.0"
PIN = "configs/uploaded_activity_v1.json"


def activity_pin(root):
    root = Path(root)
    p = json.loads(regular(root / PIN).read_text())
    if p.get("schema_version") != "uploaded-activity-pin-1.0":
        raise ValueError("Unsupported activity detector pin.")
    for name, expected in {
        p["checkpoint"]["path"]: p["checkpoint"]["sha256"],
        **p["source_sha256"],
    }.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "v9" in str(relative).lower():
            raise ValueError("Unsafe activity detector binding.")
        if sha(regular(root / relative)) != expected:
            raise ValueError("The frozen activity checkpoint, feature or decoder has changed.")
    return p


def inputs(root, directory, key):
    root, directory = Path(root), Path(directory)
    p = activity_pin(root)
    video = json.loads(regular(directory / "video.json").read_text())
    poses = json.loads(regular(directory / "pose.json").read_text())
    if not (
        video["schema_version"] == "assisted-video-1.0"
        and poses["schema_version"] == "assisted-poses-1.0"
        and video["source_sha256"]
        == poses["source_sha256"]
        == sha(regular(directory / "source.video"))
        and poses["poses_sha256"] == sha(regular(directory / "poses.npz"))
        and poses["joint_order"] == p["joint_order"]
        and poses["pose_model_sha256"] == p["pose_model_sha256"]
        and poses["threshold"] == 0.5
        and poses["mode"] == "image"
    ):
        raise ValueError("Cached video/pose provenance is invalid for activity scanning.")
    implementation = [
        "analysis/activity_contract.py",
        "detection/upload.py",
        "detection/activity_bouts.py",
    ]
    try:
        runtime_versions = dict(torch=version("torch"), numpy=version("numpy"))
    except PackageNotFoundError as exc:
        raise ValueError("Install the local pose/train dependencies before scanning.") from exc
    identity = dict(
        schema_version=VERSION,
        video_id=key,
        source_sha256=video["source_sha256"],
        poses_sha256=poses["poses_sha256"],
        video_metadata_sha256=sha(directory / "video.json"),
        pose_metadata_sha256=sha(directory / "pose.json"),
        pin_sha256=sha(root / PIN),
        implementation_sha256={s: sha(Path(__file__).parents[1] / s) for s in implementation},
        runtime=dict(
            python=platform.python_version(),
            **runtime_versions,
            system=platform.system(),
            machine=platform.machine(),
            device="cpu",
            threads=2,
            deterministic_algorithms=True,
        ),
    )
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return fingerprint, identity, video, poses, p
