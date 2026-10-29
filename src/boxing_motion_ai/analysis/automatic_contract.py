"""Lightweight identity binding for offline automatic candidate analysis."""

import hashlib
import json
from pathlib import Path

from .activity_contract import inputs as activity_inputs
from .common import pin, regular, sha

VERSION = "automatic-punch-candidates-2.0"
POLICY = "configs/punch_instances_v2.json"


def inputs(root, directory, key):
    root, directory = Path(root), Path(directory)
    activity_fp, activity_identity, video, poses, _ = activity_inputs(root, directory, key)
    classifier = pin(root)
    policy = json.loads(regular(root / POLICY).read_text())
    implementation = [
        "analysis/automatic.py",
        "analysis/automatic_contract.py",
        "analysis/inference.py",
        "detection/instances.py",
        "detection/instances_v2.py",
    ]
    identity = dict(
        schema_version=VERSION,
        video_id=key,
        activity_input_fingerprint=activity_fp,
        activity_inputs=activity_identity,
        classifier_pin_sha256=sha(root / "configs/assisted_level1_v1.json"),
        classifier_checkpoint=classifier["checkpoint"],
        classifier_provenance=classifier["provenance"],
        policy_sha256=sha(root / POLICY),
        implementation_sha256={p: sha(Path(__file__).parents[1] / p) for p in implementation},
    )
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return fingerprint, identity, video, poses, policy


def activity_record(directory, key, fingerprint, expected_hash):
    path = regular(Path(directory) / f"activity-{fingerprint}.json")
    if sha(path) != expected_hash:
        raise ValueError("Activity record has changed. Retry automatic analysis.")
    value = json.loads(path.read_text())
    if not (
        value.get("schema_version") == "uploaded-activity-detection-1.0"
        and value.get("video_id") == key
        and value.get("input_fingerprint") == fingerprint
    ):
        raise ValueError("Activity record does not match this upload.")
    return value
