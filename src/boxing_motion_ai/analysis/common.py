"""Small dependency-free contracts shared by isolated analysis workers and the API."""

import hashlib
import json
import os
from pathlib import Path

MAX_BYTES = 150 * 1024 * 1024
MAX_SECONDS = 30
MAX_FRAMES = 1800
MAX_ARTIFACT_BYTES = 768 * 1024 * 1024
EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def regular(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("A required local analysis file is missing or unsafe.")
    return path


def pin(root, *, pose=False):
    root = Path(root)
    value = json.loads(regular(root / "configs/assisted_level1_v1.json").read_text())
    if value.get("schema_version") != "assisted-level1-pin-1.0":
        raise ValueError("Unsupported analysis model pin.")
    bindings = [value["pose_model"]] if pose else [value["checkpoint"], value["provenance"]]
    bindings += [dict(path=p, sha256=h) for p, h in value["source_sha256"].items()]
    for item in bindings:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or "v9" in str(relative).lower():
            raise ValueError("Unsafe analysis model binding.")
        if sha(regular(root / relative)) != item["sha256"]:
            raise ValueError("The pinned model or feature implementation has changed.")
    return value
