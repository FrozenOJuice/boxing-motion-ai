"""Bounded metadata reads. No directory discovery, asset serving or deserialization of models."""

import hashlib
import json
import math
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from boxing_motion_ai.models import relative_path

_RESERVED = re.compile(r"(?i)(?:^|[^a-z0-9])v9(?:[^0-9]|$)")
MAX_BYTES = 16 * 1024 * 1024


class ServiceError(ValueError):
    """Stable error code for a future transport adapter; no absolute paths in responses."""

    def __init__(self, status, code):
        super().__init__(code)
        self.status = status
        self.code = code


def safe_relative(value):
    if not isinstance(value, str) or not value or _RESERVED.search(value):
        raise ServiceError("invalid", "reserved_or_invalid_path")
    if "%" in value or "\x00" in value or any(p in ("", ".", "..") for p in value.split("/")):
        raise ServiceError("invalid", "invalid_path")
    try:
        relative_path(value)
    except ValueError as exc:
        raise ServiceError("invalid", "invalid_path") from exc
    return value


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Nonfinite JSON value")

    def finite(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Nonfinite JSON value")
        return result

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite)


class MetadataStore:
    """Explicit root, regular files only, no symlinks even in parent directories.

    POSIX dir-fd traversal also prevents symlink replacement races. The repository
    already targets macOS/Linux; Windows support is not claimed by this boundary.
    """

    def __init__(self, root):
        self.root = Path(os.path.abspath(root))
        if _RESERVED.search(str(self.root)):
            raise ServiceError("invalid", "reserved_root")

    @contextmanager
    def open_regular(self, path):
        """Internal trusted-path reader, also used for registered dashboard media."""
        safe_relative(path)
        parts = (*self.root.parts[1:], *PurePosixPath(path).parts)
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
            file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            with os.fdopen(file_fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ServiceError("invalid", "not_bounded_regular_metadata")
                yield stream
        except FileNotFoundError as exc:
            raise ServiceError("missing", "artifact_not_found") from exc
        except OSError as exc:
            raise ServiceError("invalid", "unsafe_or_unreadable_artifact") from exc
        finally:
            os.close(fd)

    def read(self, path, sha256=None):
        safe_relative(path)
        if PurePosixPath(path).suffix not in (".json", ".jsonl", ".yaml", ".yml"):
            raise ServiceError("invalid", "metadata_only")
        with self.open_regular(path) as stream:
            if os.fstat(stream.fileno()).st_size > MAX_BYTES:
                raise ServiceError("invalid", "not_bounded_regular_metadata")
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ServiceError("invalid", "metadata_too_large")
        actual = hashlib.sha256(raw).hexdigest()
        if sha256 is not None and actual != sha256:
            raise ServiceError("stale", "artifact_hash_mismatch")
        return raw.decode("utf-8"), actual

    def json(self, path, sha256=None):
        raw, actual = self.read(path, sha256)
        return strict_json(raw), actual
