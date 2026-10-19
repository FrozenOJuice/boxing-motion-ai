"""Read real media metadata without decoding or inventing a frame count."""

import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

from .models import VideoMetadata


def positive(value, cast=float):
    try:
        result = cast(value)
        return result if math.isfinite(result) and result > 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def probe(path: Path, root: Path, issue) -> VideoMetadata:
    rel = path.relative_to(root).as_posix()
    meta = VideoMetadata(video_id=rel, path=rel)
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path.resolve()),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        if not isinstance(stream, dict) or not isinstance(data.get("format", {}), dict):
            raise ValueError("Unexpected ffprobe stream/format structure")
        meta.duration_seconds = positive(stream.get("duration")) or positive(
            data.get("format", {}).get("duration")
        )
        meta.width = positive(stream.get("width"), int)
        meta.height = positive(stream.get("height"), int)
        meta.frame_count = positive(stream.get("nb_frames"), int)
        meta.frame_count_source = "ffprobe.nb_frames" if meta.frame_count else None
        meta.frame_rate_rational = stream.get("avg_frame_rate")
        meta.nominal_frame_rate = stream.get("r_frame_rate")
        meta.time_base = stream.get("time_base")
        try:
            meta.fps = positive(Fraction(meta.frame_rate_rational))
        except (ValueError, TypeError, ZeroDivisionError):
            pass
        required = [meta.duration_seconds, meta.width, meta.height, meta.frame_count, meta.fps]
        meta.probe_status = "ok" if all(v is not None for v in required) else "partial"
        if meta.probe_status == "partial":
            issue(
                "partial_metadata",
                "error",
                rel,
                "ffprobe omitted metadata; exact frame bounds need a counted/verified frame total",
            )
        if meta.nominal_frame_rate != meta.frame_rate_rational:
            issue(
                "timebase_review",
                "warning",
                rel,
                "Nominal and average FPS differ; inspect VFR/timestamps before temporal alignment",
            )
    except FileNotFoundError:
        issue(
            "ffprobe_missing",
            "error",
            rel,
            "Install FFmpeg (ffprobe) and rerun; metadata has not been inspected",
        )
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        issue("probe_failed", "error", rel, f"Cannot inspect video: {type(exc).__name__}: {exc}")
    return meta
