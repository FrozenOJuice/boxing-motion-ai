"""Only synthetic fixtures; no external dataset or media is required."""

import json

import pytest
from openpyxl import Workbook

from boxing_motion_ai.models import VideoMetadata


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    (root / "recording.mp4").write_bytes(b"synthetic placeholder, never decoded")

    def fake_probe(path, root, issue):
        return VideoMetadata(
            video_id="recording.mp4",
            path="recording.mp4",
            duration_seconds=4,
            fps=25,
            frame_count=100,
            width=640,
            height=480,
            probe_status="ok",
        )

    monkeypatch.setattr("boxing_motion_ai.audit.probe", fake_probe)
    return root


@pytest.fixture
def workbook():
    def make(root, rows):
        book = Workbook()
        book.active.title = "Punches"
        book.active.append(["Synthetic begin", "Synthetic finish", "Synthetic type"])
        for row in rows:
            book.active.append(row)
        book.save(root / "labels.xlsx")
        book.close()

    return make


@pytest.fixture
def mapping(tmp_path):
    value = {
        "schema_version": "1.0",
        "annotation_version": "synthetic-v1",
        "evidence": "Synthetic test layout; not verified BoxingVI schema",
        "sheets": [
            {
                "workbook": "labels.xlsx",
                "sheet": "Punches",
                "data_start_row": 2,
                "columns": {"frame_start": 1, "frame_end": 2, "punch_type": 3},
                "video": "recording.mp4",
                "frame_base": 0,
                "end_convention": "inclusive",
                "class_map": {"J": "jab", "C": "cross", "U": "unknown"},
                "boxer_id": "synthetic-boxer",
                "session_id": "synthetic-session",
            }
        ],
    }
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(value))
    return path
