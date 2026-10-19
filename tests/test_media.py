import json
import subprocess
from types import SimpleNamespace

import pytest

from boxing_motion_ai.audit import run_audit
from boxing_motion_ai.media import probe


def test_probe_real_response_shape(tmp_path, monkeypatch):
    stream = {
        "duration": "4.0",
        "width": 640,
        "height": 480,
        "nb_frames": "100",
        "avg_frame_rate": "25/1",
        "r_frame_rate": "25/1",
        "time_base": "1/12800",
    }
    calls, issues = [], []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=json.dumps({"streams": [stream]}))

    monkeypatch.setattr("boxing_motion_ai.media.subprocess.run", run)
    meta = probe(tmp_path / "video.mp4", tmp_path, lambda *args: issues.append(args))
    assert meta.frame_count == 100
    assert meta.fps == 25
    assert meta.duration_seconds == 4
    assert meta.frame_count_source == "ffprobe.nb_frames"
    assert meta.probe_status == "ok"
    assert not issues
    assert calls[0][1]["timeout"] == 60
    stream.update(nb_frames="N/A", avg_frame_rate="0/0")
    meta = probe(tmp_path / "video.mp4", tmp_path, lambda *args: issues.append(args))
    assert meta.frame_count is None  # Never estimate length as duration * FPS.
    assert meta.fps is None
    assert meta.probe_status == "partial"


@pytest.mark.parametrize(
    "error,expected",
    [
        (FileNotFoundError(), "ffprobe_missing"),
        (subprocess.TimeoutExpired("ffprobe", 60), "probe_failed"),
        (subprocess.CalledProcessError(1, "ffprobe"), "probe_failed"),
    ],
)
def test_probe_failures_report_unknown_metadata(tmp_path, monkeypatch, error, expected):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr("boxing_motion_ai.media.subprocess.run", fail)
    issues = []
    meta = probe(tmp_path / "video.mp4", tmp_path, lambda *args: issues.append(args))
    assert meta.frame_count is None
    assert meta.probe_status == "failed"
    assert issues[0][0] == expected


def test_unknown_length_cannot_pass_alignment(tmp_path, monkeypatch, workbook, mapping):
    (tmp_path / "recording.mp4").write_bytes(b"synthetic")
    workbook(tmp_path, [[0, 9, "J"]])
    monkeypatch.setattr(
        "boxing_motion_ai.media.subprocess.run",
        lambda *a, **kw: SimpleNamespace(stdout='{"streams": [{"duration": "4"}]}'),
    )
    report = run_audit(tmp_path, mapping)
    assert "unknown_frame_count" in {i["code"] for i in report["issues"]}
    assert report["structural_checks_passed"] is False


@pytest.mark.parametrize("payload", ["not JSON", '{"streams": []}', '{"streams": [null]}'])
def test_malformed_probe_output(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(
        "boxing_motion_ai.media.subprocess.run", lambda *a, **kw: SimpleNamespace(stdout=payload)
    )
    issues = []
    meta = probe(tmp_path / "video.mp4", tmp_path, lambda *args: issues.append(args))
    assert meta.probe_status == "failed"
    assert issues[0][0] == "probe_failed"
