"""Crash/restart equivalence with synthetic frames; no model or dataset needed."""

import json
from fractions import Fraction
from types import SimpleNamespace

import pytest
from openpyxl import Workbook

from boxing_motion_ai.vision.contracts import DecodedFrame
from boxing_motion_ai.vision.pipeline import extract_v5


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video, workbook, model = [source / n for n in ["V5.mkv", "V5.xlsx", "synthetic.task"]]
    video.write_bytes(b"no real video")
    model.write_bytes(b"no real weights")
    b = Workbook()
    s = b.active
    s.title = "Sheet1"
    s.append([None] * 5)
    s.append([0, 2, "Jab", "=A2-24", "=B2-24"])
    b.save(workbook)
    return video, workbook, model


def decoder(path, width):
    for i, pts in enumerate([0, 32, 69, 104, 140, 173, 207, 245]):
        yield DecodedFrame(
            i, pts, Fraction(1, 1000), 640, 360, SimpleNamespace(shape=(360, 640, 3))
        )


class Backend:
    calls = []
    fail_at = None

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, rgb, ms):
        self.calls.append(ms)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("synthetic interruption")
        return [(0.5, 0.5, 0.1, 0.9, 0.9)] * 33

    def close(self):
        pass


def run(inputs, output, **kwargs):
    return extract_v5(
        *inputs, output, decoder=decoder, backend_factory=Backend, overlay_limit=0, **kwargs
    )


@pytest.mark.parametrize("ranges", [[], [(1, 4), (6, 8)]])
def test_resume_reuses_committed_frames_and_matches_uninterrupted(inputs, tmp_path, ranges):
    Backend.calls = []
    Backend.fail_at = None
    run(inputs, tmp_path / "reference", ranges=ranges, scene_cuts=[3])
    expected = (tmp_path / "reference" / "poses.jsonl").read_bytes()
    Backend.calls = []
    Backend.fail_at = 4
    out = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="interruption"):
        run(inputs, out, ranges=ranges, scene_cuts=[3])
    prefix = (out / "poses.jsonl.partial").read_bytes()
    assert len(prefix.splitlines()) == 3
    with (out / "poses.jsonl.partial").open("ab") as stream:
        stream.write(b'{"frame_index":')
    Backend.calls = []
    Backend.fail_at = None
    result = run(inputs, out, ranges=ranges, scene_cuts=[3], resume=True)
    assert (out / "poses.jsonl").read_bytes() == expected
    assert len(Backend.calls) == len(expected.splitlines()) - 3
    assert result["resume_history"][0]["reused_records"] == 3
    assert list(out.glob("resume-torn-tail-*.bin"))
    Backend.calls = []
    assert run(inputs, out, ranges=ranges, scene_cuts=[3], resume=True) == result
    assert not Backend.calls


def test_resume_rejects_inputs_config_corrupt_records_and_completed_tampering(inputs, tmp_path):
    Backend.calls = []
    Backend.fail_at = 3
    out = tmp_path / "run"
    with pytest.raises(RuntimeError):
        run(inputs, out)
    with pytest.raises(ValueError, match="config differs"):
        run(inputs, out, resume=True, confidence=0.6)
    original = inputs[0].read_bytes()
    inputs[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="inputs differs"):
        run(inputs, out, resume=True)
    inputs[0].write_bytes(original)
    partial = out / "poses.jsonl.partial"
    contents = partial.read_bytes()
    partial.write_bytes(b"bad committed record\n" + contents)
    with pytest.raises(ValueError, match="Corrupt committed"):
        run(inputs, out, resume=True)
    partial.write_bytes(contents)
    Backend.calls = []
    Backend.fail_at = None
    run(inputs, out, resume=True)
    with (out / "poses.jsonl").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        run(inputs, out, resume=True)


def test_resume_code_change_and_video_state_rejected(inputs, tmp_path):
    Backend.calls = []
    Backend.fail_at = 2
    out = tmp_path / "run"
    with pytest.raises(RuntimeError):
        run(inputs, out, pose_mode="video")
    with pytest.raises(ValueError, match="VIDEO-mode"):
        run(inputs, out, pose_mode="video", resume=True)
    manifest = out / "manifest.json"
    m = json.loads(manifest.read_text())
    m["code"]["sha256"]["src/boxing_motion_ai/vision/decoder.py"] = "changed"
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="code changed"):
        run(inputs, out, pose_mode="video", resume=True)


def test_active_writer_lock_refuses_second_extractor(inputs, tmp_path):
    import fcntl

    output = tmp_path / "locked"
    with (tmp_path / ".locked.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already writing"):
            run(inputs, output)
    assert not output.exists()


def test_completed_batch_reuse_checks_inputs_and_settings_without_inference(inputs, tmp_path):
    from boxing_motion_ai.data.validation import completed_run_matches
    from boxing_motion_ai.vision.recording import v5_recording

    Backend.calls = []
    Backend.fail_at = None
    output = tmp_path / "complete"
    run(inputs, output, ranges=[(1, 4)])
    calls = list(Backend.calls)
    plan = tmp_path / "plan"
    plan.mkdir()
    config = {"ranges": ["1:4"], "overlay_limit": 0}
    (plan / "pose-config.json").write_text(json.dumps(config))
    assert completed_run_matches(output, *inputs, v5_recording(), plan)
    assert Backend.calls == calls
    config["confidence"] = 0.7
    (plan / "pose-config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="settings differ"):
        completed_run_matches(output, *inputs, v5_recording(), plan)
