"""Private visual report from synthetic overlays; no repeated video decoding."""

import json
from fractions import Fraction

import pytest
from openpyxl import Workbook

from boxing_motion_ai.data.sample_review import render_sample_review
from boxing_motion_ai.vision.contracts import DecodedFrame
from boxing_motion_ai.vision.pipeline import extract_recording
from boxing_motion_ai.vision.recording import RecordingConfig
from boxing_motion_ai.vision.sampling import plan_sample


def test_sample_review_and_absent_classes_are_honest(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    source = tmp_path / "source"
    source.mkdir()
    video, workbook, model = [source / n for n in ["V2.mkv", "V2.xlsx", "fake.task"]]
    video.write_bytes(b"synthetic")
    model.write_bytes(b"synthetic")
    b = Workbook()
    s = b.active
    s.title = "Sheet1"
    s.append(["start", "end", "class"])
    s.append([1, 3, "Jab"])
    s.append([5, 7, "Jab"])
    b.save(workbook)
    config = RecordingConfig(
        video_id="V2",
        header=["start", "end", "class"],
        helper_formulas={},
        class_map={"Jab": "jab"},
        frame_offset=0,
        end_assumption="inclusive",
    )
    plan = tmp_path / "plan"
    plan_sample(workbook, config, plan, 2)
    poses = tmp_path / "poses"

    def decoder(path, width):
        for i in range(9):
            yield DecodedFrame(
                i, i * 40, Fraction(1, 1000), 64, 48, np.zeros((48, 64, 3), np.uint8)
            )

    class Backend:
        def __init__(self, *args, **kwargs):
            pass

        def detect(self, *args):
            return [(0.5, 0.5, 0.1, 0.9, 0.9)] * 33

        def close(self):
            pass

    extract_recording(
        video,
        workbook,
        model,
        poses,
        ranges=[(1, 4), (5, 8)],
        recording=config,
        decoder=decoder,
        backend_factory=Backend,
        overlay_frames=[1, 2, 3, 5, 6, 7],
        overlay_limit=6,
    )
    result = render_sample_review(poses, plan, tmp_path / "review")
    assert result["classes"]["jab"]["selected_events"] == 2
    assert result["classes"]["rear_hook"]["sheet"] is None
    assert (tmp_path / "review" / "jab.jpg").is_file()
    assert not (tmp_path / "review" / "rear_hook.jpg").exists()
    m = json.loads((poses / "manifest.json").read_text())
    m["status"] = "failed"
    (poses / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="incomplete"):
        render_sample_review(poses, plan, tmp_path / "failed")
