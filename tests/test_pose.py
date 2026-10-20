"""No real footage, weights or downloads. Core tests also work without the pose extra."""

import json
from fractions import Fraction
from types import SimpleNamespace

import pytest
from openpyxl import Workbook
from typer.testing import CliRunner

from boxing_motion_ai.cli import app
from boxing_motion_ai.vision.annotations import read_v5_annotations
from boxing_motion_ai.vision.contracts import DecodedFrame, pose_record
from boxing_motion_ai.vision.pipeline import extract_v5, frame_ranges, sha256


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "V5.mkv"
    video.write_bytes(b"synthetic metadata-only media")
    model = source / "synthetic.task"
    model.write_bytes(b"synthetic test model, never inference weights")
    workbook = source / "V5.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.title = "Sheet1"
    sheet.append([None])
    for n, row in enumerate([(0, 2, "Jab"), (2, 1, "Rear Hook"), (1, 1, "Jab "), (0, 2, "Jab")], 2):
        sheet.append([*row, f"=A{n}-24", f"=B{n}-24"])
    book.save(workbook)
    book.close()
    return video, workbook, model


def make_frame(index, pts=None, time_base=Fraction(1, 1000)):
    return DecodedFrame(
        index,
        index * 33 if pts is None else pts,
        time_base,
        1920,
        1080,
        SimpleNamespace(shape=(360, 640, 3)),
    )


def fake_decode(path, max_width):
    yield from (make_frame(i) for i in range(8))


class FakeBackend:
    def __init__(self, model, confidence, mode="video"):
        self.calls = 0

    def detect(self, rgb, timestamp_ms):
        self.calls += 1
        return None if self.calls == 2 else [(0.5, 0.5, -0.1, 0.9, 0.8)] * 33

    def close(self):
        pass


def run(inputs, output, **kwargs):
    return extract_v5(
        *inputs, output, decoder=fake_decode, backend_factory=FakeBackend, overlay_limit=0, **kwargs
    )


def records(output):
    return [json.loads(line) for line in (output / "poses.jsonl").read_text().splitlines()]


def test_exact_fractional_pts_and_missing_mask():
    frame = make_frame(17, pts=511, time_base=Fraction(1, 90000))
    result = pose_record(frame, None, 0, 0.5)
    assert result["pts"] == 511 and result["time_base"] == "1/90000"
    assert result["timestamp_seconds"] == float(Fraction(511, 90000))
    assert result["frame_index"] == 17
    assert result["status"] == "missing"
    assert result["landmarks"] == [None] * 33
    assert result["valid_mask"] == [False] * 33


def test_low_confidence_outside_image_nan_and_missing_confidence_are_invalid():
    points = [(0.5, 0.5, 0.0, 0.9, 0.9)] * 33
    points[11] = (0.4, 0.2, 0.0, 0.1, 0.9)
    points[12] = (1.1, 0.2, 0.0, 0.9, 0.9)
    points[13] = (float("nan"), 0.2, 0.0, 0.9, 0.9)
    points[14] = (0.3, 0.2, 0.0, None, 0.9)
    result = pose_record(make_frame(0), points, 0, 0.5)
    assert result["status"] == "unreliable"
    assert result["valid_body_joint_count"] == 18
    assert result["landmarks"][12]["x"] == 1.1  # retained, not clipped
    assert result["landmarks"][13]["x"] is None
    assert not any(result["valid_mask"][11:15])
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="33"):
        pose_record(make_frame(0), points[:32], 0, 0.5)


def test_annotations_preserve_invalid_rows_and_unknown_conventions(inputs):
    result = read_v5_annotations(inputs[1])
    rows = result["rows"]
    assert rows[1]["raw_start"] == 2 and rows[1]["raw_end"] == 1
    assert "reversed_bounds" in rows[1]["flags"]
    assert "equal_bounds_end_convention_unverified" in rows[2]["flags"]
    assert "out_of_row_order" in rows[2]["flags"]
    assert rows[2]["source_class"] == "Jab "
    assert rows[0]["overlapping_source_rows"] == [4, 5]
    assert "duplicate" in rows[3]["flags"]
    assert rows[0]["helper_D"] == "=A2-24"
    assert result["verified_frame_base"] is None
    assert result["verified_end_convention"] is None
    assert result["canonical_class_map"] is None


def test_full_extraction_and_reproducibility(inputs, tmp_path):
    before = [sha256(p) for p in inputs]
    a, b = tmp_path / "a", tmp_path / "b"
    manifest = run(inputs, a)
    run(inputs, b)
    assert manifest["status"] == "complete" and manifest["scope"] == "full_video"
    assert manifest["source_frame_count"] == 8
    assert manifest["status_counts"] == {"ok": 7, "missing": 1}
    data = records(a)
    assert [r["frame_index"] for r in data] == list(range(8))
    assert [r["pts"] for r in data] == [i * 33 for i in range(8)]
    assert data[1]["landmarks"] == [None] * 33  # no carry from frame 0
    assert (a / "poses.jsonl").read_bytes() == (b / "poses.jsonl").read_bytes()
    assert [sha256(p) for p in inputs] == before
    assert not (a / "poses.jsonl.partial").exists()
    assert json.loads((a / "manifest.json").read_text()) == manifest


def test_ranges_keep_source_coordinates_and_reset_state(inputs, tmp_path):
    output = tmp_path / "windows"
    manifest = run(inputs, output, ranges=[(2, 4), (6, 8)])
    data = records(output)
    assert [r["frame_index"] for r in data] == [2, 3, 6, 7]
    assert [r["pts"] for r in data] == [66, 99, 198, 231]
    assert [r["estimator_timestamp_ms"] for r in data] == [0, 33, 0, 33]
    assert [r["status"] for r in data] == ["ok", "missing", "ok", "missing"]
    assert manifest["scope"] == "selected_windows"
    assert manifest["source_frame_count"] is None


@pytest.mark.parametrize("ranges", [["1:1"], ["-1:2"], ["1:4", "3:8"], ["bad"], ["1:2:3"]])
def test_invalid_ranges(ranges):
    with pytest.raises(ValueError):
        frame_ranges(ranges)


def test_out_of_video_range_fails_without_successful_output(inputs, tmp_path):
    output = tmp_path / "past-end"
    with pytest.raises(ValueError, match="beyond"):
        run(inputs, output, ranges=[(7, 12)])
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert (output / "poses.jsonl.partial").exists()
    assert not (output / "poses.jsonl").exists()


def test_inference_exception_is_failure_not_missing(inputs, tmp_path):
    class Broken(FakeBackend):
        def detect(self, rgb, timestamp_ms):
            raise RuntimeError("synthetic backend failure")

    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="synthetic"):
        extract_v5(*inputs, output, decoder=fake_decode, backend_factory=Broken, overlay_limit=0)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["processed_frame_count"] == 0
    assert manifest["status_counts"] == {}


def test_preserve_outputs_and_raw_inputs(inputs, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="exists"):
        run(inputs, output)
    with pytest.raises(ValueError, match="outside source"):
        run(inputs, inputs[0].parent / "output")
    with pytest.raises(ValueError, match="outside raw"):
        run(inputs, tmp_path / "raw" / "output")


def test_timestamp_collisions_fail_instead_of_changing_source_time(inputs, tmp_path):
    def fast_decode(path, max_width):
        yield make_frame(0, 0, Fraction(1, 10000))
        yield make_frame(1, 1, Fraction(1, 10000))

    output = tmp_path / "collision"
    with pytest.raises(ValueError, match="milliseconds"):
        extract_v5(
            *inputs, output, decoder=fast_decode, backend_factory=FakeBackend, overlay_limit=0
        )
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_cli_help_and_missing_inputs(tmp_path):
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["extract-poses", "--help"])
    assert result.exit_code == 0 and "--range" in result.output
    args = ["extract-poses"]
    for name in ["video", "workbook", "model", "output"]:
        args += [f"--{name}", str(tmp_path / name)]
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "must all exist" in result.output
    assert not (tmp_path / "output").exists()


def test_real_decoder_preserves_synthetic_vfr_pts(tmp_path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    from boxing_motion_ai.vision.decoder import decode_video

    path = tmp_path / "vfr.mkv"
    pts = [0, 33, 70, 110, 180]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for timestamp in pts:
            frame = av.VideoFrame.from_ndarray(
                np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    decoded = list(decode_video(path, 64))
    assert [f.index for f in decoded] == list(range(5))
    assert [f.time for f in decoded] == [Fraction(n, 1000) for n in pts]
    assert decoded[0].rgb.shape == (48, 64, 3)


def test_config_defaults_validation_and_overrides(tmp_path):
    pytest.importorskip("yaml")
    from boxing_motion_ai.vision.config import load_config

    assert load_config(None).max_width == 960
    config = tmp_path / "pose.yaml"
    config.write_text("ranges: ['2:4']\nmax_width: 640\nconfidence: 0.6\n")
    loaded = load_config(config, max_width=800, overlay_limit=0)
    assert loaded.max_width == 800 and loaded.ranges == ["2:4"]
    assert loaded.confidence == 0.6 and loaded.overlay_limit == 0
    config.write_text("max_wdth: 640\n")
    with pytest.raises(ValueError, match="Extra inputs"):
        load_config(config)
    config.write_text("confidence: .nan\n")
    with pytest.raises(ValueError):
        load_config(config)
    config.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(config)


def test_overlay_and_contact_sheet_are_real_readable_images(tmp_path):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    from boxing_motion_ai.vision.overlay import contact_sheet, save_overlay

    frame = make_frame(0)
    frame.rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    record = pose_record(frame, None, 0, 0.5)
    path = tmp_path / "missing.jpg"
    save_overlay(frame, record, path, [])
    with Image.open(path) as image:
        assert image.size == (640, 434)
        assert image.getpixel((320, 300)) == (0, 0, 0)  # no imaginary skeleton
    contact_sheet([path, path], tmp_path / "sheet.jpg")
    with Image.open(tmp_path / "sheet.jpg") as image:
        assert image.size == (1440, 326)


def test_non_sequential_decoder_indices_fail(inputs, tmp_path):
    def broken(path, max_width):
        yield make_frame(0)
        yield make_frame(2)

    with pytest.raises(ValueError, match="non-sequential"):
        extract_v5(
            *inputs, tmp_path / "gap", decoder=broken, backend_factory=FakeBackend, overlay_limit=0
        )


def test_excel_integral_float_frames_preserve_values_and_get_candidate_indices(inputs):
    import re
    from zipfile import ZipFile

    path = inputs[1]
    with ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    name = "xl/worksheets/sheet1.xml"
    entries[name] = re.sub(rb"<v>(\d+)</v>", rb"<v>\1.0</v>", entries[name])
    with ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    result = read_v5_annotations(path)
    row = result["rows"][0]
    assert type(row["raw_start"]) is float and row["raw_start"] == 0.0
    assert row["candidate_start_index"] == 0 and type(row["candidate_start_index"]) is int
    assert not any("invalid_frame_values" in r["flags"] for r in result["rows"])
    assert "reversed_bounds" in result["rows"][1]["flags"]
    assert "duplicate" in result["rows"][3]["flags"]


def test_scene_cut_resets_backend_before_first_postcut_frame(inputs, tmp_path):
    output = tmp_path / "cut"
    manifest = run(inputs, output, scene_cuts=[3], pose_mode="image")
    data = records(output)
    assert [r["scene_id"] for r in data] == [0, 0, 0, 1, 1, 1, 1, 1]
    assert data[3]["scene_cut_before"] is True and data[2]["scene_cut_before"] is False
    assert data[3]["estimator_timestamp_ms"] == 0
    assert data[3]["status"] == "ok" and data[4]["status"] == "missing"
    assert manifest["config"]["pose_mode"] == "image"
    assert data[2]["segment_id"] != data[3]["segment_id"]


def test_image_backend_never_calls_video_api(monkeypatch):
    import sys

    from boxing_motion_ai.vision.mediapipe_pose import MediaPipePose

    calls = []

    class Detector:
        def detect(self, image):
            calls.append("image")
            return SimpleNamespace(pose_landmarks=[])

        def detect_for_video(self, *args):
            raise AssertionError("Temporal API must never be used in IMAGE mode")

        def close(self):
            pass

    class Base:
        Delegate = SimpleNamespace(CPU="CPU")

        def __init__(self, **kwargs):
            pass

    mp = SimpleNamespace(
        Image=lambda **kwargs: object(),
        ImageFormat=SimpleNamespace(SRGB="RGB"),
        tasks=SimpleNamespace(
            BaseOptions=Base,
            vision=SimpleNamespace(
                PoseLandmarkerOptions=lambda **kwargs: kwargs,
                RunningMode=SimpleNamespace(IMAGE="image"),
                PoseLandmarker=SimpleNamespace(create_from_options=lambda options: Detector()),
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    backend = MediaPipePose("synthetic.task", mode="image")
    assert backend.detect(None, 0) is None
    assert backend.detect(None, 100) is None
    assert calls == ["image", "image"]
