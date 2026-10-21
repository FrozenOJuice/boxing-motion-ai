"""Synthetic cached media/timing evidence; no downloads or real decoding."""

import json

import pytest
from openpyxl import Workbook

from boxing_motion_ai.data.alignment import diagnose_alignment, nearest_index
from boxing_motion_ai.vision.pipeline import sha256


def test_alignment_reports_hypotheses_flags_hashes_and_portable_files(tmp_path):
    raw = tmp_path / "raw"
    (raw / "Annotation_files").mkdir(parents=True)
    (raw / "RGB_videos").mkdir()
    video = raw / "RGB_videos" / "V1.mkv"
    video.write_bytes(b"no actual media")
    workbook = raw / "Annotation_files" / "V1.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.title = "Sheet1"
    sheet.append(["start", "end", "class"])
    sheet.append([10, 12, "Jab"])
    sheet.append([20, 19, "Jab"])
    book.save(workbook)
    (raw / "RGB_videos" / "V1.info.json").write_text(
        json.dumps({"id": "synthetic", "channel_id": "channel", "title": "Synthetic"})
    )
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    source = {
        "id": "V1",
        "youtube_id": "synthetic",
        "original_url": "https://example.test/synthetic",
        "workbook": "Annotation_files/V1.xlsx",
        "media_path": "RGB_videos/V1.mkv",
        "media_sha256": sha256(video),
        "workbook_sha256": sha256(workbook),
    }
    (evidence / "source-manifest.json").write_text(json.dumps({"recordings": [source]}))
    (evidence / "V1.ffprobe.json").write_text(
        json.dumps(
            {
                "frames": [{"pts_time": f"{n * 0.04:.3f}"} for n in range(30)],
                "streams": [
                    {
                        "avg_frame_rate": "25/1",
                        "r_frame_rate": "25/1",
                        "width": 640,
                        "height": 360,
                        "time_base": "1/1000",
                    }
                ],
                "format": {"duration": "1.2"},
            }
        )
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "video_id": "V1",
                "header": ["start", "end", "class"],
                "helper_formulas": {},
                "class_map": {"Jab": "jab"},
                "frame_offset": 0,
                "end_assumption": "inclusive",
            }
        )
    )
    before = [sha256(video), sha256(workbook)]
    result = diagnose_alignment(raw, evidence, tmp_path / "report", [config])["recordings"]["V1"]
    assert result["video"]["frame_count"] == 30 and result["source_event_rows"] == 2
    hypotheses = result["anchors"][0]["hypotheses"]
    assert hypotheses["raw_zero_base"]["pts_seconds"] == 0.4
    assert hypotheses["time_at_30_fps"]["nearest_decoded_index"] == 8
    assert result["boxer_id"] is None and result["channel_id"] == "channel"
    assert "reversed_bounds" in result["issues"][0]["flags"]
    assert (tmp_path / "report" / "V1-anchors.md").is_file()
    assert before == [sha256(video), sha256(workbook)]
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="different inputs"):
        diagnose_alignment(raw, evidence, tmp_path / "wrong", [config])
    assert not (tmp_path / "wrong").exists()


def test_nearest_index_ties_lower_and_clamps_only_hypothesis_lookup():
    from fractions import Fraction

    times = [Fraction(0), Fraction(2), Fraction(4)]
    assert nearest_index(times, Fraction(1)) == 0
    assert nearest_index(times, Fraction(-10)) == 0
    assert nearest_index(times, Fraction(10)) == 2


def test_hypothesis_sheets_keep_actual_frame_indices_and_pts(tmp_path):
    from fractions import Fraction

    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from boxing_motion_ai.data.hypothesis_review import render_hypotheses
    from boxing_motion_ai.vision.contracts import DecodedFrame

    root = tmp_path / "raw"
    (root / "RGB_videos").mkdir(parents=True)
    video = root / "RGB_videos" / "V1.mkv"
    video.write_bytes(b"synthetic media identity")
    pts = [0, 40, 85, 125, 170, 210, 255, 300]
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "V1.ffprobe.json").write_text(
        json.dumps({"frames": [{"pts_time": str(n / 1000)} for n in pts]})
    )
    diagnostics = tmp_path / "alignment.json"
    diagnostics.write_text(
        json.dumps(
            {
                "recordings": {
                    "V1": {
                        "inputs": {"video_sha256": sha256(video)},
                        "anchors": [{"source_row": 2, "source_class": "Jab", "raw_bounds": [2, 4]}],
                    }
                }
            }
        )
    )

    def decoder(path, width, ranges):
        for i, t in enumerate(pts):
            selected = any(a <= i < b for a, b in ranges)
            yield DecodedFrame(
                i,
                t,
                Fraction(1, 1000),
                64,
                36,
                np.zeros((36, 64, 3), np.uint8) if selected else None,
            )

    result = render_hypotheses(root, evidence, diagnostics, "V1", [2], tmp_path / "review", decoder)
    assert result["cases"][0]["indices"] == [2, 3, 4]
    assert result["cases"][2]["indices"] == [2, 2, 3]
    assert all(r["pts"] == pts[r["frame_index"]] for r in result["decoded_frames"])
    assert (tmp_path / "review" / "row-2.jpg").is_file()
    assert (tmp_path / "review" / "review.md").is_file()
    assert video.read_bytes() == b"synthetic media identity"
