"""Synthetic video-independent preparation, quality, cut and provenance checks."""

import json
from types import SimpleNamespace

import pytest
from openpyxl import Workbook
from typer.testing import CliRunner

from boxing_motion_ai.cli import app
from boxing_motion_ai.data.preparation import PreparationConfig, prepare_v5
from boxing_motion_ai.vision.contracts import DecodedFrame, pose_record
from boxing_motion_ai.vision.pipeline import sha256


@pytest.fixture
def prepared_inputs(tmp_path):
    pytest.importorskip("numpy")
    from fractions import Fraction

    raw = tmp_path / "raw"
    raw.mkdir()
    workbook = raw / "V5.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.title = "Sheet1"
    sheet.append([None])
    events = [
        (1, 2, "Jab"),
        (4, 5, "Cross"),
        (7, 7, "Jab"),
        (9, 8, "Rear Hook"),
        (10, 12, "Lead Hook"),
        (14, 15, "Jab "),
    ]
    for n, values in enumerate(events, 2):
        sheet.append([*values, f"=A{n}-24", f"=B{n}-24"])
    book.save(workbook)
    book.close()
    run = tmp_path / "poses"
    run.mkdir()
    with (run / "poses.jsonl").open("w") as stream:
        for i in range(16):
            frame = DecodedFrame(
                i, i * 33, Fraction(1, 1000), 640, 360, SimpleNamespace(shape=(360, 640, 3))
            )
            points = [(0.5, 0.5, 0.1, 0.9, 0.9)] * 33
            if i == 4:
                points[15] = (0.5, 0.5, 0.1, 0.1, 0.9)
            if i == 5:
                points = None
            record = pose_record(frame, points, i * 33, 0.5)
            record.update(
                schema_version="pose-1.1",
                scene_id=int(i >= 11),
                segment_id=int(i >= 11),
                scene_cut_before=(i == 11),
            )
            stream.write(json.dumps(record) + "\n")
    manifest = {
        "status": "complete",
        "scope": "full_video",
        "source_frame_count": 16,
        "processed_frame_count": 16,
        "config": {"pose_mode": "image", "confidence": 0.5, "scene_cuts": [11]},
        "inputs": {"workbook": {"sha256": sha256(workbook)}, "video": {"sha256": "synthetic"}},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run, workbook


def test_candidate_counts_masks_unknown_and_preservation(prepared_inputs, tmp_path):
    import numpy as np

    run, workbook = prepared_inputs
    hashes = [sha256(run / "poses.jsonl"), sha256(workbook)]
    output = tmp_path / "prepared"
    result = prepare_v5(run, workbook, output)
    rows = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    assert result["status"] == "complete" and result["annotation_row_count"] == 6
    assert rows[0]["array_slice_half_open"] == [1, 3]
    assert rows[0]["annotation_eligible"] and rows[0]["pose_clean_candidate"]
    assert rows[1]["annotation_eligible"] and not rows[1]["pose_clean_candidate"]
    assert rows[1]["pose_quality"]["unreliable_frames"] == 1
    assert rows[1]["pose_quality"]["missing_frames"] == 1
    assert "equal_bounds_end_convention_unverified" in rows[2]["exclusion_reasons"]
    assert "reversed_bounds" in rows[3]["exclusion_reasons"]
    assert rows[3]["raw_start"] == 9 and rows[3]["raw_end"] == 8
    assert "crosses_confirmed_scene_cut" in rows[4]["exclusion_reasons"]
    assert not rows[4]["annotation_eligible"]
    assert rows[5]["source_class"] == "Jab "
    assert rows[5]["punch_type_provisional"] is None
    assert all(r["split"] is None and r["split_group"] == "V5" for r in rows)
    with np.load(output / "poses.npz", allow_pickle=False) as arrays:
        assert arrays["frame_index"].tolist() == list(range(16))
        assert arrays["pts"].tolist() == [i * 33 for i in range(16)]
        assert np.isnan(arrays["coordinates"][5]).all()
        assert not arrays["valid_mask"][5].any()
        assert arrays["coordinates"][4, 15, 0] == 0.5  # invalid, still raw
        assert not arrays["valid_mask"][4, 15]
        assert arrays["coordinates"].dtype == np.float64
    unknown = json.loads((output / "unknown-intervals.json").read_text())["intervals"]
    assert unknown[0] == {"frame_start": 0, "frame_stop_exclusive": 1, "state": "unknown"}
    assert all(u["state"] == "unknown" for u in unknown)
    counts = result["class_counts"]["cross"]
    assert counts["annotation_eligible"] == 1 and counts["eligible_with_any_pose_issue"] == 1
    assert "cross" in result["classes_without_pose_clean_candidates"]
    assert result["training_readiness"] == "needs_pose_quality_review_and_independent_split"
    assert counts["eligible_all_arm_joints_valid"] == 0
    assert (
        counts["eligible_with_unreliable_pose"] == 1 and counts["eligible_with_missing_pose"] == 1
    )
    assert "not certified" in (output / "preparation.md").read_text()
    assert hashes == [sha256(run / "poses.jsonl"), sha256(workbook)]


def test_reproducible_candidate_output_and_known_row_exclusion(prepared_inputs, tmp_path):
    run, workbook = prepared_inputs
    a, b = tmp_path / "a", tmp_path / "b"
    config = PreparationConfig(excluded_source_rows=[2], scene_guard_frames=0)
    prepare_v5(run, workbook, a, config)
    prepare_v5(run, workbook, b, config)
    assert (a / "candidates.jsonl").read_bytes() == (b / "candidates.jsonl").read_bytes()
    rows = [json.loads(line) for line in (a / "candidates.jsonl").read_text().splitlines()]
    assert not rows[0]["pose_clean_candidate"]
    assert "known_questionable_source_row" in rows[0]["exclusion_reasons"]


def test_reject_mismatched_workbook_and_sample_scope(prepared_inputs, tmp_path):
    run, workbook = prepared_inputs
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["scope"] = "selected_windows"
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="full-video"):
        prepare_v5(run, workbook, tmp_path / "sample")
    manifest["scope"] = "full_video"
    manifest["inputs"]["workbook"]["sha256"] = "wrong"
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash differs"):
        prepare_v5(run, workbook, tmp_path / "mismatch")
    assert not (tmp_path / "mismatch").exists()


@pytest.mark.parametrize("mutation", ["truncate", "reorder", "mask", "cut", "video"])
def test_reject_corrupt_or_unsafe_pose_sequence(prepared_inputs, tmp_path, mutation):
    run, workbook = prepared_inputs
    records = [json.loads(line) for line in (run / "poses.jsonl").read_text().splitlines()]
    if mutation == "truncate":
        records.pop()
    elif mutation == "reorder":
        records[0], records[1] = records[1], records[0]
    elif mutation == "mask":
        records[4]["valid_mask"][15] = True
    elif mutation == "cut":
        records[11]["segment_id"] = 0
    elif mutation == "video":
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["config"]["pose_mode"] = "video"
        (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "poses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        prepare_v5(run, workbook, output)
    assert not output.exists()


def test_preparation_cli_and_existing_output(prepared_inputs, tmp_path):
    run, workbook = prepared_inputs
    runner = CliRunner()
    assert runner.invoke(app, ["prepare-v5", "--help"]).exit_code == 0
    output = tmp_path / "cli"
    result = runner.invoke(
        app,
        [
            "prepare-v5",
            "--pose-run",
            str(run),
            "--workbook",
            str(workbook),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    with pytest.raises(ValueError, match="already exists"):
        prepare_v5(run, workbook, output)
    with pytest.raises(ValueError, match="outside"):
        prepare_v5(run, workbook, workbook.parent / "prepared")


def test_scene_guard_excludes_nearby_non_crossing_candidates(prepared_inputs, tmp_path):
    run, workbook = prepared_inputs
    output = tmp_path / "guard"
    prepare_v5(run, workbook, output, PreparationConfig(scene_guard_frames=7))
    rows = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    assert "near_confirmed_scene_cut" in rows[1]["exclusion_reasons"]
    assert "crosses_confirmed_scene_cut" not in rows[1]["exclusion_reasons"]
    assert not rows[1]["annotation_eligible"]
    assert rows[0]["annotation_eligible"]


def test_failed_array_write_never_reports_completed_dataset(prepared_inputs, tmp_path, monkeypatch):
    import numpy as np

    run, workbook = prepared_inputs
    output = tmp_path / "failed"

    def fail(*args, **kwargs):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="synthetic"):
        prepare_v5(run, workbook, output)
    report = json.loads((output / "preparation.json").read_text())
    assert report["status"] == "failed"
    assert "write failure" in report["error"]
    assert not (output / "preparation.md").exists()
