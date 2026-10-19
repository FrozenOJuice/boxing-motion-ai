import json
from copy import deepcopy

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from boxing_motion_ai.audit import run_audit, validate_events, write_report
from boxing_motion_ai.boxingvi import Mapping
from boxing_motion_ai.cli import app
from boxing_motion_ai.models import BackgroundState, Event, PunchType

runner = CliRunner()


def codes(report):
    return {i["code"] for i in report["issues"]}


def test_missing_and_empty_inputs_write_honest_reports(tmp_path):
    root = tmp_path / "absent"
    for present in (False, True):
        if present:
            root.mkdir()
        result = runner.invoke(
            app,
            [
                "audit",
                "--dataset",
                "boxingvi",
                "--root",
                str(root),
                "--output",
                str(tmp_path / "report"),
            ],
        )
        assert result.exit_code == 1
        report = json.loads((tmp_path / "report/audit.json").read_text())
        assert report["status"] == "missing_input"
        assert report["dataset"]["found"] is False
        assert report["root_directory_exists"] is present
        assert report["counts"] == {"files": 0, "xlsx": 0, "videos": 0, "parsed_events": 0}
        assert {"missing_annotations", "missing_videos"} <= codes(report)
        markdown = (tmp_path / "report/audit.md").read_text()
        assert "unverified" in markdown
        assert "missing_input" in markdown
        assert "Original" not in report["class_balance_note"]


def test_cli_help_errors_and_output_safety(tmp_path):
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert "--mapping" in runner.invoke(app, ["audit", "--help"]).output
    assert runner.invoke(app, ["audit", "--dataset", "imaginary"]).exit_code == 2
    result = runner.invoke(
        app,
        [
            "audit",
            "--dataset",
            "boxingvi",
            "--root",
            str(tmp_path),
            "--output",
            str(tmp_path / "reports"),
        ],
    )
    assert result.exit_code == 2
    assert "outside" in result.output


def test_valid_mapping_counts_coverage_provenance(dataset, workbook, mapping, tmp_path):
    workbook(dataset, [[0, 9, "J"], [20, 29, "C"], [90, 99, "U"]])
    report = run_audit(dataset, mapping)
    assert report["structural_checks_passed"]
    assert report["status"] == "needs_review"
    assert report["phase_0_gate"] == "not_verified"
    assert report["class_balance_parsed_rows"]["jab"] == 1
    assert report["class_balance_parsed_rows"]["unknown"] == 1
    assert report["raw_class_counts_in_mapped_rows"] == {"J": 1, "C": 1, "U": 1}
    assert report["annotation_coverage"][0]["in_bounds_labeled_frames"] == 30
    assert report["annotation_coverage"][0]["unreviewed_or_unlabeled_frames"] == 70
    assert report["annotation_coverage"][0]["reviewed_no_punch_frames"] is None
    assert report["events"][0]["source_row"] == 2
    assert report["candidate_split_groups"]["boxer_id"] == {"synthetic-boxer": ["recording.mp4"]}
    assert len(report["mapping"]["sha256"]) == 64
    output = tmp_path / "report"
    write_report(report, output)
    assert json.loads((output / "audit.json").read_text()) == report
    assert str(dataset) not in (output / "audit.json").read_text()
    text = (output / "audit.md").read_text()
    for term in (
        "labels.xlsx",
        "recording.mp4",
        "Annotation coverage",
        "Class balance",
        "Candidate split groups",
        "Level 2",
        "Outstanding manual checks",
    ):
        assert term in text
    assert run_audit(dataset, mapping) == report


def test_unmapped_workbook_is_inspected_without_guesses(dataset, workbook):
    workbook(dataset, [[1, 5, "Jab"]])
    report = run_audit(dataset)
    assert "schema_needed" in codes(report)
    assert report["counts"]["parsed_events"] == 0
    assert report["sheets"][0]["nonempty_rows"] == 2
    assert report["sheets"][0]["preview"][1]["cells"] == ["1", "5", "Jab"]


def test_errors_duplicates_overlaps_and_union(dataset, workbook, mapping):
    workbook(
        dataset,
        [
            [0, 9, "J"],
            [0, 9, "J"],
            [5, 14, "C"],
            [95, 100, "J"],
            [10, 5, "J"],
            [True, 10, "J"],
            [1.5, 9, "J"],
            ["=1+1", 9, "J"],
            [10, 15, "unsupported"],
            [-1, 3, "J"],
        ],
    )
    report = run_audit(dataset, mapping)
    assert {
        "duplicate_event",
        "suspect_overlap",
        "out_of_bounds",
        "malformed_row",
        "unrecognized_class",
    } <= codes(report)
    assert report["annotation_coverage"][0]["in_bounds_labeled_frames"] == 15
    assert report["counts"]["parsed_events"] == 4
    assert report["status"] == "incomplete"


def test_missing_media_identities_and_missing_sheet(dataset, workbook, mapping):
    workbook(dataset, [[0, 9, "J"]])
    data = json.loads(mapping.read_text())
    data["sheets"][0].pop("boxer_id")
    data["sheets"][0].pop("session_id")
    mapping.write_text(json.dumps(data))
    (dataset / "recording.mp4").unlink()
    report = run_audit(dataset, mapping)
    assert {"missing_media", "missing_identity"} <= codes(report)
    data["sheets"][0]["sheet"] = "Absent"
    mapping.write_text(json.dumps(data))
    assert "missing_sheet" in codes(run_audit(dataset, mapping))


def test_corrupt_and_invalid_mapping_inputs(dataset, mapping):
    (dataset / "bad.xlsx").write_bytes(b"not a ZIP")
    report = run_audit(dataset, mapping)
    assert {"unreadable_workbook", "missing_workbook"} <= codes(report)
    mapping.write_text("{bad json")
    assert "invalid_mapping" in codes(run_audit(dataset, mapping))


@pytest.mark.parametrize(
    "change",
    [
        {"workbook": "../private.xlsx"},
        {"video": "/absolute.mp4"},
        {"video": "C:\\video.mp4"},
        {"frame_base": 7},
        {"end_convention": "guess"},
        {"columns": {"frame_start": 1, "frame_end": 1, "punch_type": 3}},
        {"class_map": {"X": "no_punch"}},
    ],
)
def test_mapping_contract_rejects_unsafe_or_ambiguous_config(mapping, change):
    data = json.loads(mapping.read_text())
    data["sheets"][0].update(change)
    with pytest.raises(ValidationError):
        Mapping.model_validate(data)


@pytest.mark.parametrize(
    "base,end,start,finish,expected",
    [
        (0, "inclusive", 0, 99, (0, 100)),
        (1, "inclusive", 1, 100, (0, 100)),
        (0, "exclusive", 0, 100, (0, 100)),
        (1, "exclusive", 1, 101, (0, 100)),
    ],
)
def test_frame_conventions(dataset, workbook, mapping, base, end, start, finish, expected):
    workbook(dataset, [[start, finish, "J"]])
    data = json.loads(mapping.read_text())
    data["sheets"][0].update(frame_base=base, end_convention=end)
    mapping.write_text(json.dumps(data))
    report = run_audit(dataset, mapping)
    event = Event.model_validate(report["events"][0])
    assert event.zero_based_interval() == expected
    assert "out_of_bounds" not in codes(report)
    assert report["annotation_coverage"][0]["labeled_fraction"] == 1


def test_event_duplicate_ids_and_timing(dataset, workbook, mapping):
    workbook(dataset, [[0, 9, "J"]])
    event = Event.model_validate(run_audit(dataset, mapping)["events"][0])
    issues = []
    validate_events([event, deepcopy(event)], {}, lambda *args: issues.append(args))
    assert "duplicate_id" in {i[0] for i in issues}
    with pytest.raises(ValidationError):
        Event.model_validate({**event.model_dump(), "frame_onset": 8, "frame_peak": 2})
    with pytest.raises(ValidationError):
        Event.model_validate({**event.model_dump(), "punch_type": "no_punch"})
    assert BackgroundState.NO_PUNCH.value not in [p.value for p in PunchType]


def test_symlink_outside_dataset_not_read(dataset, tmp_path):
    (tmp_path / "private.xlsx").write_bytes(b"not dataset data")
    (dataset / "linked.xlsx").symlink_to(tmp_path / "private.xlsx")
    report = run_audit(dataset)
    assert "symlink_skipped" in codes(report)
    assert report["counts"]["xlsx"] == 0


def test_unrelated_files_do_not_claim_dataset_found(tmp_path):
    (tmp_path / "README.txt").write_text("Synthetic unrelated content")
    report = run_audit(tmp_path)
    assert report["dataset"]["found"] is False
    assert report["status"] == "missing_input"
    assert report["counts"]["files"] == 1


def test_duplicate_mapping_does_not_duplicate_parsed_rows(dataset, workbook, mapping):
    workbook(dataset, [[0, 9, "J"]])
    data = json.loads(mapping.read_text())
    data["sheets"].append(deepcopy(data["sheets"][0]))
    mapping.write_text(json.dumps(data))
    report = run_audit(dataset, mapping)
    assert "duplicate_mapping" in codes(report)
    assert report["counts"]["parsed_events"] == 0


def test_incomplete_row_reports_location(dataset, workbook, mapping):
    workbook(dataset, [[0, None, "J"]])
    report = run_audit(dataset, mapping)
    diagnostic = next(i for i in report["issues"] if i["code"] == "malformed_row")
    assert diagnostic["location"] == "labels.xlsx:Punches:2"
    assert report["counts"]["parsed_events"] == 0


def test_cli_output_failure_is_actionable(tmp_path):
    output = tmp_path / "file"
    output.write_text("Cannot serve as a directory")
    result = runner.invoke(
        app,
        [
            "audit",
            "--dataset",
            "boxingvi",
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 2
    assert "Cannot read inputs or write reports" in result.output
