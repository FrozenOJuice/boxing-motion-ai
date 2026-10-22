"""Synthetic review diagnostics: no source video, workbook or checkpoint access."""

import json
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")
yaml = pytest.importorskip("yaml")

from boxing_motion_ai.training.forensic_report import (  # noqa: E402
    summarize,
    validate_review,
    write_review,
)
from boxing_motion_ai.training.forensic_review import (  # noqa: E402
    dense_review,
    pose_stats,
    render_sheet,
    review_indices,
    select_cases,
)
from boxing_motion_ai.vision.pipeline import sha256  # noqa: E402


def arrays(n=30):
    return dict(
        frame_index=np.arange(n),
        pts=np.arange(n) * 1001,
        time_base_num=np.ones(n, int),
        time_base_den=np.full(n, 30000),
        scene_id=np.zeros(n, int),
        segment_id=np.zeros(n, int),
        coordinates=np.full((n, 33, 3), 0.5),
        confidence=np.ones((n, 33, 2)),
        valid_mask=np.ones((n, 33), bool),
    )


def case(row=1):
    return dict(
        id=f"V4:{row}",
        recording_id="V4",
        source_row=row,
        expected="rear_hook",
        punch_type_provisional="rear_hook",
        candidate_start_index=10,
        candidate_end_index=15,
        approved_candidate=True,
        raw_start=10,
        raw_end=15,
        raw_bounds=[10, 15],
        sheet="V4-row-1.jpg",
        prediction=dict(target="rear_hook", prediction="lead_hook"),
    )


def assessment():
    return dict(
        family="consistent",
        side="ambiguous",
        boundary="consistent",
        pose="aligned",
        observation="Visible curved arm motion; stance cannot be certified.",
    )


def config():
    return dict(
        schema_version="label-pose-review-1.0",
        status="assistant_visual_review_not_human_ground_truth",
        diagnostic_only=True,
        annotation_overrides=[],
        new_exclusions=[],
        human_review_status="pending",
        candidate_evaluation_subset=["V4:1"],
        priority_disputes=[],
        assessments={"V4:1": assessment()},
    )


def test_selection_is_stratified_unique_deterministic_without_confidence_ranking():
    rows = [case(i) for i in range(1, 10)]
    for r in rows:
        r["candidate_start_index"] = r["source_row"] * 10
    predictions = [
        dict(
            source_row=r["source_row"],
            target="rear_hook",
            prediction="rear_hook" if r["source_row"] % 2 else "lead_hook",
        )
        for r in rows
    ]
    selected = select_cases(rows, predictions, [0, 0, 0, 3, 0, 0])
    assert len({c["source_row"] for c in selected}) == 3
    assert selected == select_cases(list(reversed(rows)), predictions, [0, 0, 0, 3, 0, 0])
    lookup = {p["source_row"]: p for p in predictions}
    assert [lookup[c["source_row"]]["prediction"] == "rear_hook" for c in selected] == [
        False,
        True,
        False,
    ]
    with pytest.raises(ValueError, match="support"):
        select_cases(rows, predictions, [1, 0, 0, 0, 0, 0])
    predictions[0]["target"] = "jab"
    with pytest.raises(ValueError, match="source label"):
        select_cases(rows, predictions, [0, 0, 0, 1, 0, 0])
    with pytest.raises(ValueError, match="exactly"):
        select_cases(rows, predictions[:-1], [0, 0, 0, 1, 0, 0])


def test_context_uses_exact_pts_and_stops_at_cuts_gaps_and_segments():
    a = arrays()
    # 6 frames at 29.97 would be .2002s; it must not round to 30 FPS.
    indices = review_indices(a, 10, 15)
    assert min(indices) == 5 and max(indices) == 20
    a["scene_id"][:8] = 1
    assert min(review_indices(a, 10, 15)) == 8
    a["segment_id"][18:] = 1
    assert max(review_indices(a, 10, 15)) == 17
    a["frame_index"][:10] -= 1
    assert min(review_indices(a, 10, 15)) == 10
    a["pts"][11] = a["pts"][10]
    with pytest.raises(ValueError, match="increase"):
        review_indices(a, 10, 15)
    with pytest.raises(ValueError, match="repair"):
        review_indices(arrays(), 15, 10)


def test_short_interval_does_not_invent_frames_or_trust_missing_scores():
    a = arrays(2)
    assert review_indices(a, 0, 0) == [0, 1]
    a["confidence"][:, 13] = np.nan
    a["valid_mask"][:, 13] = False
    stats = pose_stats(a, 0, 1, include_failures=True)["L"]["13"]
    assert stats["valid_frames"] == 0 and stats["median_visibility"] is None
    assert stats["nonfinite_scores"] == 2 and stats["both_scores_ge_080"] == 0
    json.dumps(stats, allow_nan=False)


def test_sheet_keeps_labels_and_predictions_hidden_and_preserves_arrays(tmp_path):
    a = arrays()
    a["coordinates"][10, 13] = np.nan
    a["valid_mask"][10, 13] = False
    before = {k: v.copy() for k, v in a.items()}
    output = tmp_path / "sheet.jpg"
    images = {i: Image.new("RGB", (160, 90), "gray") for i in (10, 15)}
    times = [Fraction(int(p), 30000) for p in a["pts"]]
    render_sheet("V4", case(), a, [10, 15], images, times, output)
    with Image.open(output) as rendered:
        assert rendered.size == (1120, 992)
    for key in a:
        np.testing.assert_equal(a[key], before[key])


def test_assessments_are_not_certified_labels_or_automatic_corrections():
    c, cfg = case(), config()
    validate_review([c], cfg)
    result = summarize([c], cfg["assessments"])
    assert result["source_label_comparison"] == {"within_family_side_disagreement": 1}
    assert result["clear_cases_with_source_label_disagreement"] == ["V4:1"]
    assert result["human_verified_cases"] == 0 and not result["new_exclusions"]
    for key, bad in [
        ("annotation_overrides", [dict(label="jab")]),
        ("new_exclusions", ["V4:1"]),
        ("human_review_status", "verified"),
    ]:
        changed = deepcopy(cfg)
        changed[key] = bad
        with pytest.raises(ValueError, match="cannot correct"):
            validate_review([c], changed)
    cfg["assessments"]["V4:1"]["family"] = "model_is_right"
    with pytest.raises(ValueError, match="evidence"):
        validate_review([c], cfg)


def test_reserved_recording_rejected_before_dense_source_reads(tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        dense_review(tmp_path / "nonexistent.json", Path("reports/new-review"), ["V9:1"])
    c = case()
    c.update(id="V9:1", recording_id="V9")
    cfg = config()
    cfg["assessments"] = {"V9:1": assessment()}
    with pytest.raises(ValueError, match="Reserved"):
        validate_review([c], cfg)


def test_report_writing_is_portable_pending_and_does_not_modify_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    review = Path("reports/source")
    review.mkdir(parents=True)
    cases_path = review / "cases.json"
    cases_path.write_text(json.dumps([case()]))
    (review / "protected.json").write_text("{}")
    root = Path("data/processed/boxingvi/V4/candidates-full-v1")
    root.mkdir(parents=True)
    np.savez(root / "poses.npz", **arrays())
    cfg = config()
    cfg["case_manifest_sha256"] = sha256(cases_path)
    config_path = Path("assessment.yaml")
    config_path.write_text(yaml.safe_dump(cfg))
    before = {p: sha256(p) for p in (cases_path, root / "poses.npz", config_path)}
    output = Path("reports/assessment")
    dense = review / "dense.json"
    dense.write_text(json.dumps([dict(id="V4:1", sheet="dense-1.jpg", frames=[10, 11])]))
    write_review(cases_path, config_path, output, dense_path=dense)
    queue = (output / "human-review.html").read_text()
    assert "../source/dense-1.jpg" in queue
    assert "Reveal hypotheses, not ground truth" in queue
    assert "../source/index.html#V4:1" in (output / "REVIEW.md").read_text()
    pending = json.loads((output / "human-review.json").read_text())
    assert pending[0]["status"] == "pending_human_review"
    assert pending[0]["observed_family"] is None
    assert before == {p: sha256(p) for p in before}
    with pytest.raises(ValueError, match="fresh"):
        write_review(cases_path, config_path, output)
    cases_path.write_text("[]")
    with pytest.raises(ValueError, match="manifest changed"):
        write_review(cases_path, config_path, "reports/changed-input")
