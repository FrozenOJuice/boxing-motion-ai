"""Synthetic retention and descriptive association tests; no classifier training."""

import json

import pytest
from typer.testing import CliRunner

from boxing_motion_ai.cli import app
from boxing_motion_ai.data.pose_quality import (
    ANCHORS,
    POLICIES,
    analyze_arrays,
    assess_quality,
    mutual_information,
    retain,
    score_validity,
)
from boxing_motion_ai.data.preparation import CLASS_MAP
from boxing_motion_ai.vision.pipeline import sha256


@pytest.fixture
def quality_inputs():
    np = pytest.importorskip("numpy")
    n = 24
    xyz = np.full((n, 33, 3), 0.5)
    conf = np.full((n, 33, 2), 0.99)
    candidates = []
    for i, label in enumerate(CLASS_MAP.values()):
        a, stop = i * 4, i * 4 + 4
        if label in {"cross", "rear_hook", "rear_uppercut"}:
            conf[a:stop, [13, 15], 0] = 0.1
        candidates.append(
            {
                "source_row": i + 2,
                "punch_type_provisional": label,
                "annotation_eligible": True,
                "exclusion_reasons": [],
                "array_slice_half_open": [a, stop],
            }
        )
    valid = score_validity(xyz, conf, 0.5)
    arrays = {
        "frame_index": np.arange(24),
        "coordinates": xyz,
        "confidence": conf,
        "valid_mask": valid,
        "pose_status": np.where(valid[:, 11:].all(axis=1), 0, 1),
        "scene_id": np.zeros(n, int),
    }
    return arrays, candidates


def test_partial_policy_keeps_usable_arm_without_promoting_invalid_joints(quality_inputs):
    np = pytest.importorskip("numpy")
    arrays, rows = quality_inputs
    before = {name: value.copy() for name, value in arrays.items()}
    result = analyze_arrays(arrays, rows, permutations=3)
    assert set(result["by_class"]) == set(CLASS_MAP.values())
    for label in ["cross", "rear_hook", "rear_uppercut"]:
        counts = result["by_class"][label]["retention"]
        assert counts["strict_body"] == counts["strict_arms"] == 0
        assert counts["mask_aware70"] == 1
        assert counts["mask_aware80"] == 0
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])
    assert (
        result["by_class"]["cross"]["joints"]["left_elbow"]["failure_reason_fractions"][
            "visibility_below_0.5"
        ]
        == 1.0
    )
    assert (
        result["by_class"]["cross"]["joints"]["left_elbow"]["failure_reason_fractions"][
            "presence_below_0.5"
        ]
        == 0
    )


def test_anchor_and_complete_arm_failures_not_covered_up(quality_inputs):
    arrays, rows = quality_inputs
    valid = arrays["valid_mask"][:4].copy()
    missing = arrays["pose_status"][:4] == 2
    valid[0, ANCHORS[0]] = False
    assert not retain(valid, missing, POLICIES["mask_aware70"])
    valid[:] = True
    valid[:, [13, 14]] = False
    assert not retain(valid, missing, POLICIES["mask_aware70"])
    assert retain(valid, missing, POLICIES["anchors_only_diagnostic"])
    with pytest.raises(ValueError, match="Empty"):
        retain(valid[:0], missing[:0], POLICIES["strict_body"])


def test_missing_frame_limit_and_honest_nan_quantiles(quality_inputs):
    np = pytest.importorskip("numpy")
    valid = np.ones((20, 33), dtype=bool)
    missing = np.zeros(20, dtype=bool)
    missing[:3] = True
    assert not retain(valid, missing, POLICIES["mask_aware70"])
    arrays, rows = quality_inputs
    arrays["coordinates"][4:8, 13] = np.nan
    arrays["confidence"][4:8, 13] = np.nan
    arrays["valid_mask"] = score_validity(arrays["coordinates"], arrays["confidence"], 0.5)
    result = analyze_arrays(arrays, rows, permutations=2)
    elbow = result["by_class"]["cross"]["joints"]["left_elbow"]
    assert elbow["visibility_q10_q50_q90"] == [None, None, None]
    assert elbow["failure_reason_fractions"]["nonfinite_coordinates"] == 1
    json.dumps(result, allow_nan=False)


def test_mi_constant_independent_and_perfect_association():
    pytest.importorskip("numpy")
    assert mutual_information([0, 0, 0, 0], [0, 1, 0, 1]) == pytest.approx(0)
    assert mutual_information([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0)
    assert mutual_information([0, 1, 0, 1], [0, 1, 0, 1]) == pytest.approx(1)
    with pytest.raises(ValueError):
        mutual_information([], [])


def test_no_class_silently_dropped_and_cut_crossing_rejected(quality_inputs):
    arrays, rows = quality_inputs
    with pytest.raises(ValueError, match="Missing source class"):
        analyze_arrays(arrays, rows[:-1], 2)
    arrays["scene_id"][2:] = 1
    with pytest.raises(ValueError, match="crosses"):
        analyze_arrays(arrays, rows, 2)


def test_sensitivity_does_not_change_saved_mask(quality_inputs):
    np = pytest.importorskip("numpy")
    arrays, rows = quality_inputs
    original = arrays["valid_mask"].copy()
    relaxed = score_validity(arrays["coordinates"], arrays["confidence"], 0.05)
    assert relaxed.all()
    np.testing.assert_array_equal(original, arrays["valid_mask"])
    with pytest.raises(ValueError, match="match"):
        arrays["valid_mask"] = relaxed
        analyze_arrays(arrays, rows, 2)


def test_report_writing_indices_reproducibility_and_preservation(quality_inputs, tmp_path):
    np = pytest.importorskip("numpy")
    arrays, rows = quality_inputs
    dataset = tmp_path / "data"
    dataset.mkdir()
    np.savez_compressed(dataset / "poses.npz", **arrays)
    (dataset / "candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "status": "complete",
        "npz_sha256": sha256(dataset / "poses.npz"),
        "split_status": "single synthetic group",
        "mapping_uncertainty": {"verified_frame_base": None},
    }
    (dataset / "preparation.json").write_text(json.dumps(manifest))
    hashes = {p.name: sha256(p) for p in dataset.iterdir()}
    a, b = tmp_path / "a", tmp_path / "b"
    first = assess_quality(dataset, a, 3)
    second = assess_quality(dataset, b, 3)
    assert first == second
    assert (a / "policy-index.json").read_bytes() == (b / "policy-index.json").read_bytes()
    assert json.loads((a / "policy-index.json").read_text())["selected_source_rows"] == list(
        range(2, 8)
    )
    assert "mask_aware70" in (a / "policies.md").read_text()
    assert hashes == {p.name: sha256(p) for p in dataset.iterdir()}
    with pytest.raises(ValueError, match="new report"):
        assess_quality(dataset, a, 3)
    manifest["npz_sha256"] = "wrong"
    (dataset / "preparation.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash"):
        assess_quality(dataset, tmp_path / "bad", 3)
    assert not (tmp_path / "bad").exists()


def test_cli_help_and_missing_inputs(tmp_path):
    runner = CliRunner()
    assert runner.invoke(app, ["assess-pose-quality", "--help"]).exit_code == 0
    result = runner.invoke(
        app,
        [
            "assess-pose-quality",
            "--dataset",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "output"),
        ],
    )
    assert result.exit_code == 1 and "failed" in result.output
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("source_offset", [0, 100])
def test_renderer_uses_exact_frames_and_rejects_wrong_source(
    quality_inputs, tmp_path, monkeypatch, source_offset
):
    from fractions import Fraction

    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from boxing_motion_ai.data import quality_review
    from boxing_motion_ai.vision.contracts import DecodedFrame

    arrays, rows = quality_inputs
    arrays["frame_index"] += source_offset
    arrays.update(
        pts=arrays["frame_index"] * 33,
        time_base_num=np.ones(24, int),
        time_base_den=np.full(24, 1000),
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    np.savez_compressed(dataset / "poses.npz", **arrays)
    video = tmp_path / "V5.mkv"
    video.write_bytes(b"synthetic no footage")
    (dataset / "preparation.json").write_text(
        json.dumps({"inputs": {"video_sha256": sha256(video)}})
    )
    output = tmp_path / "report"
    output.mkdir()
    (output / "quality.json").write_text(json.dumps(analyze_arrays(arrays, rows, 2)))

    def decoder(path, width):
        for i in range(source_offset + 24):
            yield DecodedFrame(
                i, i * 33, Fraction(1, 1000), 64, 36, np.zeros((36, 64, 3), np.uint8)
            )

    monkeypatch.setattr(quality_review, "decode_video", decoder)
    used = quality_review.render_quality_review(dataset, video, output)
    assert used == 18
    evidence = json.loads((output / "visual-review/source-frame-manifest.json").read_text())
    assert len(evidence["frames"]) == 18
    assert all(v["pts"] == 33 * v["frame_index"] for v in evidence["frames"])
    with pytest.raises(ValueError, match="already exists"):
        quality_review.render_quality_review(dataset, video, output)
    video.write_bytes(b"wrong source")
    with pytest.raises(ValueError, match="hash"):
        quality_review.render_quality_review(dataset, video, output)


def test_comparison_balances_before_quality_and_preserves_inputs(quality_inputs, tmp_path):
    from boxing_motion_ai.data.comparison import compare_recordings

    np = pytest.importorskip("numpy")
    arrays, rows = quality_inputs
    for i, row in enumerate(rows):
        row.update(
            source_class=list(CLASS_MAP)[i],
            flags=[],
            candidate_start_index=i * 4,
            candidate_end_index=i * 4 + 3,
        )
    datasets = []
    for video_id in ["V4", "V5"]:
        folder = tmp_path / video_id
        folder.mkdir()
        np.savez_compressed(folder / "poses.npz", **arrays)
        (folder / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (folder / "preparation.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "video_id": video_id,
                    "boxer_id": None,
                    "scope": "selected_windows",
                    "class_map": CLASS_MAP,
                    "npz_sha256": sha256(folder / "poses.npz"),
                }
            )
        )
        datasets.append(folder)
    hashes = [{p.name: sha256(p) for p in d.iterdir()} for d in datasets]
    result = compare_recordings(datasets, tmp_path / "comparison", 2)
    assert set(result["recordings"]) == {"V4", "V5"}
    assert (tmp_path / "comparison" / "comparison.md").is_file()
    assert result["recordings"]["V4"]["selected_source_rows"] == list(range(2, 8))
    assert all(
        v["retention"]["mask_aware70"] == 1
        for v in result["recordings"]["V4"]["quality"]["by_class"].values()
    )
    assert hashes == [{p.name: sha256(p) for p in d.iterdir()} for d in datasets]
    with pytest.raises(ValueError, match="distinct"):
        compare_recordings([datasets[0], datasets[0]], tmp_path / "duplicate", 2)


def test_absent_classes_explicit_zero_without_fake_joint_statistics(quality_inputs):
    arrays, rows = quality_inputs
    report = analyze_arrays(arrays, rows[:1], permutations=2, required_classes=())
    assert report["by_class"]["jab"]["eligible"] == 1
    cross = report["by_class"]["cross"]
    assert cross["eligible"] == 0 and cross["retention"]["mask_aware70"] == 0
    assert cross["joints"]["left_elbow"]["invalid_frame_fraction"] is None
    assert len(report["candidate_decisions"]) == 1
