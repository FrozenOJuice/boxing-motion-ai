"""Synthetic expansion safety and provenance, never source videos or V9."""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from boxing_motion_ai.data.expansion_review import (
    select_review_rows,
    summarize_rows,
    timing_hypotheses,
    write_review_html,
)


def row(n, start, end, label="Jab", overlaps=()):
    return {
        "source_row": n,
        "raw_start": start,
        "raw_end": end,
        "source_class": label,
        "candidate_start_index": start,
        "candidate_end_index": end,
        "flags": ["provisional_mapping", "class_unverified", "boundaries_unverified"]
        + (["suspect_overlap_inclusive_candidate"] if overlaps else []),
        "overlapping_source_rows": list(overlaps),
    }


def test_overlap_and_shared_boundary_stay_excluded_and_input_unchanged():
    rows = [
        row(2, 10, 20, overlaps=[3]),
        row(3, 20, 30, overlaps=[2, 4]),
        row(4, 28, 35, overlaps=[3]),
        row(5, 40, 50),
        row(6, 60, 80, "Jab "),
    ]
    original = json.dumps(rows)
    report = summarize_rows(rows, 100, {"Jab": "jab"})
    assert [p["kind"] for p in report["overlap_pairs"]] == ["shared_endpoint", "strict_overlap"]
    assert report["by_class"]["jab"]["structurally_eligible"] == 1
    assert report["by_class"]["rear_uppercut"]["source"] == 0
    assert report["unmapped_rows"] == [6]
    assert not any(r["training_approved"] for r in report["rows"])
    assert json.dumps(rows) == original


def test_timing_alternatives_use_actual_pts_and_do_not_certify_offsets():
    times = [Fraction(i * 1001, 30000) for i in range(2100)]
    hypotheses = timing_hypotheses(row(2, 2000, 2010), times)
    assert hypotheses["raw_zero_based"]["indices_inclusive_hypothesis"] == [2000, 2010]
    assert hypotheses["raw_one_based"]["indices_inclusive_hypothesis"] == [1999, 2009]
    assert hypotheses["annotation_clock_30"]["indices_inclusive_hypothesis"] == [1998, 2008]
    assert hypotheses["annotation_clock_30000/1001"]["indices_inclusive_hypothesis"] == [2000, 2010]
    outside = timing_hypotheses(row(3, 3000, 3010), times)
    assert not outside["annotation_clock_30"]["in_bounds"]
    assert outside["annotation_clock_30"]["indices_inclusive_hypothesis"] is None


def test_review_selection_spans_each_class_includes_flagged_rows():
    rows = [row(i + 2, i * 10, i * 10 + 5) for i in range(9)]
    selected = select_review_rows(rows, {"Jab": "jab"})
    assert [r["source_row"] for r in selected] == [2, 6, 10]


def test_fully_nested_different_class_boundaries_are_selected_for_review():
    rows = [row(2, 100, 120, "Rear Uppercut", [3]), row(3, 105, 108, "Lead Uppercut", [2])]
    selected = select_review_rows(
        rows, {"Rear Uppercut": "rear_uppercut", "Lead Uppercut": "lead_uppercut"}
    )
    assert [r["source_row"] for r in selected] == [2, 3]


def test_portable_html_has_exact_frame_scrubbing_and_no_network(tmp_path):
    r = row(2, 0, 1)
    r.update(review_indices=[0, 1], hypotheses={})
    write_review_html(tmp_path, [r], [Fraction(0), Fraction(1001, 30000)])
    text = (tmp_path / "review.html").read_text()
    assert '"exact_pts_seconds": "1001/30000"' in text
    assert "frames/${f.index}.jpg" in text and "Previous frame" in text
    assert "https://" not in text and "PAYLOAD" not in text


def synthetic_expansion(tmp_path, video):
    import numpy as np
    from test_level1 import synthetic

    from boxing_motion_ai.vision.pipeline import sha256

    spec = synthetic(tmp_path, video)
    root = Path(spec["dataset"])
    if video in ("V7", "V8", "V10"):
        with np.load(root / "poses.npz", allow_pickle=False) as arrays:
            data = {k: arrays[k] for k in arrays.files}
        data["coordinates"][:, 16, 0] += {"V7": 0.01, "V8": 0.02, "V10": 0.03}[video]
        np.savez_compressed(root / "poses.npz", **data)
    manifest = json.loads((root / "preparation.json").read_text())
    manifest["schema_version"] = "recording-preparation-1.1"
    manifest["npz_sha256"] = sha256(root / "poses.npz")
    (root / "preparation.json").write_text(json.dumps(manifest))
    index = json.loads(Path(spec["policy"]).read_text())
    index["inputs"]["preparation.json"] = sha256(root / "preparation.json")
    index["inputs"]["poses.npz"] = sha256(root / "poses.npz")
    Path(spec["policy"]).write_text(json.dumps(index))
    return spec


def test_grouped_loader_preserves_recording_qualified_rows_and_unknown_identity(tmp_path):
    pytest.importorskip("torch")
    from boxing_motion_ai.training.groups import GROUPS, GroupedClips, validate_split

    specs = [synthetic_expansion(tmp_path, v) for v in ["V4", "V7", "V8"]]
    dev = synthetic_expansion(tmp_path, "V5")
    validate_split(specs, dev)
    ds = GroupedClips(specs)
    assert len(ds) == 18
    assert GROUPS["V7"] == GROUPS["V8"] != GROUPS["V5"]
    examples = ds.provenance["examples"]
    assert len({(c["recording_id"], c["source_row"]) for c in examples}) == 18
    assert all(c["boxer_id"] is None and c["session_id"] is None for c in ds.clips)
    assert ds.provenance["class_counts"]["rear_hook"] == 3
    assert not ds[0]["raw"]["valid_mask"][0, 13]


def test_v9_rejected_before_file_access(monkeypatch):
    pytest.importorskip("torch")
    from boxing_motion_ai.training.data import PoseClips
    from boxing_motion_ai.training.groups import validate_split

    def forbidden(*args, **kwargs):
        pytest.fail("Reserved input accessed")

    monkeypatch.setattr(Path, "read_text", forbidden)
    with pytest.raises(ValueError, match="V9"):
        PoseClips({}, "V9", expansion=True)
    with pytest.raises(ValueError, match="V9"):
        validate_split([{"recording": "V4"}, {"recording": "V7"}, {"recording": "V9"}], {})


def test_expansion_requires_frozen_gate_and_unchanged_hashes(tmp_path):
    pytest.importorskip("torch")
    from boxing_motion_ai.training.expansion_study import check_gate
    from boxing_motion_ai.training.groups import GROUPS

    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": "expansion-training-gate-1.0",
                "training_ready": False,
                "training_specs": [],
                "source_groups": GROUPS,
            }
        )
    )
    with pytest.raises(ValueError, match="gate"):
        check_gate(gate, [])
    from boxing_motion_ai.vision.pipeline import sha256

    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")
    gate.write_text(
        json.dumps(
            {
                "schema_version": "expansion-training-gate-1.0",
                "training_ready": True,
                "training_specs": [],
                "source_groups": GROUPS,
                "review_path": str(evidence),
                "review_sha256": sha256(evidence),
                "evidence_path": str(evidence),
                "evidence_sha256": sha256(evidence),
            }
        )
    )
    assert check_gate(gate, [])["training_ready"]
    evidence.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="evidence changed"):
        check_gate(gate, [])


def test_missing_class_allowed_only_as_part_of_expansion(tmp_path):
    pytest.importorskip("torch")
    from boxing_motion_ai.training.data import PoseClips
    from boxing_motion_ai.vision.pipeline import sha256

    spec = synthetic_expansion(tmp_path, "V7")
    root = Path(spec["dataset"])
    rows = [json.loads(line) for line in (root / "candidates.jsonl").read_text().splitlines()]
    rows[5].update(annotation_eligible=False, exclusion_reasons=["suspect_overlap"])
    (root / "candidates.jsonl").write_text("\n".join(map(json.dumps, rows)))
    index = json.loads(Path(spec["policy"]).read_text())
    index["selected_source_rows"].remove(7)
    index["inputs"]["candidates.jsonl"] = sha256(root / "candidates.jsonl")
    Path(spec["policy"]).write_text(json.dumps(index))
    ds = PoseClips(spec, "V7", expansion=True)
    assert len(ds) == 5 and ds.provenance["class_counts"]["rear_uppercut"] == 0
    assert ds.provenance["excluded_candidates"][0]["source_row"] == 7
    with pytest.raises(ValueError, match="Only V4"):
        PoseClips(spec, "V7")


def test_full_dataset_json_and_markdown_report_preserve_absent_quality(tmp_path):
    pytest.importorskip("torch")
    import numpy as np

    from boxing_motion_ai.training.data import CLASSES
    from boxing_motion_ai.training.expansion_study import dataset_report

    specs = []
    for v in ["V4", "V5", "V7", "V8"]:
        root = tmp_path / v
        root.mkdir()
        spec = {"recording": v, "dataset": str(root), "policy": str(root / "policy.json")}
        specs.append(spec)
        quality = {
            "scope": "full_video",
            "by_class": {
                c: {
                    "eligible": 0,
                    "retention": {"mask_aware70": 0},
                    "joints": {"left_elbow": {"invalid_frame_fraction": None}},
                }
                for c in CLASSES
            },
        }
        (root / "quality.json").write_text(json.dumps(quality))
        (root / "preparation.json").write_text(
            json.dumps(
                {
                    "source_frame_count": 100,
                    "unknown_outside_annotation_bounds_frames": 100,
                    "class_counts": {c: {"source_rows": 0} for c in CLASSES},
                }
            )
        )
        (root / "candidates.jsonl").write_text("")
        np.savez(
            root / "poses.npz",
            valid_mask=np.ones((100, 33), dtype=bool),
            pose_status=np.zeros(100, dtype=np.uint8),
        )
    dataset_report([s for s in specs if s["recording"] != "V5"], tmp_path, specs[1])
    report = json.loads((tmp_path / "dataset-summary.json").read_text())
    assert report["source_groups"]["V7"] == report["source_groups"]["V8"]
    assert report["recordings"]["V7"]["classes"]["rear_uppercut"]["mask_aware70"] == 0
    assert "n/a" in (tmp_path / "dataset-summary.md").read_text()


def test_flagged_pose_quality_is_diagnostic_only(tmp_path):
    pytest.importorskip("torch")
    import numpy as np

    from boxing_motion_ai.training.expansion_study import diagnostic_visibility

    valid = np.ones((10, 33), dtype=bool)
    valid[:, 13] = False
    path = tmp_path / "poses.npz"
    np.savez(path, valid_mask=valid, pose_status=np.ones(10, dtype=np.uint8))
    rows = [
        {
            "source_row": 2,
            "punch_type_provisional": "rear_uppercut",
            "annotation_eligible": False,
            "array_slice_half_open": [0, 10],
        }
    ]
    result = diagnostic_visibility(rows, path)
    d = result["rear_uppercut"]
    assert d["annotation_excluded_rows"] == d["diagnostic_mask_policy_passes"] == 1
    assert d["visibility"]["left_elbow"] == 1.0
    assert result["rear_hook"]["visibility"] is None
    assert rows[0]["annotation_eligible"] is False


def test_expansion_is_optional_and_visibility_context_cannot_be_enabled(tmp_path):
    pytest.importorskip("torch")
    import yaml

    from boxing_motion_ai.training.experiment import load_config

    config = yaml.safe_load(Path("configs/level1_initial.yaml").read_text())
    config.update(
        schema_version="level1-experiment-1.4",
        mirror_probability=0.5,
        expansion_gate="reports/synthetic/gate.json",
        runs=["tcn", "tcn_mask_only"],
    )
    config["train"] = [
        {**config["train"], "recording": v, "dataset": f"synthetic/{v}"} for v in ["V4", "V7", "V8"]
    ]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    assert load_config(path)["mirror_probability"] == 0.5
    config["context_seconds"] = 0.65
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="configuration fields"):
        load_config(path)


def test_expanded_training_smoke_preserves_recording_provenance(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import yaml

    from boxing_motion_ai.training.data import PoseClips
    from boxing_motion_ai.training.experiment import run_experiment
    from boxing_motion_ai.training.groups import GROUPS
    from boxing_motion_ai.vision.pipeline import sha256

    specs = [synthetic_expansion(tmp_path, v) for v in ["V4", "V7", "V8"]]
    development = synthetic_expansion(tmp_path, "V5")
    evidence = tmp_path / "review.json"
    evidence.write_text('{"synthetic_fixture_only": true}')
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": "expansion-training-gate-1.0",
                "training_ready": True,
                "training_specs": specs,
                "source_groups": GROUPS,
                "review_path": str(evidence),
                "review_sha256": sha256(evidence),
                "evidence_path": str(evidence),
                "evidence_sha256": sha256(evidence),
                "recordings": {
                    s["recording"]: PoseClips(s, s["recording"], expansion=True).provenance
                    for s in specs
                    if s["recording"] != "V4"
                },
            }
        )
    )
    config = yaml.safe_load(Path("configs/level1_initial.yaml").read_text())
    config.update(
        schema_version="level1-experiment-1.4",
        seed=19,
        epochs=1,
        hidden=4,
        threads=1,
        train=specs,
        development=development,
        mirror_probability=0.5,
        expansion_gate=str(gate),
        runs=["tcn", "tcn_mask_only"],
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "reports/synthetic-expanded"
    run_experiment(path, output, progress=lambda _: None)
    predictions = json.loads((output / "tcn/train_predictions.json").read_text())
    assert {r["recording_id"] for r in predictions} == {"V4", "V7", "V8"}
    assert len([r for r in predictions if r["source_row"] == 2]) == 3
    logs = [
        (output / model / "augmentation.json").read_text() for model in ["tcn", "tcn_mask_only"]
    ]
    assert logs[0] == logs[1]
    assert len(json.loads(logs[0])[0]["example_order"]) == 18
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["context_policy"]["seconds"] == 0
    assert provenance["visibility_augmentation"]["probability"] == 0
