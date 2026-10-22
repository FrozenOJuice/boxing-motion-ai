"""Synthetic full-preparation checks; no source data, downloads, inference or training."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from boxing_motion_ai.data import v6_preparation as review
from boxing_motion_ai.data.pose_quality import POLICIES
from boxing_motion_ai.vision.config import load_config
from boxing_motion_ai.vision.pipeline import sha256, write_json
from boxing_motion_ai.vision.recording import load_recording


def test_approved_full_run_keeps_original_mapping_and_independent_inference():
    original = load_recording("configs/v6_recording.yaml").model_dump()
    full = load_recording(review.RECORDING).model_dump()
    assert full == {**original, "excluded_source_rows": review.RECOMMENDED_HOLDS}
    settings = load_config(Path("configs/v6_pose_full.yaml"))
    assert settings.pose_mode == "image"
    assert settings.confidence == 0.5 and settings.max_width == 960
    assert settings.ranges == []  # Entire source, never a sample mistaken for full extraction.


def candidate(n, bounds=(0, 4), label="rear_uppercut", eligible=True):
    return dict(
        source_row=n,
        raw_start=bounds[0],
        raw_end=bounds[1] - 1,
        source_class=label,
        source_cells={"A": bounds[0], "B": bounds[1] - 1, "C": label},
        punch_type_provisional=label,
        flags=["provisional_mapping"],
        array_slice_half_open=list(bounds),
        annotation_eligible=eligible,
        exclusion_reasons=[] if eligible else ["known_questionable_source_row"],
    )


def test_selection_never_restores_holds_or_confuses_quality_with_labels():
    rows = [candidate(2), candidate(3), candidate(321, eligible=False)]
    counts = review.selection_counts(rows, [2])["rear_uppercut"]
    assert [counts[k] for k in ("source", "annotation_excluded", "pose_rejected", "retained")] == [
        3,
        1,
        1,
        1,
    ]
    for selection in ([321], [2, 2], [999]):
        with pytest.raises(ValueError):
            review.selection_counts(rows, selection)
    for n in review.RECOMMENDED_HOLDS:
        with pytest.raises(ValueError, match="cannot be restored"):
            review.selection_counts([candidate(n)], [n])


def test_source_fields_flags_and_existing_exclusions_survive_preparation():
    old = [
        dict(candidate(2), structurally_eligible=True),
        dict(candidate(3, eligible=False), structurally_eligible=False),
    ]
    new = deepcopy(old)
    review.check_annotation_preservation(new, old)
    for key, value in (
        ("raw_start", 1),
        ("source_class", "jab"),
        ("flags", []),
        ("source_cells", {}),
        ("punch_type_provisional", "jab"),
    ):
        bad = deepcopy(new)
        bad[0][key] = value
        with pytest.raises(ValueError, match="fields or flags"):
            review.check_annotation_preservation(bad, old)
    new[1]["annotation_eligible"] = True
    with pytest.raises(ValueError, match="exclusions"):
        review.check_annotation_preservation(new, old)


@pytest.fixture
def arrays():
    np = pytest.importorskip("numpy")
    n = 12
    xyz = np.full((n, 33, 3), 0.5)
    for j, xy in {
        11: (0.4, 0.3),
        12: (0.6, 0.3),
        13: (0.35, 0.4),
        14: (0.65, 0.4),
        15: (0.3, 0.5),
        16: (0.7, 0.5),
        23: (0.45, 0.7),
        24: (0.55, 0.7),
    }.items():
        xyz[:, j, :2] = xy
    return dict(
        coordinates=xyz,
        confidence=np.full((n, 33, 2), 0.9),
        valid_mask=np.ones((n, 33), bool),
        frame_index=np.arange(n),
        pts=np.arange(n) * 40,
        time_base_num=np.ones(n, int),
        time_base_den=np.full(n, 1000),
        pose_status=np.zeros(n, np.uint8),
        scene_id=np.zeros(n, int),
        segment_id=np.zeros(n, int),
    )


def test_exact_source_clock_and_validity_contract_cannot_change(arrays):
    np = pytest.importorskip("numpy")
    frames = [{"pts_time": str(i * 0.04)} for i in range(12)]
    review.check_arrays(arrays, frames)
    for field in ("frame_index", "pts", "valid_mask"):
        bad = {k: v.copy() for k, v in arrays.items()}
        bad[field][0] = False if field == "valid_mask" else 1
        with pytest.raises(ValueError):
            review.check_arrays(bad, frames)
    arrays["time_base_num"] = np.full(12, 2)
    with pytest.raises(ValueError, match="timestamps"):
        review.check_arrays(arrays, frames)


def test_geometry_is_diagnostic_and_gates_invalid_joints_and_scene_pairs(arrays):
    np = pytest.importorskip("numpy")
    row = candidate(2, bounds=(0, 12))
    arrays["coordinates"][:, 15, 0] += np.arange(12) * 0.001
    arrays["scene_id"][6:] = 1
    arrays["segment_id"][6:] = 1
    before = {k: v.copy() for k, v in arrays.items()}
    p = review.geometry_profile(arrays, [row], [2], 1)["clips"][0]
    assert p["left_wrist_speed_torso_per_second"] == pytest.approx(0.0625)
    assert p["left_wrist_speed_coverage"] == pytest.approx(10 / 11)
    for k in arrays:
        np.testing.assert_array_equal(arrays[k], before[k])
    arrays["valid_mask"][:, 15] = False
    arrays["coordinates"][:, 15] = np.nan
    p = review.geometry_profile(arrays, [row], [2], 1)["clips"][0]
    assert p["left_elbow_angle_degrees"] is None
    assert p["left_elbow_angle_coverage"] == 0
    assert p["left_wrist_speed_torso_per_second"] is None
    assert p["left_wrist_speed_coverage"] == 0


def test_whole_video_missing_runs_preserve_unknown_frames_and_exact_pts(arrays):
    arrays["pose_status"][4:8] = 2
    arrays["valid_mask"][4:8] = False
    result = review.whole_video_quality(arrays)
    assert result["status_counts"] == {0: 8, 2: 4}
    assert result["missing_run_count"] == 1
    assert result["longest_missing_runs"] == [
        {
            "frame_start": 4,
            "frame_stop_exclusive": 8,
            "frames": 4,
            "first_pts_seconds": "4/25",
            "last_pts_seconds": "7/25",
        }
    ]
    assert result["invalid_fraction_by_joint"]["left_elbow"] == pytest.approx(1 / 3)
    assert not arrays["valid_mask"][4:8].any()


def test_policy_explanation_uses_existing_limits_without_loosening_them():
    metrics = {**POLICIES["mask_aware70"], "missing_frames": 0.1}
    assert review.policy_failure_reasons(metrics) == []
    metrics.update(anchor_frames=0.94, core_entries=0.69, missing_frames=0.11)
    assert review.policy_failure_reasons(metrics) == [
        "anchor_frames",
        "core_entries",
        "missing_frames",
    ]


@pytest.fixture
def frozen(tmp_path, monkeypatch, arrays):
    np = pytest.importorskip("numpy")
    import yaml

    original = load_recording("configs/v6_recording.yaml").model_dump()
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    Path("configs/v6_recording.yaml").write_text(yaml.safe_dump(original))
    config = {**original, "excluded_source_rows": review.RECOMMENDED_HOLDS}
    Path("configs/v6_recording_provisional_v1.yaml").write_text(yaml.safe_dump(config))
    dataset, poses = review.DATASET, review.POSE_RUN
    dataset.mkdir(parents=True)
    poses.mkdir()
    arrays["confidence"][4:8, 14, 0] = 0.1
    arrays["valid_mask"][4:8, 14] = False
    arrays["confidence"][8:12, 23, 0] = 0.1
    arrays["valid_mask"][8:12, 23] = False
    arrays["pose_status"][4:] = 1
    np.savez_compressed(dataset / "poses.npz", **arrays)
    rows = [
        candidate(2, label="lead_hook"),
        candidate(3, (4, 8), "rear_hook"),
        candidate(4, (8, 12)),
        candidate(321, eligible=False),
    ]
    (dataset / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (poses / "poses.jsonl").write_text(
        json.dumps({"source_width": 1920, "source_height": 1080}) + "\n"
    )
    probe = Path("reports/boxingvi/development/V6.ffprobe.json")
    probe.parent.mkdir(parents=True)
    write_json(
        probe,
        {
            "frames": [{"pts_time": str(i * 0.04)} for i in range(12)],
            "streams": [{"width": 1920, "height": 1080}],
        },
    )
    inputs = {}
    decision = {
        "video_id": "V6",
        "evidence": "reports/synthetic-review.json",
        "source_group_id": "natebowerfitness-unresolved",
        "boxer_id": None,
        "session_id": None,
        "proposed_additional_quarantine": review.RECOMMENDED_HOLDS,
        "mapping": {"frame_offset": 0},
        "uncertainties": ["Synthetic only"],
    }
    for kind in ("video", "workbook"):
        p = Path(f"{kind}.synthetic")
        p.write_text(f"synthetic {kind}")
        inputs[kind + "_path"] = str(p)
        decision[kind + "_sha256"] = sha256(p)
    inputs["frame_inventory_sha256"] = sha256(probe)
    old = [dict(r, structurally_eligible=True) for r in rows]
    write_json(
        Path(decision["evidence"]),
        {
            "inputs": inputs,
            "inventory": {"rows": old},
            "recording_config_sha256": sha256("configs/v6_recording.yaml"),
        },
    )
    decision["evidence_sha256"] = sha256(decision["evidence"])
    review.REVIEW.write_text(yaml.safe_dump(decision))
    manifest = {
        "status": "complete",
        "scope": "full_video",
        "recording_config": config,
        "config": {
            "pose_mode": "image",
            "confidence": 0.5,
            "max_width": 960,
            "ranges_half_open": [],
            "scene_cuts": [],
        },
        "source_frame_count": 12,
        "processed_frame_count": 12,
        "poses_sha256": sha256(poses / "poses.jsonl"),
        "inputs": {k: {"sha256": decision[k + "_sha256"]} for k in ("video", "workbook")},
    }
    write_json(poses / "manifest.json", manifest)
    prep = {
        "status": "complete",
        "scope": "full_video",
        "video_id": "V6",
        "extracted_frame_count": 12,
        "npz_sha256": sha256(dataset / "poses.npz"),
        "config": {"excluded_source_rows": review.RECOMMENDED_HOLDS},
        "inputs": {
            "pose_manifest_sha256": sha256(poses / "manifest.json"),
            "poses_sha256": manifest["poses_sha256"],
        },
        "unknown_outside_annotation_bounds_frames": 0,
    }
    write_json(dataset / "preparation.json", prep)
    for v in ("V4", "V5", "V6"):
        ds = dataset if v == "V6" else Path(f"data/processed/boxingvi/{v}/candidates-full-v1")
        if v != "V6":
            ds.mkdir(parents=True)
            run = Path(f"data/processed/boxingvi/{v}/pose-full-image-v1")
            run.mkdir()
            (run / "poses.jsonl").write_bytes((poses / "poses.jsonl").read_bytes())
            for name in ("preparation.json", "poses.npz", "candidates.jsonl"):
                (ds / name).write_bytes((dataset / name).read_bytes())
        qroot = (
            review.ROOT / "quality/V6"
            if v == "V6"
            else Path(f"reports/boxingvi/expansion/quality/{v}")
        )
        qroot.mkdir(parents=True)
        hashes = {
            name: sha256(ds / name)
            for name in ("preparation.json", "poses.npz", "candidates.jsonl")
        }
        write_json(
            qroot / "quality.json",
            {
                "scope": "full_video",
                "policies": POLICIES,
                "inputs": hashes,
                "by_class": {},
                "candidate_decisions": [
                    {
                        "source_row": n,
                        "retained": {"mask_aware70": n in (2, 3)},
                        "metrics": {
                            "anchor_frames": 0.0 if n == 4 else 1.0,
                            "one_arm_frames": 1.0,
                            "core_entries": 0.9,
                            "missing_frames": 0.0,
                        },
                    }
                    for n in (2, 3, 4)
                ],
            },
        )
        write_json(
            qroot / "policy-index.json",
            {
                "video_id": v,
                "policy": "mask_aware70",
                "definition": POLICIES["mask_aware70"],
                "source_mask_threshold": 0.5,
                "max_missing_frame_fraction": 0.1,
                "inputs": hashes,
                "selected_source_rows": [2, 3],
            },
        )
    return tmp_path


def test_full_report_writing_exclusions_identity_and_original_preservation(frozen):
    inputs = {p: sha256(p) for p in frozen.rglob("*") if p.is_file()}
    result = review.finish(Path("reports/result"))
    assert result["exact_timing_verified_frames"] == 12
    assert result["class_counts"]["rear_hook"]["retained"] == 1
    assert result["class_counts"]["rear_uppercut"]["pose_rejected"] == 1
    assert result["pose_rejected_rows"] == [4]
    assert result["source_group_id"] == "natebowerfitness-unresolved"
    assert result["related_recordings_guard"] == ["V1", "V6"]
    assert result["boxer_id"] is None and result["session_id"] is None
    assert not result["training_performed"] and not result["representation_changed"]
    saved = json.loads(Path("reports/result/readiness.json").read_text())
    assert saved["class_counts"] == result["class_counts"]
    assert "rear_hook" in Path("reports/result/readiness.md").read_text()
    assert all(sha256(p) == digest for p, digest in inputs.items())
    with pytest.raises(ValueError, match="fresh directory"):
        review.finish(Path("reports/result"))


@pytest.mark.parametrize("change", ["review", "policy", "clock"])
def test_changed_evidence_policy_or_exact_clock_cannot_produce_success(frozen, change):
    if change == "review":
        Path("reports/synthetic-review.json").write_text("{}")
    elif change == "policy":
        p = review.ROOT / "quality/V6/policy-index.json"
        x = json.loads(p.read_text())
        x["source_mask_threshold"] = 0.3
        write_json(p, x)
    else:
        p = Path("reports/boxingvi/development/V6.ffprobe.json")
        x = json.loads(p.read_text())
        x["frames"][1]["pts_time"] = ".041"
        write_json(p, x)
    with pytest.raises(ValueError):
        review.finish(Path("reports/failure"))
    assert not Path("reports/failure").exists()
