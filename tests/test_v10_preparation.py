"""Synthetic V10 readiness evidence; no real media, inference or training."""

import json
from copy import deepcopy

import pytest

from boxing_motion_ai.data import v10_preparation as review
from boxing_motion_ai.data.pose_quality import POLICIES
from boxing_motion_ai.data.preparation import SOURCE_LABELS
from boxing_motion_ai.vision.pipeline import sha256, write_json
from boxing_motion_ai.vision.recording import load_recording


def candidate(n, label="rear_uppercut", eligible=True, bounds=(0, 4)):
    return {
        "source_row": n,
        "punch_type_provisional": label,
        "source_class": label,
        "annotation_eligible": eligible,
        "exclusion_reasons": [] if eligible else ["known_questionable_source_row"],
        "array_slice_half_open": list(bounds),
        "raw_start": bounds[0],
        "raw_end": bounds[1] - 1,
    }


def test_selection_keeps_exclusions_separate_from_pose_rejections():
    rows = [candidate(2), candidate(3), candidate(9, eligible=False)]
    before = deepcopy(rows)
    counts = review.summarize_selection(rows, [2])
    assert counts["rear_uppercut"] == {
        "source": 3,
        "annotation_eligible": 2,
        "annotation_excluded": 1,
        "pose_rejected": 1,
        "retained": 1,
        "retained_source_rows": [2],
    }
    assert counts["jab"]["retained"] == 0
    assert rows == before
    for selected in ([9], [2, 2], [999]):
        with pytest.raises(ValueError):
            review.summarize_selection(rows, selected)


def test_quarantine_cannot_be_cleared_by_marking_row_eligible():
    for n in review.QUARANTINED:
        with pytest.raises(ValueError, match="cannot be restored"):
            review.summarize_selection([candidate(n)], [n])


def test_exact_timing_rejects_frame_drop_one_frame_offset_and_clock_drift():
    np = pytest.importorskip("numpy")
    arrays = {
        "frame_index": np.arange(3),
        "pts": np.array([0, 40, 80]),
        "time_base_num": np.ones(3, int),
        "time_base_den": np.full(3, 1000),
    }
    frames = [{"pts_time": t} for t in ["0", "0.040", "0.080"]]
    review.check_timing(arrays, frames)
    for key, change in [("frame_index", [0, 1, 3]), ("pts", [40, 80, 120]), ("pts", [0, 40, 81])]:
        bad = {k: v.copy() for k, v in arrays.items()}
        bad[key] = np.array(change)
        with pytest.raises(ValueError):
            review.check_timing(bad, frames)


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    recording = load_recording(review.RECORDING).model_dump()
    monkeypatch.chdir(tmp_path)
    import yaml

    config = tmp_path / "recording.yaml"
    config.write_text(yaml.safe_dump(recording))
    monkeypatch.setattr(review, "RECORDING", config)
    base = tmp_path / "reports/synthetic"
    dataset, poses = base / "dataset", base / "poses"
    dataset.mkdir(parents=True)
    poses.mkdir()
    monkeypatch.setattr(review, "DATASET", dataset)
    monkeypatch.setattr(review, "POSE_RUN", poses)
    monkeypatch.setattr(review, "ROOT", base)
    monkeypatch.setattr(review, "comparison", lambda: {"synthetic": True})
    n = 24
    xyz = np.full((n, 33, 3), 0.5)
    conf = np.full((n, 33, 2), 0.9)
    valid = np.ones((n, 33), bool)
    # Partial arm remains usable; a separate clip fails reliable-anchor coverage.
    valid[4:8, 13] = False
    conf[4:8, 13, 0] = 0.1
    valid[8:12, 11] = False
    conf[8:12, 11, 0] = 0.1
    arrays = dict(
        coordinates=xyz,
        confidence=conf,
        valid_mask=valid,
        frame_index=np.arange(n),
        pts=np.arange(n) * 40,
        time_base_num=np.ones(n, int),
        time_base_den=np.full(n, 1000),
        pose_status=np.where(valid[:, 11:].all(axis=1), 0, 1),
    )
    np.savez_compressed(dataset / "poses.npz", **arrays)
    journal = poses / "poses.jsonl"
    journal.write_text("synthetic hash-preservation input\n")
    manifest = {
        "status": "complete",
        "scope": "full_video",
        "recording_config": recording,
        "config": {"pose_mode": "image", "confidence": 0.5},
        "inputs": {
            "video": {"sha256": "synthetic-video"},
            "workbook": {"sha256": "synthetic-workbook"},
        },
        "poses_sha256": sha256(journal),
        "source_frame_count": n,
    }
    write_json(poses / "manifest.json", manifest)
    rows = [candidate(i + 2, c, bounds=(i * 4, i * 4 + 4)) for i, c in enumerate(SOURCE_LABELS)] + [
        candidate(9, eligible=False)
    ]
    (dataset / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    prep = {
        "status": "complete",
        "scope": "full_video",
        "config": {"excluded_source_rows": review.QUARANTINED},
        "npz_sha256": sha256(dataset / "poses.npz"),
        "inputs": {
            "pose_manifest_sha256": sha256(poses / "manifest.json"),
            "poses_sha256": manifest["poses_sha256"],
        },
        "extracted_frame_count": n,
        "unknown_outside_annotation_bounds_frames": 0,
    }
    write_json(dataset / "preparation.json", prep)
    quality = base / "quality/V10"
    quality.mkdir(parents=True)
    index = {
        "video_id": "V10",
        "policy": "mask_aware70",
        "definition": POLICIES["mask_aware70"],
        "source_mask_threshold": 0.5,
        "max_missing_frame_fraction": 0.1,
        "selected_source_rows": [2, 3, 5, 6, 7],
        "inputs": {
            name: sha256(dataset / name)
            for name in ["preparation.json", "poses.npz", "candidates.jsonl"]
        },
    }
    write_json(quality / "policy-index.json", index)
    inputs = {"video_sha256": "synthetic-video", "workbook_sha256": "synthetic-workbook"}
    evidence = base / "evidence.json"
    write_json(evidence, {"inputs": inputs, "recording_config_sha256": sha256(config)})
    decision = base / "review.yaml"
    decision.write_text(
        yaml.safe_dump(
            {
                "video_id": "V10",
                "status": "raw_indices_supported_for_provisional_preparation",
                "excluded_source_rows": review.QUARANTINED,
                "evidence": str(evidence),
                "evidence_sha256": sha256(evidence),
                "mapping": {"frame_offset": 0, "end_assumption": "inclusive"},
                "uncertainties": ["Synthetic fixture"],
            }
        )
    )
    monkeypatch.setattr(review, "REVIEW", decision)
    probe = base / "probe.json"
    frames = [{"pts_time": str(i * 40 / 1000)} for i in range(n)]
    write_json(probe, {"frames": frames})
    monkeypatch.setattr(
        review, "source_inputs", lambda: (evidence, {"inputs": inputs}, probe, {"frames": frames})
    )
    return base, dataset, journal


def test_finish_writes_honest_portable_readiness_and_preserves_inputs(frozen_inputs):
    base, dataset, journal = frozen_inputs
    before = sha256(dataset / "poses.npz"), sha256(journal)
    result = review.finish(base / "ready")
    assert not result["training_performed"] and not result["training_authorized"]
    assert result["exact_timing_verified_frames"] == 24
    assert not result["provisional_preparation_ready"]  # no retained lead hooks
    assert result["class_counts"]["cross"]["retained"] == 1  # partial poses survive
    assert result["class_counts"]["lead_hook"]["pose_rejected"] == 1
    assert result["boxer_id"] is None and result["session_id"] is None
    assert result["excluded_annotations"][0]["source_row"] == 9
    assert "No training performed" in (base / "ready/readiness.md").read_text()
    assert json.loads((base / "ready/readiness.json").read_text())["pose_rejected_source_rows"] == [
        4
    ]
    assert before == (sha256(dataset / "poses.npz"), sha256(journal))
    with pytest.raises(ValueError, match="fresh"):
        review.finish(base / "ready")


def test_finish_rejects_changed_review_before_writing(frozen_inputs):
    base, _, _ = frozen_inputs
    (base / "evidence.json").write_text('{"inputs":{}}')
    with pytest.raises(ValueError, match="provenance"):
        review.finish(base / "bad")
    assert not (base / "bad").exists()


def test_finish_rejects_a_new_offset_without_correcting_data(frozen_inputs):
    import yaml

    base, _, _ = frozen_inputs
    path = base / "review.yaml"
    decision = yaml.safe_load(path.read_text())
    decision["mapping"]["frame_offset"] = -1
    path.write_text(yaml.safe_dump(decision))
    with pytest.raises(ValueError, match="Explicit V10 review"):
        review.finish(base / "bad")
    assert not (base / "bad").exists()
