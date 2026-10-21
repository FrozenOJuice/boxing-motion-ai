"""Synthetic report integrity, partial coverage and conservative readiness gates."""

import json

import pytest

from boxing_motion_ai.data.pose_quality import POLICIES
from boxing_motion_ai.data.preparation import CLASS_MAP
from boxing_motion_ai.data.summary import summarize_datasets
from boxing_motion_ai.vision.pipeline import sha256


@pytest.fixture
def summary_inputs(tmp_path):
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    # Metadata-only fixture: summary hashes but never interprets this synthetic archive.
    (dataset / "poses.npz").write_bytes(b"synthetic archive identity only")
    (dataset / "candidates.jsonl").write_text("{}\n")
    counts = {
        label: {
            "source_rows": 2,
            "annotation_eligible": 2,
            "quality_assessable": 1,
            "eligible_with_unreliable_pose": 1,
            "eligible_with_missing_pose": 0,
        }
        for label in CLASS_MAP.values()
    }
    counts["unmapped"] = {"source_rows": 0}
    prep = {
        "video_id": "V4",
        "boxer_id": None,
        "session_id": None,
        "status": "complete",
        "scope": "selected_windows",
        "source_frame_count": None,
        "extracted_frame_count": 6,
        "npz_sha256": sha256(dataset / "poses.npz"),
        "class_counts": counts,
        "inputs": {"video_sha256": "video", "workbook_sha256": "book"},
    }
    (dataset / "preparation.json").write_text(json.dumps(prep))
    quality = {
        "video_id": "V4",
        "source_mask_threshold": 0.5,
        "policies": POLICIES,
        "inputs": {
            n: sha256(dataset / n) for n in ["preparation.json", "poses.npz", "candidates.jsonl"]
        },
        "candidate_decisions": [],
        "by_class": {
            label: {
                "eligible": 1,
                "retention": {"mask_aware70": 1, "strict_body": 0},
                "joints": {
                    j: {"invalid_frame_fraction": 0.2} for j in ["left_elbow", "right_elbow"]
                },
            }
            for label in CLASS_MAP.values()
        },
    }
    qpath = tmp_path / "quality.json"
    qpath.write_text(json.dumps(quality))
    alignment = tmp_path / "alignment.json"
    alignment.write_text(
        json.dumps(
            {
                "recordings": {
                    "V4": {
                        "inputs": prep["inputs"],
                        "channel_id": "shared-publisher",
                        "channel_name": "Synthetic publisher",
                        "video": {"frame_count": 100},
                        "issues": [],
                        "strict_overlap_pairs": [],
                        "shared_endpoint_pairs": [[2, 3]],
                    }
                }
            }
        )
    )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "V4": {
                    "decision": "provisional",
                    "reason": "Synthetic provisional review, not certified labels",
                }
            }
        )
    )
    return dataset, qpath, alignment, review


def test_partial_not_promoted_by_pose_retention_and_unknown_ids_preserved(summary_inputs, tmp_path):
    dataset, qpath, alignment, review = summary_inputs
    hashes = {p.name: sha256(p) for p in dataset.iterdir()}
    result = summarize_datasets([dataset], [qpath], alignment, tmp_path / "summary", review)
    r = result["recordings"]["V4"]
    assert r["readiness"] == "sample_only_requires_full_extraction_or_class_review"
    assert r["boxer_id"] is None and r["session_id"] is None
    assert r["extracted_frames"] == 6 and r["source_frame_count"] == 100
    assert result["source_family_guard_groups"] == {"shared-publisher": ["V4"]}
    assert result["provisional_development_mask_retained"] == 0
    assert result["diagnostic_sample_mask_retained"] == 6
    assert result["split_assignment"] is None and not result["model_training_performed"]
    assert result["totals"]["source_rows"] == 12 and result["totals"]["pose_assessed"] == 6
    assert (tmp_path / "summary" / "dataset.md").is_file()
    assert hashes == {p.name: sha256(p) for p in dataset.iterdir()}


def test_summary_refuses_stale_quality_or_changed_policy(summary_inputs, tmp_path):
    dataset, qpath, alignment, review = summary_inputs
    q = json.loads(qpath.read_text())
    q["source_mask_threshold"] = 0.4
    qpath.write_text(json.dumps(q))
    with pytest.raises(ValueError, match="unchanged approved"):
        summarize_datasets([dataset], [qpath], alignment, tmp_path / "policy", review)
    q["source_mask_threshold"] = 0.5
    qpath.write_text(json.dumps(q))
    (dataset / "candidates.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="stale quality"):
        summarize_datasets([dataset], [qpath], alignment, tmp_path / "stale", review)
    assert not (tmp_path / "stale").exists()
