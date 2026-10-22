"""Synthetic transfer diagnostics; no real dataset, reserved inputs or training needed."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from boxing_motion_ai.training.data import CLASSES, PoseClips  # noqa: E402
from boxing_motion_ai.training.experiment import metrics  # noqa: E402
from boxing_motion_ai.training.velocity import VelocityClips  # noqa: E402
from boxing_motion_ai.training.velocity_transfer import (  # noqa: E402
    clip_diagnostics,
    error_associations,
    load_transfer_config,
    source_support,
    supported_metrics,
    write_report,
)


def test_supported_f1_keeps_absent_class_predictions_as_errors():
    original = metrics([2, 2, 3], [2, 0, 2])
    result = supported_metrics(original)
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["macro_f1"] == original["macro_f1"]
    assert result["supported_macro_f1"] == pytest.approx(0.25)
    assert result["recall_with_absence"]["jab"] is None
    assert result["recall_with_absence"]["rear_hook"] == 0
    assert result["confusion_matrix"][2][0] == 1
    assert "supported_classes" not in original


def test_source_absence_is_distinct_from_annotation_exclusion(tmp_path):
    rows = [
        dict(punch_type_provisional="jab", annotation_eligible=False),
        dict(punch_type_provisional="cross", annotation_eligible=True),
    ]
    (tmp_path / "candidates.jsonl").write_text("\n".join(map(json.dumps, rows)))
    ds = SimpleNamespace(
        provenance={"dataset": str(tmp_path), "class_counts": dict.fromkeys(CLASSES, 0)}
    )
    result = source_support(ds)
    assert result["jab"]["status"] == "no_retained_examples_after_exclusions"
    assert result["cross"]["pose_rejected"] == 1
    assert result["rear_hook"]["status"] == "absent_from_source_annotations"


@pytest.mark.parametrize("diagnostic", [False, True])
def test_reserved_recording_rejected_before_access(diagnostic):
    with pytest.raises(ValueError, match="V9 reserved"):
        PoseClips({}, "V9", diagnostic=diagnostic)
    with pytest.raises(ValueError, match="V9 reserved"):
        VelocityClips(SimpleNamespace(provenance={"recording_id": "V9"}), diagnostic=diagnostic)


def test_diagnostic_access_does_not_admit_v6_for_training():
    with pytest.raises(ValueError, match="Only V4/V5"):
        PoseClips({}, "V6")
    with pytest.raises(ValueError, match="separate gates"):
        PoseClips({}, "V6", diagnostic=True, expansion=True)
    with pytest.raises(ValueError, match="V4/V5 only"):
        VelocityClips(SimpleNamespace(provenance={"recording_id": "V6"}))


@pytest.mark.parametrize("change", ["recording", "group", "path", "seed"])
def test_transfer_configuration_rejects_changed_scope(tmp_path, change):
    from pathlib import Path

    c = yaml.safe_load(Path("configs/level1_velocity_transfer.yaml").read_text())
    if change == "recording":
        c["recordings"][0]["recording"] = "V9"
    elif change == "group":
        c["recordings"][1]["related_recordings_guard"] = ["V7"]
    elif change == "path":
        c["recordings"][0]["dataset"] = "unapproved"
    else:
        c["seeds"] = [0]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):
        load_transfer_config(path)


def test_tail_diagnostics_respect_validity_and_exact_frame_metadata():
    x = torch.zeros(3, 65)
    v = x[:, 41:].reshape(3, 8, 3)
    v[0, 0, :2] = 1000  # Invalid sentinel must never count as speed.
    v[1, 0] = torch.tensor([3.0, 4.0, 1.0])
    v[2, 0] = torch.tensor([6.0, 8.0, 1.0])
    clip = dict(
        x=x,
        mask_x=torch.ones(3, 16),
        label=2,
        source_row=12,
        raw=dict(
            frame_index=np.array([10, 11, 12]),
            pts=np.array([100, 110, 120]),
            time_base_num=np.ones(3, int),
            time_base_den=np.full(3, 100),
        ),
    )
    before = x.clone()
    d = clip_diagnostics(SimpleNamespace(clips=[clip]), 8, 50)
    assert d["all"]["valid_joint_pairs"] == 2
    assert d["all"]["tail_pairs"] == 1
    assert d["all"]["max_speed"] == 10
    assert d["largest_pairs"][0]["frame_index"] == 12
    assert d["largest_pairs"][0]["dt_seconds_exact"] == "1/10"
    assert d["by_class"]["jab"]["max_speed"] is None
    torch.testing.assert_close(x, before)
    json.dumps(d, allow_nan=False)


def test_associations_require_identical_rows_and_remain_class_stratified():
    row = dict(
        source_row=1,
        **{"class": "lead_hook"},
        frames=20,
        tail_pairs=2,
        severe_tail_pairs=0,
        original_mask_coverage=[1.0] * 8,
    )
    predictions = [dict(source_row=1, target="lead_hook", prediction="jab")]
    result = error_associations([row], predictions)
    assert result["by_class"]["lead_hook"]["with_tail"] == {"n": 1, "errors": 1}
    assert result["by_class"]["jab"]["with_tail"] == {"n": 0, "errors": 0}
    with pytest.raises(ValueError, match="examples differ"):
        error_associations([row], predictions * 2)


def test_markdown_report_writes_absent_classes_and_confusions(tmp_path):
    counts = {c: dict(source_annotations=1, retained=1, status="present") for c in CLASSES}
    counts["jab"] = dict(source_annotations=0, retained=0, status="absent_from_source_annotations")
    arm = dict(
        accuracy={"mean": 0.5},
        macro_f1={"mean": 0.1},
        supported_macro_f1={"mean": 0.2},
        per_class_recall={c: {"mean": 0.3} for c in CLASSES},
        mean_confusion_counts=np.zeros((6, 6)).tolist(),
    )
    arm["per_class_recall"]["jab"] = None
    result = dict(
        warning="Provisional",
        recordings={
            "V6": dict(
                source_support=counts, aggregate={"tcn": {"xy": arm, "xy_velocity": deepcopy(arm)}}
            )
        },
    )
    write_report(tmp_path, result)
    text = (tmp_path / "summary.md").read_text()
    assert "absent_from_source_annotations | absent | absent" in text
    assert "Rows true / columns predicted" in text
    assert "predictions into absent classes remain errors" in text
