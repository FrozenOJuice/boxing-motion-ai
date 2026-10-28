"""Synthetic adapter and pin checks; no research recordings or checkpoints are required."""

import json
from fractions import Fraction

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
from boxing_motion_ai.analysis.common import pin, sha  # noqa: E402
from boxing_motion_ai.analysis.inference import Classifier  # noqa: E402
from boxing_motion_ai.training.data import CLASSES  # noqa: E402
from boxing_motion_ai.training.models import CausalTCN  # noqa: E402


@pytest.fixture
def classifier(tmp_path):
    root = tmp_path
    (root / "configs").mkdir()
    (root / "provenance.json").write_text("{}")
    config = dict(
        train=dict(recording="V4"),
        development=dict(recording="V5"),
        hidden=32,
        dropout=0.1,
        feature_representation="xy_velocity",
    )
    torch.manual_seed(4)
    saved = dict(
        model_state=CausalTCN(65).state_dict(),
        config=config,
        seed=20261021,
        epoch=32,
        run="tcn",
        provenance_sha256=sha(root / "provenance.json"),
        feature_version="torso-core-2d-velocity-1.0",
    )
    torch.save(saved, root / "checkpoint.pt")
    value = dict(
        schema_version="assisted-level1-pin-1.0",
        channels=65,
        classes=list(CLASSES),
        feature_version=saved["feature_version"],
        seed=20261021,
        epoch=32,
        source_sha256={},
        checkpoint=dict(path="checkpoint.pt", sha256=sha(root / "checkpoint.pt")),
        provenance=dict(path="provenance.json", sha256=sha(root / "provenance.json")),
    )
    (root / "configs/assisted_level1_v1.json").write_text(json.dumps(value))
    return root, Classifier(root)


@pytest.fixture
def arrays():
    n = 12
    xyz = np.full((n, 33, 3), 0.4, dtype=np.float32)
    xyz[:, 11:13, 1] = 0.2
    xyz[:, 23:25, 1] = 0.6
    xyz[:, 15, 0] = np.linspace(0.3, 0.6, n)
    return dict(
        frame_index=np.arange(n),
        pts=np.arange(n) * 40 + 10000,
        time_base_num=np.ones(n, dtype=np.int64),
        time_base_den=np.full(n, 1000),
        coordinates=xyz,
        confidence=np.full((n, 33, 2), 0.9, dtype=np.float32),
        valid_mask=np.ones((n, 33), dtype=bool),
        pose_status=np.zeros(n, dtype=np.int64),
        scene_id=np.zeros(n, dtype=np.int64),
        segment_id=np.zeros(n, dtype=np.int64),
    )


def test_deterministic_real_model_path_and_selection_time(classifier, arrays):
    _, model = classifier
    a = model.classify(arrays, 4 / 3, 2, 10)
    b = model.classify(arrays, 4 / 3, 2, 10)
    assert a["scores"] == b["scores"]
    assert a["prediction"] in CLASSES
    assert len(a["scores"]) == 6 and "no_punch" not in a["scores"]
    assert a["source"] == "model_prediction"
    assert a["start_timestamp_exact"] == "252/25"
    assert sum(a["scores"].values()) == pytest.approx(1)
    assert a["pose_coverage"]["anchor_frames"] == 1
    assert a["inference_seconds"] >= 0


def test_clip_features_reset_at_selected_start(classifier, arrays):
    _, model = classifier
    result = model.classify(arrays, 1, 4, 12)
    altered = {k: v.copy() for k, v in arrays.items()}
    altered["coordinates"][:4, :, 0] = 0.8
    assert model.classify(altered, 1, 4, 12)["scores"] == result["scores"]


@pytest.mark.parametrize("start,stop", [(-1, 4), (3, 3), (4, 3), (0, 13), (0, 1)])
def test_invalid_selection_unavailable(classifier, arrays, start, stop):
    result = classifier[1].classify(arrays, 1, start, stop)
    assert result["status"] == "unavailable" and "scores" not in result


def test_missing_pose_and_degenerate_torso_unavailable(classifier, arrays):
    model = classifier[1]
    bad = {k: v.copy() for k, v in arrays.items()}
    bad["valid_mask"][:] = False
    bad["confidence"][:] = 0
    bad["pose_status"][:] = 2
    assert model.classify(bad, 1, 0, 12)["status"] == "unavailable"
    arrays["coordinates"][:] = 0.4
    assert "normalized" in model.classify(arrays, 1, 0, 12)["reason"]


def test_timestamp_gap_and_cut_unavailable(classifier, arrays):
    arrays["pts"][6:] += 1000
    result = classifier[1].classify(arrays, 1, 0, 12, gap=Fraction(2, 25))
    assert result["status"] == "unavailable"
    arrays["pts"][6:] -= 1000
    arrays["scene_id"][6:] = 1
    assert classifier[1].classify(arrays, 1, 0, 12)["status"] == "unavailable"


def test_long_selection_unavailable(classifier, arrays):
    arrays["pts"] *= 10
    assert "3 seconds" in classifier[1].classify(arrays, 1, 0, 12)["reason"]


def test_pin_tampering_and_reserved_path_rejected(classifier):
    root, _ = classifier
    (root / "checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        Classifier(root)
    path = root / "configs/assisted_level1_v1.json"
    value = json.loads(path.read_text())
    value["checkpoint"]["path"] = "data/V9/checkpoint.pt"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Unsafe"):
        pin(root)


def test_incompatible_schema_rejected_after_valid_hash(classifier):
    root, _ = classifier
    path = root / "checkpoint.pt"
    value = torch.load(path, weights_only=True)
    value["feature_version"] = "wrong-schema"
    torch.save(value, path)
    pin_path = root / "configs/assisted_level1_v1.json"
    binding = json.loads(pin_path.read_text())
    binding["checkpoint"]["sha256"] = sha(path)
    pin_path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="incompatible"):
        Classifier(root)
