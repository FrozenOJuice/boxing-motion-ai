"""Synthetic grouped sampling, leakage guards, duration controls and training checks."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from boxing_motion_ai.training.data import PoseClips  # noqa: E402
from boxing_motion_ai.training.limbs import LimbClips  # noqa: E402
from boxing_motion_ai.training.multi_recording import (  # noqa: E402
    ARMS,
    GROUPS,
    Pool,
    duration_control,
    folds,
    sample_probabilities,
    sampling_summary,
    settings,
    validate_fold,
)
from boxing_motion_ai.training.multi_study import fit, prediction_metrics  # noqa: E402
from boxing_motion_ai.training.velocity import VelocityClips  # noqa: E402


def clip(video, label, row, n=4):
    x = torch.zeros(n, 65)
    joints = x[:, :40].reshape(n, 8, 5)
    joints[..., 2:] = 1
    joints[..., :2] = torch.tensor(
        [[-0.3, 0], [0.3, 0], [-0.3, -0.5], [0.3, 0.5], [-0.8, -0.5], [0.3, 1], [-0.2, 1], [0.2, 1]]
    )
    x[:, 40] = 1
    return dict(
        recording_id=video,
        source_row=row,
        label=label,
        x=x,
        mask_x=torch.ones(n, 16),
        raw=dict(
            pts=np.arange(n) * 1001,
            time_base_num=np.ones(n, int),
            time_base_den=np.full(n, 30000, int),
        ),
    )


def test_logo_groups_partition_without_training_or_selection_leakage():
    fs = folds()
    assert len(fs) == 4
    for f in fs:
        validate_fold(f["train"], f["development"], f["heldout"])
        assert f["development"] == ["V5"]
        assert not ({GROUPS[v] for v in f["train"]} & {GROUPS[v] for v in f["heldout"]})
    assert next(f for f in fs if "V7" in f["heldout"])["heldout"] == ["V7", "V8"]


@pytest.mark.parametrize(
    "train,dev,held",
    [
        (["V4", "V5"], ["V5"], []),
        (["V9"], ["V5"], []),
        (["V1", "V4"], ["V5"], []),
        (["V4", "V7"], ["V5"], ["V8", "V6", "V10"]),
        (["V4"], ["V6"], []),
        (["V4", "V6", "V7", "V8"], ["V5"], ["V7", "V8", "V10"]),
    ],
)
def test_invalid_group_folds_rejected(train, dev, held):
    with pytest.raises(ValueError):
        validate_fold(train, dev, held)


def test_sampler_balances_classes_without_singleton_group_dominating():
    clips = []
    for label in range(6):
        for v, n in [("V4", 40), ("V6", 20), ("V7", 1), ("V8", 2), ("V10", 10)]:
            clips.extend(clip(v, label, 100 * label + i) for i in range(n))
    p = sample_probabilities(clips)
    a = sampling_summary(clips, p)
    assert sum(p).item() == pytest.approx(1)
    assert all(x == pytest.approx(1 / 6) for x in a["class_mass"].values())
    assert max(a["recording_mass"].values()) < 0.5
    singleton = next(
        i for i, c in enumerate(clips) if c["recording_id"] == "V7" and c["label"] == 0
    )
    assert p[singleton] < (1 / 6) / 4  # Less than equal mass for an entire source group.
    perm = list(reversed(range(len(clips))))
    torch.testing.assert_close(
        sample_probabilities([clips[i] for i in perm]), p[perm], atol=0, rtol=0
    )


def test_sampler_rejects_missing_training_class_and_duplicate_source_ids():
    with pytest.raises(ValueError, match="six classes"):
        sample_probabilities([clip("V4", 0, 1)])
    with pytest.raises(ValueError, match="duplicate"):
        Pool([SimpleNamespace(clips=[clip("V4", 0, 1)] * 2)])


def test_duration_lookup_does_not_use_evaluation_labels_or_source_statistics():
    train = [clip("V4", label, label, n=4 + label * 3) for label in range(6)]
    p = sample_probabilities(train)
    cfg = {"duration_frame_bins": [8, 15, 24], "duration_second_bins": [0.3, 0.6, 1.0]}
    evaluation = [clip("V6", 1, 42, n=20)]
    first, table = duration_control(train, p, evaluation, cfg)
    evaluation[0]["label"] = 5
    second, other = duration_control(train, p, evaluation, cfg)
    assert table == other and first[0]["probabilities"] == second[0]["probabilities"]
    assert first[0]["prediction"] == second[0]["prediction"]
    assert first[0]["target"] != second[0]["target"]


@pytest.mark.parametrize("kind", ["pose", "velocity"])
def test_explicit_multi_gate_still_rejects_v9_before_reading_inputs(kind):
    with pytest.raises(ValueError, match="V9 reserved"):
        if kind == "pose":
            PoseClips({}, "V9", multi_recording=True)
        else:
            VelocityClips(SimpleNamespace(provenance={"recording_id": "V9"}), multi_recording=True)


def test_study_settings_reject_changed_scope_and_policy(tmp_path):
    original = yaml.safe_load(Path("configs/level1_multi_recording.yaml").read_text())
    for field, value in [
        ("training_recordings", ["V5", "V9"]),
        ("logo_selection", "heldout"),
        ("mirror_probability", 0),
        ("sampler", "uniform"),
        ("logo_pilot_seed", 1),
    ]:
        c = {**original, field: value}
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(c))
        with pytest.raises(ValueError, match="V9 reserved"):
            settings(path)


def test_training_smoke_is_reproducible_and_preserves_a_b_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr("boxing_motion_ai.training.multi_study.plot", lambda *a: None)
    base = dict(
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.001,
        hidden=32,
        dropout=0.1,
        threads=1,
    )
    ds = SimpleNamespace(
        clips=[clip("V4", i, i) for i in range(6)], provenance={"recording_id": "V4"}
    )
    dev = Pool([SimpleNamespace(clips=[clip("V5", i, i) for i in range(6)])])
    train = Pool([ds])
    p = sample_probabilities(train.clips)
    m, a, log = fit(train, dev, p, base, 123, ARMS[0], "tcn", tmp_path / "first", lambda x: None)
    m2, b, log2 = fit(train, dev, p, base, 123, ARMS[0], "tcn", tmp_path / "repeat", lambda x: None)
    assert log == log2 and a["development"] == b["development"]
    from boxing_motion_ai.training.multi_verify import verify_sampling

    assert verify_sampling(train.clips, p, 123, log, batch_size=4) == 1
    broken = deepcopy(log)
    broken[0]["mirrored"][0] = not broken[0]["mirrored"][0]
    with pytest.raises(ValueError, match="replay differs"):
        verify_sampling(train.clips, p, 123, broken, batch_size=4)
    for x, y in zip(m.parameters(), m2.parameters(), strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    geom = Pool([LimbClips(ds)])
    geomdev = Pool([LimbClips(SimpleNamespace(clips=dev.clips, provenance={"recording_id": "V5"}))])
    _, other, log3 = fit(
        geom, geomdev, p, base, 123, ARMS[1], "tcn", tmp_path / "geometry", lambda x: None
    )
    assert log3 == log and other["initial_shared_sha256"] == a["initial_shared_sha256"]
    assert all(row["seen"] if "seen" in row else len(row["indices"]) == len(train) for row in log)
    bad = deepcopy(dev)
    bad.clips[0]["recording_id"] = "V9"
    with pytest.raises(ValueError):
        fit(train, bad, p, base, 123, ARMS[0], "tcn", tmp_path / "bad")
    with pytest.raises(ValueError, match="training pool only"):
        fit(train, dev, p.flip(0) * 2, base, 123, ARMS[0], "tcn", tmp_path / "bad2")


def test_prediction_metrics_keep_unsupported_predictions_as_errors():
    result = prediction_metrics([dict(target="lead_hook", prediction="jab")])
    assert result["accuracy"] == 0 and result["recall_with_absence"]["jab"] is None
    assert result["confusion_matrix"][2][0] == 1


def test_report_and_aggregation_preserve_absent_class_support(tmp_path):
    from boxing_motion_ai.training.multi_recording import MODELS
    from boxing_motion_ai.training.multi_study import summarize_case, write_report

    a = prediction_metrics(
        [
            dict(target="lead_hook", prediction="lead_hook"),
            dict(target="lead_uppercut", prediction="jab"),
        ]
    )
    b = prediction_metrics(
        [
            dict(target="lead_hook", prediction="lead_hook"),
            dict(target="lead_uppercut", prediction="lead_uppercut"),
        ]
    )
    runs = {
        "20261021": {
            arm: {v: {name: deepcopy(m) for name in MODELS} for v in ["V5", "V6", "heldout_group"]}
            for arm, m in zip(ARMS, [a, b], strict=True)
        }
    }
    aggregate = summarize_case(runs, [20261021])
    assert aggregate["V6"]["tcn"][ARMS[1]]["per_class"]["jab"]["recall"] is None
    assert aggregate["V6"]["tcn"][ARMS[1]]["supported_macro_f1"]["mean"] == 1
    assert aggregate["V6"]["tcn"][ARMS[1]]["macro_f1"]["mean"] == pytest.approx(1 / 3)
    support = {
        c: {
            "retained": s["support"],
            "status": "present" if s["support"] else "absent_from_source_annotations",
        }
        for c, s in a["per_class"].items()
    }
    result = dict(
        warning="Provisional",
        cases={"hold-V6": dict(fold={"heldout": ["V6"]}, aggregate=aggregate)},
    )
    audit = {v: dict(source_support=support) for v in ["V5", "V6"]}
    write_report(tmp_path, result, audit)
    text = (tmp_path / "summary.md").read_text()
    assert "heldout_group" in text and "absent_from_source_annotations | unsupported" in text
    assert "Rows true / columns predicted" in text


def test_absolute_config_snapshot_stays_inside_output_and_preserves_source(tmp_path):
    import json

    from boxing_motion_ai.training.multi_study import snapshot

    config = tmp_path / "input.yaml"
    config.write_text("synthetic: true\n")
    before = config.stat().st_mtime_ns
    output = tmp_path / "run"
    snapshot(output, config.resolve())
    assert config.read_text() == "synthetic: true\n"
    assert config.stat().st_mtime_ns == before
    assert (output / "source/submitted-config/input.yaml").read_text() == config.read_text()
    hashes = json.loads((output / "implementation.json").read_text())
    assert "submitted-config/input.yaml" in hashes
    assert all(not Path(p).is_absolute() and ".." not in Path(p).parts for p in hashes)
