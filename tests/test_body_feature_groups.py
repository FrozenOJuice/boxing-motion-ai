"""Group membership, matched input/optimization contracts and supported-source evaluation."""

from copy import deepcopy

import pytest
import torch
from test_video_foundation import raw

from boxing_motion_ai.training import body_feature_groups as study
from boxing_motion_ai.training.experiment import seed_all
from boxing_motion_ai.training.fullbody_classification import state_hash
from boxing_motion_ai.training.fullbody_classification_data import MAP65, clip_body


def clips(labels=range(6)):
    result = []
    for label in labels:
        r = raw(8)
        r["coordinates"][:, 15, 0] += 0.002 * label
        x, mirror, normal = clip_body(dict(raw=r), 1, [])
        result.append(
            dict(
                raw=r,
                x=x[:, MAP65],
                core65_reflected=mirror[:, MAP65],
                body209=x,
                body209_reflected=mirror,
                body_normalization=normal,
                mask_x=torch.ones(8, 16),
                label=label,
                source_row=label,
                recording_id="V4",
            )
        )
    return result


def config():
    return dict(
        hidden=32,
        dropout=0.1,
        threads=2,
        learning_rate=0.001,
        weight_decay=0.001,
        epochs=2,
        batch_size=3,
        mirror_probability=0.5,
        gradient_clip=1.0,
    )


def test_groups_partition_all_extra_channels_without_duplicates_or_core_overlap():
    flat = [i for group in study.GROUPS.values() for i in group]
    assert len(flat) == 144 and len(set(flat)) == 144
    assert not set(flat) & set(MAP65)
    assert set(flat) | set(MAP65) == set(range(209))
    assert {k: len(v) for k, v in study.GROUPS.items()} == dict(upper=60, lower=68, torso=16)


@pytest.mark.parametrize("mode", ["core65", "upper", "lower", "torso"])
def test_group_views_preserve_core_exactly_and_shared_initial_logits_rng(mode):
    cs = clips()
    original = deepcopy(cs)
    view = study.GroupView(cs, mode)
    extra = study.GROUPS.get(mode, [])
    assert torch.equal(view[0]["x"][:, :65], cs[0]["x"])
    assert torch.equal(view[0]["x"][:, 65:], cs[0]["body209"][:, extra])
    assert torch.equal(cs[0]["body209"], original[0]["body209"])
    seed_all(42, 2)
    core = study.initialize("core65", config())
    state = torch.get_rng_state()
    seed_all(42, 2)
    model = study.initialize(mode, config())
    assert torch.equal(state, torch.get_rng_state())
    mask = torch.ones(1, 8, dtype=torch.bool)
    core.eval()
    model.eval()
    assert torch.allclose(core(cs[0]["x"][None], mask), model(view[0]["x"][None], mask), atol=1e-6)


def test_supported_sampler_has_no_fake_missing_classes_and_rejects_role_leakage():
    cs = clips(range(2, 6))
    cs.append(cs[-1])
    p = study.probabilities(cs)
    assert p.sum() == pytest.approx(1)
    for label in range(2, 6):
        assert p[[c["label"] == label for c in cs]].sum() == pytest.approx(0.25)
    cs[0] = cs[0] | dict(recording_id="V5")
    with pytest.raises(Exception, match="group_training_role"):
        study.probabilities(cs)
    cs[0] = cs[0] | dict(recording_id="V6")
    with pytest.raises(Exception, match="missing_class_pooled_sampler_forbidden"):
        study.probabilities(cs)


def test_small_fit_repeat_and_matched_sampling():
    cs = clips()
    a, r, rows = study.fit(cs, "upper", 42, config())
    b, repeated, repeated_rows = study.fit(cs, "upper", 42, config())
    _, core, _ = study.fit(cs, "core65", 42, config())
    assert r == repeated and rows == repeated_rows and state_hash(a) == state_hash(b)
    assert r["sampling_mirror_sha256"] == core["sampling_mirror_sha256"]


def test_supported_metrics_exclude_untrained_targets_but_keep_six_way_outputs():
    model = study.initialize("core65", config())
    model.head.weight.data.zero_()
    model.head.bias.data.zero_()
    model.head.bias.data[0] = 10
    result, rows = study.supported_score(model, clips(), "core65", {2, 3, 4, 5}, config())
    assert result["unsupported_reference_count"] == 2 and len(rows) == 4
    assert result["supported_macro_f1"] == 0
    assert result["confusion_matrix"][2][0] == 1
    assert result["per_class"]["jab"]["support"] == 0


def test_development_checks_freeze_before_loading_any_v5(tmp_path, monkeypatch):
    from boxing_motion_ai.training.experiment import write_json

    p = tmp_path / "evidence"
    p.write_text("tampered")
    write_json(
        tmp_path / "training-freeze.json",
        dict(files_sha256={str(p): "wrong"}, code_sha256={}, input_sha256={}),
    )

    def forbidden():
        raise AssertionError("must not access V5")

    monkeypatch.setattr(study, "development_views", forbidden)
    with pytest.raises(Exception, match="body_group_frozen_input_changed"):
        study.development(tmp_path)
