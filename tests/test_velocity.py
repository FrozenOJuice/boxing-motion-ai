"""Synthetic velocity units, exact clocks, discontinuities, reflection and causality."""

from copy import deepcopy
from fractions import Fraction

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from boxing_motion_ai.training.augmentation import LEFT_RIGHT, mirror_features  # noqa: E402
from boxing_motion_ai.training.models import CausalTCN  # noqa: E402
from boxing_motion_ai.training.velocity import expand_input, joint_velocities  # noqa: E402


def sequence(n=5):
    x = torch.zeros(n, 41)
    joints = x[:, :40].reshape(n, 8, 5)
    joints[..., 4] = 1
    x[:, 40] = 1
    joints[..., 0] = torch.arange(n)[:, None] * 0.125
    joints[..., 1] = -torch.arange(n)[:, None] * 0.25
    raw = dict(
        frame_index=np.arange(n),
        pts=np.arange(n) * 1001,
        time_base_num=np.ones(n, int),
        time_base_den=np.full(n, 30000, int),
        scene_id=np.zeros(n, int),
        segment_id=np.zeros(n, int),
        pose_status=np.zeros(n, int),
    )
    return x, raw


def test_exact_elapsed_time_and_first_frame_invalid_without_mutation():
    x, raw = sequence()
    before, metadata = x.clone(), deepcopy(raw)
    v = joint_velocities(x, raw, Fraction(1, 10))
    assert not v[0].any()
    assert (v[1:, :, 2] == 1).all()
    torch.testing.assert_close(v[1:, :, 0], torch.full((4, 8), 0.125 * 30000 / 1001))
    torch.testing.assert_close(v[1:, :, 1], torch.full((4, 8), -0.25 * 30000 / 1001))
    torch.testing.assert_close(before, x)
    for k in raw:
        np.testing.assert_array_equal(metadata[k], raw[k])


@pytest.mark.parametrize(
    "kind", ["joint", "anchor", "missing", "gap", "scene", "segment", "timestamp_gap"]
)
def test_unreliable_endpoints_and_discontinuities_never_generate_velocities(kind):
    x, raw = sequence()
    if kind == "joint":
        x[2, :40].reshape(8, 5)[3] = 0
    elif kind == "anchor":
        x[2, 40] = 0
    elif kind == "missing":
        raw["pose_status"][2] = 2
    elif kind == "gap":
        raw["frame_index"][2:] += 1
    elif kind == "scene":
        raw["scene_id"][2:] = 1
    elif kind == "segment":
        raw["segment_id"][2:] = 1
    else:
        raw["pts"][2:] += 30000
    v = joint_velocities(x, raw, Fraction(1, 10))
    if kind == "joint":
        assert not v[2:4, 3].any()
        assert v[2, 0, 2] == 1
    elif kind in ("anchor", "missing"):
        assert not v[2:4].any()
    else:
        assert not v[2].any()
    assert (v[4, :, 2] == 1).all()


def test_timestamps_are_subtracted_exactly_before_float_conversion_and_must_increase():
    x, raw = sequence()
    v = joint_velocities(x, raw, Fraction(1, 10))
    raw["pts"] += 10**17
    torch.testing.assert_close(joint_velocities(x, raw, Fraction(1, 10)), v, rtol=0, atol=0)
    raw["pts"][2] = raw["pts"][1]
    with pytest.raises(ValueError, match="increasing"):
        joint_velocities(x, raw, Fraction(1, 10))


def test_mirroring_commutes_with_derivative_and_is_an_involution():
    x, raw = sequence()
    x[:, :40].reshape(5, 8, 5)[2, 3] = 0
    v = joint_velocities(x, raw, Fraction(1, 10))
    full = torch.cat((x, v.reshape(5, 24)), dim=1)
    expected = torch.cat(
        (
            mirror_features(x),
            joint_velocities(mirror_features(x), raw, Fraction(1, 10)).reshape(5, 24),
        ),
        dim=1,
    )
    torch.testing.assert_close(mirror_features(full), expected, rtol=0, atol=0)
    torch.testing.assert_close(mirror_features(mirror_features(full)), full, rtol=0, atol=0)
    masks = torch.cat((x[:, :40].reshape(5, 8, 5)[..., 4], v[..., 2]), dim=1)
    actual = mirror_features(masks, mask_only=True)
    torch.testing.assert_close(actual[:, :8], masks[:, list(LEFT_RIGHT)])
    torch.testing.assert_close(actual[:, 8:], masks[:, [i + 8 for i in LEFT_RIGHT]])


def test_future_pose_or_future_timestamp_cannot_affect_past_features():
    x, raw = sequence()
    original = joint_velocities(x, raw, Fraction(1, 10))
    x[3:] = torch.randn_like(x[3:]) * 100
    raw["pts"][3:] += 500
    changed = joint_velocities(x, raw, Fraction(1, 10))
    torch.testing.assert_close(changed[:3], original[:3], rtol=0, atol=0)


@pytest.mark.parametrize("field", ["pts", "time_base_num", "time_base_den"])
def test_fractional_or_nonpositive_time_metadata_is_rejected(field):
    x, raw = sequence()
    if field == "pts":
        raw[field] = raw[field].astype(float) + 0.25
    else:
        raw[field][2] = 0
    with pytest.raises(ValueError, match="integer timing"):
        joint_velocities(x, raw, Fraction(1, 10))


def test_velocity_config_does_not_admit_other_recordings_or_interventions(tmp_path):
    from pathlib import Path
    from types import SimpleNamespace

    import yaml

    from boxing_motion_ai.training.experiment import load_config
    from boxing_motion_ai.training.velocity import VelocityClips

    config = yaml.safe_load(Path("configs/level1_initial.yaml").read_text())
    config.update(
        schema_version="level1-experiment-1.5",
        mirror_probability=0.5,
        feature_representation="xy_velocity",
        runs=["tcn", "tcn_mask_only"],
    )
    path = tmp_path / "velocity.yaml"
    path.write_text(yaml.safe_dump(config))
    assert load_config(path)["feature_representation"] == "xy_velocity"
    for recording in ("V6", "V9"):
        with pytest.raises(ValueError, match="V4/V5 only"):
            VelocityClips(SimpleNamespace(provenance={"recording_id": recording}))
        bad = deepcopy(config)
        bad["train"]["recording"] = recording
        path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError, match="V4 train"):
            load_config(path)
    for extra in ("visibility_probability", "context_seconds"):
        path.write_text(yaml.safe_dump({**config, extra: 0.5}))
        with pytest.raises(ValueError, match="configuration fields"):
            load_config(path)


def test_input_expansion_keeps_shared_weights_initial_predictions_and_rng():
    torch.manual_seed(7)
    model = CausalTCN(41)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(2, 6, 41)
    model.eval()
    expected = model(x, torch.ones(2, 6, dtype=torch.bool))
    rng = torch.get_rng_state().clone()
    expand_input(model, 65)
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in before.items():
        actual = model.state_dict()[k]
        torch.testing.assert_close(
            actual[:, :41] if k == "input.weight" else actual, v, rtol=0, atol=0
        )
    assert not model.input.weight[:, 41:].any()
    actual = model(torch.cat((x, torch.randn(2, 6, 24)), dim=2), torch.ones(2, 6, dtype=torch.bool))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
