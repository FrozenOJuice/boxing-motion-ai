"""Physical reflection, involution, semantic preservation and isolated RNG."""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from boxing_motion_ai.data.pose_quality import CORE  # noqa: E402
from boxing_motion_ai.training.augmentation import (  # noqa: E402
    LEFT_RIGHT,
    augment_batch,
    mirror_features,
)
from boxing_motion_ai.training.data import features  # noqa: E402


@pytest.mark.parametrize("mask_only,channels", [(False, 41), (True, 8)])
def test_mirror_twice_restores_all_features_without_modifying_input(mask_only, channels):
    torch.manual_seed(11)
    x = torch.randn(3, 9, channels)
    original = x.clone()
    mirrored = mirror_features(x, mask_only=mask_only)
    torch.testing.assert_close(
        mirror_features(mirrored, mask_only=mask_only), original, rtol=0, atol=0
    )
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    assert mirrored.data_ptr() != x.data_ptr()
    if not mask_only:
        torch.testing.assert_close(mirrored[..., -1], x[..., -1])
        joints = x[..., :40].reshape(3, 9, 8, 5)
        reflected = mirrored[..., :40].reshape(3, 9, 8, 5)
        torch.testing.assert_close(reflected[..., 0], -joints[..., list(LEFT_RIGHT), 0])
        torch.testing.assert_close(reflected[..., 1:], joints[..., list(LEFT_RIGHT), 1:])


def test_normalized_mirror_equals_physical_raw_image_reflection():
    rng = np.random.default_rng(9)
    xyz = rng.uniform(0.25, 0.75, (5, 33, 3))
    xyz[:, 11:13, 1] = 0.2
    xyz[:, 23:25, 1] = 0.7
    confidence = rng.uniform(0.6, 1.0, (5, 33, 2))
    valid = np.ones((5, 33), dtype=bool)
    valid[:3, 13] = False
    valid[2:, 16] = False
    xyz[~valid] = np.nan
    confidence[~valid] = np.nan
    x, mask_x, torso = features(xyz, confidence, valid, 16 / 9)
    reflected_xyz, reflected_conf = xyz.copy(), confidence.copy()
    reflected_valid = valid.copy()
    paired = np.asarray(CORE)[list(LEFT_RIGHT)]
    reflected_xyz[:, CORE] = xyz[:, paired]
    reflected_xyz[:, CORE, 0] = 1 - reflected_xyz[:, CORE, 0]
    reflected_conf[:, CORE] = confidence[:, paired]
    reflected_valid[:, CORE] = valid[:, paired]
    expected, expected_masks, expected_torso = features(
        reflected_xyz, reflected_conf, reflected_valid, 16 / 9
    )
    actual = mirror_features(torch.from_numpy(x))
    torch.testing.assert_close(actual, torch.from_numpy(expected), atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(expected_torso, torso)
    torch.testing.assert_close(
        mirror_features(torch.from_numpy(mask_x), mask_only=True), torch.from_numpy(expected_masks)
    )
    joints = actual.numpy()[:, :40].reshape(5, 8, 5)
    assert (joints[~reflected_valid[:, CORE]] == 0).all()


def test_batch_augmentation_preserves_six_labels_timing_and_global_rng():
    torch.manual_seed(12)
    x = torch.randn(1000, 7, 41)
    mask = torch.randint(0, 2, (1000, 7, 8)).float()
    labels = torch.arange(1000) % 6
    raw = [{"pts": [0, 33], "frame_index": [0, 1]}]
    batch = dict(
        x=x,
        mask_x=mask,
        labels=labels,
        source_rows=list(range(1000)),
        clips=raw,
        time_mask=torch.ones(1000, 7, dtype=torch.bool),
    )
    state = torch.get_rng_state().clone()
    expected_generator = torch.Generator().manual_seed(16)
    expected = torch.rand(1000, generator=expected_generator) < 0.5
    augmented, chosen = augment_batch(batch, torch.Generator().manual_seed(16), 0.5)
    assert torch.equal(chosen, expected)
    assert 0.45 < chosen.float().mean() < 0.55
    assert torch.equal(state, torch.get_rng_state())
    assert augmented["labels"] is labels
    assert augmented["clips"] is raw and augmented["time_mask"] is batch["time_mask"]
    assert labels.unique().tolist() == list(range(6))
    torch.testing.assert_close(augmented["x"][chosen], mirror_features(x[chosen]))
    torch.testing.assert_close(
        augmented["mask_x"][chosen], mirror_features(mask[chosen], mask_only=True)
    )
    torch.testing.assert_close(augmented["x"][~chosen], x[~chosen])
    original, decisions = augment_batch(batch, torch.Generator().manual_seed(16), 0.0)
    assert original is batch and not decisions.any()
    with pytest.raises(ValueError, match="0.5"):
        augment_batch(batch, expected_generator, 0.75)


def test_mirror_preserves_padding_and_rejects_wrong_schema():
    x = torch.zeros(2, 6, 41)
    assert not mirror_features(x).any()
    with pytest.raises(ValueError, match="41"):
        mirror_features(torch.zeros(3, 40))
