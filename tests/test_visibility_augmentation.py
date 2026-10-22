"""Temporary occlusion invariants, label independence and source preservation."""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from boxing_motion_ai.training.augmentation import (  # noqa: E402
    augment_batch,
    visibility_batch,
)


def batch():
    generator = torch.Generator().manual_seed(12)
    n, t = 120, 12
    lengths = torch.arange(n) % 10 + 3
    time_mask = torch.arange(t)[None] < lengths[:, None]
    mask = (torch.rand(n, t, 8, generator=generator) > 0.2).float()
    mask[..., [0, 1, 6, 7]] = 1
    mask *= time_mask[..., None]
    joints = torch.rand(n, t, 8, 5, generator=generator)
    joints[..., 4] = mask
    joints *= mask[..., None]
    x = torch.cat((joints.reshape(n, t, 40), time_mask[..., None].float()), -1)
    return dict(
        x=x,
        mask_x=mask,
        labels=torch.arange(n) % 6,
        lengths=lengths,
        time_mask=time_mask,
        source_rows=list(range(n)),
        clips=[{"raw": {"pts": [0, 33], "valid_mask": [True, False]}}],
    )


def test_visibility_is_contiguous_monotone_and_preserves_torso_sources_and_padding():
    b = batch()
    original_x, original_mask = b["x"].clone(), b["mask_x"].clone()
    augmented, events = visibility_batch(b, torch.Generator().manual_seed(19), 0.5)
    assert len(events) > 30
    before = original_x[..., :40].reshape(120, 12, 8, 5)
    after = augmented["x"][..., :40].reshape(120, 12, 8, 5)
    assert (augmented["mask_x"] <= original_mask).all()
    assert (after[..., 4] <= before[..., 4]).all()
    assert (augmented["mask_x"] == after[..., 4]).all()
    assert (after[after[..., 4] == 0] == 0).all()
    assert (after != before).any()  # Ensure masking writes through the feature view.
    torch.testing.assert_close(after[..., [0, 1, 6, 7], :], before[..., [0, 1, 6, 7], :])
    torch.testing.assert_close(augmented["x"][..., -1], original_x[..., -1])
    for e in events:
        i = e["source_row"]
        assert 0 <= e["start"] < e["stop"] <= b["lengths"][i]
        assert e["stop"] - e["start"] < b["lengths"][i]
        assert set(e["core_joints"]) in ({4}, {5}, {2, 4}, {3, 5})
        assert not after[i, e["start"] : e["stop"], e["core_joints"]].any()
    for key in ("labels", "lengths", "time_mask", "source_rows", "clips"):
        assert augmented[key] is b[key]
    torch.testing.assert_close(b["x"], original_x)
    torch.testing.assert_close(b["mask_x"], original_mask)
    assert not augmented["x"][~b["time_mask"]].any()


def test_decisions_do_not_depend_on_labels_or_existing_missingness():
    b = batch()
    _, a = visibility_batch(b, torch.Generator().manual_seed(28), 0.5)
    changed = {
        **b,
        "labels": b["labels"].flip(0),
        "x": b["x"].clone(),
        "mask_x": b["mask_x"].clone(),
    }
    changed["x"][..., :40] = 0
    changed["mask_x"][...] = 0
    _, other = visibility_batch(changed, torch.Generator().manual_seed(28), 0.5)
    keys = ("source_row", "start", "stop", "side", "core_joints")
    assert [{k: e[k] for k in keys} for e in a] == [{k: e[k] for k in keys} for e in other]
    assert all(e["newly_hidden_usable_entries"] == 0 for e in other)
    # No resampling until a valid joint is found; no-op events remain in the log.
    assert len(a) == len(other)


def test_visibility_rng_is_private_and_can_be_restored_without_changing_mirrors():
    b = batch()
    mirror_a = torch.Generator().manual_seed(8)
    mirror_b = torch.Generator().manual_seed(8)
    source_a, decisions_a = augment_batch(b, mirror_a, 0.5)
    source_b, decisions_b = augment_batch(b, mirror_b, 0.5)
    assert torch.equal(decisions_a, decisions_b)
    generator = torch.Generator().manual_seed(37)
    state = generator.get_state()
    global_state = torch.get_rng_state().clone()
    x, events = visibility_batch(source_a, generator, 0.5)
    assert torch.equal(global_state, torch.get_rng_state())
    restored = torch.Generator()
    restored.set_state(state)
    repeat, repeated_events = visibility_batch(source_a, restored, 0.5)
    assert events == repeated_events
    torch.testing.assert_close(x["x"], repeat["x"])
    _, next_a = augment_batch(source_a, mirror_a, 0.5)
    _, next_b = augment_batch(source_b, mirror_b, 0.5)
    assert torch.equal(next_a, next_b)


def test_visibility_disabled_is_a_noop_and_other_probabilities_rejected():
    b = batch()
    generator = torch.Generator().manual_seed(18)
    state = generator.get_state().clone()
    x, events = visibility_batch(b, generator, 0.0)
    assert x is b and events == []
    assert torch.equal(state, generator.get_state())
    with pytest.raises(ValueError, match="0.5"):
        visibility_batch(b, generator, 0.9)
