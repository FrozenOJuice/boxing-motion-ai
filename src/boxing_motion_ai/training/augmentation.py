"""Training-only physical horizontal reflection of torso-normalized core features.

A reflected boxer changes anatomical handedness, while the lead/rear relationship
is preserved. Jab stays jab, cross stays cross, and hook/uppercut lead/rear labels
stay unchanged. This makes no assumption about the original boxer's stance.
"""

import torch

MIRROR_VERSION = "horizontal-core-reflection-1.0"
LEFT_RIGHT = (1, 0, 3, 2, 5, 4, 7, 6)


def mirror_features(x, *, mask_only=False):
    """Return a copy, for [..., T, C] or [T, C]; never change time or source data.

    Normalized hip-relative X reflects as -X (raw image X would be 1-X).
    Swap all five channels per joint: X/Y, visibility/presence and usable mask.
    The final torso-valid bit is invariant. Mask-only input swaps binary masks.
    """
    channels = (8, 16, 28) if mask_only else (41, 65, 99)
    if x.ndim < 2 or x.shape[-1] not in channels:
        raise ValueError(f"Expected temporal features ending in {channels} channels")
    indices = torch.tensor(LEFT_RIGHT, device=x.device)
    if mask_only and x.shape[-1] == 28:
        result = x.clone()
        result[..., :16] = (
            x[..., :16]
            .reshape(*x.shape[:-1], 2, 8)
            .index_select(-1, indices)
            .reshape(*x.shape[:-1], 16)
        )
        result[..., 16:] = (
            x[..., 16:].reshape(*x.shape[:-1], 2, 6).flip(-2).reshape(*x.shape[:-1], 12)
        )
        return result
    if mask_only:
        return x.reshape(*x.shape[:-1], -1, 8).index_select(-1, indices).reshape_as(x)
    result = x.clone()
    joints = x[..., :40].reshape(*x.shape[:-1], 8, 5).index_select(-2, indices)
    joints[..., 0] = -joints[..., 0]
    result[..., :40] = joints.reshape(*x.shape[:-1], 40)
    if x.shape[-1] in (65, 99):
        velocity = x[..., 41:65].reshape(*x.shape[:-1], 8, 3).index_select(-2, indices)
        velocity[..., 0] = -velocity[..., 0]
        result[..., 41:65] = velocity.reshape(*x.shape[:-1], 24)
    if x.shape[-1] == 99:
        from .limbs import REFLECT_COLUMNS

        geometry = x[..., 65:].reshape(*x.shape[:-1], 2, 17).flip(-2).clone()
        geometry[..., list(REFLECT_COLUMNS)] *= -1
        result[..., 65:] = geometry.reshape(*x.shape[:-1], 34)
    return result


def augment_batch(batch, generator, probability):
    """One Bernoulli decision per whole clip; same decision for both feature views.

    Call only on optimization batches. Evaluation loaders are never augmented.
    The private RNG does not consume initialization, dropout or shuffle randomness.
    """
    if probability not in (0.0, 0.5):
        raise ValueError("Mirror probability must be disabled (0) or fixed at 0.5")
    selected = torch.zeros(len(batch["labels"]), dtype=torch.bool)
    if probability:
        selected = torch.rand(len(selected), generator=generator) < probability
    if not selected.any():
        return batch, selected
    result = dict(batch)
    for key, mask_only in (("x", False), ("mask_x", True)):
        values = batch[key].clone()
        values[selected] = mirror_features(values[selected], mask_only=mask_only)
        result[key] = values
    return result, selected


VISIBILITY_VERSION = "temporary-forearm-occlusion-1.0"
VISIBILITY_POLICY = {
    "version": VISIBILITY_VERSION,
    "unit": "one contiguous span on one uniformly sampled side per selected clip",
    "patterns": ["wrist_only", "elbow_and_wrist"],
    "pattern_probability": [0.5, 0.5],
    "span_fraction_uniform": [0.2, 0.5],
    "rounding": "ceil; cap at length-1; minimum one frame",
    "protected_core_joints": [0, 1, 6, 7],
    "selection": "independent of labels, source visibility and mirror decisions",
    "order": "after mirroring; artificial side refers to resulting model channels",
}


def visibility_batch(batch, generator, probability):
    """Temporary class-independent occlusions on copies of optimization inputs only.

    Source-validity can only decrease; no shoulder/hip, timestamp, label, sequence
    length or raw record changes. Do not re-filter clips by the source quality
    policy after augmentation. An event on an already-invalid joint is logged as
    a no-op, without resampling according to class-correlated visibility.
    """
    if probability not in (0.0, 0.5):
        raise ValueError("Visibility probability must be disabled (0) or fixed at 0.5")
    if not probability:
        return batch, []
    if batch["x"].shape[-1] != 41:
        raise ValueError("Visibility augmentation supports original XY only; velocities disabled")
    # Fixed number of private RNG draws per clip; never inspect labels for decisions.
    random = torch.rand(len(batch["labels"]), 5, generator=generator)
    result = dict(batch)
    result["x"], result["mask_x"] = batch["x"].clone(), batch["mask_x"].clone()
    joints = result["x"][..., :40].reshape(*result["x"].shape[:-1], 8, 5)
    events = []
    import math

    for i, (coin, side_draw, pattern_draw, span_draw, start_draw) in enumerate(random.tolist()):
        n = int(batch["lengths"][i])
        if coin >= probability or n < 2:
            continue
        side = 0 if side_draw < 0.5 else 1
        selected = [4 + side] if pattern_draw < 0.5 else [2 + side, 4 + side]
        length = min(n - 1, max(1, math.ceil(n * (0.2 + 0.3 * span_draw))))
        start = int(start_draw * (n - length + 1))
        stop = start + length
        hidden_usable = int(joints[i, start:stop, selected, 4].sum())
        hidden_source = int(result["mask_x"][i, start:stop, selected].sum())
        joints[i, start:stop, selected, :] = 0
        result["mask_x"][i, start:stop, selected] = 0
        events.append(
            {
                "source_row": batch["source_rows"][i],
                "start": start,
                "stop": stop,
                "side": "left" if side == 0 else "right",
                "core_joints": selected,
                "newly_hidden_usable_entries": hidden_usable,
                "newly_hidden_source_entries": hidden_source,
            }
        )
    return result, events
