"""Small offline three-head TCN; no production imports or six-class outputs."""

import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .instances_v2 import runs

VERSION = "offline-punch-localizer-1.0"
MODES = ("pose", "rgb_motion", "fused")
CHANNELS = 162


def model_observable(a):
    """Matched evidence across arms: missing person or native motion pair is abstention."""
    return a["observable"] & (a["visual"][:, -1] == 1)


def model_input(arrays, mode):
    if mode not in MODES:
        raise ValueError("Unknown localization input mode")
    pose = arrays["pose"].copy()
    # Fixed numerical scaling only. All original validity/confidence channels stay unchanged.
    for j in range(8):
        pose[:, 41 + j * 3 : 43 + j * 3] /= 10
    visual = arrays["visual"].copy()
    if mode == "pose":
        visual[:] = 0
    elif mode == "rgb_motion":
        pose[:] = 0
    return torch.from_numpy(np.concatenate((pose, visual), axis=1))


class Block(nn.Module):
    def __init__(self, hidden, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(hidden, hidden, 3, dilation=dilation, padding=dilation)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x + self.drop(F.relu(self.norm(y)))


class Localizer(nn.Module):
    """15 native samples: at most seven past/seven future frames, no clip pooling."""

    def __init__(self, hidden=32, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(CHANNELS, hidden)
        self.blocks = nn.Sequential(*(Block(hidden, d, dropout) for d in (1, 2, 4)))
        self.head = nn.Linear(hidden, 3)

    def forward(self, x, present=None):
        h = self.input(x)
        if present is not None:
            h = h * present[..., None]
        for block in self.blocks:
            h = block(h)
            if present is not None:
                h = h * present[..., None]
        return self.head(h)


def sequences(cases, mode, *, training):
    result = []
    for name, c in cases.items():
        if training and not c["receipt"]["train_eligible"]:
            continue
        a = c["arrays"]
        x = model_input(a, mode)
        for start, stop in runs(model_observable(a), a["reset"]):
            result.append(
                dict(
                    case=name,
                    start=start,
                    stop=stop,
                    x=x[start:stop],
                    y=torch.from_numpy(a["targets"][start:stop]),
                    mask=torch.from_numpy(a["supervision_mask"][start:stop]),
                )
            )
    return result


def batch(seqs):
    n = max(len(s["x"]) for s in seqs)
    x = torch.zeros(len(seqs), n, CHANNELS)
    y = torch.zeros(len(seqs), n, 3)
    mask = torch.zeros(len(seqs), n, 3, dtype=torch.bool)
    present = torch.zeros(len(seqs), n, dtype=torch.bool)
    for i, s in enumerate(seqs):
        z = len(s["x"])
        x[i, :z], y[i, :z], mask[i, :z] = s["x"], s["y"], s["mask"]
        present[i, :z] = True
    return x, y, mask, present


def loss(logits, y, mask, weights):
    if not mask.any() or not mask.any(dim=(0, 1)).all():
        raise ValueError("Every localization head needs eligible supervision")
    y = torch.where(mask, y, 0)
    values = F.binary_cross_entropy_with_logits(logits, y, pos_weight=weights, reduction="none")
    normalization = ((1 - y) + weights * y) * mask
    return ((values * mask).sum((0, 1)) / normalization.sum((0, 1))).mean()


def fit(cases, mode, config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    model = Localizer(config["hidden"], config["dropout"])
    x, y, mask, present = batch(sequences(cases, mode, training=True))
    positive = (y * mask).sum((0, 1))
    if (positive <= 0).any():
        raise ValueError("No eligible positive activity/boundary target")
    weights = ((mask.sum((0, 1)) - positive) / positive).clamp(
        max=config["maximum_positive_weight"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    model.eval()
    with torch.no_grad():
        initial = float(loss(model(x, present), y, mask, weights))
    curves = []
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        value = loss(model(x, present), y, mask, weights)
        if not torch.isfinite(value):
            raise ValueError("Nonfinite localization loss")
        value.backward()
        norm = nn.utils.clip_grad_norm_(
            model.parameters(), config["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        model.eval()
        with torch.no_grad():
            measured = float(loss(model(x, present), y, mask, weights))
        curves.append(dict(epoch=epoch + 1, training_loss=measured, gradient_norm=float(norm)))
    return model.eval(), dict(
        initial_loss=initial,
        final_loss=curves[-1]["training_loss"],
        curves=curves,
        positive_weights=weights.tolist(),
        supervised_observable_frames=mask.sum((0, 1)).tolist(),
        positive_target_mass=positive.tolist(),
        batch_shape=list(x.shape),
        batch_tensor_bytes=sum(t.numel() * t.element_size() for t in (x, y, mask)),
    )


@torch.no_grad()
def predict(model, case, mode):
    a = case["arrays"]
    values = np.full((len(a["pose"]), 3), np.nan, np.float32)
    x = model_input(a, mode)
    for start, stop in runs(model_observable(a), a["reset"]):
        values[start:stop] = model(x[start:stop][None])[0].sigmoid().numpy()
    return values
