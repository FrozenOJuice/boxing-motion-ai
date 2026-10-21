"""Small clip classifiers; causal encoding, complete-clip Level 1 pooling."""

import torch
from torch import nn
from torch.nn import functional as F


def pool(x, mask):
    weights = mask[..., None].to(x.dtype)
    return (x * weights).sum(1) / weights.sum(1).clamp_min(1)


class StaticBaseline(nn.Module):
    """Order-invariant mean/std features and a linear six-class classifier."""

    def __init__(self, channels, hidden=32, dropout=0.1):
        super().__init__()
        self.head = nn.Linear(channels * 2, 6)

    def forward(self, x, mask):
        mean = pool(x, mask)
        variance = pool((x - mean[:, None]).square(), mask)
        return self.head(torch.cat((mean, (variance + 1e-8).sqrt()), dim=-1))


class CausalBlock(nn.Module):
    def __init__(self, hidden, dilation, dropout):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(hidden, hidden, 3, dilation=dilation)
        self.norm = nn.LayerNorm(hidden)  # Channels only, no future-frame/batch statistics.
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv(F.pad(x.transpose(1, 2), (self.padding, 0))).transpose(1, 2)
        return x + self.drop(F.relu(self.norm(y)))


class CausalTCN(nn.Module):
    """Three left-padded blocks, receptive field 15 frames; no temporal resampling."""

    def __init__(self, channels, hidden=32, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(channels, hidden)
        self.blocks = nn.Sequential(*(CausalBlock(hidden, d, dropout) for d in (1, 2, 4)))
        self.head = nn.Linear(hidden, 6)

    def encode(self, x):
        return self.blocks(self.input(x))

    def forward(self, x, mask):
        return self.head(pool(self.encode(x), mask))
