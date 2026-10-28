"""Existing causal encoder with a binary frame head, never complete-clip pooling."""

import torch
from torch import nn

from ..training.models import CausalTCN

VERSION = "causal-binary-tcn-1.0"


class BinaryDetector(CausalTCN):
    def __init__(self, channels=65, hidden=32, dropout=0.1):
        super().__init__(channels, hidden, dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)

    def step(self, frame, state=None, *, reset=False):
        """Bounded past-only streaming reference. Caller resets on source discontinuity."""
        if self.training:
            raise ValueError("Streaming inference requires eval mode")
        history = frame.new_empty((0, frame.numel())) if state is None or reset else state
        history = torch.cat((history, frame.reshape(1, -1)), dim=0)[-15:]
        probability = self(history[None])[0, -1].sigmoid()
        return probability, history[-14:].detach()
