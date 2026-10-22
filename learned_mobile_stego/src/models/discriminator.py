"""Discriminator on the residual I_stego - I, against residuals of ordinary sharpen/contrast edits."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 16):
        super().__init__()
        b = int(base_channels)
        self.features = nn.Sequential(
            nn.Conv2d(3, b, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(b, b * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(b * 2, b * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Linear(b * 2, 1)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(residual).mean(dim=(2, 3))).squeeze(1)
