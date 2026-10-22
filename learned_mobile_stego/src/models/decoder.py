"""ConvNet decoder. Spatial features, global average pooling, then one linear layer of codeword logits."""
from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, code_bits: int, base_channels: int = 16):
        super().__init__()
        b = int(base_channels)
        self.features = nn.Sequential(
            nn.Conv2d(3, b, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(b * 2, b * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(b * 2, b * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(b * 4, int(code_bits))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image).mean(dim=(2, 3)))

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)
