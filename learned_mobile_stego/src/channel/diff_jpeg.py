"""Differentiable JPEG approximation: 8x8 orthonormal DCT, IJG quantization tables, straight-through rounding.

This is not libjpeg. Quality uses the IJG scale (Q < 50 -> 5000/Q, else 200 - 2Q). Rounding uses
the straight-through estimator, so the forward pass is integer quantization and the backward
pass treats rounding as the identity. Gradients reach the pixels; they do not flow into Q.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_LUMA = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]
_CHROMA = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
]


def _dct_matrix(device, dtype) -> torch.Tensor:
    n = torch.arange(8, device=device, dtype=dtype)
    k = n.unsqueeze(1)
    matrix = torch.cos(math.pi * (n + 0.5) * k / 8.0)
    matrix[0] *= 1.0 / math.sqrt(2.0)
    matrix *= math.sqrt(2.0 / 8.0)
    return matrix


def _rgb_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def _ycbcr_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2] - 0.5, ycbcr[:, 2:3] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1)


def quality_table(base: torch.Tensor, quality: float) -> torch.Tensor:
    q = float(quality)
    scale = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    table = torch.floor((base * scale + 50.0) / 100.0).clamp(1, 255)
    return table.detach()


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


class DiffJPEG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("luma", torch.tensor(_LUMA, dtype=torch.float32).view(8, 8), persistent=False)
        self.register_buffer("chroma", torch.tensor(_CHROMA, dtype=torch.float32).view(8, 8), persistent=False)

    def forward(self, rgb: torch.Tensor, quality: float) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"expected NCHW RGB, got {tuple(rgb.shape)}")
        height, width = rgb.shape[-2:]
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        x = F.pad(rgb, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else rgb
        # IJG tables are calibrated for 8-bit coefficients, level-shifted by 128.
        ycbcr = _rgb_ycbcr(x) * 255.0 - 128.0
        blocks = F.unfold(ycbcr, kernel_size=8, stride=8)
        n, _, count = blocks.shape
        blocks = blocks.transpose(1, 2).reshape(n, count, 3, 8, 8).permute(0, 2, 1, 3, 4)
        matrix = _dct_matrix(rgb.device, rgb.dtype)
        coeffs = matrix @ blocks @ matrix.transpose(0, 1)
        tables = torch.stack(
            [
                quality_table(self.luma.to(dtype=rgb.dtype), quality),
                quality_table(self.chroma.to(dtype=rgb.dtype), quality),
                quality_table(self.chroma.to(dtype=rgb.dtype), quality),
            ],
            dim=0,
        ).view(1, 3, 1, 8, 8)
        quantized = _ste_round(coeffs / tables) * tables
        restored = matrix.transpose(0, 1) @ quantized @ matrix
        restored = restored.permute(0, 2, 1, 3, 4).reshape(n, count, 3 * 64).transpose(1, 2)
        folded = F.fold(restored, output_size=ycbcr.shape[-2:], kernel_size=8, stride=8)
        out = _ycbcr_rgb((folded + 128.0) / 255.0)[..., :height, :width]
        return out
