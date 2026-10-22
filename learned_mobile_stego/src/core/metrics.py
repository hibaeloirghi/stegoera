"""PSNR and SSIM for images in [0, 1]."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).square().mean().clamp(min=1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean SSIM with an 11x11 Gaussian window, computed per image then averaged."""
    if a.ndim == 3:
        a, b = a.unsqueeze(0), b.unsqueeze(0)
    radius = 5
    coords = torch.arange(-radius, radius + 1, device=a.device, dtype=a.dtype)
    kernel = torch.exp(-0.5 * (coords / 1.5) ** 2)
    kernel = kernel / kernel.sum()
    channels = a.shape[1]
    kh = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kv = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)

    def blur(x):
        x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kh, groups=channels)
        return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), kv, groups=channels)

    mu_a, mu_b = blur(a), blur(b)
    sigma_a = blur(a * a) - mu_a.square()
    sigma_b = blur(b * b) - mu_b.square()
    sigma_ab = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    value = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / ((mu_a.square() + mu_b.square() + c1) * (sigma_a + sigma_b + c2))
    return float(value.mean())
