"""Subspace carrier selection: n_carrier of the 4x64x64 site coordinates carry the PRC codeword.

score_i = stability_i * band_rank_i
    stability_i = min(|h_i|, |h'_i|) if sign(h_i) == sign(h'_i) else 0
        h  = carrier view of an image (site recovered from VAE-encoder mean of the image)
        h' = carrier view after one extra VAE decode/encode round trip of that latent
    band_rank_i = rank in [0, 1] of the local mid-frequency energy of h at coordinate i
        (difference of Gaussians sigma_lo -> sigma_hi in latent pixels, squared, smoothed)

The function only needs an image-derived (h, h'), so the receiver can recompute it from the
received image; whether that reproduces the sender's mask is measured, not assumed.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LATENT_SHAPE = (4, 64, 64)


def _gauss_kernel(sigma: float, dtype, device):
    r = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-r, r + 1, dtype=dtype, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(z: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable per-channel blur of (C, H, W) with reflect padding."""
    k = _gauss_kernel(sigma, z.dtype, z.device)
    r = (k.numel() - 1) // 2
    c = z.shape[0]
    x = z.unsqueeze(0)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x.squeeze(0)


def _rank01(v: torch.Tensor) -> torch.Tensor:
    flat = v.flatten()
    order = torch.argsort(flat, stable=True)
    ranks = torch.empty_like(flat)
    ranks[order] = torch.arange(flat.numel(), dtype=flat.dtype, device=flat.device)
    return (ranks / max(1, flat.numel() - 1)).view_as(v)


def carrier_scores(h: torch.Tensor, h_rt: torch.Tensor, sigma_lo=0.7, sigma_hi=2.0, smooth=2.0) -> dict:
    h = h.detach().to(torch.float64).reshape(LATENT_SHAPE)
    h_rt = h_rt.detach().to(torch.float64).reshape(LATENT_SHAPE)
    band = gaussian_blur(h, sigma_lo) - gaussian_blur(h, sigma_hi)
    energy = gaussian_blur(band.square(), smooth)
    band_rank = _rank01(energy)
    same = torch.sign(h) == torch.sign(h_rt)
    stability = torch.where(same, torch.minimum(h.abs(), h_rt.abs()), torch.zeros_like(h))
    return {"score": stability * band_rank, "stability": stability, "band_rank": band_rank,
            "sign_stable_frac": float(same.double().mean())}


def get_subspace_carrier_mask(h: torch.Tensor, h_rt: torch.Tensor, n_carrier: int = 2048,
                              sigma_lo=0.7, sigma_hi=2.0, smooth=2.0) -> torch.Tensor:
    """Boolean mask of shape [4, 64, 64] with exactly n_carrier True entries (top scores)."""
    s = carrier_scores(h, h_rt, sigma_lo, sigma_hi, smooth)["score"].flatten()
    n = s.numel()
    assert 0 < n_carrier <= n, n_carrier
    # Stable descending order; ties resolved by flat index so sender and receiver agree on ties.
    order = torch.argsort(-s, stable=True)
    mask = torch.zeros(n, dtype=torch.bool, device=s.device)
    mask[order[:n_carrier]] = True
    return mask.view(LATENT_SHAPE)


def carrier_indices(mask: torch.Tensor) -> torch.Tensor:
    """Ascending flat indices of the mask; codeword position j sits at carrier_indices[j]."""
    return torch.nonzero(mask.flatten(), as_tuple=False).flatten()


def mask_overlap(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.flatten().bool(), b.flatten().bool()
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return {"intersection": inter, "iou": inter / max(1, union), "recall_of_a": inter / max(1, int(a.sum()))}
