"""Differentiable copy of the PRC parity-detector margin. OUR PROPOSAL (not a paper object).

Same arithmetic as PRC-Watermark src/prc.py Detect and prc_scores.detection_margin:

    s_i      = erf(h_i / sqrt(2 v (1+v)))          v = sigma^2 (variance), Gunn App. C.1 Fact 1
    values_i = (1-2*noise_rate) * (1-2*otp_i) * s_i
    Pi_w     = prod_{i in w} values_i              (weight-t checks)
    stat     = sum_w log((1+Pi_w)/2)
    thr      = sqrt(2 C log(1/F)) + 0.5 sum_w log_prod_w
    margin   = stat - thr          Detect is True iff margin >= 0

Differences from Detect (documented, deliberate):
    * Pi is clamped to [-1+eps, 1-eps] so the logs stay finite for autograd.
      With eps=1e-12 this is bit-for-bit the same unless some |Pi| rounds to 1.
    * Everything runs in torch float64 so gradients flow to the carrier.

The margin is a smooth function of the carrier. It is NOT the decoder: it says nothing about
whether Decode returns the right payload. Original Detect/Decode stay the final evaluators.
"""
from __future__ import annotations

import numpy as np
import torch


class TorchDetector:
    def __init__(self, decoding_key, device="cpu", clip_eps: float = 1e-12):
        (
            _g, parity_check_matrix, one_time_pad, fpr, noise_rate, _test_bits, _gdim, _bp, t,
        ) = decoding_key
        r = parity_check_matrix.shape[0]
        self.r, self.t = r, t
        self.idx = torch.from_numpy(parity_check_matrix.indices.reshape(r, t).astype(np.int64)).to(device)
        signed = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float))
        self.signed = torch.from_numpy(np.asarray(signed, dtype=np.float64)).to(device)
        self.fpr = float(fpr)
        self.clip_eps = float(clip_eps)

    @staticmethod
    def soft_bits(h: torch.Tensor, sigma_squared: float) -> torch.Tensor:
        v = float(sigma_squared)
        return torch.erf(h.flatten() / float(np.sqrt(2.0 * v * (1.0 + v))))

    def margin(self, h: torch.Tensor, sigma_squared: float):
        """Returns (margin, statistic, threshold) as 0-dim tensors. h is the carrier, any shape."""
        soft = self.soft_bits(h.to(torch.float64), sigma_squared)
        values = self.signed * soft
        pi = values[self.idx].prod(dim=1)
        pi = pi.clamp(-1.0 + self.clip_eps, 1.0 - self.clip_eps)
        lp = torch.log((1.0 + pi) / 2.0)
        lm = torch.log((1.0 - pi) / 2.0)
        lprod = lp + lm
        const = 0.5 * (lp.square() + lm.square() - 0.5 * lprod.square()).sum()
        thr = torch.sqrt(2.0 * const * np.log(1.0 / self.fpr)) + 0.5 * lprod.sum()
        stat = lp.sum()
        return stat - thr, stat, thr
