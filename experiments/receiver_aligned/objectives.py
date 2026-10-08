"""Surrogate pieces for a later receiver-aligned objective. Not a paper method.

Not used by D0. tau is a codeword margin. It is not the PRC soft-bit variance.
"""
from __future__ import annotations

import torch


def code_hinge(carrier, codeword, tau: float):
    target = codeword.reshape_as(carrier).to(dtype=carrier.dtype, device=carrier.device)
    short = torch.clamp(tau - target * carrier, min=0)
    return short.square().mean()


def softplus_neg_margin(margin: torch.Tensor):
    return torch.nn.functional.softplus(-margin)
