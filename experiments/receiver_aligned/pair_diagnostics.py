"""Mixing-layer identities for the EDICT implementation, not a rewritten formula.

Forward mix matches edict_runtime.coupled_loop:
    x' = p x + (1-p) y
    y' = (1-p) x' + p y
"""
from __future__ import annotations

import torch


def mix_pair(x, y, p: float):
    x2 = p * x + (1.0 - p) * y
    y2 = (1.0 - p) * x2 + p * y
    return x2, y2


def unmix_pair(x, y, p: float):
    """Inverse of mix_pair, y then x, matching the reverse branch."""
    y0 = (y - (1.0 - p) * x) / p
    x0 = (x - (1.0 - p) * y0) / p
    return x0, y0


def common_mode_residual(x, y, eta, p: float):
    mixed = mix_pair(x, y, p)
    mixed_eta = mix_pair(x + eta, y + eta, p)
    return (mixed_eta[0] - mixed[0] - eta), (mixed_eta[1] - mixed[1] - eta)


def difference_ratio(x, y, p: float):
    x2, y2 = mix_pair(x, y, p)
    before = x - y
    after = x2 - y2
    return after, (p ** 2) * before
