"""Cover-adaptive PRC codeword selection over the coset of valid encodings of one message.

Every call to PRC `Encode(key, m)` draws fresh randomizer bits r (via galois, not numpy) and a
Bernoulli(noise_rate) error vector (numpy), so repeated calls give distinct valid encodings of m.
We pick the one closest to the host signs in |z|^2-weighted Hamming distance.

Candidates are NOT reproducible from `seed` (galois entropy); callers must save the chosen codeword
together with the key (`receiver_aligned.key_io.save_key`).
"""
from __future__ import annotations

import numpy as np
import torch

from experiments import prc_runtime as pr


def host_average_latent(x_src: torch.Tensor, y_src: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x_src.detach().to(torch.float64) + y_src.detach().to(torch.float64))


def weighted_sign_cost(codewords: torch.Tensor, host_sign: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """sum_i w_i 1[c_i != s_i] for each row of `codewords` (R, n)."""
    return ((codewords != host_sign.unsqueeze(0)).to(weight.dtype) * weight.unsqueeze(0)).sum(dim=1)


def select_optimal_coset_codeword(
    encoding_key,
    message,
    x_src: torch.Tensor,
    y_src: torch.Tensor,
    n_candidates: int = 256,
    carrier_idx: torch.Tensor | None = None,
    seed: int = 0,
):
    """Return the valid encoding c* of `message` minimising sum_i |z_i^+|^2 1[c_i != sign(z_i^+)].

    z^+ = (x_src + y_src) / 2. If `carrier_idx` (flat indices into z^+) is given, the codeword lives
    on those coordinates only and its length must equal len(carrier_idx).
    """
    _, Encode, *_ = pr.prc_modules()
    z = host_average_latent(x_src, y_src).flatten().cpu()
    if carrier_idx is not None:
        z = z[carrier_idx.cpu()]
    n_code = int(encoding_key[0].shape[0])
    assert z.numel() == n_code, (z.numel(), n_code)
    host_sign = torch.where(z >= 0, 1.0, -1.0).to(torch.float64)
    weight = z.square()

    state = np.random.get_state()
    try:
        cands = []
        for i in range(int(n_candidates)):
            np.random.seed(int(seed) + i)
            cands.append(Encode(encoding_key, message=message).to(torch.float64).flatten())
    finally:
        np.random.set_state(state)
    C = torch.stack(cands)
    cost = weighted_sign_cost(C, host_sign, weight)
    agree = (C == host_sign.unsqueeze(0)).to(torch.float64).mean(dim=1)
    best = int(torch.argmin(cost))
    w_sum = float(weight.sum())
    return {
        "codeword": C[best].clone(),
        "candidate": best,
        "cost": float(cost[best]),
        "agreement": float(agree[best]),
        "weighted_agreement": 1.0 - float(cost[best]) / w_sum,
        "candidate_agreement_mean": float(agree.mean()),
        "candidate_agreement_std": float(agree.std()),
        "candidate_agreement_max": float(agree.max()),
        "candidate0_cost": float(cost[0]),
        "candidate_cost_mean": float(cost.mean()),
        "n_candidates": int(n_candidates),
        "n_code": n_code,
    }
