"""PRC encode/detect plus a VAE-only decoder-inversion adapter (no PRC pipeline object)."""
from __future__ import annotations

import sys
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup

from .paths import PRC_ROOT, ensure_dirs

VAE_SCALE = 0.18215


def _ensure_prc_on_path():
    root = str(PRC_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def prc_modules():
    _ensure_prc_on_path()
    from src.prc import KeyGen, Encode, Detect, Decode
    import src.pseudogaussians as prc_gaussians

    return KeyGen, Encode, Detect, Decode, prc_gaussians


def keygen(n=4 * 64 * 64, message_length=512, fpr=1e-5, t=3, seed=0):
    """Sample PRC keys. Seeded via numpy (galois uses np.random)."""
    ensure_dirs()
    _ensure_prc_on_path()
    np.random.seed(seed)
    KeyGen, *_ = prc_modules()
    return KeyGen(n, message_length=message_length, false_positive_rate=fpr, t=t)


def encode_codeword(encoding_key, message=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    _, Encode, *_rest = prc_modules()
    *_, prc_gaussians = prc_modules()
    codeword = Encode(encoding_key, message=message)
    z = prc_gaussians.sample(codeword)
    return codeword, z


def bits_from_message_length(message_length, rng_seed=0):
    if message_length <= 0:
        return None
    rng = np.random.default_rng(rng_seed)
    return rng.integers(0, 2, size=message_length).tolist()


def recover_soft_bits(z_flat, variance=1.5):
    *_, prc_gaussians = prc_modules()
    z = z_flat.detach().to(torch.float64).flatten().cpu()
    return prc_gaussians.recover_posteriors(z, variances=float(variance)).flatten().cpu()


def detect_decode(decoding_key, soft_bits):
    _, _, Detect, Decode, _ = prc_modules()
    detected = bool(Detect(decoding_key, soft_bits))
    decoded = Decode(decoding_key, soft_bits)
    decode_ok = decoded is not None
    return detected, decode_ok, decoded


def sign_agreement(recovered_z, codeword):
    rec = recovered_z.detach().to(torch.float64).flatten().cpu()
    c = codeword.detach().to(torch.float64).flatten().cpu()
    agree = (torch.sign(rec) == torch.sign(c)).float().mean().item()
    flip = 1.0 - agree
    return agree, flip


def decoder_inv_vae(vae, image, n_steps=100, lr=0.1, scale=VAE_SCALE, device=None):
    """PRC-style decoder inversion against an arbitrary VAE (EDICT's AutoencoderKL).

    image: (1,3,512,512) in [-1,1]. Returns z at diffusion scale (times 0.18215), shape (1,4,64,64).
    Init is encoder.sample(), then Adam 100 steps on ||D(z)-x||, cosine warmup — same as
    InversableStableDiffusionPipeline.decoder_inv, without that pipeline class.
    """
    if device is None:
        device = image.device
    image = image.to(device=device)
    # We optimize only z. Computing/accumulating gradients for every VAE weight is
    # unnecessary and was the dominant receiver cost.
    vae.requires_grad_(False)
    with torch.no_grad():
        z0 = vae.encode(image).latent_dist.sample() * scale
    z = z0.detach().clone().to(dtype=torch.float32 if image.dtype != torch.float64 else image.dtype)
    # Keep native VAE dtype (EDICT VAE is float64).
    z = z.detach().clone().requires_grad_(True)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    opt = torch.optim.Adam([z], lr=lr)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=n_steps)
    for _ in range(n_steps):
        x_pred = vae.decode(z / scale).sample
        loss = loss_fn(x_pred, image.to(dtype=x_pred.dtype))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return z.detach()


def detection_margin(decoding_key, posteriors, false_positive_rate=None):
    """Return (statistic, threshold, margin) using the same parity score as Detect.

    margin = sum_j log((1+Pi_j)/2) - threshold. Detect is True iff margin >= 0.
    """
    (
        _generator_matrix,
        parity_check_matrix,
        one_time_pad,
        false_positive_rate_key,
        noise_rate,
        _test_bits,
        _g,
        _max_bp_iter,
        t,
    ) = decoding_key
    fpr = false_positive_rate_key if false_positive_rate is None else false_positive_rate
    signed = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float))
    values = signed * posteriors.detach().to(torch.float64).flatten().cpu().numpy()
    r = parity_check_matrix.shape[0]
    pi = np.prod(values[parity_check_matrix.indices.reshape(r, t)], axis=1)
    log_plus = np.log((1 + pi) / 2)
    log_minus = np.log((1 - pi) / 2)
    log_prod = log_plus + log_minus
    const = 0.5 * np.sum(np.power(log_plus, 2) + np.power(log_minus, 2) - 0.5 * np.power(log_prod, 2))
    threshold = float(np.sqrt(2 * const * np.log(1 / fpr)) + 0.5 * log_prod.sum())
    statistic = float(log_plus.sum())
    return statistic, threshold, statistic - threshold


def select_codeword(encoding_key, message, host_sign, host_weight, n_candidates, seed):
    """Pick the valid encoding of `message` closest to the host in weighted Hamming distance.

    Cost is sum_i |host_i| * 1[c_i != sign(host_i)]. Numpy RNG is restored afterward.
    """
    _, Encode, *_rest = prc_modules()
    sign = host_sign.detach().to(torch.float64).flatten().cpu()
    sign = torch.where(sign == 0, torch.ones_like(sign), torch.sign(sign))
    weight = host_weight.detach().to(torch.float64).flatten().cpu().abs()
    state = np.random.get_state()
    best = None
    try:
        for i in range(int(n_candidates)):
            np.random.seed(int(seed) + i)
            codeword = Encode(encoding_key, message=message)
            flat = codeword.detach().to(torch.float64).flatten().cpu()
            mismatch = flat != sign
            cost = float((weight * mismatch.to(weight.dtype)).sum().item())
            agreement = float((~mismatch).to(torch.float64).mean().item())
            weighted_agreement = 1.0 - cost / float(weight.sum().item())
            if best is None or cost < best["cost"]:
                best = {
                    "codeword": codeword,
                    "cost": cost,
                    "agreement": agreement,
                    "weighted_agreement": weighted_agreement,
                    "candidate": i,
                }
    finally:
        np.random.set_state(state)
    return best


def calibrate_variance(z_flat, decoding_key, grid=(0.5, 1.0, 1.5, 2.0, 3.0)):
    """Pick the recover_posteriors variance that maximizes Detect's log-plus sum proxy.

    Detect already thresholds; we reuse Detect's boolean plus a simple |soft_bit| mean.
    """
    best = None
    records = []
    for var in grid:
        sb = recover_soft_bits(z_flat, variance=var)
        det, dec, _ = detect_decode(decoding_key, sb)
        mean_abs = float(sb.abs().mean().item())
        rec = {"variance": var, "detected": det, "decoded": dec, "mean_abs_soft": mean_abs}
        records.append(rec)
        score = (int(det) + int(dec), mean_abs)
        if best is None or score > best[0]:
            best = (score, rec, sb)
    return best[1], best[2], records
