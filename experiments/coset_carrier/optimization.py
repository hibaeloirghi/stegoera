"""Dual-loss stego objective with a capped detector term, and the S-dependent EDICT mixing weight.

L_total = L_det + lambda_code * L_code + lambda_vis * L_perceptual (+ optional nu_mse * MSE)
    L_det  = max(0, M_target - M(z_hat))                 hard cap: zero value and zero gradient once M >= M_target
    L_code = mean_{i in carrier} max(0, gamma - c*_i z_hat_i)
    L_perc = LPIPS(I_stego, I_edit)
"""
from __future__ import annotations

import dataclasses

import torch


def edict_mix_for_steps(steps: int, p_ref: float = 0.93, steps_ref: int = 50) -> float:
    """p(S) = p_ref ** (steps_ref / S): keeps the total mixing contraction p^S fixed across S."""
    return float(p_ref) ** (float(steps_ref) / float(steps))


@dataclasses.dataclass
class StegoLossConfig:
    m_target: float = 20.0
    gamma: float = 0.5
    lambda_code: float = 20.0
    lambda_vis: float = 20.0
    nu_mse: float = 0.0
    sigma_squared: float = 1.5


def capped_detector_loss(margin: torch.Tensor, m_target: float) -> torch.Tensor:
    return torch.relu(m_target - margin)


def coordinate_hinge_loss(z_carrier: torch.Tensor, c_carrier: torch.Tensor, gamma: float) -> torch.Tensor:
    return torch.relu(gamma - c_carrier * z_carrier).mean()


def compute_stego_loss(h, codeword, carrier_idx, detector, img, img_ref, lpips_fn, cfg: StegoLossConfig):
    """h: recovered site carrier (any shape, 16384 entries). codeword: c* on the carrier
    (len == len(carrier_idx)) or on all coordinates (len == h.numel()). The detector key length
    decides whether M is scored on the carrier coordinates or on the full vector."""
    flat = h.flatten()
    z_hat = flat[carrier_idx]
    c = codeword.to(device=flat.device, dtype=flat.dtype).flatten()
    c_carrier = c if c.numel() == z_hat.numel() else c[carrier_idx]
    det_input = z_hat if detector.signed.numel() == z_hat.numel() else flat
    margin, stat, thr = detector.margin(det_input, cfg.sigma_squared)

    l_det = capped_detector_loss(margin, cfg.m_target)
    l_code = coordinate_hinge_loss(z_hat, c_carrier, cfg.gamma)
    l_vis = lpips_fn(img, img_ref) if lpips_fn is not None else flat.new_zeros(())
    mse = (img.clamp(-1, 1) - img_ref.clamp(-1, 1)).square().mean() if img is not None else flat.new_zeros(())
    total = l_det + cfg.lambda_code * l_code + cfg.lambda_vis * l_vis + cfg.nu_mse * mse
    with torch.no_grad():
        agree = float((torch.sign(z_hat) == torch.sign(c_carrier)).double().mean())
        frac_margin = float((c_carrier * z_hat >= cfg.gamma).double().mean())
    return {"total": total, "l_det": l_det, "l_code": l_code, "l_vis": l_vis, "pixel_mse": mse,
            "margin": margin, "statistic": stat, "threshold": thr, "z_hat": z_hat,
            "carrier_sign_agreement": agree, "carrier_frac_at_gamma": frac_margin}
