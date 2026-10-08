"""R1: receiver-aligned single-S4-site problem. OUR PROPOSAL, not a paper method.

Notation (matches the handoff):
    q_k      baseline pair at S4 site k (saved by D0 as `s4_pair`).
    Q_k(q)   denoising suffix, forward steps k+1..n-1, to the final pair.
    I(q)     decode(final x), float image (no PNG in the loop).
    R_k(I)   surrogate receiver: posterior-MEAN encoder -> duplicate (z,z) -> reverse k_rev steps
             with the matched edit prompt -> state at the same site. Differentiable.
             The TRUE receiver (decoder inversion, PNG) is evaluated separately.
    h_k(I)   fixed carrier = recovered site-x / site_scale, site_scale = 1.0 (documented, fixed).

Variables: q_marked = q + (eta_x, eta_y), zero initialization.

Loss (all terms reported separately):
    L_code    = a * mean(max(0, tau - c*h)^2)  +  b * m0 * softplus(-margin(soft(h)) / m0)
    L_quality = mu * LPIPS(I, I_base_edit) + nu * mean((I - I_base_edit)^2)
    L_pair    = beta * mean((x_final - y_final)^2)
    L_energy  = c * ((mean(h^2) - v_ref) / v_ref)^2      v_ref = energy of the baseline site carrier (constant)
    L_total   = L_code + L_energy + L_quality + L_pair

c is the genuine PRC Encode output for the 64-bit message. tau (a carrier margin, carrier units)
and sigma^2 (soft-bit variance, fixed in advance, NOT tuned on the sample) are different parameters.
`margin` is the torch copy of the original Detect statistic (prc_torch). Detect/Decode on the
true receiver are the final evaluators; the margin alone does not guarantee the payload.
m0 only sets where softplus saturates; for margin << -m0 the term is ~ b * |margin|.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.receiver_aligned import diff_edict as de
from experiments.receiver_aligned.prc_torch import TorchDetector


@dataclass
class R1Config:
    k: int = 20
    sigma_squared: float = 1.5
    site_scale: float = 1.0
    tau: float = 0.3
    a_hinge: float = 0.1
    b_margin: float = 1.0
    m0: float = 20.0
    mu_lpips: float = 20.0
    nu_mse: float = 5.0
    beta_pair: float = 0.0
    c_energy: float = 50.0   # anchor on carrier energy; blocks the margin->0 amplitude-shrink shortcut


class R1Problem:
    def __init__(self, ef, chain, eps_fn, detector: TorchDetector, codeword, q, img_ref, lpips: de.LpipsLoss, cfg: R1Config):
        self.ef, self.chain, self.eps, self.det = ef, chain, eps_fn, detector
        self.c = codeword.reshape(1, 4, 64, 64).to(device=q[0].device, dtype=torch.float64)
        self.q = [v.detach() for v in q]
        self.img_ref = img_ref.detach()
        self.lpips = lpips
        self.cfg = cfg
        # Fixed constant: energy of the baseline site carrier. Not tuned on any sample's key score.
        self.v_ref = float(self.q[0].square().mean().item()) / (cfg.site_scale ** 2)

    def evaluate(self, eta_x, eta_y):
        cfg = self.cfg
        site = [self.q[0] + eta_x, self.q[1] + eta_y]
        fin = self.chain.forward_steps(site, cfg.k + 1, self.chain.n, self.eps)
        img = de.decode_image(self.ef, fin[0])
        z = de.encode_mean(self.ef, img)
        rec = self.chain.reverse_to_site([z, z], cfg.k, self.eps)
        h = rec[0] / cfg.site_scale
        margin, stat, thr = self.det.margin(h, cfg.sigma_squared)
        hinge = torch.clamp(cfg.tau - self.c * h, min=0).square().mean()
        l_code = cfg.a_hinge * hinge + cfg.b_margin * cfg.m0 * torch.nn.functional.softplus(-margin / cfg.m0)
        # Detect returns margin == 0 (True) for all-zero soft bits and random carriers sit near -20, so
        # shrinking |h| raises the margin with NO code content. Anchor the carrier energy to its baseline value.
        l_energy = cfg.c_energy * ((h.square().mean() - self.v_ref) / self.v_ref) ** 2
        lp = self.lpips(img, self.img_ref)
        mse = (img.clamp(-1, 1) - self.img_ref.clamp(-1, 1)).square().mean()
        l_q = cfg.mu_lpips * lp + cfg.nu_mse * mse
        pair_mse = (fin[0] - fin[1]).square().mean()
        l_pair = cfg.beta_pair * pair_mse
        total = l_code + l_energy + l_q + l_pair
        return {
            "total": total, "l_code": l_code, "l_energy": l_energy, "carrier_energy": h.square().mean(), "l_quality": l_q, "l_pair": l_pair,
            "margin": margin, "statistic": stat, "threshold": thr, "hinge": hinge,
            "lpips": lp, "pixel_mse": mse, "pair_mse": pair_mse,
            "h": h, "fin": fin, "img": img, "rec": rec,
        }


def build(ef, unet32, unet_dtype: str, cfg: R1Config, run: dict):
    """Assemble the R1 problem from D0's saved baseline states. No sender recomputation."""
    from experiments import prc_runtime as pr
    from experiments.paths import RESULTS

    dev = ef.device
    raw = RESULTS / "receiver_aligned" / "raw_observations"
    base = torch.load(raw / "baseline_final_pair.pt", map_location="cpu")
    q = [v.to(device=dev, dtype=torch.float64) for v in base["s4_pair"]]
    fin_ref = [base["x"].to(device=dev, dtype=torch.float64), base["y"].to(device=dev, dtype=torch.float64)]
    chain = de.Chain(ef, steps=int(run["steps"]), strength=float(run["s"]), mix=float(run["mix_weight"]))
    assert int(base["s4_index"]) == cfg.k, (base["s4_index"], cfg.k)
    assert int(chain.t_fwd[cfg.k]) == int(base["s4_timestep"]), (int(chain.t_fwd[cfg.k]), base["s4_timestep"])
    eps = de.GuidedEps(ef, run["edit_prompt"], float(run["guidance_scale"]), unet_dtype, unet32)
    seed = int(run["seed"])
    enc_key, dec_key = pr.keygen(
        n=4 * 64 * 64, message_length=int(run["key_message_length"]),
        fpr=float(run["fpr"]), t=int(run["prc_t"]), seed=seed)
    message = pr.bits_from_message_length(int(run["message_length"]), rng_seed=seed + 7)
    codeword, _z = pr.encode_codeword(enc_key, message=message, seed=seed + 99)
    det = TorchDetector(dec_key, device=dev)
    with torch.no_grad():
        img_ref = de.decode_image(ef, fin_ref[0])
    lp = de.LpipsLoss(dev)
    problem = R1Problem(ef, chain, eps, det, codeword, q, img_ref, lp, cfg)
    return {
        "problem": problem, "chain": chain, "q": q, "fin_ref": fin_ref, "eps": eps, "det": det,
        "enc_key": enc_key, "dec_key": dec_key, "message": message, "codeword": codeword,
        "base": base, "img_ref": img_ref, "lpips": lp,
    }
