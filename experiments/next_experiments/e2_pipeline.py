"""E2 pipeline: Codeword-only vs saturated-detector objectives for receiver-aligned optimization."""
from __future__ import annotations

import dataclasses
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from experiments.receiver_aligned import diff_edict as de
from experiments.receiver_aligned.prc_torch import TorchDetector
from experiments.receiver_aligned.key_io import load_key, key_hash
from experiments.paths import RESULTS


@dataclasses.dataclass
class E2Config:
    k: int = 20                         # S4 site index (step 20, timestep 380)
    sigma_squared: float = 1.5          # soft-bit variance v
    site_scale: float = 1.0             # carrier normalization
    tau: float = 0.5                    # target codeword margin in h
    a_cw: float = 10.0                  # weight on L_cw = mean(max(0, tau - c*h)^2)
    b_det: float = 0.0                  # weight on L_det (0.0 for Trial A; >0 for Trial B)
    m_target: float = 20.0              # saturation target margin for Trial B
    tau_det: float = 5.0                # smoothing width for detector margin saturation
    mu_lpips: float = 20.0              # LPIPS quality penalty vs unwatermarked edit
    nu_mse: float = 5.0                 # pixel MSE penalty vs unwatermarked edit
    beta_pair: float = 0.0              # pair consistency penalty
    c_energy: float = 50.0              # anchor on carrier energy


class E2Problem:
    def __init__(self, ef, chain, eps_fn, detector: TorchDetector, codeword, q, img_ref, lpips: de.LpipsLoss, cfg: E2Config):
        self.ef = ef
        self.chain = chain
        self.eps = eps_fn
        self.det = detector
        if isinstance(codeword, np.ndarray):
            cw_tensor = torch.from_numpy(codeword)
        else:
            cw_tensor = codeword
        self.c = cw_tensor.reshape(1, 4, 64, 64).to(device=q[0].device, dtype=torch.float64)
        self.q = [v.detach() for v in q]
        self.img_ref = img_ref.detach()
        self.lpips = lpips
        self.cfg = cfg
        # Reference carrier energy from baseline site
        self.v_ref = float(self.q[0].square().mean().item()) / (cfg.site_scale ** 2)

    def evaluate(self, eta_x, eta_y, compute_diagnostics: bool = True):
        cfg = self.cfg
        site = [self.q[0] + eta_x, self.q[1] + eta_y]
        fin = self.chain.forward_steps(site, cfg.k + 1, self.chain.n, self.eps)
        img = de.decode_image(self.ef, fin[0])
        z = de.encode_mean(self.ef, img)
        rec = self.chain.reverse_to_site([z, z], cfg.k, self.eps)
        h = rec[0] / cfg.site_scale

        # Codeword loss L_cw
        hinge_coords = torch.clamp(cfg.tau - self.c * h, min=0)
        l_cw = hinge_coords.square().mean()

        # Detector margin and saturated loss L_det
        margin, stat, thr = self.det.margin(h, cfg.sigma_squared)
        if cfg.b_det > 0:
            # Saturated penalty: softplus((m_target - margin) / tau_det)
            # Once margin >= m_target, penalty saturates and gradient vanishes
            l_det = F.softplus((cfg.m_target - margin) / cfg.tau_det)
        else:
            l_det = margin * 0.0

        # Carrier energy anchor
        carrier_energy = h.square().mean()
        l_energy = cfg.c_energy * ((carrier_energy - self.v_ref) / self.v_ref) ** 2

        # Image quality constraints vs baseline edit
        lp = self.lpips(img, self.img_ref)
        mse = (img.clamp(-1, 1) - self.img_ref.clamp(-1, 1)).square().mean()
        l_q = cfg.mu_lpips * lp + cfg.nu_mse * mse

        # Pair consistency penalty
        pair_mse = (fin[0] - fin[1]).square().mean()
        l_pair = cfg.beta_pair * pair_mse

        # Total objective
        total = cfg.a_cw * l_cw + cfg.b_det * l_det + l_q + l_energy + l_pair

        # Carrier-level gradient diagnostics (fast arithmetic at h)
        diag = {}
        with torch.no_grad():
            # Derivative of hinge^2 = 2 * (tau - c*h) * (-c)
            # Derivative of mean wrt h is (1/N) * (-2*c*(tau - c*h)_+)
            g_cw = (cfg.a_cw / h.numel()) * (-2.0 * self.c.flatten() * hinge_coords.flatten())
            diag["norm_g_cw"] = float(g_cw.norm())
            diag["norm_g_det"] = 0.0
            diag["cos_cw_det"] = 0.0
            # Derivative of energy anchor: 2 * c_energy * ((E - v_ref)/v_ref) * (2*h / (N*v_ref))
            g_energy = (4.0 * cfg.c_energy * (carrier_energy - self.v_ref) / (self.v_ref ** 2 * h.numel())) * h.flatten()
            diag["norm_g_energy"] = float(g_energy.norm())

        return {
            "total": total,
            "l_cw": l_cw,
            "l_det": l_det,
            "l_quality": l_q,
            "l_energy": l_energy,
            "l_pair": l_pair,
            "carrier_energy": carrier_energy,
            "margin": margin,
            "statistic": stat,
            "threshold": thr,
            "lpips": lp,
            "pixel_mse": mse,
            "pair_mse": pair_mse,
            "h": h,
            "fin": fin,
            "img": img,
            "rec": rec,
            "diag": diag,
        }


def build_e2(ef, unet32, unet_dtype: str, cfg: E2Config, run_cfg: dict):
    dev = ef.device
    raw = RESULTS / "receiver_aligned" / "raw_observations"
    base = torch.load(raw / "baseline_final_pair.pt", map_location="cpu")
    q = [v.to(device=dev, dtype=torch.float64) for v in base["s4_pair"]]
    fin_ref = [base["x"].to(device=dev, dtype=torch.float64), base["y"].to(device=dev, dtype=torch.float64)]
    chain = de.Chain(ef, steps=int(run_cfg["steps"]), strength=float(run_cfg["s"]), mix=float(run_cfg["mix_weight"]))
    assert int(base["s4_index"]) == cfg.k, (base["s4_index"], cfg.k)

    eps = de.GuidedEps(ef, run_cfg["edit_prompt"], float(run_cfg["guidance_scale"]), unet_dtype, unet32)

    # Load saved key from run 7793436 for direct scientific control
    key_file = Path(run_cfg.get("key_file", RESULTS / "receiver_aligned/saved_states/r1_codeword_7793436_key.pkl"))
    enc_key, dec_key, message, codeword = load_key(key_file)
    kh = key_hash(enc_key)

    det = TorchDetector(dec_key, device=dev)
    with torch.no_grad():
        img_ref = de.decode_image(ef, fin_ref[0])
    lp = de.LpipsLoss(dev)

    problem = E2Problem(ef, chain, eps, det, codeword, q, img_ref, lp, cfg)
    return {
        "problem": problem, "chain": chain, "q": q, "fin_ref": fin_ref, "eps": eps, "det": det,
        "enc_key": enc_key, "dec_key": dec_key, "message": message, "codeword": codeword,
        "key_hash": kh, "base": base, "img_ref": img_ref, "lpips": lp,
    }
