#!/usr/bin/env python3
"""CPU audit: soft-bit convention, shaping anomaly, idealized-carrier requirement curve.

No model load. Reads cached sender inversion and saved shaping artifacts.
Writes results/receiver_aligned/audit_cpu.json and audit_carrier_requirement.csv.

Three parts
-----------
1. Soft-bit convention. Shows that prc_gaussians.recover_posteriors(z, variances=v)
   is erf(z / sqrt(2 v (1+v))) and that v is sigma^2 (Gunn et al. App. C.1, Fact 1),
   not sigma.
2. Shaping anomaly (lambda=0.05, 0.15). Nominal lambda is not the perturbation size.
   Reports host/correction stats and the geometry of the correction relative to eps.
3. Idealized-carrier requirement curve. A carrier h = a z + sqrt(1-a^2) w with the real
   key. This is NOT an EDICT channel. It says how much codeword evidence the
   unchanged PRC detector/decoder needs on a unit-variance Gaussian-like carrier.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.paths import RESULTS, ensure_dirs
from experiments import edict_runtime as er
from experiments import prc_runtime as pr
from experiments.io_utils import append_csv, save_json
from experiments.receiver_aligned.prc_scores import detection_margin, exact_message_match
from experiments.receiver_aligned.state_io import code_hash, git_head
from experiments.run_shape_stego import shape_toward


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def part1_soft_bit_convention():
    z = torch.linspace(-4, 4, 17, dtype=torch.float64)
    out = {}
    for v in (0.5, 1.0, 1.5):
        got = pr.recover_soft_bits(z, variance=v).numpy()
        want = erf(z.numpy() / np.sqrt(2.0 * v * (1.0 + v)))
        # Gunn Fact 1 with sigma = sqrt(v): erf(z / sqrt(2 sigma^2 (1+sigma^2))).
        sigma = np.sqrt(v)
        paper = erf(z.numpy() / np.sqrt(2.0 * sigma ** 2 * (1.0 + sigma ** 2)))
        out[str(v)] = {
            "code_param_v": v,
            "interpreted_as": "sigma_squared (variance)",
            "sigma_std": float(sigma),
            "max_abs_diff_vs_erf_2v(1+v)": float(np.abs(got - want).max()),
            "max_abs_diff_vs_paper_with_sigma_sqrt_v": float(np.abs(got - paper).max()),
        }
    return {
        "soft_bit_expression": "s_i = erf(z_i / sqrt(2 v (1+v))), v = sigma^2; then Detect multiplies by (1-2*noise_rate)*(1-2*otp_i)",
        "calibrate_variance_grid": [0.5, 1.0, 1.5, 2.0, 3.0],
        "calibrate_variance_selector": "max over grid of (int(detect)+int(decode), mean |soft bit|); smaller v gives larger |soft|, so v=0.5 wins whenever nothing differs in detect/decode",
        "per_v": out,
    }


def part2_shaping(cfg_dir: Path):
    cache = torch.load(RESULTS / "edit_stego" / "sender_inversion_s0.8_S50_p0.93_seed1.pt", map_location="cpu")
    x_t, y_t = cache["latent_pair"]
    x0 = cache["x0"]
    alpha = cache["alpha_t"]
    eps = er.ddim_eps(x_t, x0, alpha)
    sel = torch.load(RESULTS / "shape_stego" / "selected_codeword_n1024_seed1.pt", map_location="cpu")
    code = sel["codeword"].reshape_as(eps).to(torch.float64)
    sd = RESULTS / "shape_stego"
    rows = {}
    rebuilt = er.compose_xt(x0, eps, alpha)
    base = {
        "alpha_t": float(alpha),
        "t_final": int(cache["t_final"]),
        "sqrt_alpha": float(alpha ** 0.5),
        "sqrt_1m_alpha": float((1 - alpha) ** 0.5),
        "latent_scaling": "x0 = 0.18215*E(I); eps=(x_t - sqrt(a) x0)/sqrt(1-a)",
        "host_eps_rms": float(eps.square().mean().sqrt()),
        "host_eps_max_abs": float(eps.abs().max()),
        "host_eps_var": float(eps.var()),
        "host_x_t_rms": float(x_t.square().mean().sqrt()),
        "x0_rms": float(x0.square().mean().sqrt()),
        "rebuilt_pair_vs_cached_max_abs_lambda0": float((rebuilt - x_t).abs().max()),
        "selected_codeword_hash": tensor_hash(code),
        "selected_candidate": int(sel["candidate"]),
        "selected_agreement_with_sign_eps": float(sel["agreement"]),
        "cached_inversion_hash": sha(RESULTS / "edit_stego" / "sender_inversion_s0.8_S50_p0.93_seed1.pt"),
        "lpips_reference": str(RESULTS / "edit_stego" / "baseline.png"),
        "lpips_reference_hash": sha(RESULTS / "edit_stego" / "baseline.png"),
        "config_hash": sha(sd / "config.json"),
        "seed_codeword_sign_agreement_with_eps": float((torch.sign(eps) == code).double().mean()),
    }
    rng = np.random.default_rng(0)
    for lam in (0.05, 0.15):
        tag = f"shape_top_prc_l{lam:.2f}"
        eps_s, delta = shape_toward(eps, code, lam, 1.0)
        x_marked = er.compose_xt(x0, eps_s, alpha)
        d_pair = x_marked - x_t
        cos = float((delta * eps).sum() / (delta.norm() * eps.norm()))
        # Same-RMS references that the additive screen used or that bracket it.
        rms = float(delta.square().mean().sqrt())
        g = torch.from_numpy(np.abs(rng.standard_normal(eps.shape))).to(torch.float64)
        z_ref = code * g
        # blend with gamma matched to the same RMS of (eps_used - eps)
        def blend(gamma):
            return ((1 - gamma ** 2) ** 0.5) * eps + gamma * z_ref
        gamma_match = None
        for gm in np.linspace(0.01, 0.9, 400):
            if float((blend(float(gm)) - eps).square().mean().sqrt()) >= rms:
                gamma_match = float(gm)
                break
        d_blend = blend(gamma_match) - eps if gamma_match is not None else None
        rows[tag] = {
            "lambda": lam,
            "tau": 1.0,
            "nominal_lambda_is_not_amplitude": True,
            "correction_rms": rms,
            "correction_max_abs": float(delta.abs().max()),
            "correction_mean": float(delta.mean()),
            "modified_fraction": float((delta != 0).double().mean()),
            "pair_shift_rms_x_marked_minus_x_t": float(d_pair.square().mean().sqrt()),
            "pair_shift_expected_sqrt_1m_alpha_times_rms": float((1 - alpha) ** 0.5 * rms),
            "injection_energy_mean_sq": float(d_pair.square().mean()),
            "cos_correction_vs_eps": cos,
            "eps_shaped_var_over_eps_var": float(eps_s.var() / eps.var()),
            "eps_shaped_norm_ratio": float(eps_s.norm() / eps.norm()),
            "sign_flip_fraction": float((torch.sign(eps_s) != torch.sign(eps)).double().mean()),
            "sign_agreement_with_codeword_after": float((torch.sign(eps_s) == code).double().mean()),
            "sign_agreement_with_codeword_before": float((torch.sign(eps) == code).double().mean()),
            "blend_gamma_with_same_rms": gamma_match,
            "blend_cos_correction_vs_eps": None if d_blend is None else float((d_blend * eps).sum() / (d_blend.norm() * eps.norm())),
            "blend_var_ratio": None if gamma_match is None else float(blend(gamma_match).var() / eps.var()),
            "image_hash": sha(sd / f"{tag}.png") if (sd / f"{tag}.png").is_file() else None,
            "task_hash": sha(sd / f"{tag}_task.json"),
        }
        mj = sd / f"{tag}_matched.json"
        if mj.is_file():
            m = json.loads(mj.read_text())
            rows[tag].update({
                "recorded_lpips_vs_edit": m.get("lpips_vs_unwatermarked_edit"),
                "recorded_pixel_mse_vs_edit": m.get("pixel_mse_vs_unwatermarked_edit"),
                "recorded_sender_margin": m.get("sender_margin"),
                "recorded_receiver_margin": m.get("margin"),
                "recorded_injection_energy": m.get("injection_energy"),
            })
    # Reference LPIPS values of the additive blend screen, for scale only.
    rows["_additive_blend_reference_from_screen"] = {
        "top_gamma_0.10": {"lpips_vs_edit": 0.026},
        "top_gamma_0.20": {"lpips_vs_edit": 0.087},
        "top_gamma_0.35": {"lpips_vs_edit": 0.176},
        "source": "REPORT of edit_stego.csv (existing); not recomputed here",
    }
    return base, rows


def part3_requirement_curve(out_csv: Path, decode_grid, a_grid, noise_sd, seeds):
    t0 = time.time()
    enc_key, dec_key = pr.keygen(n=4 * 64 * 64, message_length=512, fpr=1e-5, t=3, seed=1)
    message = pr.bits_from_message_length(64, rng_seed=1 + 7)
    codeword, z = pr.encode_codeword(enc_key, message=message, seed=1 + 99)
    keygen_s = time.time() - t0
    n = z.numel()
    results = []
    for a in a_grid:
        for seed in seeds:
            rng = np.random.default_rng(1000 + seed)
            w = torch.from_numpy(rng.standard_normal(n)).to(torch.float64)
            h = a * z + ((1 - a ** 2) ** 0.5) * w
            if noise_sd > 0:
                h = h + noise_sd * torch.from_numpy(rng.standard_normal(n)).to(torch.float64)
            for v in (0.5, 1.5):
                soft = pr.recover_soft_bits(h, variance=v)
                stat, thr, margin = detection_margin(dec_key, soft, clip=False)
                det = bool(pr.prc_modules()[2](dec_key, soft))
                dec_non_none = None
                exact = None
                if a in decode_grid and seed == seeds[0]:
                    payload = pr.prc_modules()[3](dec_key, soft)
                    dec_non_none = payload is not None
                    exact = exact_message_match(payload, message)
                agree = float((torch.sign(h) == torch.sign(codeword)).double().mean())
                row = {
                    "a": a, "seed": seed, "obs_noise_sd": noise_sd, "sigma_squared": v,
                    "sigma": float(np.sqrt(v)), "statistic": stat, "threshold": thr,
                    "margin": margin, "detect": det, "margin_matches_detect": det == (margin >= 0),
                    "decode_non_none": dec_non_none, "exact_message_match": exact,
                    "sign_agreement": agree, "carrier_rms": float(h.square().mean().sqrt()),
                    "corr_with_codeword": float((h * codeword).mean() / h.std()),
                }
                results.append(row)
                append_csv(out_csv, row)
                print(json.dumps(row), flush=True)
    return {"keygen_seconds": keygen_s, "n": n, "rows": len(results)}


def main():
    ensure_dirs()
    out = RESULTS / "receiver_aligned"
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "code_hash": code_hash(Path(__file__).resolve().parent),
        "experiment_git": git_head(ROOT),
        "prc_git": git_head(ROOT / "PRC-Watermark"),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    print(json.dumps(meta), flush=True)
    p1 = part1_soft_bit_convention()
    print(json.dumps(p1, indent=1), flush=True)
    base, rows = part2_shaping(out)
    print(json.dumps(base, indent=1), flush=True)
    print(json.dumps(rows, indent=1), flush=True)
    save_json({"meta": meta, "soft_bit_convention": p1, "shaping_host": base, "shaping": rows}, out / "audit_cpu.json")
    csv_path = out / "audit_carrier_requirement.csv"
    if csv_path.exists():
        csv_path = out / f"audit_carrier_requirement_{int(time.time())}.csv"
    a_grid = [0.0, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.0]
    info = part3_requirement_curve(csv_path, decode_grid={0.30, 0.40, 0.50, 0.60, 0.80, 1.0}, a_grid=a_grid, noise_sd=0.13, seeds=[0, 1])
    print("PART3", json.dumps(info), flush=True)
    print("AUDIT_DONE")


if __name__ == "__main__":
    main()
