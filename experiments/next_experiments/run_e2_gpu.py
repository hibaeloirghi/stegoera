#!/usr/bin/env python3
"""Experiment E2: Codeword-Only vs Saturated-Detector Objectives (GPU).

Trial A: Codeword-margin objective + quality/perturbation constraints; NO detector reward (b_det = 0).
Trial B: Same objective, but with detector penalty that saturates at m_target = +20 (b_det > 0).

Both trials run on the principal case:
- Photo: imagenet_dog_1.jpg
- Source inversion: NULL condition
- Edit prompt: "A golden retriever"
- Matched operational receiver
- S4 site at step 20 (timestep 380)
- Key and codeword loaded from saved_states/r1_codeword_7793436_key.pkl
- Float32 UNet for optimization, float64 for true channel evaluation through saved/reloaded PNG
- Selection rule: declared final iterate (no cherry-picking)
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PRC-Watermark"))

from experiments.paths import PHOTOS, RESULTS, ensure_dirs
from experiments.io_utils import load_config, save_json, seed_everything
from experiments.image_utils import load_edict_image, pil_to_tensor, tensor_to_pil
from experiments.metrics import clip_score, lpips_distance, pixel_mse
from experiments import edict_runtime as er
from experiments import prc_runtime as pr
from experiments.receiver_aligned import diff_edict as de
from experiments.receiver_aligned.run_d0 import score_vector
from experiments.receiver_aligned.key_io import key_hash
from experiments.receiver_aligned.prc_scores import detection_margin, exact_message_match
from experiments.receiver_aligned.state_io import code_hash, git_head, save_state
from experiments.next_experiments.e2_pipeline import E2Config, build_e2

LPIPS_TARGET = 0.03
_ROWS = []


def append_csv(path, row):
    _ROWS.append(dict(row))
    fields = list(dict.fromkeys(k for r in _ROWS for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(_ROWS)


def sha_text(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main():
    if len(sys.argv) < 2:
        print("Usage: run_e2_gpu.py <config.json>")
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    cfg = load_config(cfg_path)
    ensure_dirs()
    seed_everything(int(cfg["seed"]))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    job = os.environ.get("SLURM_JOB_ID", "local")
    ch = code_hash(Path(__file__).resolve().parent)
    cfg_hash = sha_text(cfg_path)
    exp = cfg["optim"]["experiment_id"]
    out = RESULTS / "next_experiments"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "saved_states").mkdir(parents=True, exist_ok=True)
    (out / "raw_observations").mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{exp}_{job}.csv"

    meta = {
        "job_id": job, "code_hash": ch, "config_hash": cfg_hash,
        "experiment_git": git_head(ROOT), "edict_git": git_head(ROOT / "EDICT"),
        "prc_git": git_head(ROOT / "PRC-Watermark"),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "dtype_pair_vae": "float64", "dtype_unet_optimization": cfg["optim"]["unet_dtype"],
        "dtype_unet_true_eval": "float64", "model": "CompVis/stable-diffusion-v1-4", "tf32": False,
        "seed": cfg["seed"], "selection_rule": cfg["optim"]["selection_rule"],
    }
    save_json({**meta, "config": cfg}, out / f"{exp}_meta_{job}.json")
    print(json.dumps(meta), flush=True)

    ef = er.load_models(device=cfg["device"])
    u32 = de.load_unet32(cfg["device"])
    de.freeze(ef, u32)

    e2_cfg = E2Config(**cfg["e2"])
    opt_cfg = cfg["optim"]
    B = build_e2(ef, u32, opt_cfg["unet_dtype"], e2_cfg, cfg)
    problem, chain, q, fin_ref = B["problem"], B["chain"], B["q"], B["fin_ref"]
    codeword, message, dec_key, det = B["codeword"], B["message"], B["dec_key"], B["det"]
    kh = B["key_hash"]
    k, n = e2_cfg.k, chain.n
    t_site = int(chain.t_fwd[k])
    carrier_def = f"recovered site-x at S4 step {k} / {e2_cfg.site_scale} (surrogate: encoder mean; true: decoder inversion), dim 16384"
    baseline_png = Path(cfg["baseline_png"])
    original = load_edict_image(PHOTOS / cfg["image"])

    def base_row(iteration, stage):
        return {
            "experiment_id": exp, "job_id": job, "code_hash": ch, "config_hash": cfg_hash,
            "photo_id": cfg["image"], "edit_prompt": cfg["edit_prompt"], "inversion_condition": "null",
            "receiver_condition": "matched", "site": "S4", "active_step": k, "scheduler_timestep": t_site,
            "carrier_definition": carrier_def, "dimension": 16384,
            "pair_mode": "eta_x,eta_y free" if opt_cfg["optimize_y"] else "common-mode",
            "beta": e2_cfg.beta_pair, "message_length": cfg["message_length"],
            "key_id": kh, "F": cfg["fpr"], "t": cfg["prc_t"],
            "sigma": float(np.sqrt(e2_cfg.sigma_squared)), "sigma_squared": e2_cfg.sigma_squared,
            "tau": e2_cfg.tau, "a_cw": e2_cfg.a_cw, "b_det": e2_cfg.b_det, "m_target": e2_cfg.m_target,
            "iteration": iteration, "channel_stage": stage,
        }

    ex = torch.zeros((1, 4, 64, 64), dtype=torch.float64, device=ef.device, requires_grad=True)
    ey = torch.zeros((1, 4, 64, 64), dtype=torch.float64, device=ef.device, requires_grad=True)
    params = [ex, ey] if opt_cfg["optimize_y"] else [ex]
    opt = torch.optim.Adam(params, lr=float(opt_cfg["lr"]))
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    curve = []

    print("\n--- Starting Optimization ---", flush=True)
    for it in range(int(opt_cfg["iters"]) + 1):
        t_it = time.time()
        ey_eff = ey if opt_cfg["optimize_y"] else ex
        res = problem.evaluate(ex, ey_eff, compute_diagnostics=True)
        h = res["h"].detach()
        soft = pr.recover_soft_bits(h, variance=e2_cfg.sigma_squared)
        detect_orig = bool(pr.prc_modules()[2](dec_key, soft))
        cw_tensor = torch.as_tensor(codeword.detach().cpu().numpy() if hasattr(codeword, "detach") else codeword).flatten()
        sign_agree = float((torch.sign(h.flatten().cpu()) == torch.sign(cw_tensor)).double().mean())
        site_err = float((res["rec"][0].detach() - (q[0] + ex.detach())).square().mean())

        row = base_row(it, "surrogate_fp32_float_encoder_mean")
        row.update({
            "surrogate_margin": float(res["margin"]), "true_margin": None,
            "detect": detect_orig, "decode_non_none": None, "exact_message_match": None, "overall_success": None,
            "sign_agreement": sign_agree,
            "terminal_pair_MSE": float(res["pair_mse"]),
            "terminal_x_recovery_MSE": None,
            "site_carrier_error_oracle": site_err,
            "LPIPS_vs_edit": float(res["lpips"]), "pixel_MSE_vs_edit": float(res["pixel_mse"]),
            "LPIPS_vs_original": None, "edit_alignment": None,
            "L_total": float(res["total"]), "L_cw": float(res["l_cw"]), "L_det": float(res["l_det"]),
            "L_quality": float(res["l_quality"]), "L_pair": float(res["l_pair"]), "L_energy": float(res["l_energy"]),
            "carrier_energy": float(res["carrier_energy"]),
            "norm_g_cw": res["diag"].get("norm_g_cw", None),
            "norm_g_det": res["diag"].get("norm_g_det", None),
            "cos_cw_det": res["diag"].get("cos_cw_det", None),
            "norm_g_energy": res["diag"].get("norm_g_energy", None),
            "eta_x_rms": float(ex.detach().square().mean().sqrt()), "eta_y_rms": float(ey_eff.detach().square().mean().sqrt()),
            "status": "evaluated_no_step" if it == int(opt_cfg["iters"]) else "ok",
        })

        if it < int(opt_cfg["iters"]):
            opt.zero_grad(set_to_none=True)
            res["total"].backward()
            row["grad_norm_x"] = float(ex.grad.norm())
            row["grad_norm_y"] = float(ey.grad.norm()) if opt_cfg["optimize_y"] else None
            opt.step()

        torch.cuda.synchronize()
        row["time"] = time.time() - t_it
        row["peak_memory"] = int(torch.cuda.max_memory_allocated())
        curve.append(row)
        append_csv(csv_path, row)
        print(json.dumps({
            "it": it, "margin": round(row["surrogate_margin"], 2), "detect": row["detect"],
            "agree": round(row["sign_agreement"], 4), "LPIPS": round(row["LPIPS_vs_edit"], 4),
            "MSE": round(row["pixel_MSE_vs_edit"], 5), "L_cw": round(row["L_cw"], 4),
            "L_det": round(row["L_det"], 4), "norm_g_cw": round(row["norm_g_cw"], 4) if row["norm_g_cw"] else 0,
            "norm_g_det": round(row["norm_g_det"], 4) if row["norm_g_det"] else 0,
            "cos_cw_det": round(row["cos_cw_det"], 4) if row["cos_cw_det"] else 0,
            "eta_x_rms": round(row["eta_x_rms"], 4), "time": round(row["time"], 1)
        }), flush=True)
        del res

    t_opt = time.time() - t_start
    eta_final = [ex.detach().clone(), (ey if opt_cfg["optimize_y"] else ex).detach().clone()]
    save_state(out / "saved_states" / f"{exp}_{job}_eta.pt", {"eta_x": eta_final[0].cpu(), "eta_y": eta_final[1].cpu(), "cfg": cfg, "meta": meta})

    # ------------------------------------------------------------------ True channel: float64
    print("\n--- Running True Evaluation on Saved/Reloaded PNG (fp64) ---", flush=True)
    del B, problem, opt, ex, ey
    gc.collect()
    u32.cpu()
    del u32
    torch.cuda.empty_cache()

    eps64 = de.GuidedEps(ef, cfg["edit_prompt"], float(cfg["guidance_scale"]), "float64")
    t_true = time.time()
    with torch.no_grad():
        site = [q[0] + eta_final[0], q[1] + eta_final[1]]
        fin = chain.forward_steps(site, k + 1, n, eps64)
        stego = tensor_to_pil(de.decode_image(ef, fin[0]))

    png_path = out / "images" / f"{exp}_{job}_stego.png"
    stego.save(png_path)
    reloaded = Image.open(png_path).convert("RGB")
    base_img = Image.open(baseline_png).convert("RGB")

    def true_channel(pil_img):
        pt = pil_to_tensor(pil_img, device=ef.device, dtype=torch.float64)
        z_ = pr.decoder_inv_vae(ef.vae, pt, n_steps=int(cfg["decoder_inv_steps"]))
        with torch.no_grad():
            rec_ = chain.reverse_to_site([z_, z_.clone()], k, eps64)
        return z_, rec_, rec_[0] / e2_cfg.site_scale

    z_inv, rec, h_true = true_channel(reloaded)
    with torch.no_grad():
        scored = score_vector(h_true, dec_key, codeword, message)
        term_x = float((z_inv - fin[0]).square().mean())
        site_err_true = float((rec[0] - site[0]).square().mean())
        site_err_base = float((rec[0] - q[0]).square().mean())

    primary, floor = scored["primary"], scored["grid_floor"]
    lp_edit, lp_edit_err = lpips_distance(base_img, reloaded, device=cfg["device"])
    lp_orig, lp_orig_err = lpips_distance(original, reloaded, device=cfg["device"])
    clip_val, clip_err = clip_score(reloaded, cfg["edit_prompt"], device=cfg["device"])
    mse_edit = pixel_mse(base_img, reloaded)
    success = bool(primary["detect"] and primary["exact_message_match"] and lp_edit is not None and lp_edit <= LPIPS_TARGET)

    final = base_row(int(opt_cfg["iters"]), "true_png_decoder_inversion_fp64")
    final.update({
        "surrogate_margin": curve[-1]["surrogate_margin"], "true_margin": primary["margin"],
        "statistic": primary["statistic"], "threshold": primary["threshold"],
        "detect": primary["detect"], "decode_non_none": primary["decode_non_none"],
        "exact_message_match": primary["exact_message_match"], "overall_success": success,
        "sign_agreement": primary["sign_agreement"],
        "true_margin_sigma_squared_0.5": floor["margin"], "detect_sigma_squared_0.5": floor["detect"],
        "exact_message_match_sigma_squared_0.5": floor["exact_message_match"],
        "terminal_pair_MSE": float((fin[0] - fin[1]).square().mean()),
        "terminal_x_recovery_MSE": term_x, "site_carrier_error_oracle": site_err_true,
        "site_carrier_error_vs_baseline_site": site_err_base,
        "LPIPS_vs_edit": lp_edit, "pixel_MSE_vs_edit": mse_edit, "LPIPS_vs_original": lp_orig,
        "edit_alignment": clip_val, "time": time.time() - t_true, "peak_memory": int(torch.cuda.max_memory_allocated()),
        "status": "ok", "optimization_wall_s": t_opt,
    })
    append_csv(csv_path, final)
    print("TRUE_CHANNEL", json.dumps({k_: final[k_] for k_ in (
        "true_margin", "detect", "decode_non_none", "exact_message_match", "overall_success", "sign_agreement",
        "LPIPS_vs_edit", "pixel_MSE_vs_edit", "LPIPS_vs_original", "edit_alignment", "terminal_pair_MSE",
        "terminal_x_recovery_MSE", "site_carrier_error_oracle")}), flush=True)

    save_state(out / "raw_observations" / f"{exp}_{job}_true_site_carrier.pt", {"vector": h_true.detach().cpu(), "stage": "true_png_decoder_inversion_fp64", "key_hash": kh})
    save_state(out / "saved_states" / f"{exp}_{job}_final_pair.pt", {"x": fin[0].cpu(), "y": fin[1].cpu(), "site": [v.cpu() for v in site]})

    # ---- Control 1: wrong, independent key on the stego carrier (same carrier, same PNG)
    print("Evaluating wrong-key control...", flush=True)
    enc2, dec2 = pr.keygen(n=4 * 64 * 64, message_length=int(cfg["key_message_length"]), fpr=float(cfg["fpr"]), t=int(cfg["prc_t"]), seed=int(cfg["seed"]) + 1000)
    msg2 = pr.bits_from_message_length(int(cfg["message_length"]), rng_seed=int(cfg["seed"]) + 1007)
    cw2, _ = pr.encode_codeword(enc2, message=msg2, seed=int(cfg["seed"]) + 1099)
    ctrl_key = score_vector(h_true, dec2, cw2, msg2)["primary"]
    ctrl_row = base_row(int(opt_cfg["iters"]), "control_wrong_key_on_stego_png")
    ctrl_row.update({
        "key_id": key_hash(enc2), "true_margin": ctrl_key["margin"], "statistic": ctrl_key["statistic"],
        "threshold": ctrl_key["threshold"], "detect": ctrl_key["detect"],
        "decode_non_none": ctrl_key["decode_non_none"], "exact_message_match": ctrl_key["exact_message_match"],
        "overall_success": False, "sign_agreement": ctrl_key["sign_agreement"], "status": "control_wrong_key",
    })
    append_csv(csv_path, ctrl_row)

    # ---- Control 2: baseline unwatermarked edit PNG with true key
    print("Evaluating baseline-PNG control...", flush=True)
    _, _, h_base = true_channel(base_img)
    base_eval = score_vector(h_base, dec_key, codeword, message)["primary"]
    base_row_entry = base_row(int(opt_cfg["iters"]), "control_baseline_edit_png")
    base_row_entry.update({
        "true_margin": base_eval["margin"], "statistic": base_eval["statistic"], "threshold": base_eval["threshold"],
        "detect": base_eval["detect"], "decode_non_none": base_eval["decode_non_none"],
        "exact_message_match": base_eval["exact_message_match"], "overall_success": False,
        "sign_agreement": base_eval["sign_agreement"], "status": "control_baseline_png",
    })
    append_csv(csv_path, base_row_entry)

    print("\nE2 Trial Completed Successfully! CSV saved to:", csv_path, flush=True)


if __name__ == "__main__":
    main()
