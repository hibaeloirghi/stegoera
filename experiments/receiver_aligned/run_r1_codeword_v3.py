#!/usr/bin/env python3
"""R1 codeword smoke v3 (copy of v2 + [v3] codeword-sign loss term, per-job union-schema CSV; v2 (versioned copy of run_r1_smoke.py; v1 stays untouched because it was queued).
Changes vs v1: the PRC key is saved and hashed (KeyGen is not reproducible from its seed),
true-channel controls are added in the SAME process/key: the unwatermarked baseline PNG and a
wrong independent key on the stego carrier.

R1 smoke: receiver-aligned optimization at ONE S4 site of the dog -> "A golden retriever" edit.

Gradient pass: fp32 stock-diffusers UNet inside a float64 EDICT/VAE pipeline (validated by
run_gradcheck.py). Surrogate receiver: encoder posterior mean -> duplicate -> reverse to the site.

Final evaluation is the TRUE channel in float64, none of it differentiable:
    regenerate the stego image from q+eta with the fp64 UNet,
    save PNG, reload it,
    decoder inversion (50 steps, repo implementation),
    duplicate, reverse to the same site with the fp64 UNet,
    original Detect / Decode / exact message comparison.

Selection rule (declared before running): report the FINAL iterate only. No best-iteration choice.
The surrogate per-iteration curve and the true final row are written to separate channel_stage values.
"""
from __future__ import annotations

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

from experiments.paths import PHOTOS, RESULTS, ensure_dirs
from experiments.io_utils import load_config, save_json, seed_everything
from experiments.image_utils import load_edict_image, pil_to_tensor, tensor_to_pil
from experiments.metrics import clip_score, lpips_distance, pixel_mse
from experiments import edict_runtime as er
from experiments import prc_runtime as pr
from experiments.receiver_aligned import diff_edict as de
from experiments.receiver_aligned.r1_pipeline import R1Config, build
from experiments.receiver_aligned.run_d0 import score_vector
from experiments.receiver_aligned.key_io import key_hash, save_key
from experiments.receiver_aligned.prc_scores import detection_margin, exact_message_match
from experiments.receiver_aligned.state_io import code_hash, git_head, save_state

CFG_DEFAULT = ROOT / "experiments/configs/receiver_aligned/r1_smoke.json"
LPIPS_TARGET = 0.03


_ROWS = []


def append_csv(path, row):
    """[v3] Per-job file; header = union of all keys (v2 reused the first row's header and misaligned later rows)."""
    import csv as _csv
    _ROWS.append(dict(row))
    fields = list(dict.fromkeys(k for r in _ROWS for k in r))
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(_ROWS)


def sha_text(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CFG_DEFAULT
    cfg = load_config(cfg_path)
    ensure_dirs()
    seed_everything(int(cfg["seed"]))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    job = os.environ.get("SLURM_JOB_ID", "local")
    ch = code_hash(Path(__file__).resolve().parent)
    cfg_hash = sha_text(cfg_path)
    exp = cfg["optim"]["experiment_id"]
    out = RESULTS / "receiver_aligned"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "saved_states").mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{exp}_{job}.csv"
    meta = {"job_id": job, "code_hash": ch, "config_hash": cfg_hash, "experiment_git": git_head(ROOT),
            "edict_git": git_head(ROOT / "EDICT"), "prc_git": git_head(ROOT / "PRC-Watermark"),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "dtype_pair_vae": "float64", "dtype_unet_optimization": cfg["optim"]["unet_dtype"],
            "dtype_unet_true_eval": "float64", "model": "CompVis/stable-diffusion-v1-4", "tf32": False,
            "seed": cfg["seed"], "selection_rule": cfg["optim"]["selection_rule"]}
    save_json({**meta, "config": cfg}, out / f"{exp}_meta_{job}.json")
    print(json.dumps(meta), flush=True)

    ef = er.load_models(device=cfg["device"])
    u32 = de.load_unet32(cfg["device"])
    de.freeze(ef, u32)
    r1 = R1Config(**cfg["r1"])
    opt_cfg = cfg["optim"]
    B = build(ef, u32, opt_cfg["unet_dtype"], r1, cfg)
    problem, chain, q, fin_ref = B["problem"], B["chain"], B["q"], B["fin_ref"]
    codeword, message, dec_key, det = B["codeword"], B["message"], B["dec_key"], B["det"]
    k, n = r1.k, chain.n
    kh = save_key(out / "saved_states" / f"{exp}_{job}_key.pkl", B["enc_key"], dec_key, message, codeword)
    print("key_hash", kh, flush=True)
    t_site = int(chain.t_fwd[k])
    carrier_def = f"recovered site-x at S4 step {k} / {r1.site_scale} (surrogate: encoder mean; true: decoder inversion), dim 16384"
    baseline_png = Path(cfg["baseline_png"])
    original = load_edict_image(PHOTOS / cfg["image"])

    def base_row(iteration, stage):
        return {
            "experiment_id": exp, "job_id": job, "code_hash": ch, "config_hash": cfg_hash,
            "photo_id": cfg["image"], "edit_prompt": cfg["edit_prompt"], "inversion_condition": "null",
            "receiver_condition": "matched", "site": "S4", "active_step": k, "scheduler_timestep": t_site,
            "carrier_definition": carrier_def, "dimension": 16384,
            "pair_mode": "eta_x,eta_y free" if opt_cfg["optimize_y"] else "common-mode",
            "beta": r1.beta_pair, "message_length": cfg["message_length"],
            "key_id": kh, "F": cfg["fpr"], "t": cfg["prc_t"],
            "sigma": float(np.sqrt(r1.sigma_squared)), "sigma_squared": r1.sigma_squared, "tau": r1.tau,
            "iteration": iteration, "channel_stage": stage,
        }

    ex = torch.zeros((1, 4, 64, 64), dtype=torch.float64, device=ef.device, requires_grad=True)
    ey = torch.zeros((1, 4, 64, 64), dtype=torch.float64, device=ef.device, requires_grad=True)
    params = [ex, ey] if opt_cfg["optimize_y"] else [ex]
    opt = torch.optim.Adam(params, lr=float(opt_cfg["lr"]))
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    curve = []
    for it in range(int(opt_cfg["iters"]) + 1):
        t_it = time.time()
        ey_eff = ey if opt_cfg["optimize_y"] else ex
        res = problem.evaluate(ex, ey_eff)
        h = res["h"].detach()
        soft = pr.recover_soft_bits(h, variance=r1.sigma_squared)
        detect_orig = bool(pr.prc_modules()[2](dec_key, soft))
        sign_agree = float((torch.sign(h.flatten().cpu()) == torch.sign(codeword.flatten())).double().mean())
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
            "L_total": float(res["total"]), "L_code": float(res["l_code"]), "L_sign": float(res["l_sign"]), "L_quality": float(res["l_quality"]),
            "L_pair": float(res["l_pair"]), "L_energy": float(res["l_energy"]), "carrier_energy": float(res["carrier_energy"]), "hinge": float(res["hinge"]),
            "eta_x_rms": float(ex.detach().square().mean().sqrt()), "eta_y_rms": float(ey_eff.detach().square().mean().sqrt()),
            "status": "evaluated_no_step" if it == int(opt_cfg["iters"]) else "ok",
            "LPIPS_note": "tensor-path LPIPS (256 bilinear antialias), float image, not the PNG metric",
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
        print(json.dumps({k_: row[k_] for k_ in ("iteration", "surrogate_margin", "detect", "sign_agreement", "LPIPS_vs_edit", "pixel_MSE_vs_edit",
                                                   "terminal_pair_MSE", "site_carrier_error_oracle", "eta_x_rms", "eta_y_rms", "L_code", "L_quality", "time")}), flush=True)
        del res
    t_opt = time.time() - t_start
    eta_final = [ex.detach().clone(), (ey if opt_cfg["optimize_y"] else ex).detach().clone()]
    save_state(out / "saved_states" / f"{exp}_{job}_eta.pt", {"eta_x": eta_final[0].cpu(), "eta_y": eta_final[1].cpu(), "cfg": cfg, "meta": meta})

    # ------------------------------------------------------------------ true channel, float64
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
        """Saved/reloaded PNG -> decoder inversion -> duplicate -> reverse to the site (fp64)."""
        pt = pil_to_tensor(pil_img, device=ef.device, dtype=torch.float64)
        z_ = pr.decoder_inv_vae(ef.vae, pt, n_steps=int(cfg["decoder_inv_steps"]))
        with torch.no_grad():
            rec_ = chain.reverse_to_site([z_, z_.clone()], k, eps64)
        return z_, rec_, rec_[0] / r1.site_scale

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
        "LPIPS_note": "PIL-path LPIPS on saved/reloaded PNG vs baseline.png",
    })
    append_csv(csv_path, final)
    print("TRUE_CHANNEL", json.dumps({k_: final[k_] for k_ in (
        "true_margin", "detect", "decode_non_none", "exact_message_match", "overall_success", "sign_agreement",
        "LPIPS_vs_edit", "pixel_MSE_vs_edit", "LPIPS_vs_original", "edit_alignment", "terminal_pair_MSE",
        "terminal_x_recovery_MSE", "site_carrier_error_oracle")}), flush=True)
    save_state(out / "raw_observations" / f"{exp}_{job}_true_site_carrier.pt", {"vector": h_true.detach().cpu(), "stage": "true_png_decoder_inversion_fp64", "key_hash": kh})
    save_state(out / "saved_states" / f"{exp}_{job}_final_pair.pt", {"x": fin[0].cpu(), "y": fin[1].cpu(), "site": [v.cpu() for v in site]})

    # ---- control 1: wrong, independent key on the stego carrier (same carrier, same PNG)
    enc2, dec2 = pr.keygen(n=4 * 64 * 64, message_length=int(cfg["key_message_length"]), fpr=float(cfg["fpr"]), t=int(cfg["prc_t"]), seed=int(cfg["seed"]) + 1000)
    kh2 = key_hash(enc2)
    soft_w = pr.recover_soft_bits(h_true, variance=r1.sigma_squared)
    stat_w, thr_w, margin_w = detection_margin(dec2, soft_w, clip=False)
    det_w, dec_w, payload_w = pr.detect_decode(dec2, soft_w)
    ctrl = base_row(int(opt_cfg["iters"]), "control_wrong_key_true_png")
    ctrl.update({"key_id": kh2, "true_margin": margin_w, "statistic": stat_w, "threshold": thr_w, "detect": det_w,
                 "decode_non_none": dec_w, "exact_message_match": exact_message_match(payload_w, message),
                 "status": "ok", "LPIPS_vs_edit": lp_edit, "note": "single wrong key; one trial is not an FPR estimate"})
    append_csv(csv_path, ctrl)
    print("CONTROL_WRONG_KEY", json.dumps({"margin": margin_w, "detect": det_w, "decode_non_none": dec_w, "key": kh2}), flush=True)

    # ---- control 2: unwatermarked baseline edit, same key, same true-channel pipeline
    z_b, rec_b, h_b = true_channel(base_img)
    with torch.no_grad():
        scored_b = score_vector(h_b, dec_key, codeword, message)
    pb = scored_b["primary"]
    ctrl_b = base_row(int(opt_cfg["iters"]), "control_unwatermarked_baseline_true_png")
    ctrl_b.update({"true_margin": pb["margin"], "statistic": pb["statistic"], "threshold": pb["threshold"], "detect": pb["detect"],
                   "decode_non_none": pb["decode_non_none"], "exact_message_match": pb["exact_message_match"],
                   "sign_agreement": pb["sign_agreement"], "LPIPS_vs_edit": 0.0, "pixel_MSE_vs_edit": 0.0,
                   "site_carrier_error_oracle": float((rec_b[0] - q[0]).square().mean()),
                   "status": "ok", "note": "baseline.png, historical file (byte-identical to the D0 regeneration)"})
    append_csv(csv_path, ctrl_b)
    print("CONTROL_BASELINE", json.dumps({k_: ctrl_b[k_] for k_ in ("true_margin", "detect", "decode_non_none", "exact_message_match", "sign_agreement")}), flush=True)
    # image triplet: original / baseline edit / stego edit
    trip = Image.new("RGB", (3 * 512, 512))
    for i_, im_ in enumerate([original, Image.open(baseline_png).convert("RGB"), reloaded]):
        trip.paste(im_.resize((512, 512)), (i_ * 512, 0))
    trip.save(out / "images" / f"{exp}_{job}_triplet_original_baseline_stego.png")
    print("R1_SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
