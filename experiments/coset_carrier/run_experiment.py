#!/usr/bin/env python3
"""Coset-selected codeword + subspace carrier + capped dual loss, one photo (GPU; run under SLURM).

    python experiments/coset_carrier/run_experiment.py \
        --config experiments/coset_carrier/configs/embedding.yaml \
        --image_path experiments/photos/imagenet_dog_1.jpg

Sender: VAE-sample x0 -> EDICT inversion with base prompt (production coupled_loop, fp64) ->
generation with edit prompt; free perturbation (eta_x, eta_y) of the S4 pair at forward step k.
Surrogate receiver for gradients: encoder mean -> duplicate -> reverse to site k (fp32 UNet).
True receiver: saved/reloaded PNG -> decoder inversion -> duplicate -> reverse to site k (fp64).
Carrier coordinates: oracle (sender's mask) and blind (mask recomputed from the received image).
"""
from __future__ import annotations

import argparse
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
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.paths import RESULTS, ensure_dirs
from experiments.io_utils import save_json, seed_everything
from experiments.image_utils import load_edict_image, pil_to_tensor, tensor_to_pil
from experiments.metrics import clip_score, lpips_distance
from experiments import edict_runtime as er
from experiments import prc_runtime as pr
from experiments.receiver_aligned import diff_edict as de
from experiments.receiver_aligned.prc_torch import TorchDetector
from experiments.receiver_aligned.prc_scores import exact_message_match
from experiments.receiver_aligned.key_io import key_hash, save_key
from experiments.receiver_aligned.state_io import code_hash, git_head, save_state
from experiments.coset_carrier.prc_coset import select_optimal_coset_codeword
from experiments.coset_carrier.carrier import carrier_indices, carrier_scores, get_subspace_carrier_mask, mask_overlap
from experiments.coset_carrier.optimization import StegoLossConfig, compute_stego_loss, edict_mix_for_steps

_ROWS: list[dict] = []


def append_csv(path, row):
    _ROWS.append(dict(row))
    fields = list(dict.fromkeys(k for r in _ROWS for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(_ROWS)


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB"), dtype=np.float64)
    mse = float(np.mean((x - y) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def score(vec, dec_key, codeword, message):
    """Original Detect/Decode on a carrier vector, at v = 1.5 (primary) and 0.5 (grid floor)."""
    vec = vec.detach().to(torch.float64).flatten().cpu()
    c = codeword.detach().to(torch.float64).flatten().cpu()
    out = {"sign_agreement": float((torch.sign(vec) == torch.sign(c)).double().mean())}
    for tag, var in (("v1.5", 1.5), ("v0.5", 0.5)):
        soft = pr.recover_soft_bits(vec, variance=var)
        _stat, _thr, margin = pr.detection_margin(dec_key, soft)
        detected, decoded_ok, payload = pr.detect_decode(dec_key, soft)
        ber = None
        if payload is not None:
            got = np.asarray(payload).reshape(-1)[: len(message)].astype(int)
            ber = float(np.mean(got != np.asarray(message, dtype=int)))
        out[tag] = {"margin": float(margin), "detect": bool(detected), "decode_non_none": bool(decoded_ok),
                    "exact_message_match": exact_message_match(payload, message), "message_BER": ber}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image_path", required=True)
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    ensure_dirs()
    seed = int(cfg["seed"])
    seed_everything(seed)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    E, P, C, O = cfg["edict"], cfg["prc"], cfg["carrier"], cfg["optim"]
    S, s, G = int(E["steps"]), float(E["strength"]), float(E["guidance_scale"])
    p = edict_mix_for_steps(S, float(E["p_ref"])) if E["mix_weight"] == "auto" else float(E["mix_weight"])
    base_prompt, edit_prompt = E["base_prompt"], E["edit_prompt"]
    loss_cfg = StegoLossConfig(**cfg["loss"])
    lpips_target = float(cfg["lpips_target"])

    job = os.environ.get("SLURM_JOB_ID", "local")
    exp = cfg["experiment_id"]
    out = RESULTS / "coset_carrier"
    for sub in ("images", "saved_states", "raw_observations"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{exp}_{job}.csv"
    meta = {
        "job_id": job, "code_hash": code_hash(Path(__file__).resolve().parent),
        "config_hash": hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:16],
        "experiment_git": git_head(ROOT), "edict_git": git_head(ROOT / "EDICT"), "prc_git": git_head(ROOT / "PRC-Watermark"),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "image_path": str(args.image_path),
        "S": S, "s": s, "p": p, "p_rule": "0.93**(50/S)" if E["mix_weight"] == "auto" else "fixed", "G": G,
        "base_prompt": base_prompt, "edit_prompt": edit_prompt,
        "dtype_pair_vae": "float64", "dtype_unet_optimization": O["unet_dtype"], "dtype_unet_true_eval": "float64",
        "model": "CompVis/stable-diffusion-v1-4", "tf32": False,
        "selection_rule": "final iterate; no best-iterate picking",
    }

    ef = er.load_models(device=cfg["device"])
    u32 = de.load_unet32(cfg["device"])
    de.freeze(ef, u32)
    dev = ef.device
    chain = de.Chain(ef, steps=S, strength=s, mix=p)
    n = chain.n
    k = int(round(float(E["k_fraction"]) * (n - 1)))
    meta.update({"n_active_steps": n, "t_limit": S - int(S * s), "t_fwd": [int(t) for t in chain.t_fwd],
                 "site_k": k, "site_timestep": int(chain.t_fwd[k])})
    print(json.dumps(meta), flush=True)

    eps64 = de.GuidedEps(ef, edit_prompt, G, "float64")
    eps32 = de.GuidedEps(ef, edit_prompt, G, O["unet_dtype"], u32)

    # ---------------------------------------------------------------- sender baseline (fp64)
    original = load_edict_image(args.image_path)
    with torch.no_grad():
        inv = er.invert_image(ef, original, prompt=base_prompt, steps=S, mix_weight=p,
                              init_image_strength=s, guidance_scale=G, seed=seed)
        top = [v.detach() for v in inv["latent_pair"]]
        q = chain.forward_steps(top, 0, k + 1, eps64)
        fin_ref = chain.forward_steps(q, k + 1, n, eps64)
        gen = er.generate_from_pair(ef, top, prompt=edit_prompt, steps=S, mix_weight=p,
                                    init_image_strength=s, guidance_scale=G, seed=seed)
        reg = max(float((gen["x"] - fin_ref[0]).abs().max()), float((gen["y"] - fin_ref[1]).abs().max()))
        img_ref = de.decode_image(ef, fin_ref[0])
        recon = tensor_to_pil(de.decode_image(ef, inv["x0"]))
    q = [v.detach() for v in q]
    baseline = tensor_to_pil(img_ref)
    base_png = out / "images" / f"{exp}_{job}_baseline.png"
    baseline.save(base_png)
    baseline = Image.open(base_png).convert("RGB")
    ident = {
        "chain_vs_production_generation_max_abs": reg,
        "LPIPS_baseline_vs_vae_recon": lpips_distance(baseline, recon, device=cfg["device"])[0],
        "PSNR_baseline_vs_vae_recon": psnr(baseline, recon),
        "LPIPS_baseline_vs_original": lpips_distance(baseline, original, device=cfg["device"])[0],
        "PSNR_baseline_vs_original": psnr(baseline, original),
        "LPIPS_vae_recon_vs_original": lpips_distance(recon, original, device=cfg["device"])[0],
    }
    print("IDENTITY_CONTROL", json.dumps(ident), flush=True)
    meta["identity_control"] = ident

    def mask_views(img_tensor):
        """(h, h') for the carrier mask: site from encoder mean, and after one more VAE round trip."""
        with torch.no_grad():
            z = de.encode_mean(ef, img_tensor.clamp(-1, 1))
            h = chain.reverse_to_site([z, z.clone()], k, eps64)[0]
            z2 = de.encode_mean(ef, de.decode_image(ef, z).clamp(-1, 1))
            h2 = chain.reverse_to_site([z2, z2.clone()], k, eps64)[0]
        return h, h2

    base_t = pil_to_tensor(baseline, device=dev, dtype=torch.float64)
    h_b, h_b_rt = mask_views(base_t)
    mk = dict(sigma_lo=float(C["sigma_lo"]), sigma_hi=float(C["sigma_hi"]), smooth=float(C["smooth"]))
    n_carrier = int(C["n_carrier"])
    mask = get_subspace_carrier_mask(h_b, h_b_rt, n_carrier=n_carrier, **mk)
    idx = carrier_indices(mask).to(dev)
    sc = carrier_scores(h_b, h_b_rt, **mk)
    full_block = C["prc_block"] == "full"
    n_code = 4 * 64 * 64 if full_block else n_carrier

    # ---------------------------------------------------------------- key + coset selection
    enc_key, dec_key = pr.keygen(n=n_code, message_length=int(P["key_message_length"]), fpr=float(P["fpr"]),
                                 t=int(P["t"]), seed=seed)
    message = pr.bits_from_message_length(int(P["message_length"]), rng_seed=seed + 7)
    t0 = time.time()
    sel = select_optimal_coset_codeword(enc_key, message, q[0], q[1], n_candidates=int(P["coset_candidates"]),
                                        carrier_idx=None if full_block else idx.cpu(), seed=seed + 99)
    c_star = sel["codeword"].to(dev)
    c_on_carrier = c_star[idx] if full_block else c_star
    kh = key_hash(enc_key)
    save_key(out / "saved_states" / f"{exp}_{job}_key.pkl", enc_key, dec_key, message, c_star.cpu())
    save_state(out / "saved_states" / f"{exp}_{job}_mask.pt", {"mask": mask.cpu(), "idx": idx.cpu()})
    zplus = 0.5 * (q[0] + q[1])
    init = {
        "coset_selected_candidate": sel["candidate"], "coset_seconds": time.time() - t0,
        "init_agree_sender_site_selected": sel["agreement"],
        "init_weighted_agree_sender_site_selected": sel["weighted_agreement"],
        "init_agree_sender_site_candidates_mean": sel["candidate_agreement_mean"],
        "init_agree_sender_site_candidates_sd": sel["candidate_agreement_std"],
        "init_agree_sender_site_candidates_max": sel["candidate_agreement_max"],
        "init_agree_receiver_view_baseline_carrier": float((torch.sign(h_b.flatten()[idx]) == torch.sign(c_on_carrier)).double().mean()),
        "carrier_sender_site_vs_receiver_view_sign_agree": float((torch.sign(zplus.flatten()[idx]) == torch.sign(h_b.flatten()[idx])).double().mean()),
        "mask_sign_stable_frac_all_coords": sc["sign_stable_frac"], "n_code": n_code, "n_carrier": n_carrier,
        "prc_block": C["prc_block"], "key_id": kh,
    }
    print("COSET", json.dumps(init), flush=True)
    meta["initial"] = init
    save_json({**meta, "config": cfg}, out / f"{exp}_meta_{job}.json")

    # ---------------------------------------------------------------- optimization (surrogate)
    det = TorchDetector(dec_key, device=dev)
    lp = de.LpipsLoss(dev)
    ex = torch.zeros_like(q[0], requires_grad=True)
    ey = torch.zeros_like(q[1], requires_grad=True)
    params = [ex, ey] if O["optimize_y"] else [ex]
    opt = torch.optim.Adam(params, lr=float(O["lr"]))
    iters = int(O["iters"])
    t_opt = time.time()
    last = None
    for it in range(iters + 1):
        t_it = time.time()
        ey_eff = ey if O["optimize_y"] else ex
        fin = chain.forward_steps([q[0] + ex, q[1] + ey_eff], k + 1, n, eps32)
        img = de.decode_image(ef, fin[0])
        z = de.encode_mean(ef, img)
        h = chain.reverse_to_site([z, z], k, eps32)[0]
        res = compute_stego_loss(h, c_star, idx, det, img, img_ref, lp, loss_cfg)
        row = {"experiment_id": exp, "job_id": job, "stage": "surrogate", "iteration": it,
               "margin": float(res["margin"]), "carrier_sign_agreement": res["carrier_sign_agreement"],
               "carrier_frac_at_gamma": res["carrier_frac_at_gamma"], "L_total": float(res["total"]),
               "L_det": float(res["l_det"]), "L_code": float(res["l_code"]), "L_vis": float(res["l_vis"]),
               "pixel_MSE": float(res["pixel_mse"]),
               "eta_x_rms": float(ex.detach().square().mean().sqrt()), "eta_y_rms": float(ey_eff.detach().square().mean().sqrt())}
        if it < iters:
            opt.zero_grad(set_to_none=True)
            res["total"].backward()
            row["grad_norm_x"] = float(ex.grad.norm())
            opt.step()
        torch.cuda.synchronize()
        row["time"] = time.time() - t_it
        row["peak_memory"] = int(torch.cuda.max_memory_allocated())
        append_csv(csv_path, row)
        print(json.dumps({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in row.items()
                          if k_ not in ("experiment_id", "job_id", "stage")}), flush=True)
        last = row
        del res, fin, img, z, h
    t_opt = time.time() - t_opt
    site = [q[0] + ex.detach(), q[1] + (ey if O["optimize_y"] else ex).detach()]
    save_state(out / "saved_states" / f"{exp}_{job}_eta.pt", {"eta_x": ex.detach().cpu(), "eta_y": (site[1] - q[1]).cpu()})

    # ---------------------------------------------------------------- true channel (fp64, PNG)
    del opt, ex, ey, lp, eps32
    u32.cpu()
    del u32
    gc.collect()
    torch.cuda.empty_cache()
    t_true = time.time()
    with torch.no_grad():
        fin = chain.forward_steps(site, k + 1, n, eps64)
        stego = tensor_to_pil(de.decode_image(ef, fin[0]))
    png = out / "images" / f"{exp}_{job}_stego.png"
    stego.save(png)
    stego = Image.open(png).convert("RGB")

    def receive(pil_img):
        pt = pil_to_tensor(pil_img, device=dev, dtype=torch.float64)
        z_inv = pr.decoder_inv_vae(ef.vae, pt, n_steps=int(cfg["receiver"]["decoder_inv_steps"]))
        with torch.no_grad():
            h_true = chain.reverse_to_site([z_inv, z_inv.clone()], k, eps64)[0]
        h_rx, h_rx_rt = mask_views(pt)
        mask_rx = get_subspace_carrier_mask(h_rx, h_rx_rt, n_carrier=n_carrier, **mk)
        return h_true, mask_rx

    def carrier_vec(h_true, use_idx):
        return h_true.flatten() if full_block else h_true.flatten()[use_idx]

    h_true, mask_rx = receive(stego)
    idx_rx = carrier_indices(mask_rx).to(dev)
    ov = mask_overlap(mask, mask_rx)
    oracle = score(carrier_vec(h_true, idx), dec_key, c_star, message)
    blind = score(carrier_vec(h_true, idx_rx), dec_key, c_star, message)
    carrier_agree_true = float((torch.sign(h_true.flatten()[idx]) == torch.sign(c_on_carrier)).double().mean())
    lp_edit = lpips_distance(baseline, stego, device=cfg["device"])[0]
    lp_orig = lpips_distance(original, stego, device=cfg["device"])[0]
    clip_val = clip_score(stego, edit_prompt, device=cfg["device"])[0] if edit_prompt else None
    clip_base = clip_score(baseline, edit_prompt, device=cfg["device"])[0] if edit_prompt else None
    save_state(out / "raw_observations" / f"{exp}_{job}_true_site.pt", {"h_true": h_true.cpu(), "mask_rx": mask_rx.cpu(), "key_id": kh})

    # controls: baseline PNG with the true key; stego with an independent key
    h_base_true, mask_base_rx = receive(baseline)
    ctrl_base = score(carrier_vec(h_base_true, carrier_indices(mask_base_rx).to(dev)), dec_key, c_star, message)
    enc2, dec2 = pr.keygen(n=n_code, message_length=int(P["key_message_length"]), fpr=float(P["fpr"]), t=int(P["t"]), seed=seed + 1000)
    cw2, _ = pr.encode_codeword(enc2, message=message, seed=seed + 1099)
    ctrl_key = score(carrier_vec(h_true, idx), dec2, cw2, message)

    def ok(sc_):
        return bool(sc_["v1.5"]["detect"] and sc_["v1.5"]["exact_message_match"] and lp_edit is not None and lp_edit <= lpips_target)

    final = {
        "initial_sign_agreement_sender_site": init["init_agree_sender_site_selected"],
        "initial_sign_agreement_receiver_view_baseline": init["init_agree_receiver_view_baseline_carrier"],
        "final_sign_agreement_surrogate": last["carrier_sign_agreement"],
        "final_sign_agreement_true_oracle_mask": oracle["sign_agreement"],
        "final_sign_agreement_true_blind_mask": blind["sign_agreement"],
        "final_carrier_sign_agreement_true_on_sender_mask": carrier_agree_true,
        "final_margin_surrogate": last["margin"],
        "final_M_true_oracle_v1.5": oracle["v1.5"]["margin"], "final_M_true_blind_v1.5": blind["v1.5"]["margin"],
        "oracle": oracle, "blind": blind,
        "mask_overlap_sender_vs_receiver": ov,
        "LPIPS_vs_edit": lp_edit, "PSNR_vs_edit": psnr(baseline, stego),
        "LPIPS_vs_original": lp_orig, "PSNR_vs_original": psnr(original, stego),
        "CLIP_stego": clip_val, "CLIP_baseline": clip_base,
        "terminal_pair_MSE": float((fin[0] - fin[1]).square().mean()),
        "success_blind_receiver": ok(blind), "success_oracle_mask": ok(oracle),
        "control_baseline_png_true_key": ctrl_base, "control_wrong_key_on_stego": ctrl_key,
        "optimization_wall_s": t_opt, "true_eval_wall_s": time.time() - t_true,
    }
    append_csv(csv_path, {"experiment_id": exp, "job_id": job, "stage": "true_png_decoder_inversion_fp64",
                          "iteration": iters, **{k_: v for k_, v in final.items() if not isinstance(v, dict)},
                          "oracle_detect": oracle["v1.5"]["detect"], "oracle_exact": oracle["v1.5"]["exact_message_match"],
                          "oracle_BER": oracle["v1.5"]["message_BER"], "blind_detect": blind["v1.5"]["detect"],
                          "blind_exact": blind["v1.5"]["exact_message_match"], "blind_BER": blind["v1.5"]["message_BER"],
                          "mask_iou": ov["iou"]})
    save_json({**meta, "config": cfg, "final": final}, out / f"{exp}_summary_{job}.json")
    print("FINAL_METRICS", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
