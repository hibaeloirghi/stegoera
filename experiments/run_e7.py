#!/usr/bin/env python3
"""E7: per-coordinate mixing p_i chosen by PRC bits. Phase 1 null_null. Exploratory."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from experiments.paths import RESULTS, PHOTOS, ensure_dirs
from experiments.io_utils import load_config, save_json, append_csv, seed_everything
from experiments.image_utils import load_edict_image, pil_to_tensor
from experiments.metrics import pixel_mse, lpips_distance
from experiments import edict_runtime as er
from experiments import prc_runtime as pr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure_dirs()
    seed_everything(cfg.get("seed", 1))
    outdir = RESULTS / "e7"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, outdir / "config.json")
    assert cfg.get("condition") == "null_null"

    ef = er.load_models()
    img = load_edict_image(PHOTOS / cfg["image"])
    S = cfg.get("steps", 50)
    s = cfg.get("init_image_strength", 0.8)
    G = cfg.get("guidance_scale", 3)
    seed = cfg.get("seed", 1)
    p_lo = float(cfg.get("p_lo", 0.90))
    p_hi = float(cfg.get("p_hi", 0.96))
    p_uniform = float(cfg.get("mix_weight", 0.93))
    prompt = ""
    mlen = int(cfg.get("message_length", 64))

    enc_key, dec_key = pr.keygen(n=4 * 64 * 64, message_length=512, fpr=cfg.get("fpr", 1e-5), t=3, seed=seed)
    message = pr.bits_from_message_length(mlen, rng_seed=seed + 7)
    codeword, _z = pr.encode_codeword(enc_key, message=message, seed=seed + 99)
    c = codeword.reshape(1, 4, 64, 64)
    # c in {+1,-1}; +1 -> p_hi, -1 -> p_lo
    p_map = torch.where(c > 0, torch.tensor(p_hi), torch.tensor(p_lo)).to(dtype=torch.float64)

    inv0 = er.invert_image(ef, img, prompt=prompt, steps=S, mix_weight=p_uniform, init_image_strength=s, guidance_scale=G, seed=seed)
    top = [t.clone() for t in inv0["latent_pair"]]

    gen_u = er.generate_from_pair(ef, top, prompt, steps=S, mix_weight=p_uniform, init_image_strength=s, guidance_scale=G, seed=seed)
    img_u = er.decode_latent(ef, gen_u["x"])
    p_map = p_map.to(device=top[0].device)
    gen_m = er.generate_from_pair(ef, top, prompt, steps=S, mix_weight=p_map, init_image_strength=s, guidance_scale=G, seed=seed)
    img_m = er.decode_latent(ef, gen_m["x"])
    img_u.save(outdir / "uniform_p.png")
    img_m.save(outdir / "modulated_p.png")

    im_t = pil_to_tensor(img_m, device=ef.device)
    x0_hat = pr.decoder_inv_vae(ef.vae, im_t, n_steps=cfg.get("decoder_inv_steps", 20))
    rec_u = er.coupled_loop(ef, prompt, reverse=True, init_latent=x0_hat, steps=S, mix_weight=p_uniform, init_image_strength=s, guidance_scale=G, seed=seed)
    rec_hi = er.coupled_loop(ef, prompt, reverse=True, init_latent=x0_hat, steps=S, mix_weight=p_hi, init_image_strength=s, guidance_scale=G, seed=seed)
    rec_lo = er.coupled_loop(ef, prompt, reverse=True, init_latent=x0_hat, steps=S, mix_weight=p_lo, init_image_strength=s, guidance_scale=G, seed=seed)

    # Proxy soft bit: which of (p_lo, p_hi) inverse stays closer to the paired y in cosine.
    def xy_div(out):
        return float(out["cosine"][-1]) if out["cosine"] else 0.0

    # Per-coordinate: compare rec_hi.x vs rec_lo.x distance to rec_u.x as a crude proxy (block-pooled).
    d_hi = (rec_hi["x"] - rec_u["x"]).abs()
    d_lo = (rec_lo["x"] - rec_u["x"]).abs()
    # Guess +1 if p_hi inverse is closer to uniform-p inverse
    guess = torch.where(d_hi < d_lo, torch.ones_like(d_hi), -torch.ones_like(d_hi))
    c_dev = c.to(device=guess.device, dtype=guess.dtype)
    bit_acc = float((guess == c_dev).float().mean().item())
    lpips_val, lpips_err = lpips_distance(img_u, img_m, device=str(ef.device))

    row = {
        "condition": "null_null",
        "p_lo": p_lo,
        "p_hi": p_hi,
        "p_uniform": p_uniform,
        "xy_div_uniform_inv": xy_div(rec_u),
        "xy_div_hi_inv": xy_div(rec_hi),
        "xy_div_lo_inv": xy_div(rec_lo),
        "xy_div_mod_generate": gen_m["cosine"][-1] if gen_m["cosine"] else None,
        "xy_div_uni_generate": gen_u["cosine"][-1] if gen_u["cosine"] else None,
        "proxy_coord_bit_acc": bit_acc,
        "pixel_mse_mod_vs_uni": pixel_mse(img_u, img_m),
        "pixel_mse_mod_vs_orig": pixel_mse(img, img_m),
        "lpips_mod_vs_uni": lpips_val,
        "lpips_error": lpips_err,
        "stream": "x",
        "note": "H4: expect low per-coordinate acc; this proxy is not PRC Detect",
    }
    append_csv(outdir / "e7.csv", row)
    save_json(row, outdir / "summary.json")
    print(row)


if __name__ == "__main__":
    main()
