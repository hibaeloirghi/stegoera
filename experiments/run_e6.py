#!/usr/bin/env python3
"""E6: PRC-structured coupling-layer injection. Phase 1 null_null. V1/V2/V3 at S4."""
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
from experiments.hooks import make_additive_hook, make_window_prc_hook


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure_dirs()
    seed_everything(cfg.get("seed", 1))
    outdir = RESULTS / "e6"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, outdir / "config.json")
    assert cfg.get("condition") == "null_null"

    ef = er.load_models()
    img = load_edict_image(PHOTOS / cfg["image"])
    S = cfg.get("steps", 50)
    s = cfg.get("init_image_strength", 0.8)
    pmix = cfg.get("mix_weight", 0.93)
    G = cfg.get("guidance_scale", 3)
    seed = cfg.get("seed", 1)
    n_fwd = int(S * s)
    prompt = ""
    site_default = cfg.get("site", "S4")
    sites_v4 = cfg.get("sites_v4", [site_default])
    lam = float(cfg.get("lambda_k", 0.5))
    mlen = int(cfg.get("message_length", 64))

    inv0 = er.invert_image(ef, img, prompt=prompt, steps=S, mix_weight=pmix, init_image_strength=s, guidance_scale=G, seed=seed)
    top = [t.clone() for t in inv0["latent_pair"]]
    enc_key, dec_key = pr.keygen(n=4 * 64 * 64, message_length=512, fpr=cfg.get("fpr", 1e-5), t=3, seed=seed)
    message = pr.bits_from_message_length(mlen, rng_seed=seed + 7)

    k_mid = n_fwd // 2
    window = cfg.get("window", [k_mid - 1, k_mid, k_mid + 1])
    window = [int(x) for x in window if 0 <= int(x) < n_fwd]

    variants = []
    # V1 single step; V4 is V1 repeated at each requested site.
    codeword, z = pr.encode_codeword(enc_key, message=message, seed=seed + 99)
    eta_v1 = lam * z.reshape_as(top[0]).to(device=top[0].device, dtype=top[0].dtype)
    for site in sites_v4:
        variants.append(("V1", site, make_additive_hook(site, k_mid, eta_v1, informed=False, n_forward=n_fwd), z, eta_v1.pow(2).mean().item()))
    # V2 same signs, fresh |g| per step
    etas_v2 = []
    for j, _k in enumerate(window):
        g = torch.randn_like(top[0])
        etas_v2.append(lam * torch.sign(z.reshape_as(top[0])).to(top[0].device) * g.abs())
    variants.append(("V2", site_default, make_window_prc_hook(site_default, window, etas_v2, informed=False, n_forward=n_fwd), z, float(sum(e.pow(2).mean().item() for e in etas_v2))))
    # V3 fresh codeword each step
    etas_v3 = []
    for j, _k in enumerate(window):
        cw_j, z_j = pr.encode_codeword(enc_key, message=message, seed=seed + 200 + j)
        etas_v3.append(lam * z_j.reshape_as(top[0]).to(device=top[0].device, dtype=top[0].dtype))
    variants.append(("V3", site_default, make_window_prc_hook(site_default, window, etas_v3, informed=False, n_forward=n_fwd), z, float(sum(e.pow(2).mean().item() for e in etas_v3))))

    for name, site, hook, z_ref, eta_energy in variants:
        gen = er.generate_from_pair(
            ef, top, prompt, steps=S, mix_weight=pmix, init_image_strength=s,
            guidance_scale=G, seed=seed, step_hook=hook,
        )
        stego = er.decode_latent(ef, gen["x"])
        stego.save(outdir / f"{name}_{site}.png")
        im_t = pil_to_tensor(stego, device=ef.device)
        x0_hat = pr.decoder_inv_vae(ef.vae, im_t, n_steps=cfg.get("decoder_inv_steps", 20))
        rec = er.coupled_loop(
            ef, prompt, reverse=True, init_latent=x0_hat, steps=S, mix_weight=pmix,
            init_image_strength=s, guidance_scale=G, seed=seed, step_hook=None,
        )
        u = rec["x"] / ((1.0 - rec["alpha_t"]) ** 0.5)
        sigma_rec, soft, grid = pr.calibrate_variance(u, dec_key)
        det, dec_ok, decoded_msg = pr.detect_decode(dec_key, soft)
        agree, flip = pr.sign_agreement(u, z_ref.reshape_as(u))
        lpips_val, lpips_err = lpips_distance(img, stego, device=str(ef.device))
        bit_acc = None
        if decoded_msg is not None and message is not None:
            recm = np.array(decoded_msg)[: len(message)]
            if recm.size == len(message):
                bit_acc = float((recm == np.array(message)).mean())
        row = {
            "condition": "null_null",
            "variant": name,
            "site": site,
            "lambda_k": lam,
            "eta_energy": eta_energy,
            "detect": det,
            "decode": dec_ok,
            "sign_agree": agree,
            "sign_flip": flip,
            "bit_acc": bit_acc,
            "sigma": sigma_rec["variance"],
            "pixel_mse": pixel_mse(img, stego),
            "lpips": lpips_val,
            "lpips_error": lpips_err,
            "stream": "x",
        }
        append_csv(outdir / "e6.csv", row)
        print(row)


if __name__ == "__main__":
    main()
