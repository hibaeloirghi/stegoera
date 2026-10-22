#!/usr/bin/env python3
"""E5: Gaussian eta inside the coupling step; ignorant vs informed receiver. Phase 1 null_null."""
import argparse
import sys
from pathlib import Path

import torch

WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from experiments.paths import RESULTS, PHOTOS, ensure_dirs
from experiments.io_utils import load_config, save_json, append_csv, seed_everything
from experiments.image_utils import load_edict_image, pil_to_tensor
from experiments.metrics import pixel_mse
from experiments import edict_runtime as er
from experiments import prc_runtime as pr
from experiments.hooks import make_additive_hook


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure_dirs()
    seed_everything(cfg.get("seed", 1))
    outdir = RESULTS / "e5"
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
    ks = cfg.get("k_values", [n_fwd // 2])
    sites = cfg.get("sites", ["S4"])
    scales = cfg.get("scales", [1.0])
    prompt = ""

    inv0 = er.invert_image(ef, img, prompt=prompt, steps=S, mix_weight=pmix, init_image_strength=s, guidance_scale=G, seed=seed)
    x0 = inv0["x0"]
    top = [t.clone() for t in inv0["latent_pair"]]

    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 11)
    eta0 = torch.randn(top[0].shape, generator=g, dtype=torch.float64)

    for k in ks:
        for site in sites:
            for scale in scales:
                eta = (eta0 * float(scale)).to(device=top[0].device, dtype=top[0].dtype)
                hook_add = make_additive_hook(site, int(k), eta, informed=False, n_forward=n_fwd)
                gen = er.generate_from_pair(
                    ef, top, prompt, steps=S, mix_weight=pmix, init_image_strength=s,
                    guidance_scale=G, seed=seed, step_hook=hook_add,
                )
                hook_sub = make_additive_hook(site, int(k), eta, informed=True, n_forward=n_fwd)
                # H3 control: invert the generated pair itself (no VAE). Informed subtract
                # must recover `top` up to floating-point error.
                rec_ign_lat = er.coupled_loop(
                    ef, prompt, reverse=True, init_latent=gen["latent_pair"], steps=S, mix_weight=pmix,
                    init_image_strength=s, guidance_scale=G, seed=seed, step_hook=None,
                )
                rec_inf_lat = er.coupled_loop(
                    ef, prompt, reverse=True, init_latent=gen["latent_pair"], steps=S, mix_weight=pmix,
                    init_image_strength=s, guidance_scale=G, seed=seed, step_hook=hook_sub,
                )
                stego = er.decode_latent(ef, gen["x"])
                tag = f"k{k}_{site}_sc{scale}"
                stego.save(outdir / f"{tag}.png")
                im_t = pil_to_tensor(stego, device=ef.device)
                x0_hat = pr.decoder_inv_vae(ef.vae, im_t, n_steps=cfg.get("decoder_inv_steps", 20))
                rec_ign = er.coupled_loop(
                    ef, prompt, reverse=True, init_latent=x0_hat, steps=S, mix_weight=pmix,
                    init_image_strength=s, guidance_scale=G, seed=seed, step_hook=None,
                )
                rec_inf = er.coupled_loop(
                    ef, prompt, reverse=True, init_latent=x0_hat, steps=S, mix_weight=pmix,
                    init_image_strength=s, guidance_scale=G, seed=seed, step_hook=hook_sub,
                )

                row = {
                    "condition": "null_null",
                    "k": k,
                    "n_fwd": n_fwd,
                    "site": site,
                    "scale": scale,
                    "offset_ignorant_latent_mean": float((rec_ign_lat["x"] - top[0]).abs().mean().item()),
                    "offset_ignorant_latent_max": float((rec_ign_lat["x"] - top[0]).abs().max().item()),
                    "offset_informed_latent_mean": float((rec_inf_lat["x"] - top[0]).abs().mean().item()),
                    "offset_informed_latent_max": float((rec_inf_lat["x"] - top[0]).abs().max().item()),
                    "offset_ignorant_vae_mean": float((rec_ign["x"] - top[0]).abs().mean().item()),
                    "offset_ignorant_vae_max": float((rec_ign["x"] - top[0]).abs().max().item()),
                    "offset_informed_vae_mean": float((rec_inf["x"] - top[0]).abs().mean().item()),
                    "offset_informed_vae_max": float((rec_inf["x"] - top[0]).abs().max().item()),
                    "pixel_mse_stego_vs_orig": pixel_mse(img, stego),
                    "stream": "x",
                    "eta_l2": float(eta.pow(2).mean().sqrt().item()),
                }
                append_csv(outdir / "e5.csv", row)
                print(row)


if __name__ == "__main__":
    main()
