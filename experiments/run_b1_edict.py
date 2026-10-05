#!/usr/bin/env python3
"""B1: EDICT example edit + null-prompt reconstruction. GPU."""
import argparse
import sys
from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from experiments.paths import RESULTS, PHOTOS, ensure_dirs
from experiments.io_utils import load_config, save_json, append_csv, seed_everything
from experiments.image_utils import load_edict_image
from experiments.metrics import pixel_mse, latent_mse
from experiments import edict_runtime as er


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure_dirs()
    seed_everything(cfg.get("seed", 1))
    outdir = RESULTS / "b1"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, outdir / "config.json")

    ef = er.load_models(device=cfg.get("device", "cuda"))
    img_path = Path(cfg["image"]).expanduser()
    if not img_path.is_file():
        img_path = PHOTOS / cfg["image"]
    orig = load_edict_image(img_path)
    orig.save(outdir / "orig.png")

    S = cfg.get("steps", 50)
    pmix = cfg.get("mix_weight", 0.93)
    s = cfg.get("init_image_strength", 0.8)
    G = cfg.get("guidance_scale", 3)
    seed = cfg.get("seed", 1)

    # Reconstruction: invert then generate with the same null prompt.
    inv = er.invert_image(
        ef, orig, prompt=cfg.get("recon_prompt", ""),
        steps=S, mix_weight=pmix, init_image_strength=s,
        guidance_scale=G, seed=seed,
    )
    gen = er.generate_from_pair(
        ef, inv["latent_pair"], prompt=cfg.get("recon_prompt", ""),
        steps=S, mix_weight=pmix, init_image_strength=s,
        guidance_scale=G, seed=seed,
    )
    recon_x = er.decode_latent(ef, gen["x"])
    recon_y = er.decode_latent(ef, gen["y"])
    recon_x.save(outdir / "recon_x.png")
    recon_y.save(outdir / "recon_y.png")
    vae_only = er.decode_latent(ef, inv["x0"])
    vae_only.save(outdir / "vae_roundtrip.png")

    # Edit
    inv_e = er.invert_image(
        ef, orig, prompt=cfg["base_prompt"],
        steps=S, mix_weight=pmix, init_image_strength=s,
        guidance_scale=G, seed=seed,
    )
    gen_e = er.generate_from_pair(
        ef, inv_e["latent_pair"], prompt=cfg["edit_prompt"],
        steps=S, mix_weight=pmix, init_image_strength=s,
        guidance_scale=G, seed=seed,
    )
    edit_x = er.decode_latent(ef, gen_e["x"])
    edit_y = er.decode_latent(ef, gen_e["y"])
    edit_x.save(outdir / "edit_x.png")
    edit_y.save(outdir / "edit_y.png")

    row = {
        "status": "ok",
        "image": str(img_path),
        "steps": S,
        "mix_weight": pmix,
        "s": s,
        "guidance": G,
        "t_final_recon": inv["t_final"],
        "alpha_t_recon": inv["alpha_t"],
        "pixel_mse_recon_vs_orig": pixel_mse(orig, recon_x),
        "pixel_mse_recon_vs_vae": pixel_mse(vae_only, recon_x),
        "pixel_mse_vae_vs_orig": pixel_mse(orig, vae_only),
        "latent_mse_x0_vs_recon_x": latent_mse(inv["x0"], gen["x"]),
        "cosine_xy_final_recon": inv["cosine"][-1] if inv["cosine"] else None,
        "dtype": str(ef.unet.dtype),
        "device": str(next(ef.unet.parameters()).device),
    }
    append_csv(outdir / "b1.csv", row)
    save_json(row, outdir / "summary.json")
    print(json_dumps(row))


def json_dumps(row):
    import json
    return json.dumps(row, indent=2, default=str)


if __name__ == "__main__":
    main()
