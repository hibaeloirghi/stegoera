#!/usr/bin/env python3
"""E2: EDICT null-prompt inversion on real photos; plus mismatched-conditioning edits."""
import argparse
import sys
from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from experiments.paths import RESULTS, PHOTOS, ensure_dirs
from experiments.io_utils import load_config, save_json, append_csv, seed_everything
from experiments.image_utils import load_edict_image, list_photos
from experiments.metrics import pixel_mse
from experiments import edict_runtime as er


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure_dirs()
    seed_everything(cfg.get("seed", 1))
    outdir = RESULTS / "e2"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, outdir / "config.json")

    ef = er.load_models(device=cfg.get("device", "cuda"))
    photos = list_photos(PHOTOS, names=cfg.get("photos"))
    strengths = cfg.get("s_values", [0.5, 0.6, 0.7, 0.8, 1.0])
    S = cfg.get("steps", 50)
    pmix = cfg.get("mix_weight", 0.93)
    G = cfg.get("guidance_scale", 3)
    seed = cfg.get("seed", 1)

    for img_path in photos:
        orig = load_edict_image(img_path)
        orig.save(outdir / f"{img_path.stem}_orig.png")
        x0 = er.encode_image_to_x0(ef, orig, generator=torch_cuda_seed(seed))
        vae_only = er.decode_latent(ef, x0)
        vae_only.save(outdir / f"{img_path.stem}_vae.png")
        for s in strengths:
            inv = er.invert_image(
                ef, orig, prompt="",
                steps=S, mix_weight=pmix, init_image_strength=s,
                guidance_scale=G, seed=seed,
            )
            gen = er.generate_from_pair(
                ef, inv["latent_pair"], prompt="",
                steps=S, mix_weight=pmix, init_image_strength=s,
                guidance_scale=G, seed=seed,
            )
            recon = er.decode_latent(ef, gen["x"])
            recon.save(outdir / f"{img_path.stem}_s{s}_recon.png")
            cosine_path = outdir / f"{img_path.stem}_s{s}_cosine.csv"
            with open(cosine_path, "w") as f:
                f.write("step,cosine_xy\n")
                for i, c in enumerate(inv["cosine"]):
                    f.write(f"{i},{c}\n")
            row = {
                "image": img_path.name,
                "s": s,
                "S": S,
                "p": pmix,
                "G": G,
                "t_final": inv["t_final"],
                "alpha_t": inv["alpha_t"],
                "pixel_mse_vs_orig": pixel_mse(orig, recon),
                "pixel_mse_vs_vae": pixel_mse(vae_only, recon),
                "cosine_xy_final": inv["cosine"][-1] if inv["cosine"] else None,
                "n_steps_run": len(inv["cosine"]),
            }
            append_csv(outdir / "e2.csv", row)
            print(row)

        # mismatched conditioning: invert null, regenerate with a target prompt
        if cfg.get("mismatch_prompts"):
            s = cfg.get("mismatch_s", 0.8)
            inv = er.invert_image(
                ef, orig, prompt="",
                steps=S, mix_weight=pmix, init_image_strength=s,
                guidance_scale=G, seed=seed,
            )
            for prompt in cfg["mismatch_prompts"]:
                for g in cfg.get("mismatch_guidance", [1, 3]):
                    gen = er.generate_from_pair(
                        ef, inv["latent_pair"], prompt=prompt,
                        steps=S, mix_weight=pmix, init_image_strength=s,
                        guidance_scale=g, seed=seed,
                    )
                    im = er.decode_latent(ef, gen["x"])
                    safe = prompt.replace(" ", "_")[:40]
                    im.save(outdir / f"{img_path.stem}_mismatch_g{g}_{safe}.png")


def torch_cuda_seed(seed):
    import torch
    return torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    main()
