#!/usr/bin/env python3
"""B3: hook=None must match unmodified invert/generate (max abs latent diff)."""
import argparse
import sys
from pathlib import Path

import torch

WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from experiments.paths import RESULTS, PHOTOS, ensure_dirs
from experiments.io_utils import load_config, save_json, append_csv, seed_everything
from experiments.image_utils import load_edict_image
from experiments import edict_runtime as er


def max_abs_pair(a, b):
    return float(max(
        (a[0] - b[0]).abs().max().item(),
        (a[1] - b[1]).abs().max().item(),
    ))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure_dirs()
    seed_everything(cfg.get("seed", 1))
    outdir = RESULTS / "b3"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, outdir / "config.json")

    ef = er.load_models(device=cfg.get("device", "cuda"))
    img = load_edict_image(PHOTOS / cfg["image"])
    S = cfg.get("steps", 8)
    s = cfg.get("init_image_strength", 1.0)
    pmix = cfg.get("mix_weight", 0.93)
    G = cfg.get("guidance_scale", 3)
    seed = cfg.get("seed", 1)
    prompt = ""  # null_null
    x0 = er.encode_image_to_x0(ef, img, generator=torch.cuda.manual_seed(seed))
    pair0 = [x0.clone(), x0.clone()]

    inv_omit = er.coupled_loop(ef, prompt, reverse=True, init_latent=[p.clone() for p in pair0], steps=S, mix_weight=pmix, init_image_strength=s, guidance_scale=G, seed=seed)
    inv_none = er.coupled_loop(ef, prompt, reverse=True, init_latent=[p.clone() for p in pair0], steps=S, mix_weight=pmix, init_image_strength=s, guidance_scale=G, seed=seed, step_hook=None)

    def noop(*_a, **_k):
        return None

    inv_noop = er.coupled_loop(ef, prompt, reverse=True, init_latent=[p.clone() for p in pair0], steps=S, mix_weight=pmix, init_image_strength=s, guidance_scale=G, seed=seed, step_hook=noop)

    pair_up = ef.coupled_stablediffusion(
        prompt,
        reverse=True,
        init_image=[p.clone() for p in pair0],
        steps=S,
        mix_weight=pmix,
        init_image_strength=s,
        guidance_scale=G,
        seed=seed,
        return_latents=True,
    )

    gen_omit = er.generate_from_pair(ef, inv_omit["latent_pair"], prompt, steps=S, mix_weight=pmix, init_image_strength=s, guidance_scale=G, seed=seed)
    gen_up = ef.coupled_stablediffusion(
        prompt,
        reverse=False,
        fixed_starting_latent=[p.clone() for p in pair_up],
        steps=S,
        mix_weight=pmix,
        init_image_strength=s,
        guidance_scale=G,
        seed=seed,
        return_latents=True,
    )

    row = {
        "condition": "null_null",
        "status": "ok",
        "max_abs_omit_vs_none": max_abs_pair(inv_omit["latent_pair"], inv_none["latent_pair"]),
        "max_abs_omit_vs_noop_hook": max_abs_pair(inv_omit["latent_pair"], inv_noop["latent_pair"]),
        "max_abs_experiments_vs_edict_invert": max_abs_pair(inv_omit["latent_pair"], pair_up),
        "max_abs_experiments_vs_edict_generate": max_abs_pair(gen_omit["latent_pair"], gen_up),
        "steps": S,
        "s": s,
        "stream": "x",
    }
    append_csv(outdir / "b3.csv", row)
    save_json(row, outdir / "summary.json")
    print(row)
    # Bit-identical vs omitting the kwarg is required. Match vs edict_functions is diagnostic.
    if row["max_abs_omit_vs_none"] != 0.0 or row["max_abs_omit_vs_noop_hook"] != 0.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
