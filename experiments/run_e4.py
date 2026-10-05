#!/usr/bin/env python3
"""E4: Robustness of E3 outputs (JPEG, resize, noise, crop). Receiver only."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from experiments.paths import RESULTS, ensure_dirs
from experiments.io_utils import load_config, save_json, append_csv, seed_everything
from experiments.image_utils import jpeg_roundtrip, resize_roundtrip, center_crop_roundtrip, add_gaussian_noise_pil, pil_to_tensor
from experiments import edict_runtime as er
from experiments import prc_runtime as pr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure_dirs()
    seed_everything(cfg.get("seed", 1))
    e3dir = RESULTS / "e3"
    outdir = RESULTS / "e4"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, outdir / "config.json")

    # Select best (gamma,s) per prompt from e3.csv among control==correct
    import csv
    e3csv = e3dir / "e3.csv"
    if not e3csv.is_file():
        print("missing", e3csv)
        sys.exit(1)
    rows = list(csv.DictReader(e3csv.open()))
    candidates = [r for r in rows if r.get("control") == "correct"]
    best = {}
    for r in candidates:
        key = r["prompt"]
        score = (r["detect"] == "True", r["decode"] == "True", float(r.get("sign_agree_u_vs_c") or 0))
        if key not in best or score > best[key][0]:
            best[key] = (score, r)
    chosen = [v[1] for v in best.values()]
    if not chosen:
        chosen = candidates[:1]
    save_json(chosen, outdir / "chosen.json")

    ef = er.load_models(device=cfg.get("device", "cuda"))
    S = cfg.get("steps", 50)
    pmix = cfg.get("mix_weight", 0.93)
    seed = cfg.get("seed", 1)
    n = 4 * 64 * 64
    fpr = cfg.get("fpr", 1e-5)
    t_sparse = cfg.get("prc_t", 3)
    # Recreate keys with the same seed as E3 (not the original photo).
    mlen_default = int(cfg.get("message_length", 64))
    enc_key, dec_key = pr.keygen(n=n, message_length=max(mlen_default, 1), fpr=fpr, t=t_sparse, seed=cfg.get("seed", 1))

    attacks = []
    for q in cfg.get("jpeg_qualities", [90, 70, 50, 20]):
        attacks.append(("jpeg", q))
    if cfg.get("resize", True):
        attacks.append(("resize0.5", None))
    if cfg.get("gaussian_sigma") is not None:
        attacks.append(("gauss", cfg.get("gaussian_sigma", 0.02)))
    if cfg.get("center_crop", True):
        attacks.append(("centercrop", cfg.get("crop_frac", 0.875)))

    for rec in chosen:
        img = Image.open(rec["image"]).convert("RGB")
        prompt = rec["prompt"]
        s = float(rec["s"])
        G = float(rec.get("G_recv") or rec.get("G_send") or 3)
        for name, param in attacks:
            if name == "jpeg":
                atk = jpeg_roundtrip(img, int(param))
                label = f"jpeg{param}"
            elif name == "resize0.5":
                atk = resize_roundtrip(img, 0.5)
                label = "resize0.5"
            elif name == "gauss":
                atk = add_gaussian_noise_pil(img, sigma=float(param), rng=0)
                label = f"gauss{param}"
            else:
                atk = center_crop_roundtrip(img, crop_frac=float(param))
                label = f"centercrop{param}"
            atk_path = outdir / f"{Path(rec['image']).stem}_{label}.png"
            atk.save(atk_path)
            im_t = pil_to_tensor(atk, device=ef.device)
            x0_hat = pr.decoder_inv_vae(ef.vae, im_t, n_steps=cfg.get("decoder_inv_steps", 100))
            inv = er.coupled_loop(
                ef, prompt, reverse=True, init_latent=x0_hat,
                steps=S, mix_weight=pmix, init_image_strength=s,
                guidance_scale=G, seed=seed,
            )
            u = inv["x"] / ((1.0 - inv["alpha_t"]) ** 0.5)
            sigma_rec, soft, grid = pr.calibrate_variance(u, dec_key)
            det, dec_ok, _ = pr.detect_decode(dec_key, soft)
            row = {
                "source_image": rec["image"],
                "prompt": prompt,
                "gamma": rec.get("gamma"),
                "s": s,
                "attack": label,
                "detect": det,
                "decode": dec_ok,
                "sigma": sigma_rec["variance"],
                "note": "PRC paper did not evaluate geometric attacks; centercrop is extra",
            }
            append_csv(outdir / "e4.csv", row)
            print(row)


if __name__ == "__main__":
    main()
