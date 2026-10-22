#!/usr/bin/env python3
"""E1: Autoencoder-only PRC channel (VAE + uint8 + optional JPEG/resize). No UNet."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from experiments.paths import RESULTS, ensure_dirs
from experiments.io_utils import load_config, save_json, append_csv, seed_everything
from experiments.image_utils import tensor_to_pil, jpeg_roundtrip, resize_roundtrip, pil_to_tensor
from experiments import edict_runtime as er
from experiments import prc_runtime as pr
from experiments.metrics import pixel_mse


def spatial_sign_error(z_hat, codeword):
    rec = torch.sign(z_hat.detach().to(torch.float64).cpu())
    c = torch.sign(codeword.detach().to(torch.float64).cpu()).reshape_as(rec)
    err = (rec != c).float()
    if err.ndim == 4:
        err_map = err[0].mean(dim=0).numpy()  # (64,64) avg over channels
    else:
        err_map = err.numpy()
    return err_map, float(err.mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure_dirs()
    seed_everything(cfg.get("seed", 0))
    outdir = RESULTS / "e1"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, outdir / "config.json")

    ef = er.load_models(device=cfg.get("device", "cuda"))
    n = 4 * 64 * 64
    jpeg_qs = cfg.get("jpeg_qualities", [None, 90, 70, 50, 20])
    msg_lens = cfg.get("message_lengths", [0, 64, 256, 512])
    sparsities = cfg.get("sparsity_t", [2, 3])
    do_resize = cfg.get("resize_roundtrip", [False, True])

    rows = []
    for t in sparsities:
        for mlen in msg_lens:
            enc_key, dec_key = pr.keygen(
                n=n, message_length=cfg.get("key_message_length", 512), fpr=cfg.get("fpr", 1e-5), t=t, seed=cfg.get("seed", 0) + 17 * t + mlen,
            )
            message = pr.bits_from_message_length(mlen, rng_seed=cfg.get("seed", 0) + mlen) if mlen else None
            codeword, z = pr.encode_codeword(enc_key, message=message, seed=cfg.get("seed", 0) + 99)
            z_img = z.reshape(1, 4, 64, 64).to(device=ef.device, dtype=torch.float64)
            with torch.no_grad():
                decoded = ef.vae.decode(z_img / er.VAE_SCALE).sample
            pil = tensor_to_pil(decoded)
            pil.save(outdir / f"raw_t{t}_m{mlen}.png")

            for q in jpeg_qs:
                for rz in do_resize:
                    im = pil
                    attack = "none"
                    if q is not None:
                        im = jpeg_roundtrip(im, q)
                        attack = f"jpeg{q}"
                    if rz:
                        im = resize_roundtrip(im, 0.5)
                        attack = attack + "+resize0.5" if attack != "none" else "resize0.5"
                    im_t = pil_to_tensor(im, device=ef.device, dtype=decoded.dtype)
                    z_hat = pr.decoder_inv_vae(ef.vae, im_t, n_steps=cfg.get("decoder_inv_steps", 100))
                    agree, flip = pr.sign_agreement(z_hat, z_img)
                    err_map, _ = spatial_sign_error(z_hat, z_img)
                    np.save(outdir / f"err_t{t}_m{mlen}_{attack}.npy", err_map)
                    # simple heatmap
                    try:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(3, 3))
                        plt.imshow(err_map, vmin=0, vmax=1, cmap="magma")
                        plt.axis("off")
                        plt.tight_layout()
                        plt.savefig(outdir / f"err_t{t}_m{mlen}_{attack}.png", dpi=120)
                        plt.close()
                    except Exception as exc:
                        print("plot failed", exc)

                    sigma_rec, soft, grid = pr.calibrate_variance(z_hat, dec_key)
                    det, dec_ok, decoded_msg = pr.detect_decode(dec_key, soft)
                    bit_acc = None
                    if decoded_msg is not None and message is not None:
                        rec = np.array(decoded_msg)[: len(message)]
                        bit_acc = float((rec == np.array(message)).mean()) if rec.size else None
                    row = {
                        "condition": cfg.get("condition", "null_null"),
                        "t": t,
                        "message_length": mlen,
                        "attack": attack,
                        "jpeg_q": q,
                        "resize": rz,
                        "sign_agree": agree,
                        "sign_flip": flip,
                        "sigma_calibrated": sigma_rec["variance"],
                        "detect": det,
                        "decode": dec_ok,
                        "bit_acc": bit_acc,
                        "pixel_mse_vs_raw": pixel_mse(pil, im),
                        "sigma_grid": str(grid),
                    }
                    append_csv(outdir / "e1.csv", row)
                    rows.append(row)
                    print(row)

    save_json(rows, outdir / "summary.json")


if __name__ == "__main__":
    main()
