#!/usr/bin/env python3
"""E3: Edit-time PRC injection into EDICT noising of a real photo.

Sender knows original photo. Receiver does not: decoder-inv + EDICT invert with C_target.
"""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from experiments.paths import RESULTS, PHOTOS, SCRATCH, ensure_dirs
from experiments.io_utils import load_config, save_json, append_csv, seed_everything
from experiments.image_utils import load_edict_image, list_photos, pil_to_tensor
from experiments.metrics import pixel_mse, lpips_distance, clip_score
from experiments import edict_runtime as er
from experiments import prc_runtime as pr


def task_list(cfg):
    photos = list_photos(PHOTOS, names=cfg.get("photos"))
    gammas = cfg.get("gammas", [0.0, 0.25, 0.5, 0.75, 1.0])
    ss = cfg.get("s_values", [0.5, 0.6, 0.7, 0.8])
    mlens = cfg.get("message_lengths", [0, 64, 256])
    prompts = cfg.get("target_prompts", ["a photo in the snow"])
    controls = cfg.get("controls", ["correct"])
    # control names: correct, wrong_key, gamma0 already in gammas, unwatermarked, mismatch_G
    tasks = []
    for photo, gamma, s, mlen, prompt, control in itertools.product(
        photos, gammas, ss, mlens, prompts, controls
    ):
        tasks.append(
            {
                "photo": str(photo),
                "gamma": gamma,
                "s": s,
                "message_length": mlen,
                "prompt": prompt,
                "control": control,
            }
        )
    return tasks


def apply_target_prompt(caption_or_template, photo_stem, templates_need_subject=True):
    if "{subject}" in caption_or_template:
        return caption_or_template.format(subject=photo_stem.replace("_", " "))
    return caption_or_template


def inject_prc(x_t, y_t, x0, alpha_t, codeword_z, gamma, rng):
    """Mix inversion noise with PRC pseudogaussian; shift y by the same delta x."""
    eps_inv = er.ddim_eps(x_t, x0, alpha_t)
    gshape = x_t.shape
    eps_tilde = codeword_z.reshape(gshape).to(device=x_t.device, dtype=x_t.dtype)
    if gamma == 0.0:
        eps_used = eps_inv
    else:
        eps_used = ((1.0 - gamma ** 2) ** 0.5) * eps_inv + gamma * eps_tilde
    x_tp = er.compose_xt(x0, eps_used, alpha_t)
    y_tp = y_t + (x_tp - x_t)
    return x_tp, y_tp, eps_inv, eps_used


def receiver(ef, image_pil, prompt, S, pmix, s, G, seed, dec_key, decoder_inv_steps):
    im_t = pil_to_tensor(image_pil, device=ef.device, dtype=torch.float64)
    x0_hat = pr.decoder_inv_vae(ef.vae, im_t, n_steps=decoder_inv_steps)
    inv = er.coupled_loop(
        ef, prompt, reverse=True, init_latent=x0_hat,
        steps=S, mix_weight=pmix, init_image_strength=s,
        guidance_scale=G, seed=seed,
    )
    alpha_t = inv["alpha_t"]
    x_t_hat = inv["x"]
    # Prompt-specified receiver statistic: u = x_t / sqrt(1-alpha_t)
    u = x_t_hat / ((1.0 - alpha_t) ** 0.5)
    sigma_rec, soft, grid = pr.calibrate_variance(u, dec_key)
    det, dec_ok, decoded_msg = pr.detect_decode(dec_key, soft)
    return {
        "x0_hat": x0_hat,
        "inv": inv,
        "u": u,
        "alpha_t": alpha_t,
        "t_final": inv["t_final"],
        "sigma": sigma_rec["variance"],
        "detect": det,
        "decode": dec_ok,
        "decoded_msg": decoded_msg,
        "sigma_grid": grid,
        "soft": soft,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--task-id", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure_dirs()
    seed_everything(cfg.get("seed", 1))
    outdir = RESULTS / "e3"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, outdir / "config.json")
    tasks = task_list(cfg)
    (outdir / "tasks.json").write_text(json.dumps(tasks, indent=2, default=str))
    if args.task_id is None:
        sel = tasks
    else:
        if args.task_id < 0 or args.task_id >= len(tasks):
            print(f"task-id {args.task_id} out of range n={len(tasks)}")
            sys.exit(0)
        sel = [tasks[args.task_id]]

    ef = er.load_models(device=cfg.get("device", "cuda"))
    S = cfg.get("steps", 50)
    pmix = cfg.get("mix_weight", 0.93)
    G_send = cfg.get("guidance_scale", 3)
    seed = cfg.get("seed", 1)
    n = 4 * 64 * 64
    fpr = cfg.get("fpr", 1e-5)
    t_sparse = cfg.get("prc_t", 3)
    dec_steps = cfg.get("decoder_inv_steps", 100)

    for task in sel:
        photo = Path(task["photo"])
        gamma = float(task["gamma"])
        s = float(task["s"])
        mlen = int(task["message_length"])
        prompt = apply_target_prompt(task["prompt"], photo.stem)
        control = task["control"]
        tag = f"{photo.stem}_g{gamma}_s{s}_m{mlen}_{control}_" + hashlib.md5(prompt.encode()).hexdigest()[:8]
        orig = load_edict_image(photo)

        key_seed = cfg.get("seed", 1)
        enc_key, dec_key = pr.keygen(n=n, message_length=cfg.get("key_message_length", 512), fpr=fpr, t=t_sparse, seed=key_seed)
        wrong_enc, wrong_dec = pr.keygen(n=n, message_length=cfg.get("key_message_length", 512), fpr=fpr, t=t_sparse, seed=key_seed + 12345)
        message = pr.bits_from_message_length(mlen, rng_seed=key_seed + 7) if mlen else None
        codeword, z_prc = pr.encode_codeword(enc_key, message=message, seed=key_seed + 99)

        inv = er.invert_image(
            ef, orig, prompt="",  # null-prompt noising
            steps=S, mix_weight=pmix, init_image_strength=s,
            guidance_scale=G_send, seed=seed,
        )
        x_t, y_t, x0, alpha_t = inv["x"], inv["y"], inv["x0"], inv["alpha_t"]
        definition = {
            "alpha_t_definition": "DDIMScheduler.alphas_cumprod[t_final] after reverse slice timesteps_full[t_limit:][::-1]",
            "t_final": inv["t_final"],
            "alpha_t": alpha_t,
            "t_limit": inv["t_limit"],
            "encoder": "vae.encode.latent_dist.sample * 0.18215",
        }

        use_watermark = (control != "unwatermarked") and not (control == "gamma0")
        g_use = 0.0 if control in ("unwatermarked", "gamma0") else gamma
        if control == "unwatermarked" or g_use == 0.0:
            x_tp, y_tp = x_t, y_t
            eps_inv = er.ddim_eps(x_t, x0, alpha_t)
            eps_used = eps_inv
        else:
            x_tp, y_tp, eps_inv, eps_used = inject_prc(x_t, y_t, x0, alpha_t, z_prc, g_use, None)

        gen = er.generate_from_pair(
            ef, [x_tp, y_tp], prompt=prompt,
            steps=S, mix_weight=pmix, init_image_strength=s,
            guidance_scale=G_send, seed=seed,
        )
        stego = er.decode_latent(ef, gen["x"])  # default x stream
        img_path = outdir / f"{tag}.png"
        stego.save(img_path)

        G_recv = cfg.get("mismatch_guidance", 1) if control == "mismatch_G" else G_send
        recv_key = wrong_dec if control == "wrong_key" else dec_key
        recv = receiver(ef, stego, prompt, S, pmix, s, G_recv, seed, recv_key, dec_steps)

        agree, flip = pr.sign_agreement(recv["u"], z_prc)
        lpips_val, lpips_err = lpips_distance(orig, stego, device=str(ef.device))
        clip_val, clip_err = clip_score(stego, prompt, device=str(ef.device))
        bit_acc = None
        if recv["decoded_msg"] is not None and message is not None:
            rec = np.array(recv["decoded_msg"])[: len(message)]
            if rec.size == len(message):
                bit_acc = float((rec == np.array(message)).mean())

        row = {
            "tag": tag,
            "photo": photo.name,
            "prompt": prompt,
            "control": control,
            "gamma": g_use,
            "s": s,
            "message_length": mlen,
            "G_send": G_send,
            "G_recv": G_recv,
            "t_final_send": inv["t_final"],
            "alpha_t": alpha_t,
            "t_final_recv": recv["t_final"],
            "alpha_t_recv": recv["alpha_t"],
            "stream": "x",
            "pixel_mse": pixel_mse(orig, stego),
            "lpips": lpips_val,
            "lpips_error": lpips_err,
            "clip_score": clip_val,
            "clip_error": clip_err,
            "sign_agree_u_vs_c": agree,
            "sign_flip": flip,
            "detect": recv["detect"],
            "decode": recv["decode"],
            "bit_acc": bit_acc,
            "sigma": recv["sigma"],
            "image": str(img_path),
        }
        append_csv(outdir / "e3.csv", row)
        save_json({**row, **definition}, outdir / f"{tag}.json")
        print(row)


if __name__ == "__main__":
    main()
