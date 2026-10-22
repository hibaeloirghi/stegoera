#!/usr/bin/env python3
"""PRC steganography during a prompted EDICT edit of a real image.

The base/inversion prompt is always null. The sender denoises with edit_prompt
while embedding either at the top latent or inside selected coupling layers.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from experiments.paths import RESULTS, PHOTOS, ensure_dirs
from experiments.io_utils import load_config, save_json, append_csv, seed_everything
from experiments.image_utils import load_edict_image, pil_to_tensor
from experiments.metrics import pixel_mse, lpips_distance, clip_score
from experiments import edict_runtime as er
from experiments import prc_runtime as pr
from experiments.hooks import make_additive_hook, make_window_prc_hook


def inject_top(pair, x0, alpha_t, z, gamma):
    """Blend the inversion noise with PRC pseudogaussian noise."""
    x_t, y_t = pair
    eps_inv = er.ddim_eps(x_t, x0, alpha_t)
    eps_prc = z.reshape_as(x_t).to(device=x_t.device, dtype=x_t.dtype)
    eps_used = ((1.0 - gamma**2) ** 0.5) * eps_inv + gamma * eps_prc
    x_marked = er.compose_xt(x0, eps_used, alpha_t)
    return [x_marked, y_t + (x_marked - x_t)]


def make_embedding(task, top, x0, alpha_t, z, n_fwd):
    """Return (starting pair, generation hook, injection metadata)."""
    method = task["method"]
    if method == "baseline":
        return [v.clone() for v in top], None, {"energy": 0.0}
    if method == "top":
        gamma = float(task["gamma"])
        pair = inject_top(top, x0, alpha_t, z, gamma)
        delta = pair[0] - top[0]
        return pair, None, {"energy": float(delta.square().mean().item())}

    site = task["site"]
    lam = float(task["lambda"])
    k = int(round(float(task.get("k_fraction", 0.5)) * (n_fwd - 1)))
    z_img = z.reshape_as(top[0]).to(device=top[0].device, dtype=top[0].dtype)
    if method == "coupling_single":
        eta = lam * z_img
        hook = make_additive_hook(site, k, eta, informed=False, n_forward=n_fwd)
        return [v.clone() for v in top], hook, {
            "k": k,
            "energy": float(eta.square().mean().item()),
        }
    if method == "coupling_coherent":
        radius = int(task.get("window_radius", 1))
        ks = list(range(max(0, k - radius), min(n_fwd, k + radius + 1)))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(task.get("magnitude_seed", 991)))
        signs = torch.sign(z_img)
        # Keep total injected energy comparable across window lengths.
        per_step = lam / len(ks) ** 0.5
        etas = []
        for _ in ks:
            mag = torch.randn(z_img.shape, generator=generator, dtype=torch.float64).abs()
            etas.append(per_step * signs * mag.to(device=z_img.device, dtype=z_img.dtype))
        hook = make_window_prc_hook(site, ks, etas, informed=False, n_forward=n_fwd)
        return [v.clone() for v in top], hook, {
            "k": k,
            "window": str(ks),
            "energy": float(sum(e.square().mean().item() for e in etas)),
        }
    raise ValueError(f"unknown embedding method: {method}")


def receive(ef, x0_hat, prompt, cfg, task, dec_key):
    """Recover the top x latent and apply the PRC detector/decoder."""
    inv = er.coupled_loop(
        ef,
        prompt,
        reverse=True,
        init_latent=x0_hat,
        steps=int(cfg["steps"]),
        mix_weight=float(task["mix_weight"]),
        init_image_strength=float(task["s"]),
        guidance_scale=float(task["guidance_scale"]),
        seed=int(cfg["seed"]),
    )
    u = inv["x"] / ((1.0 - inv["alpha_t"]) ** 0.5)
    sigma, soft, _ = pr.calibrate_variance(u, dec_key)
    detected, decoded, message = pr.detect_decode(dec_key, soft)
    return inv, u, sigma["variance"], detected, decoded, message


def task_tag(task):
    payload = json.dumps(task, sort_keys=True)
    return f"{task['method']}_{hashlib.md5(payload.encode()).hexdigest()[:10]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tasks = cfg["tasks"]
    if not 0 <= args.task_id < len(tasks):
        raise SystemExit(f"task-id {args.task_id} outside [0,{len(tasks)})")
    task = tasks[args.task_id]
    assert cfg["condition"] == "null_target"
    assert cfg.get("base_prompt", "") == ""

    ensure_dirs()
    seed_everything(int(cfg["seed"]))
    outdir = RESULTS / "edit_stego"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, outdir / "config.json")
    task_tag_value = task_tag(task)
    save_json(task, outdir / f"{task_tag_value}_task.json")

    ef = er.load_models(device=cfg.get("device", "cuda"))
    original = load_edict_image(PHOTOS / cfg["image"])
    edit_prompt = cfg["edit_prompt"]
    seed = int(cfg["seed"])
    steps = int(cfg["steps"])
    s = float(task["s"])
    guidance = float(task["guidance_scale"])
    mix_weight = float(task["mix_weight"])
    n_fwd = int(steps * s)

    enc_key, dec_key = pr.keygen(
        n=4 * 64 * 64,
        message_length=int(cfg.get("key_message_length", 512)),
        fpr=float(cfg.get("fpr", 1e-5)),
        t=int(cfg.get("prc_t", 3)),
        seed=seed,
    )
    message_length = int(task.get("message_length", cfg.get("message_length", 64)))
    message = pr.bits_from_message_length(message_length, rng_seed=seed + 7)
    _, z = pr.encode_codeword(enc_key, message=message, seed=seed + 99)

    # Sender null inversion is identical for every task with the same settings.
    # The array is throttled to one task, so later tasks can safely reuse task 1's cache.
    cache_tag = f"s{s}_S{steps}_p{mix_weight}_seed{seed}"
    inversion_cache = outdir / f"sender_inversion_{cache_tag}.pt"
    if task["method"] != "baseline" and inversion_cache.is_file():
        cached = torch.load(inversion_cache, map_location=ef.device)
        inversion = {
            "latent_pair": cached["latent_pair"],
            "x0": cached["x0"],
            "alpha_t": cached["alpha_t"],
            "t_final": cached["t_final"],
        }
    else:
        inversion = er.invert_image(
            ef,
            original,
            prompt="",
            steps=steps,
            mix_weight=mix_weight,
            init_image_strength=s,
            guidance_scale=guidance,
            seed=seed,
        )
        if task["method"] != "baseline":
            torch.save(
                {
                    "latent_pair": [v.detach().cpu() for v in inversion["latent_pair"]],
                    "x0": inversion["x0"].detach().cpu(),
                    "alpha_t": inversion["alpha_t"],
                    "t_final": inversion["t_final"],
                },
                inversion_cache,
            )
    starting_pair, hook, injection = make_embedding(
        task,
        inversion["latent_pair"],
        inversion["x0"],
        inversion["alpha_t"],
        z,
        n_fwd,
    )
    generated = er.generate_from_pair(
        ef,
        starting_pair,
        prompt=edit_prompt,
        steps=steps,
        mix_weight=mix_weight,
        init_image_strength=s,
        guidance_scale=guidance,
        seed=seed,
        step_hook=hook,
    )
    stego = er.decode_latent(ef, generated["x"])
    image_path = outdir / f"{task_tag_value}.png"
    stego.save(image_path)

    baseline_path = outdir / "baseline.png"
    if task["method"] == "baseline":
        stego.save(baseline_path)
        baseline = stego
    else:
        if not baseline_path.is_file():
            raise RuntimeError("baseline.png missing; run task 0 before embedding tasks")
        from PIL import Image

        baseline = Image.open(baseline_path).convert("RGB")

    lpips_orig, lpips_orig_error = lpips_distance(original, stego, device=str(ef.device))
    lpips_base, lpips_base_error = lpips_distance(baseline, stego, device=str(ef.device))
    clip_value, clip_error = clip_score(stego, edit_prompt, device=str(ef.device))
    clip_base, clip_base_error = clip_score(baseline, edit_prompt, device=str(ef.device))

    # Decoder inversion depends only on the received pixels, not receiver prompt.
    # Compute it once and reuse it for matched and null receiver inversions.
    # EDICT's vendored VAE contains explicit float64 operations, so decoder
    # inversion must stay float64. VAE weights are frozen inside decoder_inv_vae.
    image_tensor = pil_to_tensor(stego, device=ef.device, dtype=torch.float64)
    x0_hat = pr.decoder_inv_vae(
        ef.vae, image_tensor, n_steps=int(cfg.get("decoder_inv_steps", 50))
    )
    for receiver_mode, receiver_prompt in (
        ("matched", edit_prompt),
        ("null_receiver", ""),
    ):
        rec_inv, u, sigma, detected, decoded, decoded_message = receive(
            ef, x0_hat, receiver_prompt, cfg, task, dec_key
        )
        sign_agree, sign_flip = pr.sign_agreement(u, z.reshape_as(u))
        bit_accuracy = None
        if decoded_message is not None:
            recovered = np.asarray(decoded_message)[:message_length]
            if recovered.size == message_length:
                bit_accuracy = float((recovered == np.asarray(message)).mean())
        row = {
            "condition": "null_target",
            "base_prompt": "",
            "edit_prompt": edit_prompt,
            "receiver_mode": receiver_mode,
            "receiver_prompt": receiver_prompt,
            "task_id": args.task_id,
            "tag": task_tag_value,
            "method": task["method"],
            "site": task.get("site"),
            "gamma": task.get("gamma"),
            "lambda": task.get("lambda"),
            "k_fraction": task.get("k_fraction"),
            "window_radius": task.get("window_radius"),
            "injection_energy": injection.get("energy"),
            "injection_k": injection.get("k"),
            "injection_window": injection.get("window"),
            "s": s,
            "S": steps,
            "p": mix_weight,
            "G": guidance,
            "message_length": message_length,
            "stream": "x",
            "detect": detected,
            "decode": decoded,
            "bit_accuracy": bit_accuracy,
            "sign_agreement": sign_agree,
            "sign_flip": sign_flip,
            "sigma": sigma,
            "pixel_mse_vs_original": pixel_mse(original, stego),
            "lpips_vs_original": lpips_orig,
            "lpips_vs_original_error": lpips_orig_error,
            "pixel_mse_vs_unwatermarked_edit": pixel_mse(baseline, stego),
            "lpips_vs_unwatermarked_edit": lpips_base,
            "lpips_vs_unwatermarked_edit_error": lpips_base_error,
            "clip_edit_score": clip_value,
            "clip_edit_score_error": clip_error,
            "clip_unwatermarked_edit_score": clip_base,
            "clip_unwatermarked_edit_score_error": clip_base_error,
            "t_final_send": inversion["t_final"],
            "t_final_receive": rec_inv["t_final"],
            "image": str(image_path),
        }
        append_csv(outdir / "edit_stego.csv", row)
        save_json(row, outdir / f"{task_tag_value}_{receiver_mode}.json")
        print(row)


if __name__ == "__main__":
    main()
