#!/usr/bin/env python3
"""Host-aware PRC shaping during a prompted EDICT edit.

Does not replace the inversion noise with a random PRC sample. A valid codeword
is chosen by best-of-N weighted Hamming distance to the inverted noise, then
each coordinate is pushed toward that codeword only when its margin is short:

    eps'_i = eps_i + lambda * c_i * max(0, tau - c_i * eps_i)

shape_top applies this to the null-inversion noise before the edit.
shape_s4 applies the same projection, as a common-mode add, at one middle
mixing step. A random-sign control uses the same update with a non-code target.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from experiments.paths import RESULTS, PHOTOS, ensure_dirs
from experiments.io_utils import load_config, save_json, append_csv, seed_everything
from experiments.image_utils import load_edict_image, pil_to_tensor
from experiments.metrics import pixel_mse, lpips_distance, clip_score
from experiments import edict_runtime as er
from experiments import prc_runtime as pr


def shape_toward(host, codeword, lam, tau):
    """Push host coordinates toward codeword signs until margin tau."""
    target = codeword.reshape_as(host).to(device=host.device, dtype=host.dtype)
    gap = torch.clamp(tau - target * host, min=0)
    delta = float(lam) * target * gap
    return host + delta, delta


def make_s4_margin_hook(k_forward, codeword, lam, tau):
    """Common-mode S4 projection. The receiver does not subtract it."""
    state = {"energy": 0.0, "fraction_changed": 0.0}

    def hook(site_now, i, t, pair, extra):
        if extra.get("reverse") or site_now != "S4" or i != k_forward:
            return None
        x = pair[0]
        _shaped, eta = shape_toward(x, codeword, lam, tau)
        state["energy"] = float(eta.square().mean().item())
        state["fraction_changed"] = float((eta != 0).float().mean().item())
        return [x + eta, pair[1] + eta]

    hook.state = state
    return hook


def random_signs(shape, seed, device, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    bits = torch.randint(0, 2, shape, generator=generator)
    return (1 - 2 * bits).to(device=device, dtype=dtype)


def score_statistic(z_flat, decoding_key):
    """Calibrate the soft-bit variance, then report Detect and the parity margin."""
    sigma, soft, _grid = pr.calibrate_variance(z_flat, decoding_key)
    detected, decoded, message = pr.detect_decode(decoding_key, soft)
    statistic, threshold, margin = pr.detection_margin(decoding_key, soft)
    return {
        "sigma": sigma["variance"],
        "detect": bool(detected),
        "decode": bool(decoded),
        "message": message,
        "statistic": statistic,
        "threshold": threshold,
        "margin": margin,
    }


def bit_accuracy(decoded_message, message, message_length):
    if decoded_message is None:
        return None
    recovered = np.asarray(decoded_message)[:message_length]
    if recovered.size != message_length:
        return None
    return float((recovered == np.asarray(message)).mean())


def task_tag(task):
    return f"{task['method']}_{task['target']}_l{float(task['lambda']):.2f}"


def load_sender_inversion(ef, original, cfg, task, outdir):
    """Reuse the edit-screen null inversion when it is already on disk."""
    steps = int(cfg["steps"])
    s = float(task["s"])
    mix_weight = float(task["mix_weight"])
    seed = int(cfg["seed"])
    cache_tag = f"s{s}_S{steps}_p{mix_weight}_seed{seed}"
    candidates = [
        RESULTS / "edit_stego" / f"sender_inversion_{cache_tag}.pt",
        outdir / f"sender_inversion_{cache_tag}.pt",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        cached = torch.load(path, map_location=ef.device)
        return {
            "latent_pair": [v.to(ef.device) for v in cached["latent_pair"]],
            "x0": cached["x0"].to(ef.device),
            "alpha_t": cached["alpha_t"],
            "t_final": cached["t_final"],
        }
    inversion = er.invert_image(
        ef,
        original,
        prompt="",
        steps=steps,
        mix_weight=mix_weight,
        init_image_strength=s,
        guidance_scale=float(task["guidance_scale"]),
        seed=seed,
    )
    torch.save(
        {
            "latent_pair": [v.detach().cpu() for v in inversion["latent_pair"]],
            "x0": inversion["x0"].detach().cpu(),
            "alpha_t": inversion["alpha_t"],
            "t_final": inversion["t_final"],
        },
        outdir / f"sender_inversion_{cache_tag}.pt",
    )
    return inversion


def load_selected_codeword(enc_key, message, eps, cfg, outdir):
    path = outdir / (
        f"selected_codeword_n{int(cfg['n_candidates'])}_seed{int(cfg['seed'])}.pt"
    )
    if path.is_file():
        saved = torch.load(path, map_location="cpu")
        return saved
    selected = pr.select_codeword(
        enc_key,
        message,
        host_sign=eps.detach().cpu(),
        host_weight=eps.detach().cpu(),
        n_candidates=int(cfg["n_candidates"]),
        seed=int(cfg["candidate_seed"]),
    )
    payload = {
        "codeword": selected["codeword"].detach().cpu(),
        "agreement": selected["agreement"],
        "weighted_agreement": selected["weighted_agreement"],
        "cost": selected["cost"],
        "candidate": selected["candidate"],
        "n_candidates": int(cfg["n_candidates"]),
    }
    torch.save(payload, path)
    return payload


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
    outdir = RESULTS / "shape_stego"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, outdir / "config.json")
    tag = task_tag(task)
    save_json(task, outdir / f"{tag}_task.json")

    ef = er.load_models(device=cfg.get("device", "cuda"))
    original = load_edict_image(PHOTOS / cfg["image"])
    edit_prompt = cfg["edit_prompt"]
    seed = int(cfg["seed"])
    steps = int(cfg["steps"])
    s = float(task["s"])
    guidance = float(task["guidance_scale"])
    mix_weight = float(task["mix_weight"])
    n_fwd = int(steps * s)
    lam = float(task["lambda"])
    tau = float(task["tau"])

    enc_key, dec_key = pr.keygen(
        n=4 * 64 * 64,
        message_length=int(cfg.get("key_message_length", 512)),
        fpr=float(cfg.get("fpr", 1e-5)),
        t=int(cfg.get("prc_t", 3)),
        seed=seed,
    )
    message_length = int(cfg.get("message_length", 64))
    message = pr.bits_from_message_length(message_length, rng_seed=seed + 7)

    inversion = load_sender_inversion(ef, original, cfg, task, outdir)
    x_t, y_t = inversion["latent_pair"]
    eps = er.ddim_eps(x_t, inversion["x0"], inversion["alpha_t"])
    selected = load_selected_codeword(enc_key, message, eps, cfg, outdir)
    prc_codeword = selected["codeword"]
    if task["target"] == "prc":
        target = prc_codeword
    elif task["target"] == "random":
        target = random_signs(eps.shape, int(cfg["random_sign_seed"]), eps.device, eps.dtype)
    else:
        raise ValueError(f"unknown target: {task['target']}")

    hook = None
    sender_margin = None
    sender_detect = None
    injection_k = None
    if task["method"] == "shape_top":
        eps_shaped, delta = shape_toward(eps, target, lam, tau)
        x_marked = er.compose_xt(inversion["x0"], eps_shaped, inversion["alpha_t"])
        starting_pair = [x_marked, y_t + (x_marked - x_t)]
        energy = float((x_marked - x_t).square().mean().item())
        fraction_changed = float((delta != 0).float().mean().item())
        sender = score_statistic(eps_shaped, dec_key)
        sender_margin = sender["margin"]
        sender_detect = sender["detect"]
    elif task["method"] == "shape_s4":
        injection_k = int(round(float(task.get("k_fraction", 0.5)) * (n_fwd - 1)))
        hook = make_s4_margin_hook(injection_k, target, lam, tau)
        starting_pair = [v.clone() for v in inversion["latent_pair"]]
        energy = None
        fraction_changed = None
    else:
        raise ValueError(f"unknown method: {task['method']}")

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
    if hook is not None:
        energy = hook.state["energy"]
        fraction_changed = hook.state["fraction_changed"]
    stego = er.decode_latent(ef, generated["x"])
    image_path = outdir / f"{tag}.png"
    stego.save(image_path)

    baseline_path = RESULTS / "edit_stego" / "baseline.png"
    if not baseline_path.is_file():
        raise RuntimeError(f"missing unwatermarked edit baseline: {baseline_path}")
    baseline = Image.open(baseline_path).convert("RGB")

    lpips_orig, lpips_orig_error = lpips_distance(original, stego, device=str(ef.device))
    lpips_base, lpips_base_error = lpips_distance(baseline, stego, device=str(ef.device))
    clip_value, clip_error = clip_score(stego, edit_prompt, device=str(ef.device))
    clip_base, clip_base_error = clip_score(baseline, edit_prompt, device=str(ef.device))

    image_tensor = pil_to_tensor(stego, device=ef.device, dtype=torch.float64)
    x0_hat = pr.decoder_inv_vae(
        ef.vae, image_tensor, n_steps=int(cfg.get("decoder_inv_steps", 50))
    )
    for receiver_mode, receiver_prompt in (
        ("matched", edit_prompt),
        ("null_receiver", ""),
    ):
        rec_inv = er.coupled_loop(
            ef,
            receiver_prompt,
            reverse=True,
            init_latent=x0_hat,
            steps=steps,
            mix_weight=mix_weight,
            init_image_strength=s,
            guidance_scale=guidance,
            seed=seed,
        )
        u = rec_inv["x"] / ((1.0 - rec_inv["alpha_t"]) ** 0.5)
        residual = er.ddim_eps(rec_inv["x"], x0_hat, rec_inv["alpha_t"])
        scored_u = score_statistic(u, dec_key)
        scored_eps = score_statistic(residual, dec_key)
        sign_u, _flip_u = pr.sign_agreement(u, prc_codeword.reshape_as(u))
        sign_eps, _flip_eps = pr.sign_agreement(residual, prc_codeword.reshape_as(residual))
        target_u, _target_flip = pr.sign_agreement(u, target.reshape_as(u))
        row = {
            "condition": "null_target",
            "base_prompt": "",
            "edit_prompt": edit_prompt,
            "receiver_mode": receiver_mode,
            "receiver_prompt": receiver_prompt,
            "task_id": args.task_id,
            "tag": tag,
            "method": task["method"],
            "target": task["target"],
            "lambda": lam,
            "tau": tau,
            "k_fraction": task.get("k_fraction"),
            "n_candidates": int(cfg["n_candidates"]),
            "selected_candidate": selected["candidate"],
            "selected_agreement": selected["agreement"],
            "selected_weighted_agreement": selected["weighted_agreement"],
            "injection_energy": energy,
            "injection_k": injection_k,
            "fraction_changed": fraction_changed,
            "sender_detect": sender_detect,
            "sender_margin": sender_margin,
            "s": s,
            "S": steps,
            "p": mix_weight,
            "G": guidance,
            "message_length": message_length,
            "stream": "x",
            "detect": scored_u["detect"],
            "decode": scored_u["decode"],
            "bit_accuracy": bit_accuracy(scored_u["message"], message, message_length),
            "margin": scored_u["margin"],
            "statistic": scored_u["statistic"],
            "threshold": scored_u["threshold"],
            "sigma": scored_u["sigma"],
            "sign_agreement": sign_u,
            "detect_residual": scored_eps["detect"],
            "decode_residual": scored_eps["decode"],
            "bit_accuracy_residual": bit_accuracy(scored_eps["message"], message, message_length),
            "margin_residual": scored_eps["margin"],
            "statistic_residual": scored_eps["statistic"],
            "threshold_residual": scored_eps["threshold"],
            "sigma_residual": scored_eps["sigma"],
            "sign_agreement_residual": sign_eps,
            "target_sign_agreement": target_u,
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
        append_csv(outdir / "shape_stego.csv", row)
        save_json(row, outdir / f"{tag}_{receiver_mode}.json")
        print(json.dumps(row, default=str))


if __name__ == "__main__":
    main()
