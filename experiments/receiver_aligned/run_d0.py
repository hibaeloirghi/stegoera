#!/usr/bin/env python3
"""D0 channel ladder for three existing edit trajectories.

Reuses the cached null inversion. Regenerates each trajectory once and
hashes the PNG against the historical file. A hash miss is a labeled rerun,
not a claim that the original latent was recovered.

Stages, all with the matched edit prompt:
    sender: score the carrier before any pixel conversion
    A: invert the true final pair (x, y)
    B: invert (x, x), which drops y
    C_encoder: float decode, posterior sample, duplicate, invert
    C_mean: float decode, posterior mean, duplicate, invert
    C_decoder_inv: float decode, decoder inversion, duplicate, invert
    D_png: uint8 PNG round trip, decoder inversion, duplicate, invert

Soft-bit variances 0.5 and 1.5 are both recorded. The row's detect flag
uses 1.5, the pseudogaussians default. 0.5 is the sweep's grid floor.
Neither variance is selected by maximizing the key score.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.paths import PHOTOS, RESULTS, ensure_dirs
from experiments.io_utils import append_csv, load_config, save_json, seed_everything
from experiments.image_utils import load_edict_image, pil_to_tensor, tensor_to_pil
from experiments.metrics import pixel_mse, lpips_distance, clip_score
from experiments import edict_runtime as er
from experiments import prc_runtime as pr
from experiments.run_edit_stego import make_embedding
from experiments.receiver_aligned.prc_scores import detection_margin, exact_message_match
from experiments.receiver_aligned.state_io import code_hash, git_head, save_state

VAE_SCALE = 0.18215
SIGMA_PRIMARY = 1.5
SIGMA_GRID_FLOOR = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def decode_float(ef, latent):
    return ef.vae.decode(latent.to(ef.vae.dtype) / VAE_SCALE).sample


def encode_sample(ef, image, seed):
    generator = torch.Generator(device=ef.device)
    generator.manual_seed(int(seed))
    return ef.vae.encode(image.to(dtype=torch.float64)).latent_dist.sample(generator=generator) * VAE_SCALE


def encode_mean(ef, image):
    return ef.vae.encode(image.to(dtype=torch.float64)).latent_dist.mean * VAE_SCALE


def score_vector(z, decoding_key, codeword, message):
    out = {}
    for name, var in (("primary", SIGMA_PRIMARY), ("grid_floor", SIGMA_GRID_FLOOR)):
        soft = pr.recover_soft_bits(z, variance=var)
        stat, thr, margin = detection_margin(decoding_key, soft, clip=False)
        detected, decoded, payload = pr.detect_decode(decoding_key, soft)
        agree, _flip = pr.sign_agreement(z, codeword)
        out[name] = {
            "sigma": var,
            "statistic": stat,
            "threshold": thr,
            "margin": margin,
            "detect": bool(detected),
            "decode_non_none": bool(decoded),
            "exact_message_match": exact_message_match(payload, message),
            "sign_agreement": agree,
            "margin_matches_detect": bool(detected) == bool(margin >= 0.0),
        }
    return out


def capture_forward(user_hook, k, bucket):
    def hook(site, i, t, pair, extra):
        updated = user_hook(site, i, t, pair, extra) if user_hook is not None else None
        current = updated if updated is not None else pair
        if site == "S4" and (not extra.get("reverse")) and i == k:
            bucket["pair"] = [p.detach().cpu().clone() for p in current]
            bucket["timestep"] = int(t.item() if torch.is_tensor(t) else t)
            bucket["index"] = int(i)
        return updated
    return hook


def capture_reverse(k_rev, bucket):
    def hook(site, i, t, pair, extra):
        if (
            site == "S4"
            and extra.get("reverse")
            and extra.get("phase") == "before_unmix"
            and i == k_rev
        ):
            bucket["pair"] = [p.detach().cpu().clone() for p in pair]
            bucket["timestep"] = int(t.item() if torch.is_tensor(t) else t)
            bucket["index"] = int(i)
        return None
    return hook


def invert_pair(ef, pair, prompt, cfg, task, reverse_bucket, k_rev):
    return er.coupled_loop(
        ef,
        prompt,
        reverse=True,
        init_latent=pair,
        steps=int(cfg["steps"]),
        mix_weight=float(task["mix_weight"]),
        init_image_strength=float(task["s"]),
        guidance_scale=float(task["guidance_scale"]),
        seed=int(cfg["seed"]),
        step_hook=capture_reverse(k_rev, reverse_bucket),
    )


def row_from_score(base, stage, observation_name, scored, extra):
    primary = scored["primary"]
    floor = scored["grid_floor"]
    row = dict(base)
    row.update({
        "channel_stage": stage,
        "observation": observation_name,
        "sigma": primary["sigma"],
        "true_score_margin": primary["margin"],
        "statistic": primary["statistic"],
        "threshold": primary["threshold"],
        "detect": primary["detect"],
        "decode_non_none": primary["decode_non_none"],
        "exact_message_match": primary["exact_message_match"],
        "sign_agreement": primary["sign_agreement"],
        "margin_matches_detect": primary["margin_matches_detect"],
        "sigma_grid_floor": floor["sigma"],
        "margin_grid_floor": floor["margin"],
        "detect_grid_floor": floor["detect"],
        "exact_message_match_grid_floor": floor["exact_message_match"],
        "status": "ok",
    })
    row.update({
        "oracle_top_x_mse": None,
        "oracle_top_y_mse": None,
        "recovered_pair_error": None,
        "oracle_site_x_mse": None,
        "wall_time_s": None,
        "peak_gpu_bytes": None,
    })
    row.update(extra)
    return row


def main():
    parser_path = Path(__file__).resolve().parents[1] / "configs" / "receiver_aligned" / "d0.json"
    cfg = load_config(parser_path)
    ensure_dirs()
    seed_everything(int(cfg["seed"]))
    outdir = RESULTS / "receiver_aligned"
    raw_dir = outdir / "raw_observations"
    image_dir = outdir / "images"
    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    this_hash = code_hash(Path(__file__).resolve().parent)
    meta = {
        "code_hash": this_hash,
        "experiment_git": git_head(ROOT),
        "edict_git": git_head(ROOT / "EDICT"),
        "prc_git": git_head(ROOT / "PRC-Watermark"),
        "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "sigma_primary": SIGMA_PRIMARY,
        "sigma_grid_floor": SIGMA_GRID_FLOOR,
        "dtype": "float64",
        "model": "CompVis/stable-diffusion-v1-4",
    }
    save_json(meta, outdir / "d0_meta.json")
    print(json.dumps(meta))

    ef = er.load_models(device=cfg.get("device", "cuda"))
    original = load_edict_image(PHOTOS / cfg["image"])
    edit_prompt = cfg["edit_prompt"]
    steps = int(cfg["steps"])
    seed = int(cfg["seed"])
    n_fwd = int(steps * float(cfg["s"]))
    k = int(round(float(cfg["k_fraction"]) * (n_fwd - 1)))
    k_rev = (n_fwd - 1) - k

    cache = torch.load(cfg["inversion_cache"], map_location=ef.device)
    inversion = {
        "latent_pair": [v.to(ef.device) for v in cache["latent_pair"]],
        "x0": cache["x0"].to(ef.device),
        "alpha_t": cache["alpha_t"],
        "t_final": cache["t_final"],
    }
    enc_key, dec_key = pr.keygen(
        n=4 * 64 * 64,
        message_length=int(cfg["key_message_length"]),
        fpr=float(cfg["fpr"]),
        t=int(cfg["prc_t"]),
        seed=seed,
    )
    message = pr.bits_from_message_length(int(cfg["message_length"]), rng_seed=seed + 7)
    codeword, z = pr.encode_codeword(enc_key, message=message, seed=seed + 99)

    device = str(ef.device)
    csv_path = outdir / "d0_ladder.csv"
    for case in cfg["cases"]:
        task = {
            "method": case["method"],
            "s": cfg["s"],
            "guidance_scale": cfg["guidance_scale"],
            "mix_weight": cfg["mix_weight"],
        }
        for key in ("gamma", "site", "lambda", "k_fraction"):
            if key in case:
                task[key] = case[key]
        print("CASE", case["id"], flush=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        forward_bucket = {}
        pair0, user_hook, _injection = make_embedding(
            task, inversion["latent_pair"], inversion["x0"], inversion["alpha_t"], z, n_fwd
        )
        generated = er.generate_from_pair(
            ef,
            pair0,
            prompt=edit_prompt,
            steps=steps,
            mix_weight=float(cfg["mix_weight"]),
            init_image_strength=float(cfg["s"]),
            guidance_scale=float(cfg["guidance_scale"]),
            seed=seed,
            step_hook=capture_forward(user_hook, k, forward_bucket),
        )
        final_x = generated["x"].detach()
        final_y = generated["y"].detach()
        start_x = pair0[0].detach()
        start_y = pair0[1].detach()
        stego = tensor_to_pil(decode_float(ef, final_x))
        image_path = image_dir / f"{case['id']}.png"
        stego.save(image_path)
        historical = case.get("historical_png")
        png_match = None
        if historical and Path(historical).is_file():
            png_match = sha256_file(image_path) == sha256_file(Path(historical))
        float_image = decode_float(ef, final_x).detach()
        save_state(raw_dir / f"{case['id']}_final_pair.pt", {
            "x": final_x.cpu(),
            "y": final_y.cpu(),
            "s4_pair": forward_bucket.get("pair"),
            "s4_timestep": forward_bucket.get("timestep"),
            "s4_index": k,
            "alpha_t_send": inversion["alpha_t"],
            "png_matches_historical": png_match,
            "case": case["id"],
        })
        terminal_pair_error = float((final_x - final_y).square().mean().item())
        lpips_edit, lpips_edit_err = (None, "no_baseline")
        baseline_png = Path(cfg["baseline_png"])
        if case["id"] != "baseline" and baseline_png.is_file():
            lpips_edit, lpips_edit_err = lpips_distance(
                Image.open(baseline_png).convert("RGB"), stego, device=device
            )
        elif case["id"] == "baseline":
            lpips_edit, lpips_edit_err = 0.0, None
        lpips_orig, lpips_orig_err = lpips_distance(original, stego, device=device)
        clip_value, clip_err = clip_score(stego, edit_prompt, device=device)
        base = {
            "experiment_id": "d0",
            "code_hash": this_hash,
            "job_id": meta["job_id"],
            "photo_id": cfg["image"],
            "edit_prompt": edit_prompt,
            "inversion_condition": "null",
            "receiver_condition": "matched",
            "case_id": case["id"],
            "method": case["method"],
            "site": case.get("site", "top"),
            "active_step": k,
            "scheduler_timestep_s4": forward_bucket.get("timestep"),
            "carrier_definition": "x/sqrt(1-alpha) at recovered top; site x unscaled",
            "message_length": int(cfg["message_length"]),
            "F": float(cfg["fpr"]),
            "t": int(cfg["prc_t"]),
            "terminal_pair_error": terminal_pair_error,
            "LPIPS_vs_edit": lpips_edit,
            "LPIPS_vs_edit_error": lpips_edit_err,
            "pixel_MSE_vs_edit": None if case["id"] == "baseline" or not baseline_png.is_file() else pixel_mse(Image.open(baseline_png).convert("RGB"), stego),
            "LPIPS_vs_original": lpips_orig,
            "LPIPS_vs_original_error": lpips_orig_err,
            "edit_alignment": clip_value,
            "edit_alignment_error": clip_err,
            "png_matches_historical": png_match,
            "historical_png": historical,
            "oracle_note": "oracle_* columns use the sender pair and are not a receiver output",
        }
        rows = []

        def add_observation(stage, name, vector, extra=None):
            scored = score_vector(vector, dec_key, codeword, message)
            save_state(raw_dir / f"{case['id']}_{stage}_{name}.pt", {
                "vector": vector.detach().cpu(),
                "stage": stage,
                "name": name,
            })
            rows.append(row_from_score(base, stage, name, scored, extra or {}))

        # Sender-side carriers. Top blend writes eps; coupling writes the S4 x.
        if case["method"] == "top":
            eps_used = er.ddim_eps(pair0[0], inversion["x0"], inversion["alpha_t"])
            add_observation("sender", "blended_eps", eps_used)
        elif case["method"] == "baseline":
            eps_inv = er.ddim_eps(inversion["latent_pair"][0], inversion["x0"], inversion["alpha_t"])
            add_observation("sender", "inversion_eps", eps_inv)
        if "pair" in forward_bucket:
            add_observation("sender", "s4_x", forward_bucket["pair"][0])

        def run_inverse(stage, init_pair):
            bucket = {}
            inv = invert_pair(ef, init_pair, edit_prompt, cfg, task, bucket, k_rev)
            alpha = inv["alpha_t"]
            u = inv["x"] / ((1.0 - alpha) ** 0.5)
            extra = {
                "oracle_top_x_mse": float((inv["x"] - start_x).square().mean().item()),
                "oracle_top_y_mse": float((inv["y"] - start_y).square().mean().item()),
                "recovered_pair_error": float((inv["x"] - inv["y"]).square().mean().item()),
                "oracle_site_x_mse": None,
                "wall_time_s": None,
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            }
            add_observation(stage, "top_u", u, extra)
            if "pair" in bucket and "pair" in forward_bucket:
                site = bucket["pair"][0]
                extra_site = dict(extra)
                extra_site["oracle_site_x_mse"] = float(
                    (site.to(forward_bucket["pair"][0].device) - forward_bucket["pair"][0]).square().mean().item()
                )
                add_observation(stage, "s4_x", site, extra_site)
            return inv

        run_inverse("A_true_pair", [final_x, final_y])
        run_inverse("B_dup_x", [final_x, final_x.clone()])

        z_sample = encode_sample(ef, float_image, seed)
        run_inverse("C_encoder_sample", [z_sample, z_sample.clone()])
        z_mean = encode_mean(ef, float_image)
        run_inverse("C_encoder_mean", [z_mean, z_mean.clone()])
        z_inv = pr.decoder_inv_vae(
            ef.vae, float_image.to(dtype=torch.float64), n_steps=int(cfg["decoder_inv_steps"])
        )
        run_inverse("C_decoder_inv", [z_inv, z_inv.clone()])

        png_tensor = pil_to_tensor(Image.open(image_path).convert("RGB"), device=ef.device, dtype=torch.float64)
        z_png = pr.decoder_inv_vae(ef.vae, png_tensor, n_steps=int(cfg["decoder_inv_steps"]))
        run_inverse("D_png", [z_png, z_png.clone()])

        elapsed = time.time() - t0
        peak = int(torch.cuda.max_memory_allocated())
        for row in rows:
            row["wall_time_s"] = elapsed
            row["peak_gpu_bytes"] = peak
            append_csv(csv_path, row)
            print(json.dumps({k: row[k] for k in ("case_id", "channel_stage", "observation", "detect", "exact_message_match", "true_score_margin", "sign_agreement", "png_matches_historical")}), flush=True)


if __name__ == "__main__":
    main()
