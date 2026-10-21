"""L1 receiver: invert a saved DDIM final latent with the true source conditioning.

Also generates one DDIM eta=0 edit from the saved PRC noise and inverts that
final latent. Euler ancestral edits are not inverted. Sigma for the operational
score is the predeclared sigma_squared=1.5. A measured-error sigma is recorded
as a diagnostic and is not the operational detector.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.instructpix2pix_prc.ddim_invert import (  # noqa: E402
    assert_algebra_matches_scheduler,
    ddim_invert,
    ddim_invert_fixed_point,
)
from experiments.instructpix2pix_prc.run_pair import ddim_from_checkpoint, pt_to_pil, run_edit  # noqa: E402
from experiments.metrics import lpips_distance, pixel_mse  # noqa: E402
from experiments.paths import RESULTS, WORKDIR  # noqa: E402
from experiments.prc_runtime import detect_decode, detection_margin, recover_soft_bits  # noqa: E402
from experiments.receiver_aligned.key_io import key_hash, load_key  # noqa: E402

CONFIG_PATH = ROOT / "experiments" / "configs" / "instructpix2pix_prc" / "l1.json"
OUT = RESULTS / "instructpix2pix_prc" / "l1"

FIELDS = [
    "stage", "run", "receiver_level", "conditioning_provenance", "status", "failure_reason",
    "job_id", "git_rev", "code_sha256", "config_sha256",
    "model_id", "model_revision", "scheduler", "num_inference_steps", "eta",
    "guidance_scale", "image_guidance_scale", "dtype",
    "instruction", "key_hash", "message_sha256", "codeword_sha256",
    "n", "payload_length", "key_message_capacity", "t", "F",
    "sigma", "sigma_squared", "sigma_role",
    "noise_mse", "noise_correlation", "sign_agreement", "codeword_sign_agreement", "sign_errors",
    "detect", "detector_statistic", "threshold", "margin",
    "decode_non_none", "exact_message_match", "full_payload_match",
    "wrong_key_detect", "wrong_key_decode_non_none",
    "pixel_mse_vs_source", "lpips_vs_source",
    "fixed_point_iters", "fixed_point_residual_last",
    "seconds", "gpu_name",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_array(values) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def guided_noise(pipe, image_latents, prompt_embeds, guidance, image_guidance):
    def predict(latents, timestep):
        # The pipeline forward is under no_grad. Keeping the graph here filled an 11 GB GPU.
        with torch.no_grad():
            model_input = torch.cat([latents] * 3)
            model_input = pipe.scheduler.scale_model_input(model_input, timestep)
            model_input = torch.cat([model_input, image_latents], dim=1)
            noise = pipe.unet(model_input, timestep, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
            if hasattr(pipe.scheduler, "sigmas"):
                raise RuntimeError("this DDIM scheduler is not supposed to expose sigmas")
            noise_text, noise_image, noise_uncond = noise.chunk(3)
            return (
                noise_uncond
                + guidance * (noise_text - noise_image)
                + image_guidance * (noise_image - noise_uncond)
            )

    return predict


def recovery_stats(recovered, target):
    rec = recovered.detach().to(torch.float64).flatten().cpu()
    truth = target.detach().to(torch.float64).flatten().cpu()
    mse = float(torch.mean((rec - truth) ** 2).item())
    corr = float(torch.corrcoef(torch.stack([rec, truth]))[0, 1].item())
    signs = torch.sign(rec)
    target_signs = torch.sign(truth)
    errors = int((signs != target_signs).sum().item())
    agree = float((signs == target_signs).to(torch.float64).mean().item())
    oracle_sigma = float(torch.sqrt(torch.mean((rec - truth) ** 2)).item())
    return {
        "noise_mse": mse,
        "noise_correlation": corr,
        "sign_agreement": agree,
        "sign_errors": errors,
        "oracle_sigma": oracle_sigma,
        "recovered": rec,
    }


def score_soft(decoding_key, recovered, codeword, message, sigma_squared):
    soft = recover_soft_bits(recovered, variance=sigma_squared)
    detected, decode_ok, decoded = detect_decode(decoding_key, soft)
    statistic, threshold, margin = detection_margin(decoding_key, soft)
    sent = np.asarray(message, dtype=int).reshape(-1)
    if decoded is None:
        exact = False
        full = False
    else:
        got = np.asarray(decoded, dtype=int).reshape(-1)
        exact = bool(got.size >= sent.size and np.array_equal(got[: sent.size], sent))
        full = bool(got.size == sent.size and np.array_equal(got, sent))
    code = torch.as_tensor(np.asarray(codeword), dtype=torch.float64).flatten()
    rec = recovered.detach().to(torch.float64).flatten().cpu()
    code_agree = float((torch.sign(rec) == code).to(torch.float64).mean().item())
    return {
        "detect": detected,
        "detector_statistic": statistic,
        "threshold": threshold,
        "margin": margin,
        "decode_non_none": decode_ok,
        "exact_message_match": exact,
        "full_payload_match": full,
        "codeword_sign_agreement": code_agree,
        "sigma": float(np.sqrt(sigma_squared)),
        "sigma_squared": sigma_squared,
    }


def to_fp16(tensor, device):
    return tensor.to(device=device, dtype=torch.float16)


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    algebra_error = assert_algebra_matches_scheduler()
    print("algebra max abs", algebra_error, flush=True)

    key_path = WORKDIR / cfg["key_path"]
    encoding_key, decoding_key, message, _saved = load_key(key_path)
    found = key_hash(encoding_key)
    if found != cfg["expected_key_hash"]:
        (OUT / "STATUS.txt").write_text(f"FAIL\nkey hash {found}\n")
        return 2
    wrong_enc, wrong_dec, _wm, _wc = load_key(WORKDIR / cfg["wrong_key_path"])
    noise = np.load(WORKDIR / cfg["saved_noise"])
    codeword = torch.from_numpy(noise["codeword"]).flatten()
    z_prc = torch.from_numpy(noise["z_prc"]).to(torch.float32)
    if z_prc.shape != (1, 4, 64, 64):
        (OUT / "STATUS.txt").write_text(f"FAIL\nz shape {tuple(z_prc.shape)}\n")
        return 2
    if int((torch.sign(z_prc).flatten() != codeword).sum().item()) != 0:
        (OUT / "STATUS.txt").write_text("FAIL\nsaved z_prc signs do not match its codeword\n")
        return 2

    from diffusers import StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], torch_dtype=torch.float16
    ).to("cuda")
    pipe.scheduler = ddim_from_checkpoint(pipe)
    pipe.scheduler.set_timesteps(int(cfg["num_inference_steps"]), device="cuda")
    saved = torch.load(WORKDIR / cfg["saved_ddim_gaussian"], map_location="cpu")
    live = pipe.scheduler.timesteps.detach().cpu().to(torch.float32)
    if not torch.equal(live, saved["timesteps"].to(torch.float32)):
        (OUT / "STATUS.txt").write_text("FAIL\ntimesteps do not match the saved DDIM forward\n")
        return 2

    guidance = float(cfg["guidance_scale"])
    image_guidance = float(cfg["image_guidance_scale"])
    sigma_squared = float(cfg["sigma_squared"])
    device = "cuda"
    rows = []
    base = {
        "stage": "l1",
        "receiver_level": "L1",
        "conditioning_provenance": "saved forward source latent and instruction embeddings",
        "status": "recorded",
        "failure_reason": "",
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "git_rev": git_rev(),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(CONFIG_PATH),
        "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "scheduler": "DDIMScheduler",
        "num_inference_steps": int(cfg["num_inference_steps"]),
        "eta": 0.0,
        "guidance_scale": guidance,
        "image_guidance_scale": image_guidance,
        "dtype": "float16",
        "instruction": cfg["instruction"],
        "key_hash": found,
        "message_sha256": sha256_array(np.asarray(message, dtype=np.uint8)),
        "n": 16384,
        "payload_length": 64,
        "key_message_capacity": 64,
        "t": 3,
        "F": 1e-5,
        "gpu_name": torch.cuda.get_device_name(0),
        "fixed_point_iters": 0,
        "fixed_point_residual_last": "",
        "pixel_mse_vs_source": "",
        "lpips_vs_source": "",
        "codeword_sha256": "",
    }

    def invert_saved(name, final, image_latents, prompt_embeds, target, code_for_score):
        predict = guided_noise(
            pipe,
            to_fp16(image_latents, device),
            to_fp16(prompt_embeds, device),
            guidance,
            image_guidance,
        )
        started = time.perf_counter()
        recovered = ddim_invert(
            predict, pipe.scheduler, to_fp16(final, device), pipe.scheduler.timesteps
        )
        seconds = time.perf_counter() - started
        stats = recovery_stats(recovered, target)
        operational = score_soft(decoding_key, stats["recovered"], code_for_score, message, sigma_squared)
        wrong_soft = recover_soft_bits(stats["recovered"], variance=sigma_squared)
        wrong_detect, wrong_decode, _got = detect_decode(wrong_dec, wrong_soft)
        row = {
            **base,
            "run": name,
            "sigma_role": "predeclared",
            "seconds": seconds,
            "noise_mse": stats["noise_mse"],
            "noise_correlation": stats["noise_correlation"],
            "sign_agreement": stats["sign_agreement"],
            "sign_errors": stats["sign_errors"],
            "wrong_key_detect": wrong_detect,
            "wrong_key_decode_non_none": wrong_decode,
            **operational,
        }
        if name.startswith("prc"):
            row["codeword_sha256"] = sha256_array(codeword.numpy())
        torch.save(recovered.detach().to(torch.float32).cpu(), OUT / f"{name}_recovered.pt")
        np.save(OUT / f"{name}_recovered.npy", stats["recovered"].numpy())
        print(
            name,
            "corr", round(stats["noise_correlation"], 4),
            "sign", round(row["sign_agreement"], 4),
            "detect", operational["detect"],
            "exact", operational["exact_message_match"],
            flush=True,
        )
        return row, stats, predict

    gaussian_row, gaussian_stats, gaussian_predict = invert_saved(
        "gaussian_one_pass",
        saved["latent_after_step_99"],
        saved["image_latents_cfg"],
        saved["prompt_embeds"],
        saved["scaled_start"],
        codeword,
    )
    rows.append(gaussian_row)

    source = Image.open(WORKDIR / cfg["source_image"]).convert("RGB")
    run_edit.orig_prepare_latents = pipe.prepare_latents
    run_edit.orig_prepare_image_latents = pipe.prepare_image_latents
    run_edit.orig_encode_prompt = pipe._encode_prompt
    print("forward ddim prc", flush=True)
    bag = run_edit(
        pipe,
        source,
        cfg["instruction"],
        z_prc.to(device="cuda", dtype=torch.float16),
        int(cfg["num_inference_steps"]),
        guidance,
        image_guidance,
        0.0,
        None,
    )
    image = pt_to_pil(bag["image"])
    image.save(OUT / "ddim_prc.png")
    torch.save(
        {
            "raw_unscaled": z_prc,
            "scaled_start": bag["scaled"],
            "final": bag["last"][2],
            "image_latents_cfg": bag["image_latents_cfg"],
            "prompt_embeds": bag["prompt_embeds"],
            "timesteps": bag["timesteps"],
        },
        OUT / "ddim_prc_forward.pt",
    )
    prc_row, prc_stats, prc_predict = invert_saved(
        "prc_one_pass",
        bag["last"][2],
        bag["image_latents_cfg"],
        bag["prompt_embeds"],
        bag["scaled"],
        codeword,
    )
    lpips, _err = lpips_distance(source, image, device="cuda")
    prc_row["pixel_mse_vs_source"] = pixel_mse(source, image)
    prc_row["lpips_vs_source"] = lpips
    rows.append(prc_row)

    needs_solver = (not prc_row["exact_message_match"]) or gaussian_stats["noise_correlation"] < 0.99
    if needs_solver:
        target_name = "prc_fixed_point" if not prc_row["exact_message_match"] else "gaussian_fixed_point"
        predict = prc_predict if target_name.startswith("prc") else gaussian_predict
        final = bag["last"][2] if target_name.startswith("prc") else saved["latent_after_step_99"]
        target = bag["scaled"] if target_name.startswith("prc") else saved["scaled_start"]
        print("fixed point", target_name, flush=True)
        started = time.perf_counter()
        recovered, residuals = ddim_invert_fixed_point(
            predict,
            pipe.scheduler,
            to_fp16(final, device),
            pipe.scheduler.timesteps,
            max_iter=int(cfg["fixed_point_max_iter"]),
        )
        seconds = time.perf_counter() - started
        stats = recovery_stats(recovered, target)
        operational = score_soft(decoding_key, stats["recovered"], codeword, message, sigma_squared)
        wrong_soft = recover_soft_bits(stats["recovered"], variance=sigma_squared)
        wrong_detect, wrong_decode, _got = detect_decode(wrong_dec, wrong_soft)
        row = {
            **base,
            "run": target_name,
            "sigma_role": "predeclared",
            "seconds": seconds,
            "noise_mse": stats["noise_mse"],
            "noise_correlation": stats["noise_correlation"],
            "sign_agreement": stats["sign_agreement"],
            "sign_errors": stats["sign_errors"],
            "wrong_key_detect": wrong_detect,
            "wrong_key_decode_non_none": wrong_decode,
            "fixed_point_iters": len(residuals),
            "fixed_point_residual_last": residuals[-1] if residuals else "",
            "codeword_sha256": sha256_array(codeword.numpy()) if target_name.startswith("prc") else "",
            **operational,
        }
        torch.save(recovered.detach().to(torch.float32).cpu(), OUT / f"{target_name}_recovered.pt")
        rows.append(row)
        print(
            target_name,
            "corr", round(stats["noise_correlation"], 4),
            "exact", operational["exact_message_match"],
            "residual", residuals[-1] if residuals else None,
            flush=True,
        )

    oracle = score_soft(decoding_key, prc_stats["recovered"], codeword, message, max(prc_stats["oracle_sigma"] ** 2, 1e-8))
    rows.append({
        **base,
        "run": "prc_one_pass_oracle_sigma",
        "sigma_role": "oracle_measured_l1_only",
        "seconds": 0.0,
        "noise_mse": prc_stats["noise_mse"],
        "noise_correlation": prc_stats["noise_correlation"],
        "sign_agreement": prc_stats["sign_agreement"],
        "sign_errors": prc_stats["sign_errors"],
        "wrong_key_detect": "",
        "wrong_key_decode_non_none": "",
        "codeword_sha256": sha256_array(codeword.numpy()),
        **oracle,
    })

    with (OUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "approximation": (
            "Each inverse step evaluates the three-branch InstructPix2Pix denoiser "
            "at the cleaner latent and at the forward timestep, then applies the "
            "algebraic left inverse of DDIMScheduler.step at eta=0."
        ),
        "algebra_max_abs_error": algebra_error,
        "init_noise_sigma": 1.0,
        "rows": rows,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (OUT / "STATUS.txt").write_text("PASS\n")
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
