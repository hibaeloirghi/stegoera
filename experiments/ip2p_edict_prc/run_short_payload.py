"""One shard of the short-payload study: one message length and one trial.

Runs every frozen image/instruction for that shard. The photograph is
InstructPix2Pix conditioning. PRC signs are applied only to fresh noise.
PNG recovery does not receive the sender latent, the codeword, or the message.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.image_utils import load_edict_image  # noqa: E402
from experiments.ip2p_edict_prc.run_milestone import compare, decode, make_denoise  # noqa: E402
from experiments.ip2p_edict_prc.run_png_decoder_inversion import (  # noqa: E402
    encode_mode,
    invert_duplicated,
    mean_pixel_loss,
    png_to_model,
    recover_latent,
)
from experiments.ip2p_edict_prc.sampler import EdictSchedule, coupled_loop  # noqa: E402
from experiments.metrics import lpips_distance, pixel_mse  # noqa: E402
from experiments.paths import RESULTS, WORKDIR  # noqa: E402
from experiments.prc_runtime import detection_margin, detect_decode, recover_soft_bits  # noqa: E402
from experiments.receiver_aligned.key_io import key_hash, load_key  # noqa: E402

STUDY = RESULTS / "ip2p_edict_prc" / "short_payload_study"
LENGTHS = (8, 16, 32, 64)
MIX = 0.93
LEAPFROG = True
GUIDANCE = 7.5
IMAGE_GUIDANCE = 1.5
SIGMA_SQUARED = 1.5
MODEL_ID = "timbrooks/instruct-pix2pix"
MODEL_REVISION = "31519b5cb02a7fd89b906d88731cd4d6a7bbf88d"

# Frozen in PROMPT_MAPPING.csv. Do not rephrase after results.
CASES = [
    ("imagenet_dog_1", "small", "Add a red collar to the dog.", "EDICT/experiment_images/imagenet_dog_1.jpg"),
    ("imagenet_dog_1", "style", "Render the dog as an oil painting.", "EDICT/experiment_images/imagenet_dog_1.jpg"),
    ("imagenet_cat", "small", "Change the cat's collar to red.", "EDICT/experiment_images/imagenet_cat.jpg"),
    ("imagenet_cat", "style", "Render the cat as a watercolor painting.", "EDICT/experiment_images/imagenet_cat.jpg"),
    ("church", "small", "Make a car in front of the church red.", "EDICT/experiment_images/church.jpg"),
    ("church", "style", "Change the church scene to nighttime with warm lighting.", "EDICT/experiment_images/church.jpg"),
    ("fountain", "small", "Add a person sitting on a nearby bench.", "EDICT/experiment_images/fountain.jpg"),
    ("fountain", "style", "Make the park snowy.", "EDICT/experiment_images/fountain.jpg"),
    ("imagenet_lake", "small", "Add a small red boat to the lake.", "EDICT/experiment_images/imagenet_lake.jpg"),
    ("imagenet_lake", "style", "Render the lake in the style of Van Gogh.", "EDICT/experiment_images/imagenet_lake.jpg"),
    ("imagenet_dog_1", "reference", "Turn the dog into a golden retriever.", "EDICT/experiment_images/imagenet_dog_1.jpg"),
    ("imagenet_dog_1", "reference", "Add a red ball next to the dog.", "EDICT/experiment_images/imagenet_dog_1.jpg"),
]

FIELDS = [
    "length", "trial", "image_id", "role", "instruction", "watermark", "key_id",
    "receiver", "iteration", "primary_endpoint", "recovered_stream",
    "detect", "statistic", "threshold", "margin",
    "decode_non_none", "exact_message_match", "payload_ber",
    "initial_noise_mse", "initial_noise_correlation",
    "codeword_sign_agreement", "codeword_sign_errors",
    "sigma", "sigma_squared",
    "final_xy_mse", "final_xy_max_abs",
    "terminal_latent_mse", "terminal_latent_correlation",
    "pixel_loss", "pixel_mse_vs_source", "lpips_vs_source",
    "runtime", "peak_memory", "status",
]


def shard_from_env():
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    length = LENGTHS[task // 3]
    trial = task % 3
    return length, trial


def send(denoise, schedule, noise):
    """Fresh noise copied into both EDICT streams. The source latent is not the start."""
    start = [noise.clone(), noise.clone()]
    final, _trace = coupled_loop(
        denoise, schedule, start, reverse=False, mix_weight=MIX, leapfrog=LEAPFROG,
    )
    return [item.detach().to(dtype=torch.float64).cpu() for item in final]


def invert_pair(denoise, schedule, pair):
    """Matched inverse of a known final pair. Used only for receiver A."""
    started = time.perf_counter()
    recovered, _trace = coupled_loop(
        denoise, schedule,
        [pair[0].to(device="cuda"), pair[1].to(device="cuda")],
        reverse=True, mix_weight=MIX, leapfrog=LEAPFROG,
    )
    elapsed = time.perf_counter() - started
    return [item.detach().to(dtype=torch.float64).cpu() for item in recovered], elapsed


def score_stream(latent, z_target, decoding_key, codeword, message):
    geometry = compare(latent, z_target)
    soft = recover_soft_bits(latent, variance=SIGMA_SQUARED)
    detected, decode_ok, decoded = detect_decode(decoding_key, soft)
    statistic, threshold, margin = detection_margin(decoding_key, soft)
    sent = np.asarray(message, dtype=int).reshape(-1)
    if decoded is None:
        exact = False
        ber = ""
    else:
        got = np.asarray(decoded, dtype=int).reshape(-1)
        if got.size != sent.size:
            exact = False
            ber = ""
        else:
            ber = float(np.mean(got != sent))
            exact = bool(ber == 0.0)
    code = torch.as_tensor(np.asarray(codeword), dtype=torch.float64).flatten()
    flat = latent.detach().to(torch.float64).flatten().cpu()
    signs = torch.sign(flat) == code
    return {
        "detect": detected,
        "statistic": statistic,
        "threshold": threshold,
        "margin": margin,
        "decode_non_none": decode_ok,
        "exact_message_match": exact,
        "payload_ber": ber,
        "initial_noise_mse": geometry["mse"],
        "initial_noise_correlation": geometry["correlation"],
        "codeword_sign_agreement": float(signs.to(torch.float64).mean().item()),
        "codeword_sign_errors": int((~signs).sum().item()),
    }


def main():
    length, trial = shard_from_env()
    out = STUDY / "runs" / f"length_{length}" / f"trial_{trial}"
    image_dir = out / "images"
    obs_dir = out / "observations"
    out.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(exist_ok=True)
    obs_dir.mkdir(exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()

    noise_file = np.load(STUDY / "trials" / f"length_{length}_trial_{trial}.npz")
    z_prc = torch.from_numpy(np.asarray(noise_file["z_prc"])).to(dtype=torch.float64)
    gaussian = torch.from_numpy(np.asarray(noise_file["gaussian"])).to(dtype=torch.float64)
    message = [int(bit) for bit in np.asarray(noise_file["message"]).reshape(-1)]
    codeword = np.asarray(noise_file["codeword"]).reshape(-1)

    from diffusers import AutoencoderKL, StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16
    ).to("cuda")
    scale = float(pipe.vae.config.scaling_factor)
    vae_loss = AutoencoderKL.from_config(pipe.vae.config)
    vae_loss.load_state_dict({key: value.detach().float().cpu() for key, value in pipe.vae.state_dict().items()})
    vae_loss = vae_loss.to(device="cuda", dtype=torch.float32).eval()
    vae_loss.requires_grad_(False)
    schedule = EdictSchedule(50, device="cpu", dtype=torch.float64)
    recovery_config = {
        "scaling_factor": scale,
        "iterations": 100,
        "checkpoints": [0, 25, 50, 100],
        "learning_rate": 0.1,
        "warmup_steps": 10,
    }
    cfg = {"mix_weight": MIX, "leapfrog": LEAPFROG}
    print("gpu", torch.cuda.get_device_name(0), "length", length, "trial", trial, flush=True)
    rows = []
    sources = {}

    def one_edit(kind, noise, image_id, role, instruction, photo):
        if photo not in sources:
            sources[photo] = load_edict_image(WORKDIR / photo, size=512)
            sources[photo].save(image_dir / f"{image_id}__source.png")
        source = sources[photo]
        processed = pipe.image_processor.preprocess(source)
        prompt_embeds = pipe._encode_prompt(instruction, "cuda", 1, True, negative_prompt="")
        image_latents = pipe.prepare_image_latents(processed, 1, 1, prompt_embeds.dtype, "cuda", True)
        denoise = make_denoise(pipe.unet, image_latents, prompt_embeds, GUIDANCE, IMAGE_GUIDANCE)
        final = send(denoise, schedule, noise.to(device="cuda"))
        gap = final[0] - final[1]
        xy_mse = float(gap.pow(2).mean().item())
        xy_max = float(gap.abs().max().item())
        slug = f"{image_id}__{role}__{kind}"
        decode(pipe, final[0].to(device="cuda")).save(image_dir / f"{slug}_stream0.png")
        decode(pipe, final[1].to(device="cuda")).save(image_dir / f"{slug}_stream1.png")
        png_path = image_dir / f"{slug}_stream0.png"
        held = []
        recovered_a, seconds_a = invert_pair(denoise, schedule, final)
        held.append(("A_final_pair", "", False, None, recovered_a, seconds_a, ""))
        recovered_b, seconds_b = invert_duplicated(denoise, schedule, final[0], cfg)
        held.append(("B_exact_x", "", False, None, recovered_b, seconds_b, ""))
        reloaded = png_to_model(png_path)
        z_mode = encode_mode(pipe.vae, reloaded, scale)
        recovered_c, seconds_c = invert_duplicated(denoise, schedule, z_mode, cfg)
        held.append(("C_vae_mode", 0, False, z_mode.cpu(), recovered_c, seconds_c, mean_pixel_loss(vae_loss, z_mode, reloaded, scale)))
        _z0, saved_steps, _trace, finite = recover_latent(reloaded, pipe.vae, vae_loss, recovery_config)
        if not finite:
            raise RuntimeError(f"non-finite decoder inversion for {slug}")
        for iteration, z_hat in saved_steps.items():
            recovered_d, seconds_d = invert_duplicated(denoise, schedule, z_hat, cfg)
            primary = int(iteration) == 100
            held.append((
                "D_decoder_inversion", int(iteration), primary, z_hat,
                recovered_d, seconds_d, mean_pixel_loss(vae_loss, z_hat, reloaded, scale),
            ))
        torch.save(
            {"final_x": final[0], "receivers": [
                {"receiver": item[0], "iteration": item[1], "streams": item[4]} for item in held
            ]},
            obs_dir / f"{slug}.pt",
        )
        edited = decode(pipe, final[0].to(device="cuda"))
        source_mse = pixel_mse(source, edited)
        source_lpips, lpips_note = lpips_distance(source, edited, device="cuda")
        return {
            "held": held,
            "final_x": final[0],
            "xy_mse": xy_mse,
            "xy_max": xy_max,
            "pixel_mse_vs_source": source_mse,
            "lpips_vs_source": "" if source_lpips is None else source_lpips,
            "lpips_note": lpips_note,
            "png_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
            "recovered_a": recovered_a,
        }

    # Keys are loaded only after each edit's inverses return, inside the scoring loop below.
    for image_id, role, instruction, photo in CASES:
        print("prc", image_id, role, flush=True)
        bundle = one_edit("prc", z_prc, image_id, role, instruction, photo)
        _enc, decoding_key, _probe_message, _probe_code = load_key(STUDY / "keys" / f"length_{length}.pkl")
        for receiver, iteration, primary, z_hat, recovered, seconds, pixel_loss in bundle["held"]:
            terminal = {"mse": "", "correlation": ""}
            if z_hat is not None:
                terminal = compare(z_hat, bundle["final_x"])
            for stream, latent in enumerate(recovered):
                scored = score_stream(latent, z_prc, decoding_key, codeword, message)
                rows.append({
                    "length": length, "trial": trial, "image_id": image_id, "role": role,
                    "instruction": instruction, "watermark": "prc", "key_id": key_hash(_enc),
                    "receiver": receiver, "iteration": iteration,
                    "primary_endpoint": primary and stream == 1,
                    "recovered_stream": stream,
                    **scored,
                    "sigma": float(np.sqrt(SIGMA_SQUARED)),
                    "sigma_squared": SIGMA_SQUARED,
                    "final_xy_mse": bundle["xy_mse"],
                    "final_xy_max_abs": bundle["xy_max"],
                    "terminal_latent_mse": terminal["mse"],
                    "terminal_latent_correlation": terminal["correlation"],
                    "pixel_loss": pixel_loss,
                    "pixel_mse_vs_source": bundle["pixel_mse_vs_source"],
                    "lpips_vs_source": bundle["lpips_vs_source"],
                    "runtime": seconds,
                    "peak_memory": int(torch.cuda.max_memory_allocated()),
                    "status": "ok",
                })
        pair_rows = [row for row in rows if row["image_id"] == image_id and row["instruction"] == instruction and row["receiver"] == "A_final_pair" and row["watermark"] == "prc"]
        if any((not row["exact_message_match"]) or row["initial_noise_correlation"] < 0.99 for row in pair_rows):
            _write(out, rows, length, trial, failed=True)
            print("FAIL pair control", image_id, role, flush=True)
            return 2
        print("pair control ok", image_id, role, flush=True)

    if length == 64:
        for image_id, role, instruction, photo in CASES:
            print("gaussian", image_id, role, flush=True)
            gaussian_bundles[(image_id, instruction)] = one_edit(
                "gaussian", gaussian, image_id, role, instruction, photo,
            )
        for key_length in LENGTHS:
            _enc_g, key_g, _probe, _code_g = load_key(STUDY / "keys" / f"length_{key_length}.pkl")
            other = np.load(STUDY / "trials" / f"length_{key_length}_trial_{trial}.npz")
            other_message = [int(bit) for bit in np.asarray(other["message"]).reshape(-1)]
            other_code = np.asarray(other["codeword"]).reshape(-1)
            for image_id, role, instruction, photo in CASES:
                print("score gaussian", image_id, role, "key", key_length, flush=True)
                bundle = gaussian_bundles[(image_id, instruction)]
                for receiver, iteration, primary, z_hat, recovered, seconds, pixel_loss in bundle["held"]:
                    terminal = {"mse": "", "correlation": ""}
                    if z_hat is not None:
                        terminal = compare(z_hat, bundle["final_x"])
                    for stream, latent in enumerate(recovered):
                        scored = score_stream(latent, gaussian, key_g, other_code, other_message)
                        rows.append({
                            "length": key_length, "trial": trial, "image_id": image_id, "role": role,
                            "instruction": instruction, "watermark": "gaussian", "key_id": key_hash(_enc_g),
                            "receiver": receiver, "iteration": iteration,
                            "primary_endpoint": False,
                            "recovered_stream": stream,
                            **scored,
                            "sigma": float(np.sqrt(SIGMA_SQUARED)),
                            "sigma_squared": SIGMA_SQUARED,
                            "final_xy_mse": bundle["xy_mse"],
                            "final_xy_max_abs": bundle["xy_max"],
                            "terminal_latent_mse": terminal["mse"],
                            "terminal_latent_correlation": terminal["correlation"],
                            "pixel_loss": pixel_loss,
                            "pixel_mse_vs_source": bundle["pixel_mse_vs_source"],
                            "lpips_vs_source": bundle["lpips_vs_source"],
                            "runtime": seconds,
                            "peak_memory": int(torch.cuda.max_memory_allocated()),
                            "status": "ok",
                        })
    _write(out, rows, length, trial, failed=False)
    print("PASS", flush=True)
    return 0


gaussian_bundles = {}


def _write(out, rows, length, trial, failed):
    with (out / "per_trial_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out / "manifest.json").write_text(json.dumps({
        "length": length,
        "trial": trial,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "array_task": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "failed_pair_control": failed,
        "rows": len(rows),
    }, indent=2) + "\n")
    (out / "STATUS.txt").write_text("FAIL\n" if failed else "PASS\n")


if __name__ == "__main__":
    sys.exit(main())
