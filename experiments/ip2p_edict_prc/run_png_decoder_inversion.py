"""Source-known PNG receiver for the saved golden-retriever edit.

The operational input is the existing prc_stream0.png. Latent recovery sees
that image, the VAE, and the recovery settings. It does not see the sender
latent, the other stream, the initial noise, the codeword, or the message.

The inverse then sees the duplicated recovered latent and the source
conditioning saved with the forward pass. Scoring loads the key only after
every inverse has returned.
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
from PIL import Image
from transformers import get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.image_utils import tensor_to_pil  # noqa: E402
from experiments.ip2p_edict_prc.run_milestone import compare, make_denoise  # noqa: E402
from experiments.ip2p_edict_prc.sampler import EdictSchedule, coupled_loop  # noqa: E402
from experiments.paths import RESULTS, WORKDIR  # noqa: E402
from experiments.prc_runtime import detection_margin, detect_decode, recover_soft_bits  # noqa: E402
from experiments.receiver_aligned.key_io import key_hash, load_key  # noqa: E402

CONFIG_PATH = ROOT / "experiments" / "configs" / "ip2p_edict_prc" / "png_decoder_inversion.json"
OUT = RESULTS / "ip2p_edict_prc" / "png_decoder_inversion"

RECEIVER_FIELDS = [
    "experiment_id", "job_id", "code_hash", "config_hash",
    "image_hash", "conditioning_hash", "key_id",
    "input_stage", "recovery_method", "iteration",
    "VAE_dtype", "pair_dtype", "UNet_dtype",
    "learning_rate", "pixel_loss",
    "terminal_latent_MSE", "terminal_latent_correlation",
    "recovered_stream",
    "initial_noise_MSE", "initial_noise_correlation", "sign_agreement",
    "sigma", "sigma_squared",
    "detect", "statistic", "threshold", "margin",
    "decode_non_none", "exact_message_match",
    "runtime", "peak_memory", "status",
]
TRACE_FIELDS = [
    "input_stage", "iteration", "pixel_loss_mean", "loss_sum",
    "grad_norm", "latent_norm", "update_norm", "learning_rate", "finite",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def fail(reason: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STATUS.txt").write_text("FAIL\n" + reason + "\n")
    print("FAIL:", reason, flush=True)
    return 2


def png_to_model(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (512, 512):
        raise RuntimeError(f"PNG size {image.size}")
    array = np.asarray(image).astype(np.float32) / 255.0
    array = array * 2.0 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def encode_mode(vae, image, scale: float) -> torch.Tensor:
    """Posterior mode in generation-latent units. No sampling."""
    with torch.no_grad():
        encoded = vae.encode(image.to(device="cuda", dtype=vae.dtype)).latent_dist.mode()
    return (encoded * scale).to(dtype=torch.float64)


def mean_pixel_loss(vae_loss, z, target, scale: float) -> float:
    with torch.no_grad():
        prediction = vae_loss.decode((z.detach().to(device="cuda", dtype=torch.float32)) / scale).sample
        target = target.detach().to(device=prediction.device, dtype=prediction.dtype)
        return float((prediction - target).pow(2).mean().item())


def recover_latent(image, vae_encode, vae_loss, recovery_config):
    """Optimize only a latent so its decode matches image.

    image is (1, 3, 512, 512) in the same units as VAE decode. This function
    has no key, message, codeword, or sender-latent argument.
    """
    scale = float(recovery_config["scaling_factor"])
    iterations = int(recovery_config["iterations"])
    checkpoints = set(int(item) for item in recovery_config["checkpoints"])
    target = image.detach().to(device="cuda", dtype=torch.float32)
    z0 = encode_mode(vae_encode, image, scale)
    z = z0.detach().to(dtype=torch.float32).clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=float(recovery_config["learning_rate"]))
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=int(recovery_config["warmup_steps"]),
        num_training_steps=iterations,
    )
    loss_fn = torch.nn.MSELoss(reduction="sum")
    saved = {}
    trace = []
    for step_index in range(iterations + 1):
        prediction = vae_loss.decode(z / scale).sample
        loss = loss_fn(prediction, target.to(dtype=prediction.dtype))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = z.grad
        finite = bool(torch.isfinite(loss).item() and grad is not None and torch.isfinite(grad).all().item())
        grad_norm = float(grad.detach().norm().item()) if grad is not None else float("nan")
        before = z.detach().clone()
        if step_index in checkpoints:
            saved[step_index] = before.to(dtype=torch.float64).cpu()
        pixel_mean = float((prediction.detach() - target.to(dtype=prediction.dtype)).pow(2).mean().item())
        row = {
            "iteration": step_index,
            "pixel_loss_mean": pixel_mean,
            "loss_sum": float(loss.detach().item()),
            "grad_norm": grad_norm,
            "latent_norm": float(before.norm().item()),
            "update_norm": 0.0,
            "learning_rate": float(opt.param_groups[0]["lr"]),
            "finite": finite,
        }
        if not finite:
            trace.append(row)
            return z0.cpu(), saved, trace, False
        if step_index == iterations:
            trace.append(row)
            break
        opt.step()
        sched.step()
        row["update_norm"] = float((z.detach() - before).norm().item())
        trace.append(row)
    return z0.cpu(), saved, trace, True


def invert_duplicated(denoise, schedule, z, cfg):
    """Matched inverse of (z, z). z is the recovered latent only."""
    pair = [
        z.detach().to(device="cuda", dtype=torch.float64).clone(),
        z.detach().to(device="cuda", dtype=torch.float64).clone(),
    ]
    started = time.perf_counter()
    recovered, _trace = coupled_loop(
        denoise, schedule, pair, reverse=True,
        mix_weight=float(cfg["mix_weight"]), leapfrog=bool(cfg["leapfrog"]),
    )
    elapsed = time.perf_counter() - started
    return [item.detach().to(dtype=torch.float64).cpu() for item in recovered], elapsed


def channel_diagnostics(pipe, true_x, png_path: Path):
    """Rebuild the float decoder image from saved x and compare it with the PNG.

    true_x is used only to name where clipping and uint8 conversion happen.
    The returned images are what later recovery calls are allowed to see.
    """
    scale = float(pipe.vae.config.scaling_factor)
    with torch.no_grad():
        raw = pipe.vae.decode(
            (true_x.to(device="cuda", dtype=pipe.vae.dtype) / scale)
        ).sample
    display = raw / 2 + 0.5
    excess = torch.where(display < 0, -display, torch.where(display > 1, display - 1, torch.zeros_like(display)))
    clipped = display.clamp(0, 1) * 2 - 1
    reconstructed = np.asarray(tensor_to_pil(raw))
    received = np.asarray(Image.open(png_path).convert("RGB"))
    differ = reconstructed.astype(np.int16) - received.astype(np.int16)
    stats = {
        "decoder_output_min": float(raw.min().item()),
        "decoder_output_max": float(raw.max().item()),
        "clip_fraction": float(((display < 0) | (display > 1)).float().mean().item()),
        "clip_max_excess": float(excess.max().item()),
        "clip_mean_excess": float(excess.mean().item()),
        "uint8_exact_match": bool(np.array_equal(reconstructed, received)),
        "uint8_mismatched_pixels": int(np.any(differ != 0, axis=2).sum()),
        "uint8_max_abs_channel_diff": int(np.abs(differ).max()),
        "vae_dtype_for_this_decode": str(pipe.vae.dtype),
        "scaling_factor": scale,
        "uint8_rule": "(image / 2 + 0.5).clamp(0, 1) * 255, round to nearest",
    }
    images = {
        "float_decoder": raw.detach().to(dtype=torch.float32).cpu(),
        "clipped_display": clipped.detach().to(dtype=torch.float32).cpu(),
        "uint8_png": png_to_model(png_path),
    }
    tensor_to_pil(raw).save(OUT / "reconstruction_images" / "diagnostic_float_decode_displayed.png")
    return images, stats


def score_stream(recovered, z_prc, decoding_key, codeword, message, sigma_squared):
    geometry = compare(recovered, z_prc)
    soft = recover_soft_bits(recovered, variance=sigma_squared)
    detected, decode_ok, decoded = detect_decode(decoding_key, soft)
    statistic, threshold, margin = detection_margin(decoding_key, soft)
    sent = np.asarray(message, dtype=int).reshape(-1)
    if decoded is None:
        exact = False
        payload_hash = ""
    else:
        got = np.asarray(decoded, dtype=int).reshape(-1)
        exact = bool(got.size == sent.size and np.array_equal(got, sent))
        payload_hash = sha256_bytes(got.tobytes())
    return {
        "initial_noise_MSE": geometry["mse"],
        "initial_noise_correlation": geometry["correlation"],
        "sign_agreement": geometry["sign_agreement"],
        "detect": detected,
        "statistic": statistic,
        "threshold": threshold,
        "margin": margin,
        "decode_non_none": decode_ok,
        "exact_message_match": exact,
        "payload_sha256": payload_hash,
    }


def main():
    global OUT
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG_PATH
    cfg = json.loads(config_path.read_text())
    OUT = RESULTS / "ip2p_edict_prc" / cfg.get("output_dir", "png_decoder_inversion")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latent_checkpoints").mkdir(exist_ok=True)
    (OUT / "recovered_observations").mkdir(exist_ok=True)
    (OUT / "reconstruction_images").mkdir(exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()

    png_path = WORKDIR / cfg["received_png"]
    code_hash = sha256_bytes(
        (ROOT / "experiments/ip2p_edict_prc/sampler.py").read_bytes()
        + Path(__file__).read_bytes()
    )
    config_hash = sha256_file(config_path)
    image_hash = sha256_file(png_path)
    job_id = os.environ.get("SLURM_JOB_ID", "")

    saved = torch.load(WORKDIR / cfg["saved_pairs"], map_location="cpu")
    if not torch.equal(
        EdictSchedule(int(cfg["num_inference_steps"])).timesteps.cpu(),
        saved["timesteps"].to(torch.float64).cpu(),
    ):
        return fail("saved timesteps do not match the EDICT schedule")

    from diffusers import StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], torch_dtype=torch.float16
    ).to("cuda")
    scale = float(pipe.vae.config.scaling_factor)
    if abs(scale - 0.18215) > 1e-8:
        return fail(f"unexpected scaling factor {scale}")
    images, channel = channel_diagnostics(pipe, saved["prc_final"][0], png_path)
    images["gaussian_uint8_png"] = png_to_model(WORKDIR / cfg["gaussian_png"])
    del saved["prc_final"]
    del saved["prc_recovered"]
    del saved["gaussian_recovered"]

    image_latents = saved["image_latents"].to(device="cuda", dtype=torch.float16)
    prompt_embeds = saved["prompt_embeds"].to(device="cuda", dtype=torch.float16)
    conditioning_hash = sha256_bytes(
        saved["image_latents"].numpy().tobytes() + saved["prompt_embeds"].numpy().tobytes()
    )
    denoise = make_denoise(
        pipe.unet, image_latents, prompt_embeds,
        float(cfg["guidance_scale"]), float(cfg["image_guidance_scale"]),
    )
    schedule = EdictSchedule(int(cfg["num_inference_steps"]), device="cpu", dtype=torch.float64)
    from diffusers import AutoencoderKL

    vae_loss = AutoencoderKL.from_config(pipe.vae.config)
    vae_loss.load_state_dict({key: value.detach().float().cpu() for key, value in pipe.vae.state_dict().items()})
    vae_loss = vae_loss.to(device="cuda", dtype=torch.float32).eval()
    vae_loss.requires_grad_(False)
    recovery_config = {
        "scaling_factor": scale,
        "iterations": int(cfg["iterations"]),
        "checkpoints": cfg["checkpoints"],
        "learning_rate": float(cfg["learning_rate"]),
        "warmup_steps": int(cfg["warmup_steps"]),
    }
    print("gpu", torch.cuda.get_device_name(0), "vae_loss", vae_loss.dtype, flush=True)

    held = []
    traces = []

    def add_mode(stage, image, image_id):
        z0 = encode_mode(pipe.vae, image, scale)
        loss = mean_pixel_loss(vae_loss, z0, image, scale)
        token = f"{stage}_vae_mode_0"
        torch.save(z0.cpu(), OUT / "latent_checkpoints" / f"{token}.pt")
        recovered, elapsed = invert_duplicated(denoise, schedule, z0, cfg)
        torch.save(
            {"x": recovered[0], "y": recovered[1], "x_minus_y": recovered[0] - recovered[1], "z_hat": z0.cpu()},
            OUT / "recovered_observations" / f"{token}.pt",
        )
        held.append({
            "input_stage": stage, "recovery_method": "vae_mode", "iteration": 0,
            "z_hat": z0.cpu(), "recovered": recovered, "pixel_loss": loss,
            "runtime": elapsed, "image_hash": image_id, "learning_rate": 0.0,
            "VAE_dtype": "float16",
        })
        print(stage, "vae_mode inverted", round(elapsed, 2), flush=True)
        return z0

    png_image = images["uint8_png"]
    baseline_z = add_mode("uint8_png", png_image, image_hash)

    # Score the baseline before spending the optimization, then drop the key.
    _enc, decoding_key, message, key_codeword = load_key(WORKDIR / cfg["key_path"])
    if key_hash(_enc) != cfg["expected_key_hash"]:
        return fail(f"key hash {key_hash(_enc)}")
    noise_file = np.load(WORKDIR / cfg["saved_noise"])
    codeword = np.asarray(noise_file["codeword"]).reshape(-1)
    if not np.array_equal(np.asarray(key_codeword).reshape(-1), codeword):
        return fail("saved codeword does not match the PRC noise")
    z_prc = torch.from_numpy(np.asarray(noise_file["z_prc"])).to(dtype=torch.float64)
    message_hash = sha256_bytes(np.asarray(message, dtype=np.uint8).tobytes())
    codeword_hash = sha256_bytes(np.asarray(codeword, dtype=np.float64).tobytes())
    baseline_scores = [
        score_stream(item, z_prc, decoding_key, codeword, message, float(cfg["sigma_squared"]))
        for item in held[0]["recovered"]
    ]
    reproduces = all(
        (not row["decode_non_none"]) and (not row["exact_message_match"]) and row["initial_noise_correlation"] < 0.5
        for row in baseline_scores
    )
    print("baseline reproduces previous failure", reproduces, flush=True)
    del decoding_key, message, codeword, z_prc, noise_file, key_codeword, _enc
    if cfg.get("require_mode_failure", True) and not reproduces:
        return fail("ordinary VAE-mode receiver did not reproduce the previous PNG failure")

    def add_inversion(stage, image, image_id):
        _z0, saved_steps, trace, finite = recover_latent(image, pipe.vae, vae_loss, recovery_config)
        for row in trace:
            row["input_stage"] = stage
            traces.append(row)
        if not finite:
            return False
        for iteration, z_hat in saved_steps.items():
            if stage != "uint8_png" and iteration not in (0, 100):
                torch.save(z_hat, OUT / "latent_checkpoints" / f"{stage}_decoder_inversion_{iteration}.pt")
                continue
            if stage == "uint8_png" and iteration == 0:
                torch.save(z_hat, OUT / "latent_checkpoints" / f"{stage}_decoder_inversion_{iteration}.pt")
            token = f"{stage}_decoder_inversion_{iteration}"
            torch.save(z_hat, OUT / "latent_checkpoints" / f"{token}.pt")
            recovered, elapsed = invert_duplicated(denoise, schedule, z_hat, cfg)
            torch.save(
                {"x": recovered[0], "y": recovered[1], "x_minus_y": recovered[0] - recovered[1], "z_hat": z_hat},
                OUT / "recovered_observations" / f"{token}.pt",
            )
            prediction = vae_loss.decode((z_hat.to(device="cuda", dtype=torch.float32)) / scale).sample
            tensor_to_pil(prediction).save(OUT / "reconstruction_images" / f"{token}.png")
            held.append({
                "input_stage": stage, "recovery_method": "decoder_inversion", "iteration": iteration,
                "z_hat": z_hat, "recovered": recovered,
                "pixel_loss": mean_pixel_loss(vae_loss, z_hat, image, scale),
                "runtime": elapsed, "image_hash": image_id, "learning_rate": float(cfg["learning_rate"]),
                "VAE_dtype": "float32_loss_copy",
            })
            print(stage, "inversion", iteration, "inverted", round(elapsed, 2), flush=True)
        return True

    targets = [
        ("uint8_png", png_image, image_hash),
        ("float_decoder", images["float_decoder"], sha256_bytes(images["float_decoder"].numpy().tobytes())),
        ("clipped_display", images["clipped_display"], sha256_bytes(images["clipped_display"].numpy().tobytes())),
        ("gaussian_uint8_png", images["gaussian_uint8_png"], sha256_file(WORKDIR / cfg["gaussian_png"])),
    ]
    for stage, image, image_id in targets:
        if stage != "uint8_png":
            add_mode(stage, image, image_id)
        if not add_inversion(stage, image.to(device="cuda"), image_id):
            return fail(f"non-finite gradient during {stage}")

    # Evaluator-only quantities. Recovery and the inverse have already returned.
    saved_eval = torch.load(WORKDIR / cfg["saved_pairs"], map_location="cpu")
    true_x = saved_eval["prc_final"][0].to(dtype=torch.float64)
    gaussian_x = saved_eval["gaussian_final"][0].to(dtype=torch.float64)
    _enc, decoding_key, message, key_codeword = load_key(WORKDIR / cfg["key_path"])
    noise_file = np.load(WORKDIR / cfg["saved_noise"])
    codeword = np.asarray(noise_file["codeword"]).reshape(-1)
    z_prc = torch.from_numpy(np.asarray(noise_file["z_prc"])).to(dtype=torch.float64)
    _wrong_enc, wrong_dec, _wm, _wc = load_key(WORKDIR / cfg["wrong_key_path"])
    raw_soft = recover_soft_bits(z_prc, variance=float(cfg["sigma_squared"]))
    raw_detect, raw_decode_ok, raw_decoded = detect_decode(decoding_key, raw_soft)
    sent = np.asarray(message, dtype=int).reshape(-1)
    raw_exact = raw_decoded is not None and np.array_equal(np.asarray(raw_decoded, dtype=int).reshape(-1), sent)

    sigma_squared = float(cfg["sigma_squared"])
    rows = []
    wrong_key_rows = []
    peak = int(torch.cuda.max_memory_allocated())
    for item in held:
        reference = gaussian_x if item["input_stage"] == "gaussian_uint8_png" else true_x
        terminal = compare(item["z_hat"], reference)
        for stream, recovered in enumerate(item["recovered"]):
            scored = score_stream(recovered, z_prc, decoding_key, codeword, message, sigma_squared)
            rows.append({
                "experiment_id": cfg.get("experiment_id", "ip2p_edict_prc_png_decoder_inversion"),
                "job_id": job_id,
                "code_hash": code_hash,
                "config_hash": config_hash,
                "image_hash": item["image_hash"],
                "conditioning_hash": conditioning_hash,
                "key_id": cfg["expected_key_hash"],
                "input_stage": item["input_stage"],
                "recovery_method": item["recovery_method"],
                "iteration": item["iteration"],
                "VAE_dtype": item["VAE_dtype"],
                "pair_dtype": "float64",
                "UNet_dtype": "float16",
                "learning_rate": item["learning_rate"],
                "pixel_loss": item["pixel_loss"],
                "terminal_latent_MSE": terminal["mse"],
                "terminal_latent_correlation": terminal["correlation"],
                "recovered_stream": stream,
                "initial_noise_MSE": scored["initial_noise_MSE"],
                "initial_noise_correlation": scored["initial_noise_correlation"],
                "sign_agreement": scored["sign_agreement"],
                "sigma": float(np.sqrt(sigma_squared)),
                "sigma_squared": sigma_squared,
                "detect": scored["detect"],
                "statistic": scored["statistic"],
                "threshold": scored["threshold"],
                "margin": scored["margin"],
                "decode_non_none": scored["decode_non_none"],
                "exact_message_match": scored["exact_message_match"],
                "runtime": item["runtime"],
                "peak_memory": peak,
                "status": "ok",
            })
            if item["input_stage"] == "uint8_png" and item["recovery_method"] == "decoder_inversion" and item["iteration"] == 100:
                soft = recover_soft_bits(recovered, variance=sigma_squared)
                wrong_detect, wrong_decode, _wrong = detect_decode(wrong_dec, soft)
                wrong_key_rows.append({
                    "recovered_stream": stream,
                    "wrong_key_detect": wrong_detect,
                    "wrong_key_decode_non_none": wrong_decode,
                })
            print(
                item["input_stage"], item["recovery_method"], item["iteration"], stream,
                "exact", rows[-1]["exact_message_match"], "detect", rows[-1]["detect"],
                "corr", round(rows[-1]["initial_noise_correlation"], 4),
                flush=True,
            )

    with (OUT / "recovery_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(traces)
    with (OUT / "receiver_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECEIVER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    png_final = [
        row for row in rows
        if row["input_stage"] == "uint8_png" and row["recovery_method"] == "decoder_inversion" and row["iteration"] == 100
    ]
    manifest = {
        "receiver_level": "source-known PNG, not blind recovery",
        "received_file": cfg["received_png"],
        "received_png_sha256": image_hash,
        "source_png_sha256": sha256_file(WORKDIR / cfg["source_png"]),
        "conditioning_sha256": conditioning_hash,
        "instruction": cfg["instruction"],
        "key_id": cfg["expected_key_hash"],
        "codeword_sha256": codeword_hash,
        "message_sha256": message_hash,
        "sampler_sha256": sha256_file(ROOT / "experiments/ip2p_edict_prc/sampler.py"),
        "receiver_sha256": sha256_file(Path(__file__)),
        "code_hash": code_hash,
        "config_hash": config_hash,
        "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "timestep_grid": "EDICT arange 980, 960, ..., 0",
        "mix_weight": cfg["mix_weight"],
        "leapfrog": cfg["leapfrog"],
        "guidance_scale": cfg["guidance_scale"],
        "image_guidance_scale": cfg["image_guidance_scale"],
        "pair_dtype": "float64",
        "unet_dtype": "float16",
        "vae_sender_dtype": "float16",
        "vae_loss_dtype": "float32 copy; live float16 VAE was not cast",
        "loss": "sum of squared decode error; pixel_loss column is the mean",
        "init": "posterior mode times scaling_factor; PRC decoder_inv samples instead",
        "learning_rate": cfg["learning_rate"],
        "warmup_steps": cfg["warmup_steps"],
        "iterations": cfg["iterations"],
        "channel": channel,
        "baseline_reproduced": True,
        "baseline_init_max_abs_vs_float64_mode": float((
            torch.load(OUT / "latent_checkpoints" / "uint8_png_decoder_inversion_0.pt", map_location="cpu")
            - baseline_z.cpu()
        ).abs().max().item()),
        "raw_noise_positive_control": {
            "detect": raw_detect, "decode_non_none": raw_decode_ok, "exact_message_match": bool(raw_exact),
        },
        "wrong_key_final_png": wrong_key_rows,
        "png_final_exact_message": [bool(row["exact_message_match"]) for row in png_final],
        "job_id": job_id,
        "gpu": torch.cuda.get_device_name(0),
        "peak_memory": peak,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (OUT / "STATUS.txt").write_text("PASS\n")
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
