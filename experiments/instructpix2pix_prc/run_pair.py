"""One ordinary InstructPix2Pix edit and one PRC-noise edit.

Uses the checkpoint's Euler ancestral scheduler. A third image repeats the
ordinary Gaussian initial noise with DDIM eta=0. That image is a scheduler
probe. None of these rows is a receiver.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.image_utils import load_edict_image  # noqa: E402
from experiments.metrics import lpips_distance, pixel_mse  # noqa: E402
from experiments.paths import RESULTS, WORKDIR  # noqa: E402
from experiments.prc_runtime import prc_modules  # noqa: E402
from experiments.receiver_aligned.key_io import key_hash, load_key, save_key  # noqa: E402

CONFIG_PATH = ROOT / "experiments" / "configs" / "instructpix2pix_prc" / "pair.json"
OUT = RESULTS / "instructpix2pix_prc" / "pair"
IMAGES = OUT / "images"
STATES = OUT / "saved_states"

FIELDS = [
    "stage", "run", "receiver_level", "status", "failure_reason",
    "job_id", "git_rev", "code_sha256", "config_sha256",
    "model_id", "model_revision", "scheduler", "num_inference_steps", "eta",
    "guidance_scale", "image_guidance_scale", "dtype",
    "prediction_type", "timestep_spacing", "init_noise_sigma", "has_sigmas",
    "source", "instruction",
    "key_hash", "message_sha256", "codeword_sha256",
    "n", "payload_length", "key_message_capacity", "t", "F",
    "noise_shape", "noise_scale", "fp16_sign_mismatches",
    "pixel_mse_vs_source", "lpips_vs_source", "lpips_error",
    "clip_instruction", "clip_edit_noun", "directional_clip",
    "paired_seed_lpips_vs_euler_gaussian",
    "frac_decode_below_m1", "frac_decode_above_p1", "frac_png_saturated",
    "nsfw", "seconds", "peak_mem_mib", "gpu_name",
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


def as_float_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(torch.float64).cpu().flatten()
    return torch.as_tensor(np.asarray(value), dtype=torch.float64).flatten()


def pt_to_pil(image):
    array = image.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    array = (array * 255.0).round().astype("uint8")
    return Image.fromarray(array[0])


def saturation_fraction(image: Image.Image) -> float:
    array = np.asarray(image.convert("RGB"))
    hit = (array == 0) | (array == 255)
    return float(hit.mean())


def decode_clip_stats(pipe, latents):
    scale = float(pipe.vae.config.scaling_factor)
    with torch.no_grad():
        decoded = pipe.vae.decode(latents.to(pipe.vae.dtype) / scale).sample
    decoded = decoded.float()
    return {
        "float_decode": decoded.cpu(),
        "frac_decode_below_m1": float((decoded < -1).float().mean().item()),
        "frac_decode_above_p1": float((decoded > 1).float().mean().item()),
    }


def clip_bundle(device):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    return model, processor


def embed_image(model, processor, image, device):
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        return model.get_image_features(**inputs)[0].float()


def embed_text(model, processor, text, device):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        return model.get_text_features(**inputs)[0].float()


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())


def directional_clip(edit_image, source_image, instruction, source_caption, model, processor, device):
    """Gal-style directional CLIP.

    cosine( E_image(edit) - E_image(source), E_text(instruction) - E_text(source_caption) ).
    Embeddings are the pooled CLIP features. source_caption is the fixed human
    description in the config, not a caption of the edit.
    """
    image_delta = embed_image(model, processor, edit_image, device) - embed_image(model, processor, source_image, device)
    text_delta = embed_text(model, processor, instruction, device) - embed_text(model, processor, source_caption, device)
    return cosine(image_delta, text_delta)


def run_edit(pipe, source, prompt, latents, steps, guidance, image_guidance, eta, generator):
    bag = {"states": []}

    def prepare_latents(*args, **kwargs):
        out = run_edit.orig_prepare_latents(*args, **kwargs)
        bag["scaled"] = out.detach().to(torch.float32).cpu()
        bag["init_noise_sigma"] = float(pipe.scheduler.init_noise_sigma)
        bag["timesteps"] = pipe.scheduler.timesteps.detach().to(torch.float32).cpu()
        bag["has_sigmas"] = hasattr(pipe.scheduler, "sigmas")
        return out

    def prepare_image_latents(*args, **kwargs):
        out = run_edit.orig_prepare_image_latents(*args, **kwargs)
        bag["image_latents_cfg"] = out.detach().to(torch.float32).cpu()
        return out

    def encode_prompt(*args, **kwargs):
        out = run_edit.orig_encode_prompt(*args, **kwargs)
        bag["prompt_embeds"] = out.detach().to(torch.float32).cpu()
        return out

    def callback(i, t, latents):
        state = (int(i), float(t), latents.detach().to(torch.float32).cpu())
        bag["last"] = state
        if i in (0, steps // 2, steps - 1):
            bag["states"].append(state)

    pipe.prepare_latents = prepare_latents
    pipe.prepare_image_latents = prepare_image_latents
    pipe._encode_prompt = encode_prompt
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = pipe(
            prompt=prompt,
            image=source,
            num_inference_steps=steps,
            guidance_scale=guidance,
            image_guidance_scale=image_guidance,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type="pt",
            callback=callback,
            callback_steps=1,
        )
    finally:
        pipe.prepare_latents = run_edit.orig_prepare_latents
        pipe.prepare_image_latents = run_edit.orig_prepare_image_latents
        pipe._encode_prompt = run_edit.orig_encode_prompt
    bag["seconds"] = time.perf_counter() - started
    bag["peak_mem_mib"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    bag["image"] = result.images
    bag["nsfw"] = None if result.nsfw_content_detected is None else bool(result.nsfw_content_detected[0])
    return bag


def save_state(path, bag, raw):
    payload = {
        "raw_unscaled": raw,
        "scaled_start": bag["scaled"],
        "image_latents_cfg": bag["image_latents_cfg"],
        "prompt_embeds": bag["prompt_embeds"],
        "timesteps": bag["timesteps"],
    }
    for index, timestep, state in bag["states"]:
        payload[f"latent_after_step_{index}"] = state
        payload[f"timestep_after_step_{index}"] = torch.tensor(timestep)
    torch.save(payload, path)


def panel(images, labels, path):
    width, height = images[0].size
    label_h = 28
    sheet = Image.new("RGB", (width * len(images), height + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (image, label) in enumerate(zip(images, labels)):
        sheet.paste(image.convert("RGB"), (i * width, label_h))
        draw.text((i * width + 6, 6), f"{i}: {label}", fill="black")
    sheet.save(path)


def ddim_from_checkpoint(pipe):
    from diffusers import DDIMScheduler

    cfg = pipe.scheduler.config
    return DDIMScheduler(
        num_train_timesteps=int(cfg.num_train_timesteps),
        beta_start=float(cfg.beta_start),
        beta_end=float(cfg.beta_end),
        beta_schedule=str(cfg.beta_schedule),
        prediction_type=str(cfg.prediction_type),
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        timestep_spacing="leading",
    )


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    STATES.mkdir(parents=True, exist_ok=True)
    code_sha = sha256_file(Path(__file__).resolve())
    config_sha = sha256_file(CONFIG_PATH)
    key_path = WORKDIR / cfg["key_path"]
    encoding_key, decoding_key, message, _saved_codeword = load_key(key_path)
    found_hash = key_hash(encoding_key)
    if found_hash != cfg["expected_key_hash"]:
        (OUT / "STATUS.txt").write_text(f"FAIL\nkey hash {found_hash}\n")
        return 2
    generator_matrix, _otp, test_bits, g_dim, _noise = encoding_key
    capacity = int(generator_matrix.shape[1] - len(test_bits) - int(g_dim))
    if capacity != 64 or len(message) != 64:
        (OUT / "STATUS.txt").write_text(f"FAIL\ncapacity {capacity} message {len(message)}\n")
        return 2

    _, Encode, *_rest = prc_modules()
    codeword = Encode(encoding_key, message=message)
    save_key(OUT / "pair_encoding.pkl", encoding_key, decoding_key, message, codeword)
    rng = np.random.default_rng(int(cfg["gaussian_rng_seed"]))
    gaussian = rng.standard_normal(int(codeword.numel()))
    raw_gaussian = torch.from_numpy(gaussian).reshape(1, 4, 64, 64)
    raw_prc = torch.from_numpy(np.asarray(codeword.detach().cpu().numpy()).reshape(1, 4, 64, 64) * np.abs(gaussian.reshape(1, 4, 64, 64)))
    np.savez(
        OUT / "raw_noise.npz",
        gaussian=raw_gaussian.numpy(),
        z_prc=raw_prc.numpy(),
        codeword=codeword.detach().cpu().numpy(),
        message=np.asarray(message, dtype=np.int64),
    )

    source = load_edict_image(WORKDIR / cfg["source_photo"], size=int(cfg["preprocessed_size"]))
    source.save(IMAGES / "00_source.png")
    from diffusers import StableDiffusionInstructPix2PixPipeline

    dtype = torch.float16
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], torch_dtype=dtype
    )
    pipe = pipe.to("cuda")
    pipe.vae.config.scaling_factor = float(pipe.vae.config.scaling_factor)
    if pipe.unet.config.in_channels != 8 or pipe.vae.config.latent_channels != 4:
        (OUT / "STATUS.txt").write_text("FAIL\nunexpected channel counts\n")
        return 2
    run_edit.orig_prepare_latents = pipe.prepare_latents
    run_edit.orig_prepare_image_latents = pipe.prepare_image_latents
    run_edit.orig_encode_prompt = pipe._encode_prompt
    euler_scheduler = pipe.scheduler
    ddim_scheduler = ddim_from_checkpoint(pipe)
    gpu_name = torch.cuda.get_device_name(0)
    steps = int(cfg["num_inference_steps"])
    guidance = float(cfg["guidance_scale"])
    image_guidance = float(cfg["image_guidance_scale"])
    prompt = cfg["instruction"]

    specs = [
        ("euler_gaussian", euler_scheduler, raw_gaussian, 0.0, True),
        ("euler_prc", euler_scheduler, raw_prc, 0.0, True),
        ("ddim_gaussian", ddim_scheduler, raw_gaussian, 0.0, False),
    ]
    rows = []
    pil_images = [source]
    labels = ["source"]
    euler_pil = None
    for name, scheduler, raw, eta, ancestral in specs:
        pipe.scheduler = scheduler
        supplied = raw.to(device="cuda", dtype=dtype)
        fp16 = supplied.detach().to(torch.float32).cpu()
        if name == "euler_prc":
            signs = torch.sign(fp16).flatten()
            bits = as_float_tensor(codeword)
            mismatches = int((signs != bits).sum().item())
        else:
            mismatches = 0
        generator = None
        if ancestral:
            generator = torch.Generator(device="cuda").manual_seed(int(cfg["ancestral_generator_seed"]))
        print("run", name, scheduler.__class__.__name__, flush=True)
        bag = run_edit(pipe, source, prompt, supplied, steps, guidance, image_guidance, eta, generator)
        if "last" not in bag:
            raise RuntimeError(f"{name} saved no denoising state")
        final = bag["last"][2]
        stats = decode_clip_stats(pipe, final.to("cuda"))
        image = pt_to_pil(bag["image"])
        image.save(IMAGES / f"{name}.png")
        save_state(STATES / f"{name}.pt", bag, raw.to(torch.float32))
        torch.save(stats["float_decode"], STATES / f"{name}_float_decode.pt")
        pil_images.append(image)
        labels.append(name)
        if name == "euler_gaussian":
            euler_pil = image
        row = {
            "stage": "pair",
            "run": name,
            "receiver_level": "not_run",
            "status": "recorded",
            "failure_reason": "",
            "job_id": os_job(),
            "git_rev": git_rev(),
            "code_sha256": code_sha,
            "config_sha256": config_sha,
            "model_id": cfg["model_id"],
            "model_revision": cfg["model_revision"],
            "scheduler": scheduler.__class__.__name__,
            "num_inference_steps": steps,
            "eta": eta if scheduler.__class__.__name__.startswith("DDIM") else "",
            "guidance_scale": guidance,
            "image_guidance_scale": image_guidance,
            "dtype": "float16",
            "prediction_type": scheduler.config.prediction_type,
            "timestep_spacing": getattr(scheduler.config, "timestep_spacing", ""),
            "init_noise_sigma": bag["init_noise_sigma"],
            "has_sigmas": bag["has_sigmas"],
            "source": cfg["source_photo"],
            "instruction": prompt,
            "key_hash": found_hash,
            "message_sha256": sha256_array(np.asarray(message, dtype=np.uint8)),
            "codeword_sha256": sha256_array(codeword.detach().cpu().numpy()) if name == "euler_prc" else "",
            "n": 16384,
            "payload_length": 64,
            "key_message_capacity": capacity,
            "t": 3,
            "F": 1e-5,
            "noise_shape": "[1, 4, 64, 64]",
            "noise_scale": "raw standard gaussian or c*abs(g), then one multiply by init_noise_sigma",
            "fp16_sign_mismatches": mismatches,
            "pixel_mse_vs_source": pixel_mse(source, image),
            "frac_decode_below_m1": stats["frac_decode_below_m1"],
            "frac_decode_above_p1": stats["frac_decode_above_p1"],
            "frac_png_saturated": saturation_fraction(image),
            "nsfw": bag["nsfw"],
            "seconds": bag["seconds"],
            "peak_mem_mib": bag["peak_mem_mib"],
            "gpu_name": gpu_name,
            "vae_scaling_factor": float(pipe.vae.config.scaling_factor),
            "_image": image,
        }
        rows.append(row)
        print(" done", name, "sec", round(bag["seconds"], 1), "sigma", bag["init_noise_sigma"], flush=True)

    del pipe
    torch.cuda.empty_cache()
    model, processor = clip_bundle("cuda")
    for row in rows:
        image = row.pop("_image")
        lpips, lpips_err = lpips_distance(source, image, device="cuda")
        row["lpips_vs_source"] = lpips
        row["lpips_error"] = lpips_err
        row["clip_instruction"] = cosine(embed_image(model, processor, image, "cuda"), embed_text(model, processor, prompt, "cuda"))
        row["clip_edit_noun"] = cosine(embed_image(model, processor, image, "cuda"), embed_text(model, processor, cfg["edit_noun"], "cuda"))
        row["directional_clip"] = directional_clip(image, source, prompt, cfg["source_caption"], model, processor, "cuda")
        if row["run"] == "euler_prc" and euler_pil is not None:
            paired, paired_err = lpips_distance(euler_pil, image, device="cuda")
            row["paired_seed_lpips_vs_euler_gaussian"] = paired
            row["paired_seed_lpips_note"] = "paired-seed output difference, not watermark damage"
            row["paired_lpips_error"] = paired_err
        else:
            row["paired_seed_lpips_vs_euler_gaussian"] = ""
    panel(pil_images, labels, IMAGES / "panel.png")
    with (OUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "PASS",
        "panel_order": labels,
        "directional_clip_definition": (
            "cosine(CLIP_image(edit)-CLIP_image(source), CLIP_text(instruction)-CLIP_text(source_caption)); "
            "openai/clip-vit-large-patch14 pooled features; source_caption is the fixed string in pair.json"
        ),
        "paired_seed_note": "euler_prc and euler_gaussian share Gaussian magnitudes and the ancestral generator seed",
        "scheduler_probe_note": "ddim_gaussian is not a PRC edit and not an inversion",
        "vae_scaling_factor_loaded": rows[0]["vae_scaling_factor"],
        "rows": rows,
        "code_sha256": code_sha,
        "config_sha256": config_sha,
        "git_rev": git_rev(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (OUT / "STATUS.txt").write_text("PASS\n")
    print("PASS", gpu_name, flush=True)
    return 0


def os_job():
    import os
    return os.environ.get("SLURM_JOB_ID", "")


if __name__ == "__main__":
    sys.exit(main())
