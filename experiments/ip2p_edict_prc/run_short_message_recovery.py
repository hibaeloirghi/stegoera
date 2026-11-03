"""PNG message recovery for short PRC payloads.

The dog black-and-white trial at both lengths runs first. A failed true-pair
control stops the job. A failed PNG decode does not. Receiver 2 conditions on
the edited PNG, not on the source photograph.
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
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.image_utils import load_edict_image, tensor_to_pil  # noqa: E402
from experiments.ip2p_edict_prc.run_milestone import compare, decode, make_denoise  # noqa: E402
from experiments.ip2p_edict_prc.run_png_decoder_inversion import (  # noqa: E402
    invert_duplicated,
    mean_pixel_loss,
    png_to_model,
    recover_latent,
)
from experiments.ip2p_edict_prc.sampler import EdictSchedule, coupled_loop  # noqa: E402
from experiments.metrics import lpips_distance, pixel_mse  # noqa: E402
from experiments.paths import RESULTS  # noqa: E402
from experiments.prc_runtime import detection_margin, detect_decode, recover_soft_bits  # noqa: E402
from experiments.receiver_aligned.key_io import key_hash, load_key  # noqa: E402

OUT = RESULTS / "ip2p_edict_prc" / "short_message_recovery"
SIGMA_SQUARED = 1.5
FIELDS = [
    "image_id", "instruction_id", "instruction", "length", "trial", "watermark", "key_id",
    "receiver", "recovered_stream", "primary_endpoint",
    "detect", "statistic", "threshold", "margin",
    "decode_non_none", "exact_message_match", "payload_ber",
    "initial_noise_mse", "initial_noise_correlation",
    "codeword_sign_agreement", "codeword_sign_errors",
    "pixel_loss", "pixel_mse_vs_source", "lpips_vs_source",
    "mean_saturation", "low_saturation_fraction",
    "png_sha256", "conditioning_sha256", "sender_conditioning_sha256", "conditioning_from",
    "runtime_seconds", "inversion_seconds", "peak_memory_bytes", "status",
]


def claim(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite {path}") from exc
    os.close(fd)


def save_new(path: Path, write) -> None:
    claim(path)
    temporary = path.with_name(path.name + ".partial")
    write(temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(tensor) -> str:
    array = tensor.detach().to(dtype=torch.float64).cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def stem(image_id, instruction_id, length, trial) -> str:
    name = f"{image_id}__{instruction_id}__len{length}__trial{trial}"
    required = (str(image_id), str(instruction_id), f"len{length}", f"trial{trial}")
    if any(part not in name for part in required):
        raise RuntimeError(name)
    return name


def image_condition(pipe, image_path, instruction):
    """Same preprocessing and encoding API the sender uses for a source photo."""
    photograph = load_edict_image(image_path, size=512)
    processed = pipe.image_processor.preprocess(photograph)
    prompt_embeds = pipe._encode_prompt(instruction, "cuda", 1, True, negative_prompt="")
    image_latents = pipe.prepare_image_latents(
        processed, 1, 1, prompt_embeds.dtype, "cuda", True
    )
    return image_latents, prompt_embeds, tensor_sha(image_latents)


def forward_pair(denoise, schedule, noise, cfg):
    start = [noise.clone(), noise.clone()]
    final, _trace = coupled_loop(
        denoise, schedule, start, reverse=False,
        mix_weight=float(cfg["mix_weight"]), leapfrog=bool(cfg["leapfrog"]),
    )
    return [item.detach().to(dtype=torch.float64).cpu() for item in final]


def invert_true_pair(denoise, schedule, pair, cfg):
    started = time.perf_counter()
    recovered, _trace = coupled_loop(
        denoise, schedule,
        [pair[0].to(device="cuda", dtype=torch.float64), pair[1].to(device="cuda", dtype=torch.float64)],
        reverse=True, mix_weight=float(cfg["mix_weight"]), leapfrog=bool(cfg["leapfrog"]),
    )
    elapsed = time.perf_counter() - started
    return [item.detach().to(dtype=torch.float64).cpu() for item in recovered], elapsed


def saturation(path: Path):
    array = np.asarray(Image.open(path).convert("RGB")).astype(np.float64)
    high = array.max(axis=2)
    low = array.min(axis=2)
    sat = np.where(high > 0, (high - low) / np.maximum(high, 1.0), 0.0)
    return float(sat.mean()), float((sat < 0.08).mean())


def recover_generation_latent(pipe, vae_encode, vae_loss, png_path, expected_hash, recovery_config):
    """100-step image-only inversion. No source photo and no PRC arguments."""
    digest = sha256_file(png_path)
    if digest != expected_hash:
        raise RuntimeError(f"PNG hash changed before latent recovery: {png_path}")
    image = png_to_model(png_path)
    started = time.perf_counter()
    _z0, saved, trace, finite = recover_latent(image, vae_encode, vae_loss, recovery_config)
    elapsed = time.perf_counter() - started
    if not finite or 100 not in saved:
        raise RuntimeError(f"decoder inversion failed for {png_path}")
    latent = saved[100]
    loss = mean_pixel_loss(vae_loss, latent, image, float(recovery_config["scaling_factor"]))
    return latent, loss, trace, elapsed, digest


def receiver_source_known(pipe, schedule, latent_path, source_path, png_path, expected_hash, instruction, cfg):
    digest = sha256_file(png_path)
    if digest != expected_hash:
        raise RuntimeError("source-known receiver read a different PNG than the sender saved")
    latent = torch.load(latent_path, map_location="cpu")["z"]
    image_latents, prompt_embeds, conditioning_sha = image_condition(pipe, source_path, instruction)
    denoise = make_denoise(
        pipe.unet, image_latents, prompt_embeds,
        float(cfg["guidance_scale"]), float(cfg["image_guidance_scale"]),
    )
    streams, seconds = invert_duplicated(denoise, schedule, latent, cfg)
    return streams, seconds, conditioning_sha, digest, "original_source"


def receiver_source_unknown(pipe, schedule, latent_path, source_path, png_path, expected_hash, instruction, cfg):
    """Edited PNG supplies the image condition. This is not the sender's source."""
    if Path(png_path).resolve() == Path(source_path).resolve():
        raise RuntimeError("source-unknown receiver was pointed at the source photo")
    digest = sha256_file(png_path)
    if digest != expected_hash:
        raise RuntimeError("source-unknown receiver read a different PNG than the sender saved")
    latent = torch.load(latent_path, map_location="cpu")["z"]
    image_latents, prompt_embeds, conditioning_sha = image_condition(pipe, png_path, instruction)
    denoise = make_denoise(
        pipe.unet, image_latents, prompt_embeds,
        float(cfg["guidance_scale"]), float(cfg["image_guidance_scale"]),
    )
    streams, seconds = invert_duplicated(denoise, schedule, latent, cfg)
    return streams, seconds, conditioning_sha, digest, "received_png"


def diagnostic_source_denoise(pipe, schedule, source_path, instruction, cfg):
    image_latents, prompt_embeds, conditioning_sha = image_condition(pipe, source_path, instruction)
    denoise = make_denoise(
        pipe.unet, image_latents, prompt_embeds,
        float(cfg["guidance_scale"]), float(cfg["image_guidance_scale"]),
    )
    return denoise, conditioning_sha


def score_stream(latent, z_target, decoding_key, codeword, message):
    geometry = compare(latent, z_target)
    soft = recover_soft_bits(latent, variance=SIGMA_SQUARED)
    detected, decode_ok, decoded = detect_decode(decoding_key, soft)
    statistic, threshold, margin = detection_margin(decoding_key, soft)
    if message is None or decoded is None:
        exact = False
        ber = ""
    else:
        got = np.asarray(decoded, dtype=int).reshape(-1)
        sent = np.asarray(message, dtype=int).reshape(-1)
        if got.shape != sent.shape:
            exact = False
            ber = ""
        else:
            ber = float(np.mean(got != sent))
            exact = bool(ber == 0.0)
    if codeword is None:
        agreement, errors = "", ""
    else:
        code = torch.as_tensor(np.asarray(codeword), dtype=torch.float64).reshape(-1)
        signs = torch.sign(latent.detach().to(dtype=torch.float64).cpu().reshape(-1)) == code
        agreement = float(signs.to(dtype=torch.float64).mean().item())
        errors = int((~signs).sum().item())
    return {
        "detect": bool(detected),
        "statistic": statistic,
        "threshold": threshold,
        "margin": margin,
        "decode_non_none": bool(decode_ok),
        "exact_message_match": exact,
        "payload_ber": ber,
        "initial_noise_mse": geometry["mse"],
        "initial_noise_correlation": geometry["correlation"],
        "codeword_sign_agreement": agreement,
        "codeword_sign_errors": errors,
    }


def evaluate(held, z_target, decoding_key, key_ident, codeword, message, common):
    rows = []
    for receiver, streams, seconds, conditioning_sha, conditioning_from, pixel_loss, inversion_seconds, primary in held:
        for stream, latent in enumerate(streams):
            scored = score_stream(latent, z_target, decoding_key, codeword, message)
            rows.append({
                **common,
                "key_id": key_ident,
                "receiver": receiver,
                "recovered_stream": stream,
                "primary_endpoint": bool(primary and stream == 1),
                **scored,
                "pixel_loss": pixel_loss,
                "conditioning_sha256": conditioning_sha,
                "conditioning_from": conditioning_from,
                "runtime_seconds": seconds,
                "inversion_seconds": inversion_seconds,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "status": "ok",
            })
    return rows


def write_rows(path: Path, rows) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    save_new(path, write)


def save_tensor(path: Path, payload) -> None:
    save_new(path, lambda temporary: torch.save(payload, temporary))


def save_png(image: Image.Image, path: Path) -> str:
    save_new(path, lambda temporary: image.save(temporary, format="PNG"))
    return sha256_file(path)


def generate(pipe, schedule, source_path, instruction, noise, cfg, name):
    image_latents, prompt_embeds, conditioning_sha = image_condition(pipe, source_path, instruction)
    denoise = make_denoise(
        pipe.unet, image_latents, prompt_embeds,
        float(cfg["guidance_scale"]), float(cfg["image_guidance_scale"]),
    )
    final = forward_pair(denoise, schedule, noise.to(device="cuda", dtype=torch.float64), cfg)
    image_dir = OUT / "images"
    stream_paths = []
    hashes = []
    for stream, latent in enumerate(final):
        path = image_dir / f"{name}__stream{stream}.png"
        hashes.append(save_png(decode(pipe, latent.to(device="cuda")), path))
        stream_paths.append(path)
    observation_dir = OUT / "observations"
    save_tensor(observation_dir / f"{name}__sender_x.pt", {"final_x": final[0]})
    save_tensor(observation_dir / f"{name}__sender_y.pt", {"final_y": final[1]})
    return {
        "stream_paths": stream_paths,
        "png_sha256": hashes[0],
        "sender_conditioning_sha256": conditioning_sha,
    }


def run_receivers(pipe, vae_loss, schedule, source_path, instruction, cfg, recovery_config, name, png_path, expected_hash):
    latent, pixel_loss, trace, inversion_seconds, digest = recover_generation_latent(
        pipe, pipe.vae, vae_loss, png_path, expected_hash, recovery_config
    )
    latent_path = OUT / "observations" / f"{name}__recovered_latent.pt"
    save_tensor(latent_path, {"z": latent, "trace": trace, "png_sha256": digest})
    known = receiver_source_known(
        pipe, schedule, latent_path, source_path, png_path, expected_hash, instruction, cfg
    )
    unknown = receiver_source_unknown(
        pipe, schedule, latent_path, source_path, png_path, expected_hash, instruction, cfg
    )
    denoise, diagnostic_sha = diagnostic_source_denoise(pipe, schedule, source_path, instruction, cfg)
    pair = torch.load(OUT / "observations" / f"{name}__sender_x.pt", map_location="cpu")["final_x"]
    partner = torch.load(OUT / "observations" / f"{name}__sender_y.pt", map_location="cpu")["final_y"]
    pair_streams, pair_seconds = invert_true_pair(denoise, schedule, [pair, partner], cfg)
    exact_x = torch.load(OUT / "observations" / f"{name}__sender_x.pt", map_location="cpu")["final_x"]
    exact_streams, exact_seconds = invert_duplicated(denoise, schedule, exact_x, cfg)
    held = [
        ("diagnostic_final_pair", pair_streams, pair_seconds, diagnostic_sha, "original_source", "", "", False),
        ("diagnostic_exact_x", exact_streams, exact_seconds, diagnostic_sha, "original_source", "", "", False),
        ("source_known_png", known[0], known[1], known[2], known[4], pixel_loss, inversion_seconds, True),
        ("source_unknown_png", unknown[0], unknown[1], unknown[2], unknown[4], pixel_loss, inversion_seconds, True),
    ]
    if known[3] != expected_hash or unknown[3] != expected_hash:
        raise RuntimeError("receiver PNG hash does not match the sender hash")
    return held, pixel_loss


def quality(source_path, edited_path):
    source = load_edict_image(source_path, size=512)
    edited = Image.open(edited_path).convert("RGB")
    distance, note = lpips_distance(source, edited, device="cuda")
    mean_sat, low_fraction = saturation(edited_path)
    return {
        "pixel_mse_vs_source": pixel_mse(source, edited),
        "lpips_vs_source": "" if distance is None else distance,
        "lpips_note": note,
        "mean_saturation": mean_sat,
        "low_saturation_fraction": low_fraction,
    }


def work_items(cfg):
    items = []
    for trial in cfg["trials"]:
        for length in cfg["message_lengths"]:
            for image_id, photo in cfg["images"]:
                for instruction_id, instruction in cfg["instructions"].items():
                    items.append({
                        "image_id": image_id,
                        "photo": photo,
                        "instruction_id": instruction_id,
                        "instruction": instruction,
                        "length": int(length),
                        "trial": int(trial),
                    })
    pilot = cfg["pilot"]

    def is_pilot(item):
        return (
            item["image_id"] == pilot["image_id"]
            and item["instruction_id"] == pilot["instruction_id"]
            and item["trial"] == int(pilot["trial"])
        )

    return [item for item in items if is_pilot(item)] + [item for item in items if not is_pilot(item)]


def combine_tables(row_dir: Path) -> None:
    dest = OUT / "per_trial_results.csv"
    paths = sorted(row_dir.glob("*.csv"))

    def write_combined(temporary: Path) -> None:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for path in paths:
                with path.open() as source:
                    for row in csv.DictReader(source):
                        writer.writerow(row)
    save_new(dest, write_combined)
    grouped = {}
    with dest.open() as handle:
        for row in csv.DictReader(handle):
            if row["watermark"] != "prc":
                continue
            key = (
                row["image_id"], row["instruction_id"], row["length"],
                row["receiver"], row["recovered_stream"],
            )
            bucket = grouped.setdefault(key, [0, 0, 0])
            bucket[0] += 1
            bucket[1] += row["exact_message_match"] == "True"
            bucket[2] += row["detect"] == "True"

    def write_summary(temporary: Path) -> None:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "image_id", "instruction_id", "length", "receiver", "recovered_stream",
                "trials", "exact_message", "detect",
            ])
            writer.writeheader()
            for key, counts in sorted(grouped.items()):
                writer.writerow({
                    "image_id": key[0],
                    "instruction_id": key[1],
                    "length": key[2],
                    "receiver": key[3],
                    "recovered_stream": key[4],
                    "trials": counts[0],
                    "exact_message": counts[1],
                    "detect": counts[2],
                })
    save_new(OUT / "summary.csv", write_summary)


def write_panels(cfg):
    for image_id, photo in cfg["images"]:
        source = load_edict_image(cfg_photo(photo), size=512)
        for instruction_id in cfg["instructions"]:
            control_name = stem(image_id, instruction_id, "control", 0)
            gaussian = Image.open(OUT / "images" / f"{control_name}__stream0.png").convert("RGB")
            for length in cfg["message_lengths"]:
                for trial in cfg["trials"]:
                    name = stem(image_id, instruction_id, int(length), int(trial))
                    edited = Image.open(OUT / "images" / f"{name}__stream0.png").convert("RGB")
                    header = Image.new("RGB", (1536, 28), (255, 255, 255))
                    draw = ImageDraw.Draw(header)
                    draw.text((8, 6), "source", fill=(0, 0, 0))
                    draw.text((520, 6), "gaussian edit", fill=(0, 0, 0))
                    draw.text((1032, 6), "prc edit", fill=(0, 0, 0))
                    panel = Image.new("RGB", (1536, 540), (255, 255, 255))
                    panel.paste(header, (0, 0))
                    panel.paste(source, (0, 28))
                    panel.paste(gaussian, (512, 28))
                    panel.paste(edited, (1024, 28))
                    save_png(panel, OUT / "images" / f"{name}__panel.png")


def cfg_photo(photo):
    from experiments.paths import WORKDIR
    return WORKDIR / photo


def main():
    cfg = json.loads((OUT / "frozen_config.json").read_text())
    if (OUT / "STATUS_NOISE.txt").read_text().strip() != "PASS":
        raise RuntimeError("noise job did not pass")
    if float(cfg["sigma_squared"]) != SIGMA_SQUARED:
        raise RuntimeError("frozen sigma_squared is not 1.5")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    from diffusers import AutoencoderKL, StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], torch_dtype=torch.float16
    ).to("cuda")
    scale = float(pipe.vae.config.scaling_factor)
    vae_loss = AutoencoderKL.from_config(pipe.vae.config)
    vae_loss.load_state_dict({
        key: value.detach().float().cpu() for key, value in pipe.vae.state_dict().items()
    })
    vae_loss = vae_loss.to(device="cuda", dtype=torch.float32).eval()
    vae_loss.requires_grad_(False)
    schedule = EdictSchedule(int(cfg["steps"]), device="cpu", dtype=torch.float64)
    recovery_config = dict(cfg["decoder_inversion"])
    recovery_config["scaling_factor"] = scale
    print("gpu", torch.cuda.get_device_name(0), flush=True)
    row_dir = OUT / "rows"
    pilot_done = 0
    for item in work_items(cfg):
        name = stem(item["image_id"], item["instruction_id"], item["length"], item["trial"])
        print("prc", name, flush=True)
        source_path = cfg_photo(item["photo"])
        z_path = OUT / "trials" / f"length_{item['length']}_trial_{item['trial']}_z.npy"
        noise = torch.from_numpy(np.load(z_path))
        generated = generate(
            pipe, schedule, source_path, item["instruction"], noise.to(dtype=torch.float64), cfg, name
        )
        del noise
        held, _loss = run_receivers(
            pipe, vae_loss, schedule, source_path, item["instruction"], cfg, recovery_config,
            name, generated["stream_paths"][0], generated["png_sha256"],
        )
        edited_quality = quality(source_path, generated["stream_paths"][0])
        noise = torch.from_numpy(np.load(z_path)).to(dtype=torch.float64)
        secret = np.load(OUT / "trials" / f"length_{item['length']}_trial_{item['trial']}_secret.npz")
        enc, decoding_key, _probe, _probe_code = load_key(OUT / "keys" / f"length_{item['length']}.pkl")
        rows = evaluate(
            held,
            noise,
            decoding_key,
            key_hash(enc),
            secret["codeword"],
            secret["message"],
            {
                "image_id": item["image_id"],
                "instruction_id": item["instruction_id"],
                "instruction": item["instruction"],
                "length": item["length"],
                "trial": item["trial"],
                "watermark": "prc",
                "png_sha256": generated["png_sha256"],
                "sender_conditioning_sha256": generated["sender_conditioning_sha256"],
                **{key: edited_quality[key] for key in (
                    "pixel_mse_vs_source", "lpips_vs_source", "mean_saturation", "low_saturation_fraction"
                )},
            },
        )
        write_rows(row_dir / f"{name}.csv", rows)
        pair_rows = [row for row in rows if row["receiver"] == "diagnostic_final_pair"]
        failed = any(
            (not row["exact_message_match"]) or float(row["initial_noise_correlation"]) < 0.99
            for row in pair_rows
        )
        if failed:
            (OUT / "STATUS.txt").write_text("FAIL pair control\n")
            print("FAIL pair control", name, flush=True)
            return 2
        print("pair control ok", name, flush=True)
        pilot = cfg["pilot"]
        if item["image_id"] == pilot["image_id"] and item["instruction_id"] == pilot["instruction_id"] and item["trial"] == int(pilot["trial"]):
            pilot_done += 1
            if pilot_done == len(cfg["message_lengths"]):
                print("PILOT_TECHNICAL_PASS", flush=True)
        del noise, secret, generated, held

    for image_id, photo in cfg["images"]:
        for instruction_id, instruction in cfg["instructions"].items():
            name = stem(image_id, instruction_id, "control", 0)
            print("gaussian", name, flush=True)
            source_path = cfg_photo(photo)
            z_path = OUT / "controls" / f"{name}_z.npy"
            noise = torch.from_numpy(np.load(z_path)).to(dtype=torch.float64)
            generated = generate(pipe, schedule, source_path, instruction, noise, cfg, name)
            del noise
            held, _loss = run_receivers(
                pipe, vae_loss, schedule, source_path, instruction, cfg, recovery_config,
                name, generated["stream_paths"][0], generated["png_sha256"],
            )
            edited_quality = quality(source_path, generated["stream_paths"][0])
            noise = torch.from_numpy(np.load(z_path)).to(dtype=torch.float64)
            for length in cfg["message_lengths"]:
                enc, decoding_key, _probe, _probe_code = load_key(OUT / "keys" / f"length_{length}.pkl")
                rows = evaluate(
                    held,
                    noise,
                    decoding_key,
                    key_hash(enc),
                    None,
                    None,
                    {
                        "image_id": image_id,
                        "instruction_id": instruction_id,
                        "instruction": instruction,
                        "length": int(length),
                        "trial": "control",
                        "watermark": "gaussian",
                        "png_sha256": generated["png_sha256"],
                        "sender_conditioning_sha256": generated["sender_conditioning_sha256"],
                        **{key: edited_quality[key] for key in (
                            "pixel_mse_vs_source", "lpips_vs_source", "mean_saturation", "low_saturation_fraction"
                        )},
                    },
                )
                write_rows(row_dir / f"{name}__key{length}.csv", rows)
            del noise, generated, held

    write_panels(cfg)
    combine_tables(row_dir)
    manifest = {
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "gpu": torch.cuda.get_device_name(0),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "primary_receiver_stream": 1,
        "primary_iteration": 100,
        "sigma_squared": SIGMA_SQUARED,
    }
    save_new(OUT / "manifest.json", lambda temporary: temporary.write_text(json.dumps(manifest, indent=2) + "\n"))
    (OUT / "STATUS.txt").write_text("PASS\n")
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
