"""EDICT at s=1 with Gunn's PRC sign embedding, one results file.

Five photos from EDICT/experiment_images. Each photo is inverted with a null
prompt or a caption of the photo, then both latents are signed to a PRC
codeword (z = c * |z|, magnitudes unchanged). Generation then uses a null
prompt, the caption, a small edit, or a stylistic edit.

The receiver inverts the saved PNG with the generation prompt. An oracle row
inverts the exact latent pair, before the VAE, with the same prompt.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.edict_runtime as er  # noqa: E402
import experiments.prc_runtime as pr  # noqa: E402
from experiments.image_utils import load_edict_image  # noqa: E402
from experiments.metrics import lpips_distance, pixel_mse  # noqa: E402
from experiments.receiver_aligned.prc_scores import exact_message_match  # noqa: E402

STEPS = 50
STRENGTH = 1.0
MIX = 0.93
GUIDANCE = 3.0
SEED = 1
FPR = 1e-5
PRC_T = 3
MESSAGE_LENGTH = 64
VARIANCE = 1.5
OUT = ROOT / "results" / "full_prc_sign"
IMAGES_DIR = OUT / "images"

# Photos shipped with the EDICT repo. Captions describe the file on disk.
IMAGES = [
    {
        "id": "imagenet_dog_1",
        "path": "EDICT/experiment_images/imagenet_dog_1.jpg",
        "caption": "a brown and white spaniel standing on a dirt path in front of green bushes",
        "small": "a brown and white spaniel wearing a red collar, standing on a dirt path in front of green bushes",
        "style": "an oil painting of a brown and white spaniel standing on a dirt path in front of green bushes",
    },
    {
        "id": "imagenet_cat",
        "path": "EDICT/experiment_images/imagenet_cat.jpg",
        "caption": "a white and grey cat with a collar and bell, sitting indoors against a wooden cabinet",
        "small": "a white and grey cat with a red collar and bell, sitting indoors against a wooden cabinet",
        "style": "a watercolor painting of a white and grey cat with a collar and bell, sitting indoors",
    },
    {
        "id": "church",
        "path": "EDICT/experiment_images/church.jpg",
        "caption": "a tall concrete church with a stepped spire under a cloudy sky, cars and grass in front",
        "small": "a tall concrete church with a stepped spire under a cloudy sky, with a red car in front",
        "style": "a tall concrete church with a stepped spire at night, lit by warm lights",
    },
    {
        "id": "fountain",
        "path": "EDICT/experiment_images/fountain.jpg",
        "caption": "a turquoise ornamental fountain in a white basin, in a park with trees and a dirt path",
        "small": "a turquoise ornamental fountain in a white basin, with a person sitting on a nearby bench",
        "style": "a turquoise ornamental fountain in a snowy park, trees covered in snow",
    },
    {
        "id": "imagenet_lake",
        "path": "EDICT/experiment_images/imagenet_lake.jpg",
        "caption": "a calm lake with a forested shoreline of green and autumn trees under a cloudy sky",
        "small": "a calm lake with a small red boat and a forested shoreline under a cloudy sky",
        "style": "a calm lake with a forested shoreline, painted in the style of Van Gogh",
    },
]
GUIDES = ("null", "caption")
TARGETS = ("null", "caption", "small", "style")

FIELDS = [
    "image_id", "guide", "target", "watermark", "invert_prompt", "generate_prompt",
    "steps", "strength", "mix_weight", "guidance", "t_final", "n_steps",
    "channel", "sign_agreement_x", "sign_agreement_y",
    "detect", "decode_ok", "exact_message", "message_ber", "margin",
    "wrong_key_detect", "wrong_key_margin",
    "pixel_mse_vs_original", "psnr_vs_original", "lpips_vs_original", "seconds",
]


def prompt_for(image: dict, name: str) -> str:
    if name == "null":
        return ""
    return image[name]


def sign_to_codeword(latent: torch.Tensor, codeword: torch.Tensor) -> torch.Tensor:
    """Gunn embedding: keep |z_i|, replace the sign with the codeword bit."""
    c = codeword.reshape(1, 4, 64, 64).to(device=latent.device, dtype=latent.dtype)
    return c * latent.abs()


def agreement(latent: torch.Tensor, codeword: torch.Tensor) -> float:
    signs = torch.sign(latent.detach().to(torch.float64).reshape(-1))
    bits = codeword.detach().to(device=signs.device, dtype=torch.float64).reshape(-1)
    return float((signs == bits).to(torch.float64).mean().item())


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(4.0 / mse))


def message_ber(payload, message) -> float | None:
    if payload is None:
        return None
    got = np.asarray(payload).reshape(-1)[: len(message)]
    return float(np.mean(got != np.asarray(message)))


def score_latent(latent_x, latent_y, codeword, decoding_key, wrong_decoding_key, message) -> dict:
    soft = pr.recover_soft_bits(latent_x.detach().to(torch.float64).cpu().reshape(-1), variance=VARIANCE)
    _stat, _thr, margin = pr.detection_margin(decoding_key, soft)
    detected, _ok, payload = pr.detect_decode(decoding_key, soft)
    _wstat, _wthr, wrong_margin = pr.detection_margin(wrong_decoding_key, soft)
    wrong_detected, _wok, _wpayload = pr.detect_decode(wrong_decoding_key, soft)
    return {
        "sign_agreement_x": agreement(latent_x, codeword),
        "sign_agreement_y": agreement(latent_y, codeword),
        "detect": bool(detected),
        "decode_ok": payload is not None,
        "exact_message": exact_message_match(payload, message),
        "message_ber": message_ber(payload, message),
        "margin": margin,
        "wrong_key_detect": bool(wrong_detected),
        "wrong_key_margin": wrong_margin,
    }


def edict_kwargs(prompt: str) -> dict:
    return dict(
        prompt=prompt,
        steps=STEPS,
        mix_weight=MIX,
        init_image_strength=STRENGTH,
        guidance_scale=GUIDANCE,
        seed=SEED,
        null_prompt="",
    )


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in FIELDS})
        f.flush()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "all_experiments.csv"
    if csv_path.exists():
        csv_path.unlink()

    print("loading EDICT", flush=True)
    ef = er.load_models("cuda")
    n = 4 * 64 * 64
    message = pr.bits_from_message_length(MESSAGE_LENGTH, rng_seed=SEED)
    encoding_key, decoding_key = pr.keygen(n, message_length=512, fpr=FPR, t=PRC_T, seed=SEED)
    wrong_encoding_key, wrong_decoding_key = pr.keygen(n, message_length=512, fpr=FPR, t=PRC_T, seed=SEED + 1)
    del wrong_encoding_key
    codewords = {}
    for image in IMAGES:
        codeword, _ = pr.encode_codeword(encoding_key, message)
        codewords[image["id"]] = codeword
    with (OUT / "key.pkl").open("wb") as f:
        pickle.dump(
            {
                "encoding_key": encoding_key,
                "decoding_key": decoding_key,
                "wrong_decoding_key": wrong_decoding_key,
                "message": message,
                "codewords": {k: v.detach().cpu() for k, v in codewords.items()},
                "variance": VARIANCE,
                "steps": STEPS,
                "strength": STRENGTH,
                "mix_weight": MIX,
            },
            f,
        )
    meta = {
        "images": IMAGES,
        "guides": GUIDES,
        "targets": TARGETS,
        "embedding": "z_i = c_i * |z_i| on both EDICT latents after full inversion",
    }
    (OUT / "grid.json").write_text(json.dumps(meta, indent=2))

    for image in IMAGES:
        photo = load_edict_image(ROOT / image["path"])
        codeword = codewords[image["id"]]
        for guide in GUIDES:
            invert_prompt = prompt_for(image, guide)
            print(f"INVERT {image['id']} guide={guide}", flush=True)
            t0 = time.time()
            inv = er.invert_image(ef, photo, **edict_kwargs(invert_prompt))
            print(
                f"  inverted in {time.time() - t0:.0f}s  t_final={inv['t_final']} n={len(inv['timesteps'])}",
                flush=True,
            )
            x, y = inv["latent_pair"]
            signed = [sign_to_codeword(x, codeword), sign_to_codeword(y, codeword)]
            print(
                f"  signed agreement x={agreement(signed[0], codeword):.4f} y={agreement(signed[1], codeword):.4f}",
                flush=True,
            )
            for target in TARGETS:
                generate_prompt = prompt_for(image, target)
                for watermark, pair in (("prc", signed), ("nowm", [x, y])):
                    print(f"  GEN {image['id']} {guide}->{target} {watermark}", flush=True)
                    t1 = time.time()
                    gen = er.generate_from_pair(ef, pair, **edict_kwargs(generate_prompt))
                    pil = er.decode_latent(ef, gen["x"])
                    image_path = IMAGES_DIR / f"{image['id']}__{guide}__{target}__{watermark}.png"
                    pil.save(image_path)
                    mse = pixel_mse(pil, photo)
                    lpips, lpips_err = lpips_distance(pil, photo, device="cuda")
                    if lpips_err:
                        print("  lpips", lpips_err, flush=True)
                    base = {
                        "image_id": image["id"],
                        "guide": guide,
                        "target": target,
                        "watermark": watermark,
                        "invert_prompt": invert_prompt,
                        "generate_prompt": generate_prompt,
                        "steps": STEPS,
                        "strength": STRENGTH,
                        "mix_weight": MIX,
                        "guidance": GUIDANCE,
                        "t_final": int(inv["t_final"]),
                        "n_steps": len(inv["timesteps"]),
                        "pixel_mse_vs_original": mse,
                        "psnr_vs_original": psnr_from_mse(mse),
                        "lpips_vs_original": lpips,
                    }
                    channels = []
                    if watermark == "prc":
                        kw = edict_kwargs(generate_prompt)
                        oracle = er.coupled_loop(
                            ef,
                            kw.pop("prompt"),
                            reverse=True,
                            init_latent=gen["latent_pair"],
                            **kw,
                        )
                        channels.append(("oracle_latent", oracle["latent_pair"]))
                    receive_png = watermark == "prc" or (guide == "null" and target in ("null", "style"))
                    if receive_png:
                        rec = er.invert_image(
                            ef, Image.open(image_path).convert("RGB"), **edict_kwargs(generate_prompt)
                        )
                        channels.append(("png", rec["latent_pair"]))
                    if not channels:
                        row = dict(base)
                        row.update({
                            "channel": "image_only",
                            "sign_agreement_x": "",
                            "sign_agreement_y": "",
                            "detect": "",
                            "decode_ok": "",
                            "exact_message": "",
                            "message_ber": "",
                            "margin": "",
                            "wrong_key_detect": "",
                            "wrong_key_margin": "",
                            "seconds": round(time.time() - t1, 1),
                        })
                        append_row(csv_path, row)
                        print(f"    image_only lpips={lpips} mse={mse:.4f}", flush=True)
                        continue
                    for channel, recovered in channels:
                        scores = score_latent(
                            recovered[0], recovered[1], codeword, decoding_key, wrong_decoding_key, message
                        )
                        row = dict(base)
                        row.update(scores)
                        row["channel"] = channel
                        row["seconds"] = round(time.time() - t1, 1)
                        append_row(csv_path, row)
                        print(
                            f"    {channel} agree_x={scores['sign_agreement_x']:.4f} "
                            f"detect={scores['detect']} exact={scores['exact_message']} "
                            f"ber={scores['message_ber']} margin={scores['margin']}",
                            flush=True,
                        )
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
