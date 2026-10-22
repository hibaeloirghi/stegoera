"""Source-known single-stream tests on the saved golden-retriever edit.

Test 2 inverts the exact final x duplicated as (x, x).
Test 3 reloads the saved PNG, encodes it with the VAE, duplicates that
latent, and inverts that pair.

Neither inverse receives the saved y or the original initial noise.
The noise and the codeword are loaded only after the inverse, for scoring.
The source conditioning is the conditioning saved with the forward pass.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.image_utils import tensor_to_pil  # noqa: E402
from experiments.ip2p_edict_prc.run_milestone import compare, make_denoise, score_prc  # noqa: E402
from experiments.ip2p_edict_prc.sampler import EdictSchedule, coupled_loop  # noqa: E402
from experiments.paths import RESULTS, WORKDIR  # noqa: E402
from experiments.receiver_aligned.key_io import key_hash, load_key  # noqa: E402

CONFIG_PATH = ROOT / "experiments" / "configs" / "ip2p_edict_prc" / "single_image.json"
OUT = RESULTS / "ip2p_edict_prc" / "single_image"

FIELDS = [
    "test", "stream",
    "noise_mse", "noise_max_abs", "noise_correlation",
    "noise_sign_agreement", "noise_sign_errors",
    "codeword_sign_agreement", "codeword_sign_errors",
    "detect", "decode_non_none", "exact_message_match", "margin",
    "wrong_key_detect",
]


def fail(reason: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STATUS.txt").write_text("FAIL\n" + reason + "\n")
    print("FAIL:", reason, flush=True)
    return 2


def invert_duplicated(denoise, schedule, x, cfg):
    """Matched inverse of (x, x). The other argument must not be y or the initial noise."""
    pair = [x.detach().to(dtype=torch.float64).clone(), x.detach().to(dtype=torch.float64).clone()]
    recovered, _trace = coupled_loop(
        denoise, schedule, pair, reverse=True,
        mix_weight=float(cfg["mix_weight"]), leapfrog=bool(cfg["leapfrog"]),
    )
    return recovered


def save_endpoints(name, recovered):
    x_end = recovered[0].detach().to(torch.float64).cpu()
    y_end = recovered[1].detach().to(torch.float64).cpu()
    difference = x_end - y_end
    torch.save(
        {"x": x_end, "y": y_end, "x_minus_y": difference},
        OUT / f"{name}_endpoints.pt",
    )
    return {
        "endpoint_x_minus_y_mse": float(difference.pow(2).mean().item()),
        "endpoint_x_minus_y_max_abs": float(difference.abs().max().item()),
    }


def latent_from_png(pipe, path):
    """Reload an 8-bit PNG and map it back to a diffusion latent.

    The forward save was vae.decode(z / scaling_factor), then 8-bit quantization.
    This uses the posterior mode times scaling_factor. It is not an optimization
    of the decoder, and it is not an exact inverse of decode.
    """
    image = Image.open(path).convert("RGB")
    if image.size != (512, 512):
        raise RuntimeError(f"PNG size {image.size}")
    array = np.asarray(image).astype(np.float32) / 255.0
    array = array * 2.0 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device="cuda", dtype=pipe.vae.dtype)
    mode = pipe.vae.encode(tensor).latent_dist.mode()
    scale = float(pipe.vae.config.scaling_factor)
    return (mode * scale).to(dtype=torch.float64)


def score_stream(test, stream, recovered, z_prc, decoding_key, wrong_key, codeword, message, sigma_squared):
    geometry = compare(recovered, z_prc)
    prc = score_prc(decoding_key, wrong_key, recovered, codeword, message, sigma_squared)
    return {
        "test": test,
        "stream": stream,
        "noise_mse": geometry["mse"],
        "noise_max_abs": geometry["max_abs"],
        "noise_correlation": geometry["correlation"],
        "noise_sign_agreement": geometry["sign_agreement"],
        "noise_sign_errors": geometry["sign_errors"],
        "codeword_sign_agreement": prc["sign_agreement"],
        "codeword_sign_errors": prc["sign_errors"],
        "detect": prc["detect"],
        "decode_non_none": prc["decode_non_none"],
        "exact_message_match": prc["exact_message_match"],
        "margin": prc["margin"],
        "wrong_key_detect": prc["wrong_key_detect"],
    }


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    saved = torch.load(WORKDIR / cfg["saved_pairs"], map_location="cpu")
    true_x = saved["prc_final"][0]
    true_y = saved["prc_final"][1]
    forward_difference = (true_x - true_y).to(torch.float64)
    torch.save(
        {
            "x": true_x.to(torch.float64),
            "y": true_y.to(torch.float64),
            "x_minus_y": forward_difference,
            "role": "saved forward terminal pair; not an input to tests 2 or 3",
        },
        OUT / "forward_terminal_endpoints.pt",
    )
    # From here, y is not passed to an inverse.
    del true_y

    schedule = EdictSchedule(int(cfg["num_inference_steps"]), device="cpu", dtype=torch.float64)
    if not torch.equal(schedule.timesteps.cpu(), saved["timesteps"].to(torch.float64).cpu()):
        return fail("saved timesteps do not match the EDICT schedule")

    from diffusers import StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], torch_dtype=torch.float16
    ).to("cuda")
    image_latents = saved["image_latents"].to(device="cuda", dtype=torch.float16)
    prompt_embeds = saved["prompt_embeds"].to(device="cuda", dtype=torch.float16)
    denoise = make_denoise(
        pipe.unet, image_latents, prompt_embeds,
        float(cfg["guidance_scale"]), float(cfg["image_guidance_scale"]),
    )
    print("gpu", torch.cuda.get_device_name(0), flush=True)

    held = {}
    summaries = {
        "forward_terminal_x_minus_y_mse": float(forward_difference.pow(2).mean().item()),
        "forward_terminal_x_minus_y_max_abs": float(forward_difference.abs().max().item()),
    }

    print("test2 exact x duplicated", flush=True)
    held["exact_x_duplicated"] = [
        item.detach().cpu() for item in invert_duplicated(denoise, schedule, true_x.to(device="cuda"), cfg)
    ]
    summaries["test2_endpoints"] = save_endpoints("test2_exact_x", held["exact_x_duplicated"])

    print("test3 png vae duplicated", flush=True)
    encoded = latent_from_png(pipe, WORKDIR / cfg["saved_png"])
    vae_gap = compare(encoded, true_x.to(device="cuda"))
    summaries["test3_png_vae_vs_true_x"] = vae_gap
    roundtrip = pipe.vae.decode(
        (encoded.to(device="cuda", dtype=pipe.vae.dtype) / float(pipe.vae.config.scaling_factor))
    ).sample
    tensor_to_pil(roundtrip).save(OUT / "test3_vae_roundtrip.png")
    held["png_vae_duplicated"] = [
        item.detach().cpu() for item in invert_duplicated(denoise, schedule, encoded, cfg)
    ]
    summaries["test3_endpoints"] = save_endpoints("test3_png_vae", held["png_vae_duplicated"])

    # Scoring starts only after both inverses have returned.
    _enc, decoding_key, message, key_codeword = load_key(WORKDIR / cfg["key_path"])
    if key_hash(_enc) != cfg["expected_key_hash"]:
        return fail(f"key hash {key_hash(_enc)}")
    noise_file = np.load(WORKDIR / cfg["saved_noise"])
    codeword = np.asarray(noise_file["codeword"]).reshape(-1)
    if not np.array_equal(np.asarray(key_codeword).reshape(-1), codeword):
        return fail("saved codeword does not match the PRC noise")
    z_prc = torch.from_numpy(np.asarray(noise_file["z_prc"])).to(dtype=torch.float64)
    _wrong_enc, wrong_dec, _wm, _wc = load_key(WORKDIR / cfg["wrong_key_path"])
    rows = []
    for test_name, recovered in held.items():
        for stream, latent in enumerate(recovered):
            row = score_stream(
                test_name, stream, latent, z_prc,
                decoding_key, wrong_dec, codeword, message, float(cfg["sigma_squared"]),
            )
            rows.append(row)
            print(test_name, stream, "corr", row["noise_correlation"], "exact", row["exact_message_match"], flush=True)

    with (OUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "receiver_level": "source-known single stream, not blind recovery",
        "test1": "actual final pair already recorded in results/ip2p_edict_prc/milestone/results.csv; not rerun",
        "test2": "exact saved final x duplicated to (x, x); saved y was not an inverse input",
        "test3": "prc_stream0.png reloaded, VAE posterior mode times scaling_factor, duplicated; saved y was not an inverse input",
        "scoring": "original noise and codeword are used only after the inverse returns",
        "conditioning": "image latents and instruction embeddings saved by the golden-retriever forward",
        "instruction": cfg["instruction"],
        "vae_scaling_factor": float(pipe.vae.config.scaling_factor),
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "gpu": torch.cuda.get_device_name(0),
        "summaries": summaries,
        "rows": rows,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (OUT / "STATUS.txt").write_text("PASS\n")
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
