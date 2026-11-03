"""Exact red-ball final x, duplicated, through the matched EDICT inverse.

recover_initial receives only x. The saved partner y, the original noise,
the codeword, and the message are loaded later by evaluate_recovered.
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

from experiments.ip2p_edict_prc.run_milestone import compare, make_denoise  # noqa: E402
from experiments.ip2p_edict_prc.sampler import EdictSchedule, coupled_loop  # noqa: E402
from experiments.paths import RESULTS, WORKDIR  # noqa: E402
from experiments.prc_runtime import detection_margin, detect_decode, recover_soft_bits  # noqa: E402
from experiments.receiver_aligned.key_io import key_hash, load_key  # noqa: E402

CONFIG_PATH = ROOT / "experiments" / "configs" / "ip2p_edict_prc" / "red_ball_exact_x.json"
OUT = RESULTS / "ip2p_edict_prc" / "red_ball_exact_x"

FIELDS = [
    "stream", "detect", "statistic", "threshold", "margin",
    "decode_non_none", "exact_message_match",
    "initial_noise_mse", "initial_noise_correlation",
    "codeword_sign_agreement", "codeword_sign_errors",
    "sigma", "sigma_squared", "wrong_key_detect", "wrong_key_decode_non_none",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(reason: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STATUS.txt").write_text("FAIL\n" + reason + "\n")
    print("FAIL:", reason, flush=True)
    return 2


def load_receiver_x(path: Path):
    """Return x and the saved conditioning. Do not return y."""
    saved = torch.load(path, map_location="cpu")
    return {
        "x": saved["prc_final"][0].to(dtype=torch.float64),
        "image_latents": saved["image_latents"],
        "prompt_embeds": saved["prompt_embeds"],
        "timesteps": saved["timesteps"].to(dtype=torch.float64),
    }


def recover_initial(denoise, schedule, x, cfg):
    """Matched EDICT inverse of (x, x). No y, noise, codeword, or message."""
    pair = [x.detach().to(dtype=torch.float64).clone(), x.detach().to(dtype=torch.float64).clone()]
    started = time.perf_counter()
    recovered, _trace = coupled_loop(
        denoise, schedule, pair, reverse=True,
        mix_weight=float(cfg["mix_weight"]), leapfrog=bool(cfg["leapfrog"]),
    )
    elapsed = time.perf_counter() - started
    return [item.detach().to(dtype=torch.float64).cpu() for item in recovered], elapsed


def evaluate_recovered(recovered, cfg):
    """Score after the inverse. This is the only place the key and noise are loaded."""
    saved = torch.load(WORKDIR / cfg["saved_pairs"], map_location="cpu")
    final_x = saved["prc_final"][0].to(dtype=torch.float64)
    final_y = saved["prc_final"][1].to(dtype=torch.float64)
    final_gap = final_x - final_y
    _enc, decoding_key, message, key_codeword = load_key(WORKDIR / cfg["key_path"])
    if key_hash(_enc) != cfg["expected_key_hash"]:
        raise RuntimeError(f"key hash {key_hash(_enc)}")
    noise_file = np.load(WORKDIR / cfg["saved_noise"])
    codeword = np.asarray(noise_file["codeword"]).reshape(-1)
    if not np.array_equal(np.asarray(key_codeword).reshape(-1), codeword):
        raise RuntimeError("saved codeword does not match the PRC noise")
    if not np.array_equal(np.asarray(message, dtype=int).reshape(-1), np.asarray(noise_file["message"], dtype=int)):
        raise RuntimeError("saved message does not match the key")
    z_prc = torch.from_numpy(np.asarray(noise_file["z_prc"])).to(dtype=torch.float64)
    _wrong_enc, wrong_key, _wm, _wc = load_key(WORKDIR / cfg["wrong_key_path"])
    sigma_squared = float(cfg["sigma_squared"])
    rows = []
    for stream, latent in enumerate(recovered):
        geometry = compare(latent, z_prc)
        soft = recover_soft_bits(latent, variance=sigma_squared)
        detected, decode_ok, decoded = detect_decode(decoding_key, soft)
        statistic, threshold, margin = detection_margin(decoding_key, soft)
        sent = np.asarray(message, dtype=int).reshape(-1)
        if decoded is None:
            exact = False
        else:
            got = np.asarray(decoded, dtype=int).reshape(-1)
            exact = bool(got.size == sent.size and np.array_equal(got, sent))
        code = torch.as_tensor(codeword, dtype=torch.float64).flatten()
        flat = latent.detach().to(torch.float64).flatten().cpu()
        code_signs = torch.sign(flat) == code
        wrong_detect, wrong_decode, _wrong = detect_decode(wrong_key, soft)
        rows.append({
            "stream": stream,
            "detect": detected,
            "statistic": statistic,
            "threshold": threshold,
            "margin": margin,
            "decode_non_none": decode_ok,
            "exact_message_match": exact,
            "initial_noise_mse": geometry["mse"],
            "initial_noise_correlation": geometry["correlation"],
            "codeword_sign_agreement": float(code_signs.to(torch.float64).mean().item()),
            "codeword_sign_errors": int((~code_signs).sum().item()),
            "sigma": float(np.sqrt(sigma_squared)),
            "sigma_squared": sigma_squared,
            "wrong_key_detect": wrong_detect,
            "wrong_key_decode_non_none": wrong_decode,
        })
    disagreement = recovered[0] - recovered[1]
    summary = {
        "saved_final_x_minus_y_mse": float(final_gap.pow(2).mean().item()),
        "saved_final_x_minus_y_max_abs": float(final_gap.abs().max().item()),
        "recovered_stream_disagreement_mse": float(disagreement.pow(2).mean().item()),
        "recovered_stream_disagreement_max_abs": float(disagreement.abs().max().item()),
        "message_sha256": hashlib.sha256(np.asarray(message, dtype=np.uint8).tobytes()).hexdigest(),
        "codeword_sha256": hashlib.sha256(np.asarray(codeword, dtype=np.float64).tobytes()).hexdigest(),
    }
    return rows, summary


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    receiver = load_receiver_x(WORKDIR / cfg["saved_pairs"])
    schedule = EdictSchedule(int(cfg["num_inference_steps"]), device="cpu", dtype=torch.float64)
    if not torch.equal(schedule.timesteps.cpu(), receiver["timesteps"].cpu()):
        return fail("saved timesteps do not match the EDICT schedule")

    from diffusers import StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], torch_dtype=torch.float16
    ).to("cuda")
    image_latents = receiver["image_latents"].to(device="cuda", dtype=torch.float16)
    prompt_embeds = receiver["prompt_embeds"].to(device="cuda", dtype=torch.float16)
    denoise = make_denoise(
        pipe.unet, image_latents, prompt_embeds,
        float(cfg["guidance_scale"]), float(cfg["image_guidance_scale"]),
    )
    print("gpu", torch.cuda.get_device_name(0), flush=True)
    recovered, elapsed = recover_initial(denoise, schedule, receiver["x"].to(device="cuda"), cfg)
    torch.save(
        {
            "stream0": recovered[0],
            "stream1": recovered[1],
            "stream0_minus_stream1": recovered[0] - recovered[1],
        },
        OUT / "recovered_initial_observations.pt",
    )
    del pipe
    rows, summary = evaluate_recovered(recovered, cfg)
    with (OUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "receiver_level": "source-known exact final x duplicated as (x, x); not blind recovery",
        "question": "Does replacing the unknown final y with a copy of exact final x prevent PRC.Decode?",
        "saved_pairs": cfg["saved_pairs"],
        "pairs_sha256": sha256_file(WORKDIR / cfg["saved_pairs"]),
        "sampler_sha256": sha256_file(ROOT / "experiments/ip2p_edict_prc/sampler.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "instruction": cfg["instruction"],
        "key_id": cfg["expected_key_hash"],
        "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "timestep_grid": "EDICT arange 980, 960, ..., 0",
        "mix_weight": cfg["mix_weight"],
        "leapfrog": cfg["leapfrog"],
        "guidance_scale": cfg["guidance_scale"],
        "image_guidance_scale": cfg["image_guidance_scale"],
        "pair_dtype": "float64",
        "unet_dtype": "float16",
        "sigma_squared": cfg["sigma_squared"],
        "sigma": float(np.sqrt(float(cfg["sigma_squared"]))),
        "soft_observation": "pseudogaussians.recover_posteriors variances=sigma^2",
        "inverse_seconds": elapsed,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "summary": summary,
        "rows": rows,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (OUT / "STATUS.txt").write_text("PASS\n")
    for row in rows:
        print(
            "stream", row["stream"],
            "detect", row["detect"],
            "decode", row["decode_non_none"],
            "exact", row["exact_message_match"],
            "corr", round(row["initial_noise_correlation"], 4),
            flush=True,
        )
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        sys.exit(fail(str(exc)))
