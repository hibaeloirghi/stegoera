"""CPU positive controls for InstructPix2Pix PRC noise.

Confirms the checkpoint latent shape, saves the actual PRC key, and checks
encode/decode before any GPU edit. Does not load model weights.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.instructpix2pix_prc.toy_ddim import run_toy_ddim  # noqa: E402
from experiments.paths import RESULTS  # noqa: E402
from experiments.prc_runtime import (  # noqa: E402
    detect_decode,
    keygen,
    prc_modules,
    recover_soft_bits,
)
from experiments.receiver_aligned.key_io import key_hash, load_key, save_key  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "instructpix2pix_prc" / "cpu_controls.json"
OUT = RESULTS / "instructpix2pix_prc" / "cpu_controls"
KEYS = RESULTS / "instructpix2pix_prc" / "raw_keys"
OBS = RESULTS / "instructpix2pix_prc" / "raw_observations"

FIELDS = [
    "stage", "control", "status", "failure_reason",
    "model_id", "model_revision", "code_sha256", "config_sha256",
    "n", "latent_shape", "payload_length", "key_message_capacity", "t", "F",
    "noise_rate", "sigma", "sigma_squared",
    "key_hash", "message_sha256", "codeword_sha256",
    "detect", "detector_statistic", "threshold", "margin",
    "decode_non_none", "exact_message_match", "full_payload_match", "decoded_length",
    "sign_agreement", "noise_mse", "noise_correlation",
    "seconds",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_array(values) -> str:
    array = np.asarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def fail(reason: str):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STATUS.txt").write_text("FAIL\n" + reason + "\n")
    print("FAIL:", reason, flush=True)
    return 2


def component_hashes(encoding_key):
    generator, otp, test_bits, g_dim, noise_rate = encoding_key
    return {
        "generator_sha256": sha256_array(np.asarray(generator)),
        "otp_sha256": sha256_array(np.asarray(otp)),
        "test_bits_sha256": sha256_array(np.asarray(test_bits)),
        "g_dim": int(g_dim),
        "noise_rate": float(noise_rate),
        "n": int(generator.shape[0]),
        "k": int(generator.shape[1]),
        "num_test_bits": int(len(test_bits)),
        "message_capacity": int(generator.shape[1] - len(test_bits) - int(g_dim)),
    }


def as_float_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(torch.float64).cpu().flatten()
    return torch.as_tensor(np.asarray(value), dtype=torch.float64).flatten()


def score_row(name, decoding_key, soft, z, codeword, message, meta, seconds, key_id):
    from experiments.prc_runtime import detection_margin

    soft = as_float_tensor(soft)
    detected, decode_ok, decoded = detect_decode(decoding_key, soft)
    statistic, threshold, margin = detection_margin(decoding_key, soft)
    sent = np.asarray(message, dtype=int).reshape(-1)
    if decoded is None:
        exact = False
        full = False
        decoded_len = 0
    else:
        got = np.asarray(decoded, dtype=int).reshape(-1)
        decoded_len = int(got.size)
        exact = bool(got.size >= sent.size and np.array_equal(got[: sent.size], sent))
        full = bool(got.size == sent.size and np.array_equal(got, sent))
    rec = as_float_tensor(z)
    code = as_float_tensor(codeword)
    signs = torch.sign(rec)
    agree = float((signs == code).to(torch.float64).mean().item())
    if z is not None and rec.numel() == code.numel():
        mse = float(torch.mean((rec - code * rec.abs()) ** 2).item()) if "z_prc" in name or name == "L0_reloaded_key" else ""
        corr = float(torch.corrcoef(torch.stack([rec, code]))[0, 1].item()) if rec.std() > 0 else ""
    else:
        mse, corr = "", ""
    return {
        **meta,
        "control": name,
        "status": "recorded",
        "failure_reason": "",
        "key_hash": key_id,
        "detect": detected,
        "detector_statistic": statistic,
        "threshold": threshold,
        "margin": margin,
        "decode_non_none": decode_ok,
        "exact_message_match": exact,
        "full_payload_match": full,
        "decoded_length": decoded_len,
        "sign_agreement": agree,
        "noise_mse": mse,
        "noise_correlation": corr,
        "seconds": seconds,
    }


def fetch_checkpoint(model_id: str, out_dir: Path):
    from huggingface_hub import hf_hub_download, model_info

    info = model_info(model_id)
    names = [
        "model_index.json",
        "unet/config.json",
        "vae/config.json",
        "scheduler/scheduler_config.json",
    ]
    saved = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = Path(hf_hub_download(model_id, name, revision=info.sha))
        dest = out_dir / name.replace("/", "__")
        dest.write_bytes(src.read_bytes())
        saved[name] = json.loads(dest.read_text())
    card = {
        "model_id": model_id,
        "revision": info.sha,
        "private": bool(getattr(info, "private", False)),
        "gated": getattr(info, "gated", None),
        "license": getattr(getattr(info, "card_data", None), "license", None),
        "pipeline_class": saved["model_index.json"].get("_class_name"),
        "scheduler_class": (saved["model_index.json"].get("scheduler") or [None, None])[1],
        "unet_in_channels": saved["unet/config.json"].get("in_channels"),
        "unet_sample_size": saved["unet/config.json"].get("sample_size"),
        "vae_latent_channels": saved["vae/config.json"].get("latent_channels"),
        "vae_scaling_factor": saved["vae/config.json"].get("scaling_factor"),
        "vae_block_out_channels": saved["vae/config.json"].get("block_out_channels"),
        "scheduler_config": saved["scheduler/scheduler_config.json"],
    }
    (out_dir / "checkpoint_summary.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    return card


def expected_shape(card, image_size: int):
    blocks = card["vae_block_out_channels"]
    scale = 2 ** (len(blocks) - 1)
    channels = int(card["vae_latent_channels"])
    if image_size % scale != 0:
        raise RuntimeError(f"image size {image_size} is not divisible by VAE scale {scale}")
    side = image_size // scale
    shape = [1, channels, side, side]
    if int(card["unet_in_channels"]) != channels * 2:
        raise RuntimeError(
            f"UNet in_channels={card['unet_in_channels']} but generation channels={channels}. "
            "PRC noise must use the generation channels only."
        )
    return shape, int(np.prod(shape))


def main():
    started = time.time()
    cfg = json.loads(CONFIG_PATH.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    KEYS.mkdir(parents=True, exist_ok=True)
    OBS.mkdir(parents=True, exist_ok=True)
    code_sha = sha256_file(Path(__file__).resolve())
    config_sha = sha256_file(CONFIG_PATH)
    toy_sha = sha256_file(Path(__file__).resolve().parent / "toy_ddim.py")

    print("fetching checkpoint configs", cfg["model_id"], flush=True)
    try:
        card = fetch_checkpoint(cfg["model_id"], OUT / "checkpoint")
    except Exception as exc:
        return fail(f"checkpoint_config_fetch_failed: {type(exc).__name__}: {exc}")
    try:
        shape, n = expected_shape(card, int(cfg["preprocessed_size"]))
    except Exception as exc:
        return fail(str(exc))
    print("generation shape", shape, "n", n, "scheduler", card["scheduler_class"], flush=True)

    toy = run_toy_ddim()
    (OUT / "toy_ddim.json").write_text(json.dumps(toy, indent=2) + "\n")
    if not toy["constant_predictor_exact"]:
        return fail(f"toy DDIM constant predictor did not invert: {toy}")
    print("toy DDIM constant error", toy["constant_predictor_max_abs_error"], flush=True)
    print("toy DDIM state-dependent error", toy["state_dependent_predictor_max_abs_error"], flush=True)
    print("toy one-step solve", toy["one_step_fixed_point"], flush=True)

    message_length = int(cfg["message_length"])
    fpr = float(cfg["false_positive_rate"])
    t_sparsity = int(cfg["t"])
    sigma_squared = float(cfg["sigma_squared"])
    message = np.random.default_rng(int(cfg["message_rng_seed"])).integers(0, 2, size=message_length).tolist()

    t0 = time.time()
    encoding_key, decoding_key = keygen(n=n, message_length=message_length, fpr=fpr, t=t_sparsity, seed=0)
    key_seconds = time.time() - t0
    summary = component_hashes(encoding_key)
    if summary["message_capacity"] != message_length:
        return fail(
            f"key message capacity is {summary['message_capacity']}, not the requested {message_length}"
        )
    _, Encode, _, _, _ = prc_modules()
    codeword = Encode(encoding_key, message=message)
    if codeword.numel() != n:
        return fail(f"codeword length {codeword.numel()} != n {n}")
    if not torch.equal(codeword.abs(), torch.ones_like(codeword)):
        return fail("codeword is not in {-1,+1}")

    rng = np.random.default_rng(int(cfg["gaussian_rng_seed"]))
    gaussian = rng.standard_normal(n)
    z_prc = torch.from_numpy(np.asarray(codeword.detach().cpu().numpy()) * np.abs(gaussian))
    ordinary = torch.from_numpy(rng.standard_normal(n))

    encoding_wrong, decoding_wrong = keygen(n=n, message_length=message_length, fpr=fpr, t=t_sparsity, seed=1)
    if key_hash(encoding_key) == key_hash(encoding_wrong):
        return fail("independent KeyGen calls produced the same key hash")

    key_path = KEYS / "cpu_controls_key.pkl"
    wrong_path = KEYS / "cpu_controls_wrong_key.pkl"
    save_key(key_path, encoding_key, decoding_key, message, codeword)
    save_key(wrong_path, encoding_wrong, decoding_wrong, message, codeword)
    np.savez(
        OBS / "cpu_controls_noise.npz",
        gaussian=gaussian,
        z_prc=z_prc.numpy(),
        ordinary=ordinary.numpy(),
        codeword=codeword.detach().cpu().numpy(),
        message=np.asarray(message, dtype=np.int64),
    )

    meta = {
        "stage": "cpu_controls",
        "model_id": cfg["model_id"],
        "model_revision": card["revision"],
        "code_sha256": code_sha,
        "config_sha256": config_sha,
        "n": n,
        "latent_shape": str(shape),
        "payload_length": message_length,
        "key_message_capacity": summary["message_capacity"],
        "t": t_sparsity,
        "F": fpr,
        "noise_rate": summary["noise_rate"],
        "sigma": float(np.sqrt(sigma_squared)),
        "sigma_squared": sigma_squared,
        "message_sha256": sha256_array(np.asarray(message, dtype=np.uint8)),
        "codeword_sha256": sha256_array(codeword.detach().cpu().numpy()),
    }
    correct_id = key_hash(encoding_key)
    wrong_id = key_hash(encoding_wrong)

    rows = []
    t0 = time.time()
    rows.append(score_row(
        "L0_codeword", decoding_key, codeword, codeword, codeword, message, meta, time.time() - t0, correct_id,
    ))
    t0 = time.time()
    soft_z = recover_soft_bits(z_prc, variance=sigma_squared)
    rows.append(score_row(
        "L0_z_prc", decoding_key, soft_z, z_prc, codeword, message, meta, time.time() - t0, correct_id,
    ))
    t0 = time.time()
    rows.append(score_row(
        "L0_z_prc_wrong_key", decoding_wrong, soft_z, z_prc, codeword, message, meta, time.time() - t0, wrong_id,
    ))
    t0 = time.time()
    soft_ordinary = recover_soft_bits(ordinary, variance=sigma_squared)
    rows.append(score_row(
        "L0_ordinary_noise", decoding_key, soft_ordinary, ordinary, codeword, message, meta, time.time() - t0, correct_id,
    ))

    re_enc, re_dec, re_message, re_codeword = load_key(key_path)
    if key_hash(re_enc) != correct_id:
        return fail("reloaded key hash does not match the saved key")
    t0 = time.time()
    soft_reload = recover_soft_bits(torch.from_numpy(np.load(OBS / "cpu_controls_noise.npz")["z_prc"]), variance=sigma_squared)
    rows.append(score_row(
        "L0_reloaded_key", re_dec, soft_reload, z_prc, re_codeword, re_message, meta, time.time() - t0, key_hash(re_enc),
    ))

    reasons = []
    by_name = {row["control"]: row for row in rows}
    for name in ("L0_codeword", "L0_z_prc", "L0_reloaded_key"):
        row = by_name[name]
        if not (row["detect"] and row["decode_non_none"] and row["exact_message_match"] and row["full_payload_match"]):
            reasons.append(f"{name} did not detect and exactly decode")
    if by_name["L0_z_prc"]["sign_agreement"] != 1.0:
        reasons.append(f"z_prc sign agreement is {by_name['L0_z_prc']['sign_agreement']}")
    if by_name["L0_z_prc_wrong_key"]["detect"] or by_name["L0_z_prc_wrong_key"]["decode_non_none"]:
        reasons.append("wrong key detected or decoded the PRC noise")
    if by_name["L0_ordinary_noise"]["detect"] or by_name["L0_ordinary_noise"]["decode_non_none"]:
        reasons.append("ordinary noise was detected or decoded by the correct key")
    reloaded = by_name["L0_reloaded_key"]
    original = by_name["L0_z_prc"]
    for field in ("detect", "exact_message_match", "margin", "detector_statistic"):
        if reloaded[field] != original[field]:
            reasons.append(f"reload changed {field}")

    status = "PASS" if not reasons else "FAIL"
    for row in rows:
        row["status"] = status
        row["failure_reason"] = "; ".join(reasons)
    with (OUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": status,
        "failure_reason": reasons,
        "checkpoint": card,
        "key_summary": summary,
        "key_hash": correct_id,
        "wrong_key_hash": wrong_id,
        "key_path": str(key_path),
        "wrong_key_path": str(wrong_path),
        "sign_convention": "codeword = 1 - 2 * GF(2) bits; bit 0 maps to +1 and bit 1 maps to -1",
        "encoding_randomness": "Encode does not return the random g-vector or Bernoulli noise. They are contained in the saved codeword.",
        "padding_bits": 0,
        "sigma_convention": "sigma is std(z_recovered - z). recover_posteriors variances argument is sigma_squared.",
        "toy_ddim": toy,
        "code_sha256": code_sha,
        "config_sha256": config_sha,
        "toy_sha256": toy_sha,
        "keygen_seconds": key_seconds,
        "seconds": time.time() - started,
        "rows": rows,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (OUT / "STATUS.txt").write_text(status + "\n" + "\n".join(reasons) + "\n")
    print(status, reasons, flush=True)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
