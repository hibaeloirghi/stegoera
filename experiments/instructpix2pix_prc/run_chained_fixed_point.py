"""One capped chained fixed-point inverse of the saved PRC DDIM edit.

Builds a float32 trajectory from the saved PRC initial noise and the saved
source conditioning, then inverts that trajectory's own final latent. Each
inverse step is a fixed-point solve with a fixed iteration cap. This is a
source-known latent-channel diagnostic. It does not read a PNG and it does
not hide the source latent.
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

from experiments.instructpix2pix_prc.ddim_invert import (  # noqa: E402
    assert_algebra_matches_scheduler,
    invert_ddim_step,
    make_pair_ddim,
)
from experiments.paths import RESULTS, WORKDIR  # noqa: E402
from experiments.prc_runtime import detect_decode, detection_margin, recover_soft_bits  # noqa: E402
from experiments.receiver_aligned.key_io import key_hash, load_key  # noqa: E402

CONFIG_PATH = ROOT / "experiments" / "configs" / "instructpix2pix_prc" / "chained_fixed_point.json"
OUT = RESULTS / "instructpix2pix_prc" / "chained_fixed_point"

STEP_FIELDS = [
    "inverse_depth", "forward_step", "timestep",
    "iters", "unet_evals",
    "residual_mse", "normalized_residual", "converged",
    "state_mse", "state_max_abs", "state_correlation", "state_sign_agreement",
    "degraded",
]

SUMMARY_FIELDS = [
    "check", "mse", "max_abs", "correlation", "sign_agreement", "sign_errors",
    "detect", "detector_statistic", "threshold", "margin",
    "decode_non_none", "exact_message_match", "full_payload_match",
    "wrong_key_detect", "wrong_key_decode_non_none",
    "sigma_squared", "seconds", "note",
]


def sha256_array(values) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def fail(reason: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STATUS.txt").write_text("FAIL\n" + reason + "\n")
    print("FAIL:", reason, flush=True)
    return 2


def compare(candidate, target):
    x = candidate.detach().to(torch.float64).flatten().cpu()
    y = target.detach().to(torch.float64).flatten().cpu()
    err = x - y
    corr = float(torch.corrcoef(torch.stack([x, y]))[0, 1].item())
    signs = torch.sign(x) == torch.sign(y)
    return {
        "mse": float(err.pow(2).mean().item()),
        "max_abs": float(err.abs().max().item()),
        "correlation": corr,
        "sign_agreement": float(signs.float().mean().item()),
        "sign_errors": int((~signs).sum().item()),
    }


def load_unet(cfg):
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        cfg["model_id"],
        subfolder="unet",
        revision=cfg["model_revision"],
        torch_dtype=torch.float32,
    )
    return unet.to("cuda").eval()


def predict_fn(unet, scheduler, image_latents, prompt_embeds, guidance, image_guidance):
    def predict(latents, timestep):
        with torch.no_grad():
            model_input = torch.cat([latents] * 3)
            model_input = scheduler.scale_model_input(model_input, timestep)
            model_input = torch.cat([model_input, image_latents], dim=1)
            noise = unet(
                model_input, timestep, encoder_hidden_states=prompt_embeds, return_dict=False
            )[0]
            if hasattr(scheduler, "sigmas"):
                raise RuntimeError("DDIM scheduler should not expose sigmas")
            noise_text, noise_image, noise_uncond = noise.chunk(3)
            return (
                noise_uncond
                + guidance * (noise_text - noise_image)
                + image_guidance * (noise_image - noise_uncond)
            )

    return predict


def replay(predict, scheduler, start, timesteps):
    states = [start.detach()]
    current = start
    for timestep in timesteps:
        eps = predict(current, timestep)
        current = scheduler.step(eps, timestep, current, eta=0.0, return_dict=False)[0]
        states.append(current.detach())
    return states


def solve_step(predict, scheduler, x_out, timestep, max_iter, tol):
    """Fixed-point step. The logged residual belongs to the state that is kept.

    The first candidate is the one-pass inverse. Further iterations replace it
    only while its step residual stays above tol and the iteration cap remains.
    """
    eps = predict(x_out, timestep)
    current = invert_ddim_step(scheduler, x_out, eps, timestep)
    residual_mse = None
    normalized = None
    iters = 0
    for i in range(max_iter):
        eps = predict(current, timestep)
        recon = scheduler.step(eps, timestep, current, eta=0.0, return_dict=False)[0]
        diff = recon.float() - x_out.float()
        residual_mse = float(diff.pow(2).mean().item())
        target_rms = float(x_out.float().pow(2).mean().sqrt().item())
        normalized = float(diff.pow(2).mean().sqrt().item()) / max(target_rms, 1e-12)
        iters = i + 1
        if residual_mse < tol or i == max_iter - 1:
            break
        current = invert_ddim_step(scheduler, x_out, eps, timestep)
    return current.detach(), {
        "iters": iters,
        "unet_evals": 1 + iters,
        "residual_mse": residual_mse,
        "normalized_residual": normalized,
        "converged": bool(residual_mse < tol),
    }


def is_degraded(stats, cfg):
    return bool(
        stats["mse"] > float(cfg["degrade_state_mse"])
        or stats["correlation"] < float(cfg["degrade_correlation"])
        or stats["sign_agreement"] < float(cfg["degrade_sign_agreement"])
    )


def score_prc(decoding_key, wrong_key, recovered, codeword, message, sigma_squared):
    flat = recovered.detach().to(torch.float64).flatten().cpu()
    soft = recover_soft_bits(flat, variance=sigma_squared)
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
    wrong_soft = recover_soft_bits(flat, variance=sigma_squared)
    wrong_detect, wrong_decode, _wrong = detect_decode(wrong_key, wrong_soft)
    code = torch.as_tensor(np.asarray(codeword), dtype=torch.float64).flatten()
    code_agree = float((torch.sign(flat) == code).to(torch.float64).mean().item())
    code_errors = int((torch.sign(flat) != code).sum().item())
    return {
        "detect": detected,
        "detector_statistic": statistic,
        "threshold": threshold,
        "margin": margin,
        "decode_non_none": decode_ok,
        "exact_message_match": exact,
        "full_payload_match": full,
        "wrong_key_detect": wrong_detect,
        "wrong_key_decode_non_none": wrong_decode,
        "codeword_sign_agreement": code_agree,
        "codeword_sign_errors": code_errors,
        "sigma_squared": sigma_squared,
    }


def blank_summary(check, **kwargs):
    row = {key: "" for key in SUMMARY_FIELDS}
    row["check"] = check
    row.update(kwargs)
    return row


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    algebra_error = assert_algebra_matches_scheduler()
    print("algebra max abs", algebra_error, flush=True)

    encoding_key, decoding_key, message, key_codeword = load_key(WORKDIR / cfg["key_path"])
    found = key_hash(encoding_key)
    if found != cfg["expected_key_hash"]:
        return fail(f"key hash {found}")
    if len(message) != 64:
        return fail(f"message length {len(message)}")
    noise = np.load(WORKDIR / cfg["saved_noise"])
    codeword = np.asarray(noise["codeword"]).reshape(-1)
    key_code = np.asarray(key_codeword).reshape(-1)
    saved_message = np.asarray(noise["message"], dtype=int).reshape(-1)
    if not np.array_equal(key_code, codeword):
        return fail("pair encoding codeword does not match raw_noise.npz")
    if not np.array_equal(np.asarray(message, dtype=int).reshape(-1), saved_message):
        return fail("key message does not match raw_noise.npz")
    z_prc = torch.from_numpy(np.asarray(noise["z_prc"])).to(torch.float32)
    if int((torch.sign(z_prc).flatten() != torch.from_numpy(codeword).float()).sum().item()) != 0:
        return fail("saved z_prc signs do not match its codeword")
    _wrong_enc, wrong_dec, _wrong_message, _wrong_code = load_key(WORKDIR / cfg["wrong_key_path"])

    saved = torch.load(WORKDIR / cfg["saved_forward"], map_location="cpu")
    scheduler = make_pair_ddim()
    scheduler.set_timesteps(int(cfg["num_inference_steps"]), device="cuda")
    live = scheduler.timesteps.detach().cpu().to(torch.float32)
    if not torch.equal(live, saved["timesteps"].to(torch.float32)):
        return fail("timesteps do not match the saved PRC forward")
    timesteps = list(scheduler.timesteps)
    start = saved["scaled_start"].to(device="cuda", dtype=torch.float32)
    rounding = compare(start, z_prc.to(device="cuda"))
    if rounding["sign_errors"] != 0:
        return fail(f"scaled start flipped {rounding['sign_errors']} PRC signs")

    image_latents = saved["image_latents_cfg"].to(device="cuda", dtype=torch.float32)
    prompt_embeds = saved["prompt_embeds"].to(device="cuda", dtype=torch.float32)
    print(
        "gpu", torch.cuda.get_device_name(0),
        "mem_gib", torch.cuda.get_device_properties(0).total_memory / 2**30,
        flush=True,
    )
    unet = load_unet(cfg)
    predict = predict_fn(
        unet, scheduler, image_latents, prompt_embeds,
        float(cfg["guidance_scale"]), float(cfg["image_guidance_scale"]),
    )
    print("float32 reference trajectory", flush=True)
    started = time.perf_counter()
    states = replay(predict, scheduler, start, timesteps)
    reference_seconds = time.perf_counter() - started
    saved_final_gap = compare(states[-1], saved["final"].to(device="cuda"))
    print(
        "reference seconds", round(reference_seconds, 2),
        "vs saved fp16 final mse", saved_final_gap["mse"],
        flush=True,
    )

    max_iter = int(cfg["fixed_point_max_iter"])
    tol = float(cfg["fixed_point_tol"])
    current = states[-1]
    step_rows = []
    first_degrade = None
    first_nonconverged = None
    print("chained fixed point", "max_iter", max_iter, "tol", tol, flush=True)
    started = time.perf_counter()
    for depth, forward_step in enumerate(range(len(timesteps) - 1, -1, -1), start=1):
        timestep = timesteps[forward_step]
        current, info = solve_step(predict, scheduler, current, timestep, max_iter, tol)
        stats = compare(current, states[forward_step])
        degraded = is_degraded(stats, cfg)
        if degraded and first_degrade is None:
            first_degrade = {
                "inverse_depth": depth,
                "forward_step": forward_step,
                "timestep": int(timestep),
                "state_mse": stats["mse"],
                "state_max_abs": stats["max_abs"],
                "state_correlation": stats["correlation"],
                "state_sign_agreement": stats["sign_agreement"],
                "sign_errors": stats["sign_errors"],
                "residual_mse": info["residual_mse"],
                "normalized_residual": info["normalized_residual"],
                "converged": info["converged"],
                "iters": info["iters"],
            }
        if not info["converged"] and first_nonconverged is None:
            first_nonconverged = {
                "inverse_depth": depth,
                "forward_step": forward_step,
                "timestep": int(timestep),
                "iters": info["iters"],
                "residual_mse": info["residual_mse"],
                "normalized_residual": info["normalized_residual"],
            }
        step_rows.append({
            "inverse_depth": depth,
            "forward_step": forward_step,
            "timestep": int(timestep),
            **info,
            "state_mse": stats["mse"],
            "state_max_abs": stats["max_abs"],
            "state_correlation": stats["correlation"],
            "state_sign_agreement": stats["sign_agreement"],
            "degraded": degraded,
        })
        if depth == 1 or depth % 10 == 0 or degraded and first_degrade["inverse_depth"] == depth:
            print(
                " depth", depth,
                "forward_step", forward_step,
                "iters", info["iters"],
                "residual", info["residual_mse"],
                "norm_residual", info["normalized_residual"],
                "state_mse", stats["mse"],
                "corr", stats["correlation"],
                "sign", stats["sign_agreement"],
                "converged", info["converged"],
                flush=True,
            )
    inverse_seconds = time.perf_counter() - started
    recovered = current

    print("regenerate from recovered start", flush=True)
    started = time.perf_counter()
    regenerated = replay(predict, scheduler, recovered, timesteps)
    regenerate_seconds = time.perf_counter() - started
    final_stats = compare(regenerated[-1], states[-1])
    start_stats = compare(recovered, states[0])
    z_stats = compare(recovered, z_prc.to(device="cuda"))
    print("score prc", flush=True)
    prc = score_prc(
        decoding_key, wrong_dec, recovered, codeword, message, float(cfg["sigma_squared"])
    )
    print(
        "start mse", start_stats["mse"],
        "corr", start_stats["correlation"],
        "sign", start_stats["sign_agreement"],
        "detect", prc["detect"],
        "exact", prc["exact_message_match"],
        "final mse", final_stats["mse"],
        flush=True,
    )

    with (OUT / "steps.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STEP_FIELDS)
        writer.writeheader()
        writer.writerows(step_rows)
    summary_rows = [
        blank_summary(
            "reference_vs_saved_fp16_final",
            seconds=reference_seconds,
            note="float32 replay final compared with the saved float16 PRC final",
            **saved_final_gap,
        ),
        blank_summary(
            "scaled_start_vs_z_prc",
            note="float16 rounding of the PRC noise that entered the forward",
            **rounding,
        ),
        blank_summary(
            "recovered_vs_reference_start",
            seconds=inverse_seconds,
            note="chained fixed-point result against the float32 reference initial noise",
            **start_stats,
        ),
        blank_summary(
            "recovered_vs_z_prc",
            note="same recovered noise against the saved full-precision PRC noise",
            **z_stats,
        ),
        blank_summary(
            "recovered_prc",
            sigma_squared=prc["sigma_squared"],
            sign_agreement=prc["codeword_sign_agreement"],
            sign_errors=prc["codeword_sign_errors"],
            note="operational score at sigma_squared=1.5; sign agreement is versus the codeword",
            **{key: prc[key] for key in (
                "detect", "detector_statistic", "threshold", "margin",
                "decode_non_none", "exact_message_match", "full_payload_match",
                "wrong_key_detect", "wrong_key_decode_non_none",
            )},
        ),
        blank_summary(
            "regenerated_vs_reference_final",
            seconds=regenerate_seconds,
            note="float32 forward from the recovered start, compared with the reference final latent",
            **final_stats,
        ),
    ]
    with (OUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    torch.save(recovered.detach().to(torch.float32).cpu(), OUT / "recovered_start.pt")
    n_nonconverged = sum(1 for row in step_rows if not row["converged"])
    last_within_band = None
    for row in step_rows:
        if row["degraded"]:
            break
        last_within_band = row["inverse_depth"]
    manifest = {
        "receiver_level": "L1",
        "conditioning_provenance": "saved source-image latents and saved instruction embeddings",
        "diagnostic_dtype": "float32",
        "saved_forward_dtype": "float16 values stored as float32",
        "precision_note": (
            "The reference trajectory replays the saved PRC initial latent and saved "
            "conditioning in float32. The inverse target is that replay's final latent."
        ),
        "iteration_cap": max_iter,
        "residual_tolerance_mse": tol,
        "normalized_residual": "rms(step(eps(x), t, x) - x_out) / rms(x_out), for the kept state x",
        "degrade_rule": {
            "state_mse_above": float(cfg["degrade_state_mse"]),
            "correlation_below": float(cfg["degrade_correlation"]),
            "sign_agreement_below": float(cfg["degrade_sign_agreement"]),
            "rule": "first inverse step, counting from the final latent, that trips any threshold",
        },
        "first_degrade": first_degrade,
        "last_inverse_depth_within_band": last_within_band,
        "first_nonconverged": first_nonconverged,
        "nonconverged_steps": n_nonconverged,
        "instruction": cfg["instruction"],
        "key_hash": found,
        "message_sha256": sha256_array(np.asarray(message, dtype=np.uint8)),
        "codeword_sha256": sha256_array(codeword),
        "prompt_embeds_sha256": hashlib.sha256(saved["prompt_embeds"].numpy().tobytes()).hexdigest(),
        "image_latents_sha256": hashlib.sha256(saved["image_latents_cfg"].numpy().tobytes()).hexdigest(),
        "algebra_max_abs_error": algebra_error,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "gpu": torch.cuda.get_device_name(0),
        "reference_seconds": reference_seconds,
        "inverse_seconds": inverse_seconds,
        "regenerate_seconds": regenerate_seconds,
        "residual_note": (
            "A small step residual means the kept state maps to the latent being inverted "
            "under the noise prediction at that state. It does not by itself identify another preimage."
        ),
        "summary": summary_rows,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (OUT / "STATUS.txt").write_text("PASS\n")
    print(
        "PASS",
        "first_degrade", None if first_degrade is None else first_degrade["inverse_depth"],
        "nonconverged", n_nonconverged,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
