"""Replay one historical EDICT+PRC dog1 row, or the same row at guidance 7.5.

Guidance 3 must reproduce the saved PNG row. Guidance 7.5 is the same code
with the single CFG scale changed. That scale is uncond + G*(cond - uncond)
on SD 1.4. It is not InstructPix2Pix text guidance.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import experiments.edict_runtime as er  # noqa: E402
import experiments.prc_runtime as pr  # noqa: E402
from experiments.image_utils import load_edict_image  # noqa: E402
from experiments.ip2p_edict_prc.guidance_diag.common import (  # noqa: E402
    DEPARTURE_FIELDS,
    OUT,
    PARTS,
    QUALITY_FIELDS,
    RECEIVER_FIELDS,
    TRACE_FIELDS,
    code_hash,
    departure_row,
    git_rev,
    overlay_plot,
    publish,
    require_hash,
    sha256_file,
    trace_rows,
    write_json,
    write_new_csv,
)
from experiments.metrics import clip_score, lpips_distance, pixel_mse  # noqa: E402
from experiments.receiver_aligned.prc_scores import exact_message_match  # noqa: E402
from experiments.run_full_prc_sign import IMAGES, agreement, prompt_for, sign_to_codeword  # noqa: E402


class StepTrace:
    """Record the pair after each coupled step. Return None so the pair is unchanged."""

    def __init__(self, initial):
        self.snaps = [[item.detach().to(dtype=torch.float64).cpu().clone() for item in initial]]
        self.times = [None]

    def hook(self, label, _i, t, latent_pair, info):
        reverse = bool(info.get("reverse"))
        # Forward S4 is after the mix. Reverse S2 is after both affine updates.
        if (label == "S2" and reverse) or (label == "S4" and not reverse):
            self.snaps.append([item.detach().to(dtype=torch.float64).cpu().clone() for item in latent_pair])
            self.times.append(int(t.item()) if torch.is_tensor(t) else int(t))
        return None


def loop_kwargs(guidance: float) -> dict:
    return dict(
        steps=50,
        mix_weight=0.93,
        init_image_strength=1.0,
        guidance_scale=float(guidance),
        seed=1,
        null_prompt="",
        leapfrog_steps=True,
    )


def score_stream(latent, codeword, decoding_key, wrong_key, message):
    soft = pr.recover_soft_bits(latent.detach().to(torch.float64).cpu().reshape(-1), variance=1.5)
    statistic, threshold, margin = pr.detection_margin(decoding_key, soft)
    detected, decode_ok, payload = pr.detect_decode(decoding_key, soft)
    wrong_detected, wrong_ok, _wrong_payload = pr.detect_decode(wrong_key, soft)
    return {
        "initial_noise_MSE": "",
        "initial_noise_correlation": "",
        "sign_agreement": agreement(latent, codeword),
        "detect": bool(detected),
        "statistic": statistic,
        "threshold": threshold,
        "margin": margin,
        "decode_non_none": bool(decode_ok),
        "exact_message_match": exact_message_match(payload, message),
        "wrong_key_detect": bool(wrong_detected),
        "wrong_key_decode_non_none": bool(wrong_ok),
    }


def run_direction(ef, prompt, guidance, start, reverse: bool):
    tracer = StepTrace(start)
    kwargs = loop_kwargs(guidance)
    started = time.perf_counter()
    if reverse:
        out = er.coupled_loop(
            ef, prompt, reverse=True, init_latent=start, step_hook=tracer.hook, **kwargs,
        )
    else:
        out = er.coupled_loop(
            ef, prompt, reverse=False, fixed_starting_latent=start, step_hook=tracer.hook, **kwargs,
        )
    elapsed = time.perf_counter() - started
    if len(tracer.snaps) != 51:
        raise RuntimeError(f"expected 51 snapshots, got {len(tracer.snaps)}")
    walked = [int(item.item()) if torch.is_tensor(item) else int(item) for item in out["timesteps"]]
    if tracer.times[1:] != walked:
        raise RuntimeError("hook timesteps do not match the scheduler")
    final = [item.detach().to(dtype=torch.float64).clone() for item in out["latent_pair"]]
    return final, tracer.snaps, tracer.times, elapsed


def main() -> int:
    config_path = Path(sys.argv[1])
    cfg = json.loads(config_path.read_text())
    guidance = float(cfg["guidance"])
    slug = f"old_g{guidance:g}"
    if Path(cfg["source_photo"]).name != "imagenet_dog_1.jpg":
        print("FAIL: source must be the original dog photograph", flush=True)
        return 2
    require_hash(ROOT / cfg["source_photo"], cfg["source_photo_sha256"], "source photo")
    require_hash(ROOT / cfg["key_path"], cfg["key_sha256"], "old key")
    if cfg.get("regression"):
        require_hash(ROOT / cfg["historical_png"], cfg["historical_png_sha256"], "historical png")

    image = next(item for item in IMAGES if item["id"] == cfg["image_id"])
    invert_prompt = prompt_for(image, cfg["guide"])
    generate_prompt = prompt_for(image, cfg["target"])
    if invert_prompt == "" or generate_prompt == "":
        print("FAIL: this row was chosen so both prompts are non-empty", flush=True)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    (OUT / "endpoints").mkdir(exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    print("loading EDICT", flush=True)
    ef = er.load_models("cuda")
    photo = load_edict_image(ROOT / cfg["source_photo"])
    with (ROOT / cfg["key_path"]).open("rb") as handle:
        key = pickle.load(handle)
    codeword = key["codewords"][cfg["image_id"]]
    message = key["message"]
    decoding_key = key["decoding_key"]
    wrong_key = key["wrong_decoding_key"]

    print("invert", cfg["guide"], "guidance", guidance, flush=True)
    inverted = er.invert_image(ef, photo, prompt=invert_prompt, **loop_kwargs(guidance))
    signed = [
        sign_to_codeword(inverted["latent_pair"][0], codeword),
        sign_to_codeword(inverted["latent_pair"][1], codeword),
    ]
    print("forward", flush=True)
    final, forward_snaps, forward_times, _forward_seconds = run_direction(
        ef, generate_prompt, guidance, signed, reverse=False,
    )
    print("inverse true pair", flush=True)
    _oracle, snaps_i, inverse_times, seconds_i = run_direction(
        ef, generate_prompt, guidance, final, reverse=True,
    )
    print("inverse duplicated x", flush=True)
    duplicated = [final[0].detach().clone(), final[0].detach().clone()]
    _dup, snaps_d, _times_d, seconds_d = run_direction(
        ef, generate_prompt, guidance, duplicated, reverse=True,
    )

    replay_path = OUT / "images" / f"old_g{guidance:g}_caption_small_replay.png"
    replay_image = er.decode_latent(ef, final[0])
    replay_image.save(replay_path)
    reloaded = Image.open(replay_path).convert("RGB")
    print("png receiver on replay", flush=True)
    replay_png = er.invert_image(ef, reloaded, prompt=generate_prompt, **loop_kwargs(guidance))
    historical_png = None
    if cfg.get("regression"):
        print("png receiver on historical file", flush=True)
        historical = Image.open(ROOT / cfg["historical_png"]).convert("RGB")
        historical_png = er.invert_image(ef, historical, prompt=generate_prompt, **loop_kwargs(guidance))

    edit = "dog1_caption_to_small"
    guidance_config = f"sd_cfg_{guidance:g}"
    rows = []
    rows.extend(trace_rows("old_edict_prc", edit, guidance_config, "F", forward_snaps, forward_times, forward_snaps))
    rows.extend(trace_rows("old_edict_prc", edit, guidance_config, "I", snaps_i, inverse_times, forward_snaps))
    rows.extend(trace_rows("old_edict_prc", edit, guidance_config, "D", snaps_d, inverse_times, forward_snaps))
    summary = departure_row("old_edict_prc", edit, guidance_config, rows)
    overlay_plot(rows, f"old dog1 caption to small, CFG {guidance:g}", OUT / "plots" / f"overlay_{slug}.png")
    torch.save(
        {"role": "evaluator-only forward trajectory", "forward": forward_snaps},
        OUT / "endpoints" / f"{slug}_evaluator_forward.pt",
    )

    sigma2 = 1.5
    sigma = math.sqrt(sigma2)
    job_id = os.environ.get("SLURM_JOB_ID", "")
    digest = code_hash([
        ROOT / "experiments/edict_runtime.py",
        ROOT / "experiments/run_full_prc_sign.py",
        Path(__file__),
    ])
    config_digest = sha256_file(config_path)
    image_hash = sha256_file(replay_path)
    terminal_xy = float(summary["terminal_xy_mse"])
    receiver_rows = []

    def add_level(level, pair, seconds, image_digest):
        for stream, latent in enumerate(pair):
            scored = score_stream(latent, codeword, decoding_key, wrong_key, message)
            receiver_rows.append({
                "experiment_id": f"old_dog1_caption_small_g{guidance:g}",
                "job_id": job_id,
                "code_hash": digest,
                "config_hash": config_digest,
                "image_hash": image_digest,
                "conditioning_hash": "",
                "key_id": "full_prc_sign_key",
                "receiver_level": level,
                "guidance_text": guidance,
                "guidance_image": "not_applicable_single_scale",
                "guidance_kind": "sd14_uncond_plus_scale_times_cond_minus_uncond",
                "recovered_stream": stream,
                "initial_noise_role": "not_applicable_old_pipeline_starts_from_inverted_photo",
                "sigma": sigma,
                "sigma_squared": sigma2,
                "terminal_xy_MSE": terminal_xy,
                "terminal_latent_MSE": "",
                "terminal_latent_corr": "",
                "runtime": seconds,
                "peak_memory": int(torch.cuda.max_memory_allocated()),
                "status": "recorded",
                **scored,
            })

    add_level("I_true_final_pair", snaps_i[-1], seconds_i, image_hash)
    add_level("D_exact_x_duplicated", snaps_d[-1], seconds_d, image_hash)
    add_level("replay_png", replay_png["latent_pair"], 0.0, image_hash)
    regression_ok = True
    if historical_png is not None:
        historical_hash = sha256_file(ROOT / cfg["historical_png"])
        add_level("historical_png", historical_png["latent_pair"], 0.0, historical_hash)
        historical_rows = receiver_rows[-2:]
        x_row, y_row = historical_rows
        tolerance = float(cfg["sign_agreement_tolerance"])
        regression_ok = bool(
            x_row["detect"] and x_row["decode_non_none"] and x_row["exact_message_match"]
            and abs(float(x_row["sign_agreement"]) - float(cfg["historical_sign_agreement_x"])) <= tolerance
            and abs(float(y_row["sign_agreement"]) - float(cfg["historical_sign_agreement_y"])) <= tolerance
        )
        oracle_rows = [row for row in receiver_rows if row["receiver_level"] == "I_true_final_pair"]
        if any(float(row["sign_agreement"]) < 0.999 or not row["exact_message_match"] for row in oracle_rows):
            regression_ok = False
    for row in receiver_rows:
        if cfg.get("regression") and row["receiver_level"] == "historical_png":
            row["status"] = "PASS" if regression_ok else "FAIL"
        else:
            row["status"] = "recorded"

    quality = []
    quality_error = ""
    try:
        lpips, lpips_error = lpips_distance(replay_image, photo, device="cuda")
        clip_value, clip_error = clip_score(replay_image, generate_prompt, device="cuda")
        quality = [{
            "experiment_id": f"old_dog1_caption_small_g{guidance:g}",
            "edit": edit,
            "guidance_text": guidance,
            "guidance_image": "not_applicable_single_scale",
            "role": f"old_cfg_{guidance:g}",
            "lpips_vs_source": lpips,
            "lpips_error": lpips_error or clip_error or "",
            "pixel_mse_vs_source": pixel_mse(replay_image, photo),
            "directional_clip": "",
            "clip_instruction": clip_value,
            "clip_source_caption": "",
            "source_caption": invert_prompt,
            "instruction": generate_prompt,
            "image_path": str(replay_path.relative_to(ROOT)),
        }]
    except Exception as exc:
        quality_error = repr(exc)
        print("quality failed", quality_error, flush=True)

    write_new_csv(PARTS / f"trace_{slug}.csv", TRACE_FIELDS, rows)
    write_new_csv(PARTS / f"receiver_{slug}.csv", RECEIVER_FIELDS, receiver_rows)
    write_new_csv(PARTS / f"departure_{slug}.csv", DEPARTURE_FIELDS, [summary])
    write_new_csv(PARTS / f"quality_{slug}.csv", QUALITY_FIELDS, quality)
    write_json(PARTS / f"manifest_{slug}.json", {
        "stage": slug,
        "git_rev": git_rev(),
        "job_id": job_id,
        "guidance": guidance,
        "guidance_formula": "noise_uncond + guidance * (noise_cond - noise_uncond)",
        "not_equivalent_to": "InstructPix2Pix text scale",
        "image_id": cfg["image_id"],
        "guide": cfg["guide"],
        "target": cfg["target"],
        "invert_prompt": invert_prompt,
        "generate_prompt": generate_prompt,
        "replay_png_sha256": image_hash,
        "historical_png_sha256": cfg.get("historical_png_sha256", ""),
        "replay_matches_historical_bytes": image_hash == cfg.get("historical_png_sha256", ""),
        "regression_required": bool(cfg.get("regression")),
        "regression_ok": regression_ok,
        "terminal_xy_mse": terminal_xy,
        "first_x_departure_step": summary["first_x_departure_step"],
        "gpu": torch.cuda.get_device_name(0),
        "peak_memory": int(torch.cuda.max_memory_allocated()),
        "code_hash": digest,
        "config_hash": config_digest,
        "quality_error": quality_error,
    })
    if cfg.get("regression"):
        label = "PASS" if regression_ok else "FAIL"
    else:
        label = "COMPLETE"
    (PARTS / f"status_{slug}.txt").write_text(
        f"{label} {slug} job {job_id}\n"
        f"terminal_xy_mse={terminal_xy} first_x_departure={summary['first_x_departure_step']}\n"
        f"replay_png={image_hash}\n"
    )
    publish()
    print(label, "terminal_xy", terminal_xy, flush=True)
    if cfg.get("regression") and not regression_ok:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
