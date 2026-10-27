"""One hybrid guidance config and one instruction.

The reference pair (text 7.5, image 1.5) is refused. R1 is the true final pair.
If R1 loses Detect, Decode, or the exact message, later receiver levels are not run.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

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
    publish,
    require_hash,
    sha256_bytes,
    sha256_file,
    snaps_from_trace,
    trace_rows,
    vector_corr,
    vector_mse,
    write_json,
    write_new_csv,
)
from experiments.ip2p_edict_prc.run_milestone import decode, make_denoise  # noqa: E402
from experiments.ip2p_edict_prc.run_png_decoder_inversion import (  # noqa: E402
    encode_mode,
    mean_pixel_loss,
    png_to_model,
    recover_latent,
    score_stream,
)
from experiments.ip2p_edict_prc.sampler import EdictSchedule, coupled_loop  # noqa: E402
from experiments.prc_runtime import detect_decode, recover_soft_bits  # noqa: E402
from experiments.receiver_aligned.key_io import key_hash, load_key  # noqa: E402

ALLOWED = {(3.0, 1.5), (3.0, 1.0), (7.5, 1.0), (1.5, 1.0)}


def inverse_pair(denoise, schedule, first, second, mix_weight: float):
    start = [
        first.detach().to(device="cuda", dtype=torch.float64).clone(),
        second.detach().to(device="cuda", dtype=torch.float64).clone(),
    ]
    started = time.perf_counter()
    recovered, trace = coupled_loop(
        denoise, schedule, start, reverse=True, mix_weight=mix_weight, leapfrog=True,
    )
    return recovered, snaps_from_trace(trace), time.perf_counter() - started


def message_ok(scored: dict) -> bool:
    return bool(scored["detect"] and scored["decode_non_none"] and scored["exact_message_match"])


def main() -> int:
    config_path = Path(sys.argv[1])
    cfg = json.loads(config_path.read_text())
    pair = (round(float(cfg["guidance_scale"]), 1), round(float(cfg["image_guidance_scale"]), 1))
    if pair == (7.5, 1.5):
        print("FAIL: reference guidance is not rerun", flush=True)
        return 2
    if pair not in ALLOWED:
        print("FAIL: guidance is outside G1-G4", pair, flush=True)
        return 2
    slug = cfg["experiment_id"]
    case_dir = ROOT / "results" / "ip2p_edict_prc" / cfg["output_dir"]
    if (case_dir / "STATUS.txt").exists() or (PARTS / f"receiver_{slug}.csv").exists():
        print("FAIL: refusing to overwrite", case_dir, flush=True)
        return 2
    require_hash(ROOT / cfg["saved_pairs"], cfg["saved_pairs_sha256"], "pairs")
    require_hash(ROOT / cfg["saved_noise"], cfg["saved_noise_sha256"], "noise")
    require_hash(ROOT / cfg["key_path"], cfg["key_sha256"], "key")
    require_hash(ROOT / cfg["wrong_key_path"], cfg["wrong_key_sha256"], "wrong key")
    case_dir.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()

    saved = torch.load(ROOT / cfg["saved_pairs"], map_location="cpu")
    schedule = EdictSchedule(int(cfg["num_inference_steps"]), device="cpu", dtype=torch.float64)
    if not torch.equal(schedule.timesteps.cpu(), saved["timesteps"].to(torch.float64).cpu()):
        print("FAIL: saved timesteps differ", flush=True)
        return 2
    noise = np.load(ROOT / cfg["saved_noise"])
    z_prc = torch.from_numpy(np.asarray(noise["z_prc"])).to(dtype=torch.float64)
    forward_times = [None] + [int(schedule.timesteps[i].item()) for i in range(schedule.timesteps.shape[0])]
    inverse_times = [None] + list(reversed(forward_times[1:]))
    mix = float(cfg["mix_weight"])

    from diffusers import AutoencoderKL, StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], torch_dtype=torch.float16,
    ).to("cuda")
    signature = inspect.signature(pipe.__call__)
    if "guidance_scale" not in signature.parameters or "image_guidance_scale" not in signature.parameters:
        print("FAIL: Diffusers __call__ does not accept these guidance arguments", flush=True)
        return 2
    gate = pair[0] > 1.0 and pair[1] >= 1.0
    print("diffusers_three_branch_gate", gate, "scales", pair, flush=True)
    if not gate:
        print("FAIL: Diffusers would disable three-branch guidance", flush=True)
        return 2

    image_latents = saved["image_latents"].to(device="cuda", dtype=torch.float16)
    prompt_embeds = saved["prompt_embeds"].to(device="cuda", dtype=torch.float16)
    denoise = make_denoise(pipe.unet, image_latents, prompt_embeds, pair[0], pair[1])
    print("forward", slug, flush=True)
    start = [z_prc.to(device="cuda").clone(), z_prc.to(device="cuda").clone()]
    final, forward_trace = coupled_loop(
        denoise, schedule, start, reverse=False, mix_weight=mix, leapfrog=True,
    )
    forward_snaps = snaps_from_trace(forward_trace)
    edit_path = case_dir / "edit_stream0.png"
    decode(pipe, final[0]).save(edit_path)
    stable_path = OUT / "images" / f"{cfg['edit']}__{cfg['panel_role']}.png"
    decode(pipe, final[0]).save(stable_path)

    print("R1", flush=True)
    recovered_i, snaps_i, seconds_i = inverse_pair(denoise, schedule, final[0], final[1], mix)
    r1_cpu = [item.detach().to(torch.float64).cpu() for item in recovered_i]
    r1_scores = score_level(r1_cpu, cfg, z_prc)
    r1_ok = all(message_ok(item) for item in r1_scores)
    print("R1_ok", r1_ok, flush=True)

    recovered_levels = [("R1_true_final_pair", r1_cpu, seconds_i, 0.0, 1.0)]
    snaps_d = None
    seconds_d = None
    if r1_ok:
        print("R2", flush=True)
        x_only = final[0]
        recovered_d, snaps_d, seconds_d = inverse_pair(denoise, schedule, x_only, x_only, mix)
        r2_cpu = [item.detach().to(torch.float64).cpu() for item in recovered_d]
        recovered_levels.append(("R2_exact_x_duplicated", r2_cpu, seconds_d, 0.0, 1.0))

        print("R3", flush=True)
        reloaded = png_to_model(edit_path)
        scale = float(pipe.vae.config.scaling_factor)
        z_mode = encode_mode(pipe.vae, reloaded, scale)
        mode_mse = vector_mse(z_mode, final[0])
        mode_corr = vector_corr(z_mode, final[0])
        recovered_r3, _snaps_r3, seconds_r3 = inverse_pair(denoise, schedule, z_mode, z_mode, mix)
        r3_cpu = [item.detach().to(torch.float64).cpu() for item in recovered_r3]
        recovered_levels.append(("R3_png_vae_mode", r3_cpu, seconds_r3, mode_mse, mode_corr))

        print("R4", flush=True)
        vae_loss = AutoencoderKL.from_config(pipe.vae.config)
        vae_loss.load_state_dict({key: value.detach().float().cpu() for key, value in pipe.vae.state_dict().items()})
        vae_loss = vae_loss.to(device="cuda", dtype=torch.float32).eval()
        vae_loss.requires_grad_(False)
        recovery = {
            "scaling_factor": scale,
            "iterations": int(cfg["iterations"]),
            "checkpoints": cfg["checkpoints"],
            "learning_rate": float(cfg["learning_rate"]),
            "warmup_steps": int(cfg["warmup_steps"]),
        }
        _z0, saved_steps, _adam_trace, finite = recover_latent(reloaded, pipe.vae, vae_loss, recovery)
        if not finite or 100 not in saved_steps:
            print("FAIL: frozen decoder inversion did not return iteration 100", flush=True)
            return 2
        z100 = saved_steps[100]
        pixel = mean_pixel_loss(vae_loss, z100, reloaded, scale)
        inv_mse = vector_mse(z100, final[0])
        inv_corr = vector_corr(z100, final[0])
        torch.save(z100, case_dir / "decoder_inversion_100.pt")
        recovered_r4, _snaps_r4, seconds_r4 = inverse_pair(denoise, schedule, z100, z100, mix)
        r4_cpu = [item.detach().to(torch.float64).cpu() for item in recovered_r4]
        recovered_levels.append(("R4_png_decoder_inversion_100", r4_cpu, seconds_r4, inv_mse, inv_corr))
        (case_dir / "decoder_inversion_pixel_mse.txt").write_text(str(pixel) + "\n")
        del vae_loss

    guidance_config = f"text_{pair[0]:g}_image_{pair[1]:g}"
    trace = trace_rows("hybrid", cfg["edit"], guidance_config, "F", forward_snaps, forward_times, forward_snaps)
    trace.extend(trace_rows("hybrid", cfg["edit"], guidance_config, "I", snaps_i, inverse_times, forward_snaps))
    if snaps_d is not None:
        trace.extend(trace_rows("hybrid", cfg["edit"], guidance_config, "D", snaps_d, inverse_times, forward_snaps))
    summary = departure_row("hybrid", cfg["edit"], guidance_config, trace) if snaps_d is not None else None

    image_hash = sha256_file(edit_path)
    conditioning_hash = sha256_bytes(
        saved["image_latents"].numpy().tobytes() + saved["prompt_embeds"].numpy().tobytes()
    )
    digest = code_hash([
        ROOT / "experiments/ip2p_edict_prc/sampler.py",
        ROOT / "experiments/ip2p_edict_prc/run_milestone.py",
        ROOT / "experiments/ip2p_edict_prc/run_png_decoder_inversion.py",
        Path(__file__),
    ])
    config_digest = sha256_file(config_path)
    job_id = os.environ.get("SLURM_JOB_ID", "")
    sigma2 = float(cfg["sigma_squared"])
    sigma = math.sqrt(sigma2)
    forward_rows = [row for row in trace if row["path"] == "F"]
    terminal_xy = float(forward_rows[-1]["mse_xy"])
    _enc, decoding_key, message, key_codeword = load_key(ROOT / cfg["key_path"])
    if key_hash(_enc) != cfg["expected_key_hash"]:
        print("FAIL: key hash", flush=True)
        return 2
    codeword = np.asarray(noise["codeword"]).reshape(-1)
    if not np.array_equal(np.asarray(key_codeword).reshape(-1), codeword):
        print("FAIL: codeword mismatch", flush=True)
        return 2
    _wrong_enc, wrong_key, _wm, _wc = load_key(ROOT / cfg["wrong_key_path"])
    # The scores above were a stop check. Recompute them with the same function now that
    # the key is loaded for the table. R1 was already scored; score_level loads nothing else.
    rows = []
    for level, recovered, seconds, latent_mse, latent_corr in recovered_levels:
        fresh = score_level(recovered, cfg, z_prc)
        for stream, (tensor, scored) in enumerate(zip(recovered, fresh)):
            soft = recover_soft_bits(tensor, variance=sigma2)
            wrong_detect, wrong_decode, _wrong = detect_decode(wrong_key, soft)
            rows.append({
                "experiment_id": slug,
                "job_id": job_id,
                "code_hash": digest,
                "config_hash": config_digest,
                "image_hash": image_hash,
                "conditioning_hash": conditioning_hash,
                "key_id": cfg["expected_key_hash"],
                "receiver_level": level,
                "guidance_text": pair[0],
                "guidance_image": pair[1],
                "guidance_kind": "ip2p_three_branch",
                "recovered_stream": stream,
                "initial_noise_role": "evaluator-only",
                "sigma": sigma,
                "sigma_squared": sigma2,
                "terminal_xy_MSE": terminal_xy,
                "terminal_latent_MSE": latent_mse,
                "terminal_latent_corr": latent_corr,
                "runtime": seconds,
                "peak_memory": int(torch.cuda.max_memory_allocated()),
                "status": "ok" if r1_ok else "r1_failed",
                "wrong_key_detect": wrong_detect,
                "wrong_key_decode_non_none": wrong_decode,
                **scored,
            })

    write_new_csv(PARTS / f"trace_{slug}.csv", TRACE_FIELDS, trace)
    write_new_csv(PARTS / f"receiver_{slug}.csv", RECEIVER_FIELDS, rows)
    if summary is not None:
        write_new_csv(PARTS / f"departure_{slug}.csv", DEPARTURE_FIELDS, [summary])
    publish()
    quality_rows = []
    quality_error = ""
    try:
        quality_rows = [edit_quality(cfg, stable_path)]
    except Exception as exc:
        quality_error = repr(exc)
        print("quality failed", quality_error, flush=True)
    if quality_rows:
        write_new_csv(PARTS / f"quality_{slug}.csv", QUALITY_FIELDS, quality_rows)
    torch.save(
        {"x": forward_snaps[-1][0], "y": forward_snaps[-1][1], "role": "forward terminal pair"},
        case_dir / "forward_terminal.pt",
    )
    write_json(PARTS / f"manifest_{slug}.json", {
        "experiment_id": slug,
        "git_rev": git_rev(),
        "job_id": job_id,
        "guidance_text": pair[0],
        "guidance_image": pair[1],
        "diffusers_gate": gate,
        "r1_ok": r1_ok,
        "terminal_xy_mse": terminal_xy,
        "first_x_departure_step": None if summary is None else summary["first_x_departure_step"],
        "image_hash": image_hash,
        "conditioning_hash": conditioning_hash,
        "code_hash": digest,
        "config_hash": config_digest,
        "quality_error": quality_error,
        "gpu": torch.cuda.get_device_name(0),
        "peak_memory": int(torch.cuda.max_memory_allocated()),
    })
    (PARTS / f"status_{slug}.txt").write_text(
        f"{'R1_FAIL' if not r1_ok else 'RECORDED'} {slug} job {job_id}\n"
        f"terminal_xy_mse={terminal_xy}\n"
    )
    (case_dir / "STATUS.txt").write_text(("R1_FAIL\n" if not r1_ok else "RECORDED\n"))
    publish()
    print("terminal_xy", terminal_xy, "r1", r1_ok, flush=True)
    return 0 if r1_ok else 2


def score_level(recovered, cfg, z_prc):
    _enc, decoding_key, message, key_codeword = load_key(ROOT / cfg["key_path"])
    if key_hash(_enc) != cfg["expected_key_hash"]:
        raise SystemExit("key hash mismatch")
    codeword = np.asarray(key_codeword).reshape(-1)
    sigma2 = float(cfg["sigma_squared"])
    return [
        score_stream(tensor, z_prc, decoding_key, codeword, message, sigma2)
        for tensor in recovered
    ]


def edit_quality(cfg, image_path: Path) -> dict:
    from PIL import Image
    from experiments.instructpix2pix_prc.run_pair import clip_bundle, directional_clip
    from experiments.metrics import clip_score, lpips_distance, pixel_mse

    source = Image.open(ROOT / cfg["source_png"]).convert("RGB")
    edit = Image.open(image_path).convert("RGB")
    lpips, lpips_error = lpips_distance(edit, source, device="cuda")
    model, processor = clip_bundle("cuda")
    directional = directional_clip(
        edit, source, cfg["instruction"], cfg["source_caption"], model, processor, "cuda",
    )
    clip_instruction, _err = clip_score(edit, cfg["instruction"], device="cuda")
    clip_caption, _err = clip_score(edit, cfg["source_caption"], device="cuda")
    return {
        "experiment_id": cfg["experiment_id"],
        "edit": cfg["edit"],
        "guidance_text": cfg["guidance_scale"],
        "guidance_image": cfg["image_guidance_scale"],
        "role": cfg["panel_role"],
        "lpips_vs_source": lpips,
        "lpips_error": lpips_error or "",
        "pixel_mse_vs_source": pixel_mse(edit, source),
        "directional_clip": directional,
        "clip_instruction": clip_instruction,
        "clip_source_caption": clip_caption,
        "source_caption": cfg["source_caption"],
        "instruction": cfg["instruction"],
        "image_path": str(image_path.relative_to(ROOT)),
    }


if __name__ == "__main__":
    sys.exit(main())
