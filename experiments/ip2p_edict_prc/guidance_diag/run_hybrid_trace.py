"""Replay the saved hybrid edits and log the x-y gap.

Text 7.5 and image 1.5 are not a new sender. The forward starts from the saved
PRC noise and the saved conditioning. Paths I and D then start from the saved
final tensors. The forward snapshots are compared with those inverses only
after the inverses return.
"""
from __future__ import annotations

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
    overlay_plot,
    publish,
    require_hash,
    sha256_bytes,
    sha256_file,
    snaps_from_trace,
    trace_rows,
    write_json,
    write_new_csv,
)
from experiments.ip2p_edict_prc.run_milestone import make_denoise  # noqa: E402
from experiments.ip2p_edict_prc.run_png_decoder_inversion import score_stream  # noqa: E402
from experiments.ip2p_edict_prc.sampler import EdictSchedule, coupled_loop  # noqa: E402
from experiments.prc_runtime import detect_decode, recover_soft_bits  # noqa: E402
from experiments.receiver_aligned.key_io import key_hash, load_key  # noqa: E402

SLUG = "hybrid_reference"


def fail(reason: str) -> int:
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / f"status_{SLUG}.txt").write_text(f"FAIL hybrid_reference\n{reason}\n")
    publish()
    print("FAIL:", reason, flush=True)
    return 2


def inverse_pair(denoise, schedule, first, second, mix_weight: float):
    """Matched inverse. The only latents this function sees are first and second."""
    start = [
        first.detach().to(device="cuda", dtype=torch.float64).clone(),
        second.detach().to(device="cuda", dtype=torch.float64).clone(),
    ]
    started = time.perf_counter()
    recovered, trace = coupled_loop(
        denoise, schedule, start, reverse=True, mix_weight=mix_weight, leapfrog=True,
    )
    elapsed = time.perf_counter() - started
    return recovered, snaps_from_trace(trace), elapsed


def max_abs_against(candidate, reference) -> float:
    values = []
    for left, right in zip(candidate, reference):
        values.append(float((left.detach().cpu().to(torch.float64) - right.to(torch.float64)).abs().max().item()))
    return max(values)


def main() -> int:
    config_path = Path(sys.argv[1])
    cfg = json.loads(config_path.read_text())
    if float(cfg["guidance_scale"]) != 7.5 or float(cfg["image_guidance_scale"]) != 1.5:
        return fail("reference replay must stay at text 7.5 and image 1.5")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    (OUT / "endpoints").mkdir(exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()

    for edit in cfg["edits"]:
        require_hash(ROOT / edit["saved_pairs"], edit["saved_pairs_sha256"], edit["edit"])
    require_hash(ROOT / cfg["saved_noise"], cfg["saved_noise_sha256"], "noise")

    noise_file = np.load(ROOT / cfg["saved_noise"])
    z_prc = torch.from_numpy(np.asarray(noise_file["z_prc"])).to(dtype=torch.float64)
    schedule = EdictSchedule(int(cfg["num_inference_steps"]), device="cpu", dtype=torch.float64)
    forward_times = [None] + [int(schedule.timesteps[i].item()) for i in range(schedule.timesteps.shape[0])]
    inverse_times = [None] + list(reversed(forward_times[1:]))

    from diffusers import StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], torch_dtype=torch.float16,
    ).to("cuda")
    print("gpu", torch.cuda.get_device_name(0), flush=True)

    trace_all = []
    receiver_rows = []
    departure_rows = []
    regression_flags = []
    endpoint_dir = OUT / "endpoints"
    mix = float(cfg["mix_weight"])
    held_gaussian = []

    for edit in cfg["edits"]:
        saved = torch.load(ROOT / edit["saved_pairs"], map_location="cpu")
        if not torch.equal(schedule.timesteps.cpu(), saved["timesteps"].to(torch.float64).cpu()):
            return fail(f"{edit['edit']} timesteps differ from the EDICT grid")
        image_latents = saved["image_latents"].to(device="cuda", dtype=torch.float16)
        prompt_embeds = saved["prompt_embeds"].to(device="cuda", dtype=torch.float16)
        if image_latents.shape[0] != 3 or prompt_embeds.shape[0] != 3:
            return fail(f"{edit['edit']} conditioning is not three-branch")
        denoise = make_denoise(
            pipe.unet, image_latents, prompt_embeds,
            float(cfg["guidance_scale"]), float(cfg["image_guidance_scale"]),
        )
        print(edit["edit"], "forward", flush=True)
        start = [z_prc.to(device="cuda").clone(), z_prc.to(device="cuda").clone()]
        final, forward_trace = coupled_loop(
            denoise, schedule, start, reverse=False, mix_weight=mix, leapfrog=True,
        )
        forward_snaps = snaps_from_trace(forward_trace)
        endpoint_error = max_abs_against(final, saved["prc_final"])
        print(edit["edit"], "endpoint_max_abs", endpoint_error, flush=True)
        if endpoint_error > 1e-3:
            return fail(f"{edit['edit']} replay endpoint max abs {endpoint_error} exceeds 1e-3")

        true_x = saved["prc_final"][0]
        true_y = saved["prc_final"][1]
        print(edit["edit"], "inverse true pair", flush=True)
        recovered_i, snaps_i, seconds_i = inverse_pair(denoise, schedule, true_x, true_y, mix)
        print(edit["edit"], "inverse duplicated x", flush=True)
        recovered_d, snaps_d, seconds_d = inverse_pair(denoise, schedule, true_x, true_x, mix)
        inverse_error = max_abs_against(recovered_i, saved["prc_recovered"])
        print(edit["edit"], "inverse_vs_saved_max_abs", inverse_error, flush=True)
        if inverse_error > 1e-3:
            regression_flags.append(f"{edit['edit']} inverse max abs {inverse_error}")

        guidance_config = "text_7.5_image_1.5"
        rows = []
        rows.extend(trace_rows(cfg["pipeline"], edit["edit"], guidance_config, "F", forward_snaps, forward_times, forward_snaps))
        rows.extend(trace_rows(cfg["pipeline"], edit["edit"], guidance_config, "I", snaps_i, inverse_times, forward_snaps))
        rows.extend(trace_rows(cfg["pipeline"], edit["edit"], guidance_config, "D", snaps_d, inverse_times, forward_snaps))
        trace_all.extend(rows)
        departure_rows.append(departure_row(cfg["pipeline"], edit["edit"], guidance_config, rows))
        overlay_plot(
            rows,
            f"{edit['edit']} text 7.5 image 1.5",
            OUT / "plots" / f"overlay_hybrid_{edit['edit']}.png",
        )
        torch.save(
            {
                "role": "evaluator-only forward trajectory; not an inverse input",
                "forward": forward_snaps,
                "endpoint_max_abs_vs_saved": endpoint_error,
                "inverse_max_abs_vs_saved_recovered": inverse_error,
            },
            endpoint_dir / f"hybrid_{edit['edit']}_evaluator_forward.pt",
        )
        torch.save(
            {"x": snaps_i[-1][0], "y": snaps_i[-1][1], "level": "I"},
            endpoint_dir / f"hybrid_{edit['edit']}_I_endpoints.pt",
        )
        torch.save(
            {"x": snaps_d[-1][0], "y": snaps_d[-1][1], "level": "D"},
            endpoint_dir / f"hybrid_{edit['edit']}_D_endpoints.pt",
        )
        held_gaussian.append({
            "edit": edit,
            "saved": saved,
            "recovered_i": [item.detach().to(torch.float64).cpu() for item in recovered_i],
            "recovered_d": [item.detach().to(torch.float64).cpu() for item in recovered_d],
            "seconds_i": seconds_i,
            "seconds_d": seconds_d,
            "inverse_error": inverse_error,
            "terminal_xy": rows[50]["mse_xy"],
        })

    del pipe
    torch.cuda.empty_cache()

    _enc, decoding_key, message, key_codeword = load_key(ROOT / cfg["key_path"])
    if key_hash(_enc) != cfg["expected_key_hash"]:
        return fail(f"key hash {key_hash(_enc)}")
    codeword = np.asarray(noise_file["codeword"]).reshape(-1)
    if not np.array_equal(np.asarray(key_codeword).reshape(-1), codeword):
        return fail("saved codeword does not match the PRC noise")
    _wrong_enc, wrong_key, _wm, _wc = load_key(ROOT / cfg["wrong_key_path"])
    sigma2 = float(cfg["sigma_squared"])
    sigma = math.sqrt(sigma2)
    job_id = os.environ.get("SLURM_JOB_ID", "")
    digest = code_hash([
        ROOT / "experiments/ip2p_edict_prc/sampler.py",
        ROOT / "experiments/ip2p_edict_prc/run_milestone.py",
        Path(__file__),
    ])
    config_digest = sha256_file(config_path)
    regression_ok = True

    for item in held_gaussian:
        edit = item["edit"]
        saved = item["saved"]
        conditioning_hash = sha256_bytes(
            saved["image_latents"].numpy().tobytes() + saved["prompt_embeds"].numpy().tobytes()
        )
        image_hash = sha256_file(ROOT / edit["reference_png"])
        terminal_xy = float(item["terminal_xy"])
        for level, recovered, seconds in (
            ("I_true_final_pair", item["recovered_i"], item["seconds_i"]),
            ("D_exact_x_duplicated", item["recovered_d"], item["seconds_d"]),
        ):
            for stream, tensor in enumerate(recovered):
                scored = score_stream(tensor, z_prc, decoding_key, codeword, message, sigma2)
                soft = recover_soft_bits(tensor, variance=sigma2)
                wrong_detect, wrong_decode, _wrong = detect_decode(wrong_key, soft)
                if level.startswith("I") and not (
                    scored["detect"] and scored["decode_non_none"] and scored["exact_message_match"]
                ):
                    regression_ok = False
                if edit["edit"] == "golden_retriever" and level.startswith("D") and not scored["exact_message_match"]:
                    regression_ok = False
                if edit["edit"] == "red_ball" and level.startswith("D") and (
                    scored["exact_message_match"] or scored["detect"] or scored["decode_non_none"]
                ):
                    regression_ok = False
                receiver_rows.append({
                    "experiment_id": "hybrid_reference_trace",
                    "job_id": job_id,
                    "code_hash": digest,
                    "config_hash": config_digest,
                    "image_hash": image_hash,
                    "conditioning_hash": conditioning_hash,
                    "key_id": cfg["expected_key_hash"],
                    "receiver_level": level,
                    "guidance_text": 7.5,
                    "guidance_image": 1.5,
                    "guidance_kind": "ip2p_three_branch",
                    "recovered_stream": stream,
                    "initial_noise_role": "evaluator-only",
                    "sigma": sigma,
                    "sigma_squared": sigma2,
                    "terminal_xy_MSE": terminal_xy,
                    "terminal_latent_MSE": 0.0 if level.startswith("I") or level.startswith("D") else "",
                    "terminal_latent_corr": 1.0 if level.startswith("I") or level.startswith("D") else "",
                    "runtime": seconds,
                    "peak_memory": int(torch.cuda.max_memory_allocated()),
                    "status": "pending",
                    "wrong_key_detect": wrong_detect,
                    "wrong_key_decode_non_none": wrong_decode,
                    **scored,
                })
        for stream, tensor in enumerate(saved["gaussian_recovered"]):
            scored = score_stream(tensor, z_prc, decoding_key, codeword, message, sigma2)
            receiver_rows.append({
                "experiment_id": "hybrid_reference_gaussian_saved",
                "job_id": job_id,
                "code_hash": digest,
                "config_hash": config_digest,
                "image_hash": sha256_file(ROOT / edit["gaussian_png"]),
                "conditioning_hash": conditioning_hash,
                "key_id": cfg["expected_key_hash"],
                "receiver_level": "saved_gaussian_inverse",
                "guidance_text": 7.5,
                "guidance_image": 1.5,
                "guidance_kind": "ip2p_three_branch",
                "recovered_stream": stream,
                "initial_noise_role": "evaluator-only",
                "sigma": sigma,
                "sigma_squared": sigma2,
                "terminal_xy_MSE": "",
                "terminal_latent_MSE": "",
                "terminal_latent_corr": "",
                "runtime": 0.0,
                "peak_memory": int(torch.cuda.max_memory_allocated()),
                "status": "control_existing_output",
                "wrong_key_detect": "",
                "wrong_key_decode_non_none": "",
                **scored,
            })
            if scored["detect"] or scored["exact_message_match"]:
                regression_ok = False

    if regression_flags:
        regression_ok = False
        print("regression_flags", regression_flags, flush=True)
    status = "PASS" if regression_ok else "FAIL"
    for row in receiver_rows:
        if row["status"] == "pending":
            row["status"] = status
    write_new_csv(PARTS / f"trace_{SLUG}.csv", TRACE_FIELDS, trace_all)
    write_new_csv(PARTS / f"receiver_{SLUG}.csv", RECEIVER_FIELDS, receiver_rows)
    write_new_csv(PARTS / f"departure_{SLUG}.csv", DEPARTURE_FIELDS, departure_rows)
    publish()
    try:
        quality_rows = reference_quality(cfg)
    except Exception as exc:
        quality_rows = []
        print("quality failed", repr(exc), flush=True)
    if quality_rows:
        write_new_csv(PARTS / f"quality_{SLUG}.csv", QUALITY_FIELDS, quality_rows)
    write_json(PARTS / f"manifest_{SLUG}.json", {
        "stage": "hybrid_reference_trace",
        "git_rev": git_rev(),
        "job_id": job_id,
        "code_hash": digest,
        "config_hash": config_digest,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "peak_memory": int(torch.cuda.max_memory_allocated()),
        "regression_ok": regression_ok,
        "departure": departure_rows,
        "note": "Reference text 7.5 image 1.5 was replayed from saved noise. It was not a new guidance config.",
    })
    (PARTS / f"status_{SLUG}.txt").write_text(
        f"{status} hybrid_reference job {job_id}\n"
        + "\n".join(
            f"{row['edit']} terminal_xy_mse={row['terminal_xy_mse']} first_x_departure={row['first_x_departure_step']}"
            for row in departure_rows
        )
        + "\n"
    )
    publish()
    print(status, flush=True)
    return 0 if regression_ok else 2


def reference_quality(cfg) -> list[dict]:
    """LPIPS and directional CLIP on the saved reference PNGs. No new edit."""
    try:
        from PIL import Image
        from experiments.instructpix2pix_prc.run_pair import clip_bundle, directional_clip
        from experiments.metrics import clip_score, lpips_distance, pixel_mse
    except Exception as exc:
        print("quality import failed", exc, flush=True)
        return []
    try:
        model, processor = clip_bundle("cuda")
    except Exception as exc:
        print("CLIP load failed", exc, flush=True)
        model = processor = None
    rows = []
    for edit in cfg["edits"]:
        source = Image.open(ROOT / edit["source_png"]).convert("RGB")
        reference = Image.open(ROOT / edit["reference_png"]).convert("RGB")
        lpips, lpips_error = lpips_distance(reference, source, device="cuda")
        directional = ""
        clip_instruction = ""
        clip_caption = ""
        if model is not None:
            directional = directional_clip(
                reference, source, edit["instruction"], edit["source_caption"], model, processor, "cuda",
            )
        else:
            clip_instruction, _err = clip_score(reference, edit["instruction"], device="cuda")
        caption_score, _err = clip_score(source, edit["source_caption"], device="cuda")
        edit_caption, _err = clip_score(reference, edit["source_caption"], device="cuda")
        clip_caption = edit_caption
        rows.append({
            "experiment_id": "hybrid_reference_saved_png",
            "edit": edit["edit"],
            "guidance_text": 7.5,
            "guidance_image": 1.5,
            "role": "reference",
            "lpips_vs_source": lpips,
            "lpips_error": lpips_error or "",
            "pixel_mse_vs_source": pixel_mse(reference, source),
            "directional_clip": directional,
            "clip_instruction": clip_instruction if clip_instruction != "" else clip_score(reference, edit["instruction"], device="cuda")[0],
            "clip_source_caption": clip_caption,
            "source_caption": edit["source_caption"],
            "instruction": edit["instruction"],
            "image_path": edit["reference_png"],
        })
        rows.append({
            "experiment_id": "hybrid_source",
            "edit": edit["edit"],
            "guidance_text": "",
            "guidance_image": "",
            "role": "source",
            "lpips_vs_source": 0.0,
            "lpips_error": "",
            "pixel_mse_vs_source": 0.0,
            "directional_clip": "",
            "clip_instruction": "",
            "clip_source_caption": caption_score,
            "source_caption": edit["source_caption"],
            "instruction": edit["instruction"],
            "image_path": edit["source_png"],
        })
    return rows


if __name__ == "__main__":
    sys.exit(main())
