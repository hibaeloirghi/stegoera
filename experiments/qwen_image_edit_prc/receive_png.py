"""Operational receiver: PNG bytes, decoding key, and fixed settings.

This module does not open the sender config, the source photograph, the
editing instruction, or any saved latent. The empty user string below is read
from the fixed settings file.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import torch
from PIL import Image

from experiments.qwen_image_edit_prc.editing import (
    build_schedule,
    encode_png_conditioning,
    invert_packed,
    load_pipeline,
    unpack_noise,
)
from experiments.qwen_image_edit_prc.io_util import read_png, write_json
from experiments.qwen_image_edit_prc.prc_adapter import load_decoding_key_only, score_latent


def recover(png_path: Path, decoding_key_path: Path, settings_path: Path, out_dir: Path, pipe=None):
    png_bytes, png_hash = read_png(png_path)
    settings = json.loads(Path(settings_path).read_text())
    if settings.get("user_instruction") not in ("",):
        raise RuntimeError("the fixed receiver instruction must be empty")
    if int(settings["num_inference_steps"]) != 50 or float(settings["true_cfg_scale"]) != 4.0:
        raise RuntimeError("receiver settings are not the fixed official inference configuration")
    decoding_key = load_decoding_key_only(decoding_key_path)
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if pipe is None:
        pipe = load_pipeline(settings["model_id"], settings["revision"])
    conditioning, final_estimate, height, width = encode_png_conditioning(
        pipe,
        image,
        settings["user_instruction"],
        settings["negative_prompt"],
        float(settings["true_cfg_scale"]),
    )
    schedule = build_schedule(pipe, final_estimate, int(settings["num_inference_steps"]))
    estimated_packed = invert_packed(
        pipe,
        final_estimate,
        conditioning,
        schedule,
        float(settings["true_cfg_scale"]),
    )
    estimated = unpack_noise(pipe, estimated_packed, height, width).reshape(-1)
    # message=None: this call must not see the target payload.
    scored = score_latent(decoding_key, estimated, message=None)
    elapsed = time.perf_counter() - started
    peak = None
    if torch.cuda.is_available():
        peak = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    result = {
        "role": "receiver_png",
        "png_sha256": png_hash,
        "decoded_bits": scored["decoded_bits"],
        "detect": scored["detect"],
        "decode_non_none": scored["decode_non_none"],
        "detector_statistic": scored["detector_statistic"],
        "detector_threshold": scored["detector_threshold"],
        "runtime_sec": elapsed,
        "peak_gpu_mib": peak,
        "procedure": settings["procedure"],
        "inputs": "png bytes, decoding key, fixed settings",
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "decoded.json", result)
    return result, estimated
