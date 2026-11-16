"""Photo 1 follow-up on Qwen-Image-Edit.

Three instructions share the Gaussian draw from the iPhone batch. Four more
draws repeat the eye-color instruction. Each edit records a diagnostic
inversion, which sees the source conditioning and the real instruction, and
the operational PNG receiver, which does not.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.prc_runtime import sign_agreement  # noqa: E402
from experiments.qwen_image_edit_prc.editing import (  # noqa: E402
    conditioning_from_capture,
    invert_packed,
    load_pipeline,
    pack_noise,
    park_on_cpu,
    run_forward,
    unpack_noise,
)
from experiments.qwen_image_edit_prc.geometry import block_length, edit_canvas  # noqa: E402
from experiments.qwen_image_edit_prc.io_util import (  # noqa: E402
    create_directory,
    sha256_file,
    write_json,
)
from experiments.qwen_image_edit_prc.prc_adapter import (  # noqa: E402
    key_basename,
    load_sender_key,
    score_latent,
    sign_magnitudes,
)
from experiments.qwen_image_edit_prc.receive_png import recover  # noqa: E402
from experiments.qwen_image_edit_prc.run_iphone_batch import to_unit  # noqa: E402
from experiments.qwen_image_edit_prc.run_pilot import pixel_metrics, save_png  # noqa: E402


def gap(estimate: torch.Tensor, planted: torch.Tensor, codeword: torch.Tensor) -> dict:
    flat = estimate.detach().reshape(-1).cpu()
    planted_flat = planted.detach().reshape(-1).cpu()
    if flat.numel() != planted_flat.numel():
        raise RuntimeError(f"recovered length {flat.numel()} is not the planted length {planted_flat.numel()}")
    agreement, flip = sign_agreement(flat, codeword)
    return {
        "z_max_abs": float((flat.double() - planted_flat.double()).abs().max().item()),
        "sign_agreement": agreement,
        "sign_flip_rate": flip,
    }


def main() -> None:
    cfg = json.loads(Path(sys.argv[1]).read_text())
    controls = json.loads((ROOT / cfg["controls_json"]).read_text())
    if not controls.get("passed"):
        raise RuntimeError("120-bit controls have not passed")
    photo = ROOT / cfg["image"]
    if sha256_file(photo) != cfg["sha256"]:
        raise RuntimeError("photo 1 hash does not match the config")
    image = Image.open(photo).convert("RGB")
    canvas_w, canvas_h = edit_canvas(*image.size)
    length = block_length(canvas_w, canvas_h)
    stem = key_basename(cfg["payload_length"])
    key_dir = ROOT / cfg["keys_dir"]
    _encoding, decoding_key, message, codeword = load_sender_key(key_dir / f"{stem}_sender.pkl")
    message = [int(bit) for bit in message]
    codeword = torch.as_tensor(codeword)
    if len(message) != int(cfg["payload_length"]) or int(codeword.numel()) != length:
        raise RuntimeError(f"saved key does not match photo 1 n={length}")
    original_seed = int(cfg["original_gaussian_seed"])
    extra_seeds = [int(seed) for seed in cfg["extra_seeds"]]
    if original_seed in extra_seeds or len(set(extra_seeds)) != len(extra_seeds):
        raise RuntimeError("extra seeds overlap the original photo-1 seed")
    cases = [
        {
            "role": "sweep",
            "id": spec["id"],
            "instruction": spec["instruction"],
            "gaussian_seed": original_seed,
        }
        for spec in cfg["sweep"]
    ]
    cases.extend(
        {
            "role": "seed",
            "id": f"seed{seed}",
            "instruction": cfg["failure_instruction"],
            "gaussian_seed": seed,
        }
        for seed in extra_seeds
    )
    out_root = ROOT / cfg["output_dir"]
    create_directory(out_root)
    prior = json.loads((ROOT / cfg["prior_summary"]).read_text())
    prior_eye = next(row for row in prior["images"] if row["index"] == 1)
    print(json.dumps({
        "model": cfg["model_id"],
        "photo": photo.name,
        "n": length,
        "cases": len(cases),
        "prior_eye_margin": prior_eye["prc"]["detector_margin"],
        "prior_eye_exact": prior_eye["prc"]["exact_message"],
    }), flush=True)
    pipe = load_pipeline(cfg["model_id"], cfg["revision"])
    measure = None
    try:
        import lpips

        measure = lpips.LPIPS(net="alex")
    except Exception as exc:
        write_json(out_root / "lpips_error.json", {"error": f"{type(exc).__name__}: {exc}"})
    prepared: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def planted_noise(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seed not in prepared:
            gaussian = np.random.default_rng(seed).standard_normal(length)
            unpacked = sign_magnitudes(codeword, gaussian).reshape(
                1, 1, 16, 2 * (canvas_h // 16), 2 * (canvas_w // 16)
            )
            prepared[seed] = (pack_noise(pipe, unpacked.float()), unpacked)
        return prepared[seed]

    resized = pipe.image_processor.resize(image, canvas_h, canvas_w)
    source = to_unit(pipe.image_processor.preprocess(resized, canvas_h, canvas_w))
    rows = []
    for case in cases:
        started = time.perf_counter()
        case_dir = out_root / case["id"]
        create_directory(case_dir)
        (case_dir / "instruction.txt").write_text(case["instruction"] + "\n")
        packed, unpacked = planted_noise(case["gaussian_seed"])
        images, trajectory, captured, scheduler = run_forward(
            pipe,
            image,
            case["instruction"],
            cfg["negative_prompt"],
            packed.detach().clone(),
            int(cfg["num_inference_steps"]),
            float(cfg["true_cfg_scale"]),
            float(cfg["guidance_scale"]),
        )
        png_digest, _pil = save_png(pipe, images, case_dir / "prc.png", case_dir / "prc_float.pt")
        unit = to_unit(images)
        pixels = pixel_metrics(source, unit)
        lpips_source = None
        if measure is not None:
            lpips_source = float(measure(source.unsqueeze(0) * 2 - 1, unit.unsqueeze(0) * 2 - 1).item())
        conditioning = conditioning_from_capture(captured, canvas_h, canvas_w, pipe.vae_scale_factor)
        estimated_diag = invert_packed(
            pipe, trajectory[-1], conditioning, scheduler, float(cfg["true_cfg_scale"])
        )
        unpacked_diag = unpack_noise(pipe, estimated_diag, canvas_h, canvas_w).reshape(-1)
        diagnostic = score_latent(decoding_key, unpacked_diag, message)
        diagnostic_gap = gap(unpacked_diag, unpacked, codeword)
        del images, trajectory, captured, conditioning, estimated_diag, unpacked_diag, unit
        park_on_cpu(pipe)
        receiver_result, estimated_op = recover(
            case_dir / "prc.png",
            key_dir / f"{stem}_receiver.pkl",
            ROOT / cfg["receiver_settings"],
            case_dir / "receiver_png",
            pipe=pipe,
        )
        operational = score_latent(decoding_key, estimated_op, message)
        operational_gap = gap(estimated_op, unpacked, codeword)
        del estimated_op
        park_on_cpu(pipe)
        row = {
            "role": case["role"],
            "id": case["id"],
            "instruction": case["instruction"],
            "gaussian_seed": case["gaussian_seed"],
            "canvas": [canvas_w, canvas_h],
            "n": length,
            "png_sha256": png_digest,
            "receiver_png_sha256": receiver_result["png_sha256"],
            "pixel_source_prc": pixels,
            "lpips_source_prc": lpips_source,
            "diagnostic": diagnostic,
            "diagnostic_gap": diagnostic_gap,
            "operational": operational,
            "operational_gap": operational_gap,
            "runtime_sec": time.perf_counter() - started,
        }
        if case["id"] == "eye" and case["gaussian_seed"] == original_seed:
            row["prior_png_sha256"] = prior_eye["png_prc_sha256"]
            row["prior_operational"] = prior_eye["prc"]
            row["prior_lpips_source_prc"] = prior_eye["lpips_source_prc"]
        write_json(case_dir / "manifest.json", row)
        rows.append(row)
        print(json.dumps({
            "id": case["id"],
            "lpips": lpips_source,
            "diagnostic_exact": diagnostic["exact_message"],
            "diagnostic_margin": diagnostic["detector_margin"],
            "operational_exact": operational["exact_message"],
            "operational_detect": operational["detect"],
            "operational_margin": operational["detector_margin"],
            "runtime_sec": row["runtime_sec"],
        }), flush=True)
    summary = {
        "model_id": cfg["model_id"],
        "revision": cfg["revision"],
        "photo_sha256": cfg["sha256"],
        "payload_length": int(cfg["payload_length"]),
        "original_gaussian_seed": original_seed,
        "cases": rows,
        "operational_exact_count": sum(bool(row["operational"]["exact_message"]) for row in rows),
        "diagnostic_exact_count": sum(bool(row["diagnostic"]["exact_message"]) for row in rows),
        "case_count": len(rows),
    }
    write_json(out_root / "summary.json", summary)
    print(json.dumps({
        "operational_exact": summary["operational_exact_count"],
        "diagnostic_exact": summary["diagnostic_exact_count"],
        "cases": len(rows),
    }), flush=True)


if __name__ == "__main__":
    main()
