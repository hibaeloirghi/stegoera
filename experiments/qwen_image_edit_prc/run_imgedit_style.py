"""ImgEdit-Bench style split on Qwen-Image-Edit, with one PRC key.

Each case runs the benchmark instruction, once with ordinary noise and once
with PRC signs on the same magnitudes. The operational receiver then sees
only the PNG, the decoding key, and the empty instruction. A missed message
is a recorded result. This script does not exit early because of it.
"""
from __future__ import annotations

import json
import os
import shutil
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
    load_pipeline,
    pack_noise,
    park_on_cpu,
    run_forward,
)
from experiments.qwen_image_edit_prc.geometry import block_length, edit_canvas  # noqa: E402
from experiments.qwen_image_edit_prc.io_util import (  # noqa: E402
    create_directory,
    read_png,
    sha256_file,
    write_bytes,
    write_json,
)
from experiments.qwen_image_edit_prc.prc_adapter import (  # noqa: E402
    key_basename,
    load_sender_key,
    score_latent,
    sign_magnitudes,
)
from experiments.qwen_image_edit_prc.receive_png import recover  # noqa: E402
from experiments.qwen_image_edit_prc.run_iphone_batch import (  # noqa: E402
    bits_to_text,
    caption,
    draw_panel,
    to_unit,
)
from experiments.qwen_image_edit_prc.run_pilot import pixel_metrics  # noqa: E402


def ssim_luminance(left: torch.Tensor, right: torch.Tensor) -> float:
    """Wang SSIM on the luminance of two CxHxW images in [0, 1]."""
    if left.shape != right.shape or left.ndim != 3 or left.shape[0] != 3:
        raise RuntimeError(f"SSIM expects matching CxHxW images, got {tuple(left.shape)} {tuple(right.shape)}")

    def luma(image: torch.Tensor) -> torch.Tensor:
        return 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]

    coords = torch.arange(11, dtype=torch.float32) - 5
    weights = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    weights = weights / weights.sum()
    kernel = (weights[:, None] * weights[None, :]).view(1, 1, 11, 11)
    first = luma(left.float()).view(1, 1, *left.shape[1:])
    second = luma(right.float()).view(1, 1, *right.shape[1:])

    def conv(image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(image, kernel, padding=5)

    mu_a, mu_b = conv(first), conv(second)
    var_a = conv(first * first) - mu_a.square()
    var_b = conv(second * second) - mu_b.square()
    cov = conv(first * second) - mu_a * mu_b
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    denominator = (mu_a.square() + mu_b.square() + c1) * (var_a + var_b + c2)
    return float((numerator / denominator).mean().item())


def save_png_only(pipe, image_tensor, png_path: Path):
    image = image_tensor if image_tensor.ndim == 4 else image_tensor.unsqueeze(0)
    numpy_image = pipe.image_processor.pt_to_numpy(image.float().cpu())
    pil = pipe.image_processor.numpy_to_pil(numpy_image)[0]
    buffer = __import__("io").BytesIO()
    pil.save(buffer, format="PNG")
    write_bytes(png_path, buffer.getvalue())
    _raw, digest = read_png(png_path)
    return digest, pil


def mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("aggregate received an empty or non-finite metric")
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0)), "n": int(array.size)}


def slim_score(score: dict) -> dict:
    return {key: value for key, value in score.items() if key != "decoded_bits"}


def aggregate(rows: list[dict]) -> dict:
    def rate(stream: str, field: str) -> float:
        return float(np.mean([bool(row[stream][field]) for row in rows]))

    return {
        "n": len(rows),
        "prc_detection_rate": rate("prc", "detect"),
        "prc_exact_message_rate": rate("prc", "exact_message"),
        "ordinary_detection_rate": rate("gaussian", "detect"),
        "ordinary_exact_message_rate": rate("gaussian", "exact_message"),
        "false_positive_bound_F": 1e-5,
        "lpips_ordinary_vs_prc": mean_std([row["lpips_gaussian_prc"] for row in rows]),
        "psnr_ordinary_vs_prc": mean_std([row["psnr_gaussian_prc"] for row in rows]),
        "ssim_ordinary_vs_prc": mean_std([row["ssim_gaussian_prc"] for row in rows]),
        "lpips_source_vs_prc": mean_std([row["lpips_source_prc"] for row in rows]),
        "psnr_source_vs_prc": mean_std([row["psnr_source_prc"] for row in rows]),
        "ssim_source_vs_prc": mean_std([row["ssim_source_prc"] for row in rows]),
        "sign_agreement_prc": mean_std([row["sign_agreement_prc"] for row in rows]),
        "sign_agreement_ordinary": mean_std([row["sign_agreement_gaussian"] for row in rows]),
        "ssim_definition": "luminance, Gaussian window 11, sigma 1.5, data range 1",
        "note": (
            "Detection rate and exact-message rate are separate. "
            "Ordinary edits are the unwatermarked control. "
            "LPIPS, PSNR, and SSIM between ordinary and PRC edits are the quality shift. "
            "This key uses F=1e-5. Gunn's headline table uses FPR 0.01. "
            "Sign agreement is not in Gunn's tables."
        ),
    }


def write_replacing(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def merge_shards(output_dir: Path, shard_count: int) -> None:
    rows = []
    for shard in range(shard_count):
        path = output_dir / f"shard_{shard:02d}" / "summary.json"
        if not path.exists():
            return
        summary = json.loads(path.read_text())
        rows.extend(summary["cases"])
    rows.sort(key=lambda row: int(row["key"]))
    payload = {
        "task": "imgedit_style",
        "shards": shard_count,
        **aggregate(rows),
        "cases": rows,
    }
    write_replacing(output_dir / "summary.json", payload)
    print(json.dumps({"merged": payload["n"], "prc_exact_message_rate": payload["prc_exact_message_rate"]}), flush=True)


def selected_cases(cfg: dict, catalog: dict) -> tuple[int, list[dict]]:
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is None:
        raise SystemExit("submit experiments/slurm/qwen_image_edit_prc/imgedit_style.sh so SLURM_ARRAY_TASK_ID is set")
    shard = int(task)
    shard_count = int(cfg["shard_count"])
    if shard < 0 or shard >= shard_count:
        raise RuntimeError(f"array task {shard} is outside 0..{shard_count - 1}")
    cases = catalog["cases"]
    if len(cases) != int(cfg["expected_cases"]):
        raise RuntimeError(f"cases.json has {len(cases)} rows, expected {cfg['expected_cases']}")
    mine = [case for index, case in enumerate(cases) if index % shard_count == shard]
    if len(mine) != len(cases) // shard_count:
        raise RuntimeError(f"shard {shard} has {len(mine)} cases")
    return shard, mine


def main() -> None:
    cfg = json.loads(Path(sys.argv[1]).read_text())
    settings = json.loads((ROOT / cfg["receiver_settings"]).read_text())
    if settings.get("user_instruction") not in ("",):
        raise RuntimeError("receiver instruction must stay empty")
    if (
        int(settings["num_inference_steps"]) != int(cfg["num_inference_steps"])
        or float(settings["true_cfg_scale"]) != float(cfg["true_cfg_scale"])
        or settings["negative_prompt"] != cfg["negative_prompt"]
        or settings["model_id"] != cfg["model_id"]
        or settings["revision"] != cfg["revision"]
    ):
        raise RuntimeError("edit settings do not match the fixed receiver settings")
    controls = json.loads((ROOT / cfg["controls_json"]).read_text())
    if not controls.get("passed") or int(controls["key"]["n"]) != int(cfg["expected_n"]):
        raise RuntimeError("style-split controls have not passed")
    catalog = json.loads((ROOT / cfg["cases_json"]).read_text())
    if int(catalog["n"]) != int(cfg["expected_n"]):
        raise RuntimeError("cases.json n does not match the config")
    shard, mine = selected_cases(cfg, catalog)
    output_dir = ROOT / cfg["output_dir"]
    shard_dir = output_dir / f"shard_{shard:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    finished = {}
    for case in mine:
        trial_dir = shard_dir / case["key"]
        manifest_path = trial_dir / "manifest.json"
        if manifest_path.exists():
            try:
                row = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                row = None
            if row is not None and row.get("instruction") == case["prompt"] and row.get("sha256") == case["sha256"]:
                finished[case["key"]] = row
                continue
            shutil.rmtree(trial_dir)
        elif trial_dir.exists():
            shutil.rmtree(trial_dir)
    pending = [case for case in mine if case["key"] not in finished]
    pipe = None
    measure = None
    decoding_key = message = codeword = None
    if pending:
        import lpips

        measure = lpips.LPIPS(net="alex")
        stem = key_basename(cfg["payload_length"])
        key_dir = ROOT / cfg["keys_dir"]
        _encoding, decoding_key, message, codeword = load_sender_key(key_dir / f"{stem}_sender.pkl")
        message = [int(bit) for bit in message]
        codeword = torch.as_tensor(codeword)
        if len(message) != int(cfg["payload_length"]) or int(codeword.numel()) != int(cfg["expected_n"]):
            raise RuntimeError("sender key does not match the style-split payload")
        if bits_to_text(message) != cfg["message_text"]:
            raise RuntimeError("sender key message does not match the config")
        pipe = load_pipeline(cfg["model_id"], cfg["revision"])
    rows = dict(finished)
    for case in pending:
        trial_dir = shard_dir / case["key"]
        manifest_path = trial_dir / "manifest.json"
        started = time.perf_counter()
        photo = Path(cfg["image_root"]) / case["id"]
        if sha256_file(photo) != case["sha256"]:
            raise RuntimeError(f"{photo} hash does not match cases.json")
        image = Image.open(photo).convert("RGB")
        canvas_w, canvas_h = edit_canvas(*image.size)
        length = block_length(canvas_w, canvas_h)
        if [canvas_w, canvas_h] != catalog["canvas"] or length != int(catalog["n"]):
            raise RuntimeError(f"{case['id']} canvas changed")
        create_directory(trial_dir)
        (trial_dir / "instruction.txt").write_text(case["prompt"] + "\n")
        resized = pipe.image_processor.resize(image, canvas_h, canvas_w)
        gaussian = np.random.default_rng(int(case["gaussian_seed"])).standard_normal(length)
        unpacked_shape = (1, 1, 16, 2 * (canvas_h // 16), 2 * (canvas_w // 16))
        unpacked_gaussian = torch.from_numpy(gaussian.reshape(unpacked_shape))
        unpacked_prc = sign_magnitudes(codeword, gaussian).reshape(unpacked_shape)
        packed_gaussian = pack_noise(pipe, unpacked_gaussian.float()).detach().clone()
        packed_prc = pack_noise(pipe, unpacked_prc.float()).detach().clone()
        images_g, _, _, _ = run_forward(
            pipe, image, case["prompt"], cfg["negative_prompt"], packed_gaussian,
            int(cfg["num_inference_steps"]), float(cfg["true_cfg_scale"]), float(cfg["guidance_scale"]),
        )
        images_p, _, _, _ = run_forward(
            pipe, image, case["prompt"], cfg["negative_prompt"], packed_prc,
            int(cfg["num_inference_steps"]), float(cfg["true_cfg_scale"]), float(cfg["guidance_scale"]),
        )
        png_g, pil_g = save_png_only(pipe, images_g, trial_dir / "gaussian.png")
        png_p, pil_p = save_png_only(pipe, images_p, trial_dir / "prc.png")
        receiver_key = key_dir / f"{stem}_receiver.pkl"
        settings_path = ROOT / cfg["receiver_settings"]
        _prc_result, estimated_p = recover(trial_dir / "prc.png", receiver_key, settings_path, trial_dir / "receiver_prc", pipe=pipe)
        _gaussian_result, estimated_g = recover(
            trial_dir / "gaussian.png", receiver_key, settings_path, trial_dir / "receiver_gaussian", pipe=pipe
        )
        scored_p = score_latent(decoding_key, estimated_p, message)
        scored_g = score_latent(decoding_key, estimated_g, message)
        source = to_unit(pipe.image_processor.preprocess(resized, canvas_h, canvas_w))
        unit_g = to_unit(images_g)
        unit_p = to_unit(images_p)
        pixels_g = pixel_metrics(source, unit_g)
        pixels_p = pixel_metrics(source, unit_p)
        pixels_pair = pixel_metrics(unit_g, unit_p)

        def lpips_value(left, right) -> float:
            return float(measure(left.unsqueeze(0) * 2 - 1, right.unsqueeze(0) * 2 - 1).item())

        agree_p, _flip_p = sign_agreement(estimated_p.reshape(-1), codeword)
        agree_g, _flip_g = sign_agreement(estimated_g.reshape(-1), codeword)
        recovered = bits_to_text(scored_p["decoded_bits"])
        write_bytes(
            trial_dir / "panel.png",
            draw_panel(
                resized,
                pil_g,
                pil_p,
                [
                    f"{case['key']} source\n{case['prompt']}",
                    "ordinary\n" + caption(scored_g, bits_to_text(scored_g["decoded_bits"]), lpips_value(source, unit_g)),
                    "PRC\n" + caption(scored_p, recovered, lpips_value(source, unit_p)),
                ],
            ),
        )
        row = {
            "key": case["key"],
            "image_id": case["id"],
            "sha256": case["sha256"],
            "instruction": case["prompt"],
            "gaussian_seed": int(case["gaussian_seed"]),
            "canvas": [canvas_w, canvas_h],
            "n": length,
            "png_gaussian_sha256": png_g,
            "png_prc_sha256": png_p,
            "prc": scored_p,
            "gaussian": scored_g,
            "recovered_text": recovered,
            "pixel_source_gaussian": pixels_g,
            "pixel_source_prc": pixels_p,
            "pixel_gaussian_prc": pixels_pair,
            "lpips_source_gaussian": lpips_value(source, unit_g),
            "lpips_source_prc": lpips_value(source, unit_p),
            "lpips_gaussian_prc": lpips_value(unit_g, unit_p),
            "psnr_source_gaussian": pixels_g["psnr"],
            "psnr_source_prc": pixels_p["psnr"],
            "psnr_gaussian_prc": pixels_pair["psnr"],
            "ssim_source_gaussian": ssim_luminance(source, unit_g),
            "ssim_source_prc": ssim_luminance(source, unit_p),
            "ssim_gaussian_prc": ssim_luminance(unit_g, unit_p),
            "sign_agreement_prc": agree_p,
            "sign_agreement_gaussian": agree_g,
            "runtime_sec": time.perf_counter() - started,
        }
        write_json(manifest_path, row)
        rows[case["key"]] = row
        del images_g, images_p, unit_g, unit_p, estimated_p, estimated_g
        park_on_cpu(pipe)
        print(json.dumps({
            "shard": shard,
            "key": case["key"],
            "detect": scored_p["detect"],
            "exact": scored_p["exact_message"],
            "sign_agreement": agree_p,
            "lpips_gaussian_prc": row["lpips_gaussian_prc"],
            "runtime_sec": row["runtime_sec"],
        }), flush=True)
    ordered = [rows[case["key"]] for case in mine]
    public_rows = []
    for row in ordered:
        public = dict(row)
        public["prc"] = slim_score(row["prc"])
        public["gaussian"] = slim_score(row["gaussian"])
        public_rows.append(public)
    summary = {"task": "imgedit_style", "shard": shard, **aggregate(public_rows), "cases": public_rows}
    write_replacing(shard_dir / "summary.json", summary)
    panels = [Image.open(shard_dir / case["key"] / "panel.png").convert("RGB") for case in mine]
    width = max(panel.width for panel in panels)
    scaled = []
    for panel in panels:
        height = round(panel.height * width / panel.width)
        scaled.append(panel.resize((width, height), Image.Resampling.LANCZOS))
    gallery = Image.new("RGB", (width, sum(panel.height for panel in scaled)), "white")
    y = 0
    for panel in scaled:
        gallery.paste(panel, (0, y))
        y += panel.height
    buffer = __import__("io").BytesIO()
    gallery.save(buffer, format="PNG")
    gallery_path = shard_dir / "gallery.png"
    if gallery_path.exists():
        gallery_path.unlink()
    write_bytes(gallery_path, buffer.getvalue())
    if pipe is not None:
        park_on_cpu(pipe)
    merge_shards(output_dir, int(cfg["shard_count"]))


if __name__ == "__main__":
    main()
