"""Ten iPhone edits: one ordinary noise draw and one PRC draw, then panels.

The 120-bit key is the one already checked for n=265216. Photo 9 has a
different canvas, so it uses the key checked for n=255744. The receiver still
sees only the PNG, the decoding key, and the fixed empty instruction.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.qwen_image_edit_prc.editing import (  # noqa: E402
    load_pipeline,
    pack_noise,
    park_on_cpu,
    run_forward,
)
from experiments.qwen_image_edit_prc.geometry import block_length, edit_canvas  # noqa: E402
from experiments.qwen_image_edit_prc.io_util import (  # noqa: E402
    create_directory,
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
from experiments.qwen_image_edit_prc.run_pilot import pixel_metrics, save_png  # noqa: E402


def load_prompts(path: Path) -> dict[int, str]:
    prompts = {}
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        number, instruction = text.split(".", 1)
        prompts[int(number)] = instruction.strip()
    if sorted(prompts) != list(range(1, 11)):
        raise RuntimeError(f"prompts must be numbered 1 to 10, found {sorted(prompts)}")
    return prompts


def bits_to_text(bits) -> str | None:
    if not bits or len(bits) % 8:
        return None
    raw = bytearray()
    for start in range(0, len(bits), 8):
        value = 0
        for bit in bits[start : start + 8]:
            value = (value << 1) | int(bit)
        raw.append(value)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def to_unit(image: torch.Tensor) -> torch.Tensor:
    flat = image[0].float().cpu() if image.ndim == 4 else image.float().cpu()
    if float(flat.min()) < 0:
        flat = (flat + 1) / 2
    return flat


def caption(score: dict, recovered: str | None, lpips_source: float | None) -> str:
    decoded = "None" if score["decoded_bits"] is None else (recovered or "undecodable")
    lpips_text = "n/a" if lpips_source is None else f"{lpips_source:.3f}"
    return (
        f"Detect {score['detect']}  Decode {decoded}\n"
        f"exact {score['exact_message']}  BER {score['payload_ber']}  "
        f"margin {score['detector_margin']:.1f}  LPIPS {lpips_text}"
    )


def draw_panel(source: Image.Image, gaussian: Image.Image, prc: Image.Image, lines: list[str]) -> bytes:
    if not source.size == gaussian.size == prc.size:
        raise RuntimeError("panel images differ in size")
    width, height = source.size
    header = 56
    panel = Image.new("RGB", (width * 3, height + header), "white")
    panel.paste(source, (0, header))
    panel.paste(gaussian, (width, header))
    panel.paste(prc, (width * 2, header))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
    for column, text in enumerate(lines):
        draw.multiline_text((width * column + 8, 4), text, fill="black", font=font, spacing=2)
    buffer = __import__("io").BytesIO()
    panel.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    cfg = json.loads(Path(sys.argv[1]).read_text())
    prompts = load_prompts(ROOT / cfg["prompts"])
    for index in range(1, 11):
        photo = ROOT / cfg["image_dir"] / cfg["image_pattern"].format(index=index)
        if sha256_file(photo) != cfg["sha256"][str(index)]:
            raise RuntimeError(f"{photo.name} hash does not match the config")
    for spec in cfg["keys_by_n"].values():
        report = json.loads((ROOT / spec["controls_json"]).read_text())
        if not report.get("passed"):
            raise RuntimeError(f"controls have not passed: {spec['controls_json']}")
    output_root = ROOT / "results" / "qwen_image_edit_prc"
    batch_dir = output_root / f"iphone_len{cfg['payload_length']}_trial{cfg['trial']}"
    create_directory(batch_dir)
    loaded = {}
    pipe = load_pipeline(cfg["model_id"], cfg["revision"])
    measure = None
    try:
        import lpips

        measure = lpips.LPIPS(net="alex")
    except Exception as exc:
        write_json(batch_dir / "lpips_error.json", {"error": f"{type(exc).__name__}: {exc}"})
    rows = []
    panels = []
    for index in range(1, 11):
        started = time.perf_counter()
        filename = cfg["image_pattern"].format(index=index)
        photo = ROOT / cfg["image_dir"] / filename
        digest = cfg["sha256"][str(index)]
        image = Image.open(photo).convert("RGB")
        canvas_w, canvas_h = edit_canvas(*image.size)
        length = block_length(canvas_w, canvas_h)
        spec = cfg["keys_by_n"].get(str(length))
        if spec is None:
            raise RuntimeError(f"image {index} has n={length}, which has no checked key")
        if length not in loaded:
            stem = key_basename(cfg["payload_length"])
            key_dir = ROOT / spec["keys_dir"]
            encoding_key, decoding_key, message, codeword = load_sender_key(key_dir / f"{stem}_sender.pkl")
            message = [int(bit) for bit in message]
            if len(message) != int(cfg["payload_length"]) or int(torch.as_tensor(codeword).numel()) != length:
                raise RuntimeError(f"saved key does not match n={length}")
            loaded[length] = (key_dir, decoding_key, message, codeword)
        key_dir, decoding_key, message, codeword = loaded[length]
        trial_dir = batch_dir / f"{index:02d}"
        create_directory(trial_dir)
        instruction = prompts[index]
        (trial_dir / "instruction.txt").write_text(instruction + "\n")
        write_bytes(trial_dir / "source.png", photo.read_bytes())
        resized = pipe.image_processor.resize(image, canvas_h, canvas_w)
        resized.save(trial_dir / "source_resized.png")
        gaussian = np.random.default_rng(int(cfg["gaussian_seed_base"]) + index).standard_normal(length)
        unpacked_shape = (1, 1, 16, 2 * (canvas_h // 16), 2 * (canvas_w // 16))
        unpacked_gaussian = torch.from_numpy(gaussian.reshape(unpacked_shape))
        unpacked_prc = sign_magnitudes(torch.as_tensor(codeword), gaussian).reshape(unpacked_shape)
        packed_gaussian = pack_noise(pipe, unpacked_gaussian.float())
        packed_prc = pack_noise(pipe, unpacked_prc.float())
        images_g, _, _, _ = run_forward(
            pipe, image, instruction, cfg["negative_prompt"], packed_gaussian,
            int(cfg["num_inference_steps"]), float(cfg["true_cfg_scale"]), float(cfg["guidance_scale"]),
        )
        images_p, _, _, _ = run_forward(
            pipe, image, instruction, cfg["negative_prompt"], packed_prc,
            int(cfg["num_inference_steps"]), float(cfg["true_cfg_scale"]), float(cfg["guidance_scale"]),
        )
        png_g, pil_g = save_png(pipe, images_g, trial_dir / "gaussian.png", trial_dir / "gaussian_float.pt")
        png_p, pil_p = save_png(pipe, images_p, trial_dir / "prc.png", trial_dir / "prc_float.pt")
        receiver_key = key_dir / f"{key_basename(cfg['payload_length'])}_receiver.pkl"
        settings = ROOT / cfg["receiver_settings"]
        prc_result, estimated_p = recover(trial_dir / "prc.png", receiver_key, settings, trial_dir / "receiver_prc", pipe=pipe)
        gaussian_result, estimated_g = recover(
            trial_dir / "gaussian.png", receiver_key, settings, trial_dir / "receiver_gaussian", pipe=pipe
        )
        scored_p = score_latent(decoding_key, estimated_p, message)
        scored_g = score_latent(decoding_key, estimated_g, message)
        source = to_unit(pipe.image_processor.preprocess(resized, canvas_h, canvas_w))
        unit_g = to_unit(images_g)
        unit_p = to_unit(images_p)
        pixels_g = pixel_metrics(source, unit_g)
        pixels_p = pixel_metrics(source, unit_p)
        pixels_pair = pixel_metrics(unit_g, unit_p)
        lpips_g = lpips_p = lpips_pair = None
        if measure is not None:
            def lpips_pair_fn(left, right):
                return float(measure(left.unsqueeze(0) * 2 - 1, right.unsqueeze(0) * 2 - 1).item())
            lpips_g = lpips_pair_fn(source, unit_g)
            lpips_p = lpips_pair_fn(source, unit_p)
            lpips_pair = lpips_pair_fn(unit_g, unit_p)
        recovered = bits_to_text(scored_p["decoded_bits"])
        panel_lines = [
            f"{index}. source\n{instruction}",
            "ordinary\n" + caption(scored_g, bits_to_text(scored_g["decoded_bits"]), lpips_g),
            "PRC\n" + caption(scored_p, recovered, lpips_p),
        ]
        write_bytes(trial_dir / "panel.png", draw_panel(resized, pil_g, pil_p, panel_lines))
        row = {
            "index": index,
            "file": filename,
            "sha256": digest,
            "instruction": instruction,
            "canvas": [canvas_w, canvas_h],
            "n": length,
            "png_gaussian_sha256": png_g,
            "png_prc_sha256": png_p,
            "receiver_prc_sha256": prc_result["png_sha256"],
            "receiver_gaussian_sha256": gaussian_result["png_sha256"],
            "prc": scored_p,
            "gaussian": scored_g,
            "recovered_text": recovered,
            "pixel_source_gaussian": pixels_g,
            "pixel_source_prc": pixels_p,
            "pixel_gaussian_prc": pixels_pair,
            "lpips_source_gaussian": lpips_g,
            "lpips_source_prc": lpips_p,
            "lpips_gaussian_prc": lpips_pair,
            "runtime_sec": time.perf_counter() - started,
        }
        write_json(trial_dir / "manifest.json", row)
        rows.append(row)
        panels.append(Image.open(trial_dir / "panel.png").convert("RGB"))
        del images_g, images_p, unit_g, unit_p, estimated_p, estimated_g
        park_on_cpu(pipe)
        print(json.dumps({
            "index": index,
            "exact": scored_p["exact_message"],
            "recovered_text": recovered,
            "runtime_sec": row["runtime_sec"],
        }), flush=True)
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
    write_bytes(batch_dir / "gallery.png", buffer.getvalue())
    passed = all(row["prc"]["exact_message"] for row in rows)
    write_json(batch_dir / "summary.json", {"passed_all_exact": passed, "images": rows})
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
