"""One ordinary edit and one PRC edit of imagenet_dog_1.

Sender tensors are written under diagnostics/. The PNG receiver is called
afterward and is given only the PNG, the decoding key, and the fixed settings.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.qwen_image_edit_prc.editing import (  # noqa: E402
    assert_pack_roundtrip,
    build_schedule,
    conditioning_from_capture,
    euler_update,
    invert_packed,
    lagged_inverse,
    load_pipeline,
    pack_noise,
    predict_velocity,
    run_forward,
    unpack_noise,
)
from experiments.qwen_image_edit_prc.geometry import block_length, edit_canvas  # noqa: E402
from experiments.qwen_image_edit_prc.io_util import (  # noqa: E402
    create_directory,
    read_png,
    sha256_file,
    write_bytes,
    write_json,
    write_tensor,
)
from experiments.qwen_image_edit_prc.prc_adapter import (  # noqa: E402
    load_decoding_key_only,
    load_sender_key,
    score_latent,
    sign_magnitudes,
)
from experiments.qwen_image_edit_prc.receive_png import recover  # noqa: E402


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def save_png(pipe, image_tensor, png_path: Path, float_path: Path) -> tuple[str, Image.Image]:
    image = image_tensor if image_tensor.ndim == 4 else image_tensor.unsqueeze(0)
    write_tensor(float_path, image[0].float().cpu())
    numpy_image = pipe.image_processor.pt_to_numpy(image.float().cpu())
    pil = pipe.image_processor.numpy_to_pil(numpy_image)[0]
    buffer = __import__("io").BytesIO()
    pil.save(buffer, format="PNG")
    write_bytes(png_path, buffer.getvalue())
    _raw, digest = read_png(png_path)
    return digest, pil


def pixel_metrics(left: torch.Tensor, right: torch.Tensor) -> dict:
    diff = left.float() - right.float()
    mse = float(diff.pow(2).mean().item())
    psnr = float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
    return {"mse": mse, "psnr": psnr}


def step_errors(pipe, trajectory, scheduler, conditioning, index: int, true_cfg: float) -> dict:
    sigmas = scheduler["sigmas"]
    timesteps = scheduler["timesteps"]
    current = trajectory[index]
    nxt = trajectory[index + 1]
    forward_velocity = predict_velocity(pipe, current, timesteps[index], conditioning, true_cfg)
    recomputed = euler_update(current, forward_velocity.cpu(), sigmas[index], sigmas[index + 1])
    lagged_velocity = predict_velocity(pipe, nxt, timesteps[index], conditioning, true_cfg)
    recovered = lagged_inverse(nxt, lagged_velocity.cpu(), sigmas[index], sigmas[index + 1])
    return {
        "index": index,
        "forward_recompute_max_abs": max_abs(recomputed, nxt),
        "lagged_inverse_max_abs": max_abs(recovered, current),
    }


def main() -> None:
    cfg = json.loads(Path(sys.argv[1]).read_text())
    controls = json.loads((ROOT / cfg["controls_json"]).read_text())
    if not controls.get("passed"):
        raise RuntimeError("PRC controls have not passed")
    photo = ROOT / cfg["source_photo"]
    if sha256_file(photo) != cfg["source_sha256"]:
        raise RuntimeError("source photograph hash does not match the config")
    trial = (
        f"{cfg['image_id']}__{cfg['instruction_id']}__len{cfg['payload_length']}__trial{cfg['trial']}"
    )
    trial_dir = ROOT / "results" / "qwen_image_edit_prc" / trial
    create_directory(trial_dir)
    encoding_key, decoding_key, message, codeword = load_sender_key(ROOT / cfg["keys_dir"] / "len8_sender.pkl")
    message = [int(bit) for bit in message]
    if message != [int(bit) for bit in cfg["message_bits"]]:
        raise RuntimeError("saved message does not match the config")
    image = Image.open(photo).convert("RGB")
    canvas_w, canvas_h = edit_canvas(*image.size)
    shape_n = block_length(canvas_w, canvas_h)
    if int(torch.as_tensor(codeword).numel()) != shape_n:
        raise RuntimeError("saved codeword length is not the generation-noise length")
    pipe = load_pipeline(cfg["model_id"], cfg["revision"])
    assert_pack_roundtrip(pipe, canvas_h, canvas_w)
    write_bytes(trial_dir / "source_original.jpg", photo.read_bytes())
    resized = pipe.image_processor.resize(image, canvas_h, canvas_w)
    resized.save(trial_dir / "source_resized.png")
    preprocessed = pipe.image_processor.preprocess(resized, canvas_h, canvas_w)
    write_tensor(trial_dir / "source_preprocessed.pt", preprocessed)
    template = pipe.prompt_template_encode
    (trial_dir / "template_conditional.txt").write_text(template.format(cfg["instruction"]))
    (trial_dir / "template_negative.txt").write_text(template.format(cfg["negative_prompt"]))
    (trial_dir / "instruction.txt").write_text(cfg["instruction"] + "\n")
    gaussian = np.random.default_rng(int(cfg["gaussian_seed"])).standard_normal(shape_n)
    unpacked_shape = (1, 1, 16, 2 * (canvas_h // 16), 2 * (canvas_w // 16))
    unpacked_gaussian = torch.from_numpy(gaussian.reshape(unpacked_shape))
    unpacked_prc = sign_magnitudes(torch.as_tensor(codeword), gaussian).reshape(unpacked_shape)
    packed_gaussian = pack_noise(pipe, unpacked_gaussian.float())
    packed_prc = pack_noise(pipe, unpacked_prc.float())
    diag = trial_dir / "diagnostics"
    create_directory(diag)
    write_tensor(diag / "gaussian_unpacked.pt", unpacked_gaussian)
    write_tensor(diag / "prc_unpacked.pt", unpacked_prc)
    write_tensor(diag / "gaussian_packed.pt", packed_gaussian)
    write_tensor(diag / "prc_packed.pt", packed_prc)
    write_tensor(diag / "codeword.pt", torch.as_tensor(codeword))

    def edit(packed, steps):
        return run_forward(
            pipe,
            image,
            cfg["instruction"],
            cfg["negative_prompt"],
            packed,
            steps,
            float(cfg["true_cfg_scale"]),
            float(cfg["guidance_scale"]),
        )

    started = time.perf_counter()
    short_a, short_traj_a, short_cap_a, short_sched_a = edit(packed_gaussian, 2)
    short_b, short_traj_b, _short_cap_b, _short_sched_b = edit(packed_gaussian, 2)
    short_gap = max_abs(short_a, short_b)
    conditioning = conditioning_from_capture(short_cap_a, canvas_h, canvas_w, pipe.vae_scale_factor)
    short_step = step_errors(pipe, short_traj_a, short_sched_a, conditioning, 0, float(cfg["true_cfg_scale"]))
    step_ok = short_step["forward_recompute_max_abs"] <= float(cfg["step_match_abs_tol"])
    images_a, traj_a, cap_a, sched_a = edit(packed_gaussian, int(cfg["num_inference_steps"]))
    images_b, traj_b, _cap_b, sched_b = edit(packed_gaussian, int(cfg["num_inference_steps"]))
    images_p, traj_p, cap_p, sched_p = edit(packed_prc, int(cfg["num_inference_steps"]))
    png_g, pil_g = save_png(pipe, images_a, trial_dir / "gaussian.png", trial_dir / "gaussian_float.pt")
    png_p, pil_p = save_png(pipe, images_p, trial_dir / "prc.png", trial_dir / "prc_float.pt")
    write_tensor(diag / "gaussian_trajectory.pt", torch.stack(traj_a))
    write_tensor(diag / "prc_trajectory.pt", torch.stack(traj_p))
    write_tensor(diag / "gaussian_final_latent.pt", traj_a[-1])
    write_tensor(diag / "prc_final_latent.pt", traj_p[-1])
    torch.save({"gaussian": sched_a, "prc": sched_p}, diag / "scheduler.pt")
    for name, captured in (("gaussian", cap_a), ("prc", cap_p)):
        write_tensor(diag / f"{name}_image_latents.pt", captured["image_latents"])
        write_tensor(diag / f"{name}_positive_embeds.pt", captured["positive_embeds"])
        write_tensor(diag / f"{name}_negative_embeds.pt", captured["negative_embeds"])
    panel = Image.new("RGB", (pil_g.width * 3, pil_g.height + 24), "white")
    panel.paste(resized, (0, 24))
    panel.paste(pil_g, (pil_g.width, 24))
    panel.paste(pil_p, (pil_g.width * 2, 24))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 4), "resized source", fill="black")
    draw.text((pil_g.width + 8, 4), "gaussian edit", fill="black")
    draw.text((pil_g.width * 2 + 8, 4), "PRC edit", fill="black")
    panel_buffer = __import__("io").BytesIO()
    panel.save(panel_buffer, format="PNG")
    write_bytes(trial_dir / "panel.png", panel_buffer.getvalue())
    prc_conditioning = conditioning_from_capture(cap_p, canvas_h, canvas_w, pipe.vae_scale_factor)
    checks = [
        step_errors(pipe, traj_p, sched_p, prc_conditioning, index, float(cfg["true_cfg_scale"]))
        for index in cfg["step_check_indices"]
    ]
    rebuilt = build_schedule(pipe, traj_p[-1], int(cfg["num_inference_steps"]))
    schedule_gap = max_abs(rebuilt["sigmas"], sched_p["sigmas"])
    rows = []
    if max(item["forward_recompute_max_abs"] for item in checks) <= float(cfg["step_match_abs_tol"]):
        estimated_b = invert_packed(
            pipe, traj_p[-1], prc_conditioning, sched_p, float(cfg["true_cfg_scale"])
        )
        unpacked_b = unpack_noise(pipe, estimated_b, canvas_h, canvas_w).reshape(-1)
        scored_b = score_latent(decoding_key, unpacked_b, message)
        rows.append(("diagnostic_B", scored_b, max_abs(unpacked_b, unpacked_prc.reshape(-1))))
    receiver_dir = trial_dir / "receiver_prc"
    receiver_result, estimated_c = recover(
        trial_dir / "prc.png",
        ROOT / cfg["keys_dir"] / "len8_receiver.pkl",
        ROOT / "experiments/configs/qwen_image_edit_prc/receiver_settings.json",
        receiver_dir,
        pipe=pipe,
    )
    scored_c = score_latent(decoding_key, estimated_c, message)
    wrong_key = load_decoding_key_only(ROOT / cfg["keys_dir"] / "len8_receiver.pkl")
    # Wrong-key control uses the sender wrong-key file's decoding key.
    _wrong_enc, wrong_dec, _wrong_message, _wrong_code = load_sender_key(ROOT / cfg["keys_dir"] / "len8_wrong.pkl")
    scored_wrong = score_latent(wrong_dec, estimated_c, message)
    del wrong_key
    gaussian_receiver, estimated_g = recover(
        trial_dir / "gaussian.png",
        ROOT / cfg["keys_dir"] / "len8_receiver.pkl",
        ROOT / "experiments/configs/qwen_image_edit_prc/receiver_settings.json",
        trial_dir / "receiver_gaussian",
        pipe=pipe,
    )
    scored_g = score_latent(decoding_key, estimated_g, message)
    float_g = images_a[0].float().cpu() if images_a.ndim == 4 else images_a.float().cpu()
    float_p = images_p[0].float().cpu() if images_p.ndim == 4 else images_p.float().cpu()
    source = preprocessed[0].float().cpu()
    source_01 = (source + 1) / 2 if float(source.min()) < 0 else source
    report = {
        "trial": trial,
        "png_gaussian_sha256": png_g,
        "png_prc_sha256": png_p,
        "two_step_rerun_max_abs": short_gap,
        "two_step_forward_recompute": short_step,
        "two_step_algorithm_ok": step_ok,
        "fifty_step_gaussian_rerun_max_abs": max_abs(images_a, images_b),
        "fifty_step_latent_rerun_max_abs": max_abs(traj_a[-1], traj_b[-1]),
        "schedule_rebuild_max_abs": schedule_gap,
        "step_checks": checks,
        "pixel_source_gaussian": pixel_metrics(source_01, float_g),
        "pixel_source_prc": pixel_metrics(source_01, float_p),
        "pixel_gaussian_prc": pixel_metrics(float_g, float_p),
        "receiver_prc_png_sha256": receiver_result["png_sha256"],
        "receiver_gaussian_png_sha256": gaussian_receiver["png_sha256"],
        "receiver_c": scored_c,
        "receiver_c_wrong_key": scored_wrong,
        "receiver_gaussian": scored_g,
        "diagnostic_rows": [
            {"role": role, "score": score, "sign_input_max_abs": gap} for role, score, gap in rows
        ],
        "runtime_sec": time.perf_counter() - started,
        "sender_key_is_diagnostic": True,
        "operational_inputs": "png bytes, decoding key, fixed settings",
    }
    try:
        import lpips

        measure = lpips.LPIPS(net="alex")
        def lpips_pair(left, right):
            return float(measure(left.unsqueeze(0) * 2 - 1, right.unsqueeze(0) * 2 - 1).item())
        report["lpips_source_gaussian"] = lpips_pair(source_01, float_g)
        report["lpips_source_prc"] = lpips_pair(source_01, float_p)
        report["lpips_gaussian_prc"] = lpips_pair(float_g, float_p)
    except Exception as exc:
        report["lpips_error"] = f"{type(exc).__name__}: {exc}"
    write_json(trial_dir / "manifest.json", report)
    print(json.dumps({
        "png_prc_sha256": png_p,
        "receiver_exact": scored_c["exact_message"],
        "receiver_detect": scored_c["detect"],
        "runtime_sec": report["runtime_sec"],
    }), flush=True)


if __name__ == "__main__":
    main()
