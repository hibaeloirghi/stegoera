"""PNG receivers that do not see the source photo.

image_key_model sees the edited PNG, the PRC key, and the sampler settings,
including the guidance scales used to make that PNG. image_key_model_prompt
also sees the editing instruction. Neither sees the source latent.

The three-branch formula is unchanged. Zero image latents and an empty
instruction make branches identical, so the corresponding guidance scale
cancels. That cancellation is checked below rather than special-cased.
"""
from __future__ import annotations

import fcntl
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
    code_hash,
    git_rev,
    require_hash,
    sha256_bytes,
    sha256_file,
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

OUT = ROOT / "results" / "ip2p_edict_prc" / "guidance_diag" / "blind_receiver"
PARTS = OUT / "parts"

EXISTING = {(7.5, 1.5), (3.0, 1.5), (3.0, 1.0), (7.5, 1.0), (1.5, 1.0)}
GENERATE = {
    (1.5, 1.25), (1.5, 1.5),
    (3.0, 1.25),
    (5.0, 1.0), (5.0, 1.25), (5.0, 1.5),
    (7.5, 1.25),
}
# The spaniel already has the full grid. These four are the cells rerun on new photos.
PHOTO_PAIRS = {(7.5, 1.5), (3.0, 1.5), (7.5, 1.0), (3.0, 1.0)}
FIELDS = [
    "experiment_id", "job_id", "code_hash", "config_hash", "image_hash",
    "edit", "guidance_text", "guidance_image",
    "receiver", "latent_source",
    "sees_source_image", "sees_editing_prompt", "sees_guidance_scales",
    "image_latent_abs_max", "text_scale_changes_denoiser", "image_scale_changes_denoiser",
    "recovered_stream",
    "detect", "decode_non_none", "exact_message_match",
    "sign_agreement", "statistic", "threshold", "margin",
    "initial_noise_MSE", "initial_noise_correlation", "initial_noise_role",
    "terminal_latent_MSE", "terminal_latent_corr", "terminal_latent_role",
    "sigma", "sigma_squared",
    "runtime", "peak_memory", "status",
    "wrong_key_detect", "wrong_key_decode_non_none",
    "sender_r1_exact",
]
SCORE_KEYS = [
    "initial_noise_MSE", "initial_noise_correlation", "sign_agreement",
    "detect", "statistic", "threshold", "margin",
    "decode_non_none", "exact_message_match",
]


def algebra_check() -> None:
    """The cancellation claimed for these two receivers, on dummy branch outputs."""
    def guided(text, image, uncond, text_scale, image_scale):
        return uncond + text_scale * (text - image) + image_scale * (image - uncond)

    same = torch.tensor([0.2])
    if not torch.equal(guided(same, same, same, 7.5, 1.5), same):
        raise SystemExit("empty conditioning did not collapse to the unconditional branch")
    text = torch.tensor([0.8])
    image = torch.tensor([-0.4])
    low = guided(text, image, image, 3.0, 1.0)
    high = guided(text, image, image, 3.0, 2.0)
    expected = image + 3.0 * (text - image)
    if not torch.equal(low, high) or not torch.equal(low, expected):
        raise SystemExit("image scale still changes the prompt receiver when its image branch matches empty")


def publish() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    lock = (OUT / ".publish.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    rows = []
    for path in sorted(PARTS.glob("receiver_*.csv")):
        import csv
        with path.open(newline="") as handle:
            rows.extend(list(csv.DictReader(handle)))
    destination = OUT / "receiver_results.csv"
    temporary = destination.with_suffix(".csv.tmp")
    import csv
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})
    os.replace(temporary, destination)
    lines = [path.read_text().rstrip() for path in sorted(PARTS.glob("status_*.txt"))]
    (OUT / "STATUS.txt").write_text("\n\n".join(lines) + "\n")
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()


def inverse_two(denoise, schedule, first, second, mix_weight: float):
    start = [
        first.detach().to(device="cuda", dtype=torch.float64).clone(),
        second.detach().to(device="cuda", dtype=torch.float64).clone(),
    ]
    started = time.perf_counter()
    recovered, _trace = coupled_loop(
        denoise, schedule, start, reverse=True, mix_weight=mix_weight, leapfrog=True,
    )
    return recovered, time.perf_counter() - started


def encode_three(pipe, text: str) -> torch.Tensor:
    embeds = pipe._encode_prompt(text, "cuda", 1, True, negative_prompt="")
    if tuple(embeds.shape)[:1] != (3,):
        raise SystemExit(f"prompt batch is {tuple(embeds.shape)}")
    return embeds


def branch_facts(image_latents: torch.Tensor, prompt_embeds: torch.Tensor) -> dict:
    image_same = torch.equal(image_latents[0], image_latents[1]) and torch.equal(image_latents[1], image_latents[2])
    text_same = torch.equal(prompt_embeds[0], prompt_embeds[1]) and torch.equal(prompt_embeds[1], prompt_embeds[2])
    image_branch_matches_empty = torch.equal(prompt_embeds[1], prompt_embeds[2]) and torch.equal(image_latents[1], image_latents[2])
    return {
        "image_latent_abs_max": float(image_latents.detach().abs().max().item()),
        "text_scale_changes_denoiser": (not text_same) or (not image_same),
        "image_scale_changes_denoiser": not image_branch_matches_empty,
    }


def message_ok(scored: dict) -> bool:
    return bool(scored["detect"] and scored["decode_non_none"] and scored["exact_message_match"])


def load_saved(cfg):
    require_hash(ROOT / cfg["saved_pairs"], cfg["saved_pairs_sha256"], "pairs")
    saved = torch.load(ROOT / cfg["saved_pairs"], map_location="cpu")
    image = saved["image_latents"]
    prompt = saved["prompt_embeds"]
    if tuple(image.shape) != (3, 4, 64, 64):
        raise SystemExit(f"image latent shape {tuple(image.shape)}")
    if float(image[2].abs().max().item()) != 0.0:
        raise SystemExit("saved empty image branch is not zero")
    return saved, image, prompt


def evaluator_x(cfg, saved):
    kind = cfg.get("evaluator_x", "")
    if kind == "prc_final":
        return saved["prc_final"][0].detach().to(dtype=torch.float64).cpu()
    if kind == "forward_terminal":
        blob = torch.load(ROOT / cfg["evaluator_x_path"], map_location="cpu")
        return blob["x"].detach().to(dtype=torch.float64).cpu()
    if kind == "generated":
        return None
    raise SystemExit(f"unknown evaluator_x {kind}")


def main() -> int:
    global OUT, PARTS
    algebra_check()
    config_path = Path(sys.argv[1])
    cfg = json.loads(config_path.read_text())
    pair = (float(cfg["guidance_scale"]), float(cfg["image_guidance_scale"]))
    mode = cfg["mode"]
    encode_source = cfg.get("conditioning") == "encode_source"
    if encode_source:
        if mode != "generate" or pair not in PHOTO_PAIRS:
            print("FAIL: five-image grid", pair, mode, flush=True)
            return 2
    elif mode == "receive_existing" and pair not in EXISTING:
        print("FAIL: existing PNG grid", pair, flush=True)
        return 2
    elif mode == "generate" and pair not in GENERATE:
        print("FAIL: generate grid", pair, flush=True)
        return 2
    if mode not in {"receive_existing", "generate"}:
        print("FAIL: mode", mode, flush=True)
        return 2
    OUT = ROOT / "results" / "ip2p_edict_prc" / cfg.get("output_root", "guidance_diag/blind_receiver")
    PARTS = OUT / "parts"
    slug = cfg["experiment_id"]
    case_dir = OUT / "cases" / slug
    if (case_dir / "STATUS.txt").exists() or (PARTS / f"receiver_{slug}.csv").exists():
        print("FAIL: refusing to overwrite", slug, flush=True)
        return 2
    require_hash(ROOT / cfg["saved_noise"], cfg["saved_noise_sha256"], "noise")
    require_hash(ROOT / cfg["key_path"], cfg["key_sha256"], "key")
    require_hash(ROOT / cfg["wrong_key_path"], cfg["wrong_key_sha256"], "wrong key")
    if mode == "receive_existing":
        require_hash(ROOT / cfg["png"], cfg["png_sha256"], "png")
    case_dir.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()

    if encode_source:
        require_hash(ROOT / cfg["source_photo"], cfg["source_photo_sha256"], "source photo")
        require_hash(ROOT / cfg["grid_pairs"], cfg["grid_pairs_sha256"], "grid pairs")
        grid_blob = torch.load(ROOT / cfg["grid_pairs"], map_location="cpu")
        grid_steps = grid_blob["timesteps"].to(torch.float64).cpu()
        del grid_blob
        saved = None
        exact_x = None
    else:
        saved, image_cpu, prompt_cpu = load_saved(cfg)
        exact_x = evaluator_x(cfg, saved)
        grid_steps = saved["timesteps"].to(torch.float64).cpu()
    schedule = EdictSchedule(int(cfg["num_inference_steps"]), device="cpu", dtype=torch.float64)
    if not torch.equal(schedule.timesteps.cpu(), grid_steps):
        print("FAIL: timestep grid differs from the saved sender", flush=True)
        return 2
    noise = np.load(ROOT / cfg["saved_noise"])
    z_prc = torch.from_numpy(np.asarray(noise["z_prc"])).to(dtype=torch.float64)
    _enc, _decoding_key, _message, key_codeword = load_key(ROOT / cfg["key_path"])
    if key_hash(_enc) != cfg["expected_key_hash"]:
        print("FAIL: key hash", flush=True)
        return 2
    if not np.array_equal(np.asarray(key_codeword).reshape(-1), np.asarray(noise["codeword"]).reshape(-1)):
        print("FAIL: codeword mismatch", flush=True)
        return 2
    mix = float(cfg["mix_weight"])

    from diffusers import AutoencoderKL, StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], torch_dtype=torch.float16,
    ).to("cuda")
    signature = inspect.signature(pipe.__call__)
    if "guidance_scale" not in signature.parameters or "image_guidance_scale" not in signature.parameters:
        print("FAIL: Diffusers __call__ does not accept these guidance arguments", flush=True)
        return 2
    if not (pair[0] > 1.0 and pair[1] >= 1.0):
        print("FAIL: Diffusers would disable three-branch guidance", pair, flush=True)
        return 2

    prompt_built = encode_three(pipe, cfg["instruction"])
    blind_built = encode_three(pipe, "")
    if encode_source:
        from experiments.image_utils import load_edict_image

        photo = load_edict_image(ROOT / cfg["source_photo"], size=512)
        photo_tensor = pipe.image_processor.preprocess(photo)
        sender_prompt = pipe._encode_prompt(cfg["instruction"], "cuda", 1, True, negative_prompt="")
        sender_image = pipe.prepare_image_latents(photo_tensor, 1, 1, sender_prompt.dtype, "cuda", True)
        if tuple(sender_image.shape) != (3, 4, 64, 64):
            print("FAIL: source latent shape", tuple(sender_image.shape), flush=True)
            return 2
        if float(sender_image[2].abs().max().item()) != 0.0:
            print("FAIL: empty image branch is not zero", flush=True)
            return 2
        image_cpu = sender_image.detach().to(dtype=torch.float16).cpu()
        prompt_cpu = sender_prompt.detach().cpu()
        reference_prompt = sender_prompt
    else:
        reference_prompt = prompt_cpu.to(device="cuda", dtype=prompt_built.dtype)
    prompt_error = float((prompt_built - reference_prompt).abs().max().item())
    blind_error = float((blind_built[0] - reference_prompt[1]).abs().max().item())
    if prompt_error != 0.0 or blind_error != 0.0:
        print("FAIL: re-encoded prompt differs from the sender prompt", prompt_error, blind_error, flush=True)
        return 2
    if not (torch.equal(blind_built[0], blind_built[1]) and torch.equal(blind_built[1], blind_built[2])):
        print("FAIL: empty prompt branches differ", flush=True)
        return 2
    zeros = torch.zeros((3, 4, 64, 64), dtype=torch.float16, device="cuda")
    receivers = {
        "image_key_model": (zeros, blind_built, False),
        "image_key_model_prompt": (zeros, prompt_built, True),
    }
    for name, (image_latents, prompt_embeds, _sees_prompt) in receivers.items():
        facts = branch_facts(image_latents, prompt_embeds)
        if facts["image_latent_abs_max"] != 0.0:
            print("FAIL: receiver image latent is not zero", name, flush=True)
            return 2
        if name == "image_key_model" and facts["text_scale_changes_denoiser"]:
            print("FAIL: empty receiver still depends on the text scale", flush=True)
            return 2
        if facts["image_scale_changes_denoiser"]:
            print("FAIL: image scale still changes this receiver", name, flush=True)
            return 2

    sender_r1_exact = "not_rerun"
    generated_final = None
    if mode == "generate":
        sender_image = image_cpu.to(device="cuda", dtype=torch.float16)
        sender_prompt = prompt_cpu.to(device="cuda", dtype=torch.float16)
        sender = make_denoise(pipe.unet, sender_image, sender_prompt, pair[0], pair[1])
        print("forward", slug, flush=True)
        start = [z_prc.to(device="cuda").clone(), z_prc.to(device="cuda").clone()]
        final, _forward_trace = coupled_loop(
            sender, schedule, start, reverse=False, mix_weight=mix, leapfrog=True,
        )
        png_path = case_dir / "edit_stream0.png"
        stable_path = OUT / "images" / cfg["output_png_name"]
        if stable_path.exists():
            print("FAIL: refusing to overwrite png", stable_path, flush=True)
            return 2
        decode(pipe, final[0]).save(png_path)
        print("sender_check", flush=True)
        recovered, _seconds = inverse_two(sender, schedule, final[0], final[1], mix)
        r1_cpu = [item.detach().to(torch.float64).cpu() for item in recovered]
        r1_scores = [
            score_stream(tensor, z_prc, *_key_parts(cfg), float(cfg["sigma_squared"]))
            for tensor in r1_cpu
        ]
        sender_r1_exact = "true" if all(message_ok(item) for item in r1_scores) else "false"
        print("sender_r1_exact", sender_r1_exact, flush=True)
        generated_final = final[0].detach().to(dtype=torch.float64).cpu()
        del sender, sender_image, sender_prompt, final
        if sender_r1_exact != "true":
            _write(cfg, config_path, case_dir, [], sender_r1_exact, "r1_failed")
            print("FAIL: true-pair sender check lost the message", flush=True)
            return 2
        png_path_used = png_path
    else:
        png_path_used = ROOT / cfg["png"]

    del image_cpu, prompt_cpu
    if saved is not None:
        del saved
    denoisers = {
        name: (make_denoise(pipe.unet, image_latents, prompt_embeds, pair[0], pair[1]), sees_prompt, branch_facts(image_latents, prompt_embeds))
        for name, (image_latents, prompt_embeds, sees_prompt) in receivers.items()
    }

    print("vae", flush=True)
    reloaded = png_to_model(png_path_used)
    scale = float(pipe.vae.config.scaling_factor)
    z_mode = encode_mode(pipe.vae, reloaded, scale).detach().to(dtype=torch.float64).cpu()
    print("decoder_inversion", flush=True)
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
    (case_dir / "decoder_inversion_pixel_mse.txt").write_text(str(pixel) + "\n")
    torch.save(z100, case_dir / "decoder_inversion_100.pt")
    del vae_loss

    if generated_final is not None:
        exact_x = generated_final
    latents = [("vae_mode", z_mode), ("decoder_inversion_100", z100)]
    pending = []
    for latent_name, latent in latents:
        for receiver_name, (denoise, sees_prompt, facts) in denoisers.items():
            print(receiver_name, latent_name, flush=True)
            recovered, seconds = inverse_two(denoise, schedule, latent, latent, mix)
            cpu = [item.detach().to(torch.float64).cpu() for item in recovered]
            pending.append((receiver_name, latent_name, sees_prompt, facts, cpu, seconds, latent))

    image_hash = sha256_file(png_path_used)
    digest = code_hash([
        ROOT / "experiments/ip2p_edict_prc/sampler.py",
        ROOT / "experiments/ip2p_edict_prc/run_milestone.py",
        ROOT / "experiments/ip2p_edict_prc/run_png_decoder_inversion.py",
        Path(__file__),
    ])
    rows = []
    sigma2 = float(cfg["sigma_squared"])
    _enc, _decoding_key, _message, _codeword = load_key(ROOT / cfg["key_path"])
    if key_hash(_enc) != cfg["expected_key_hash"]:
        print("FAIL: key hash", flush=True)
        return 2
    _wrong_enc, wrong_key, _wm, _wc = load_key(ROOT / cfg["wrong_key_path"])
    for receiver_name, latent_name, sees_prompt, facts, cpu, seconds, latent in pending:
        scored = [score_stream(tensor, z_prc, *_key_parts(cfg), sigma2) for tensor in cpu]
        for stream, tensor in enumerate(cpu):
            soft = recover_soft_bits(tensor, variance=sigma2)
            wrong_detect, wrong_decode, _wrong = detect_decode(wrong_key, soft)
            latent_mse = "" if exact_x is None else vector_mse(latent, exact_x)
            latent_corr = "" if exact_x is None else vector_corr(latent, exact_x)
            row = {
                "experiment_id": slug,
                "job_id": os.environ.get("SLURM_JOB_ID", ""),
                "code_hash": digest,
                "config_hash": sha256_file(config_path),
                "image_hash": image_hash,
                "edit": cfg["edit"],
                "guidance_text": pair[0],
                "guidance_image": pair[1],
                "receiver": receiver_name,
                "latent_source": latent_name,
                "sees_source_image": False,
                "sees_editing_prompt": sees_prompt,
                "sees_guidance_scales": True,
                "recovered_stream": stream,
                "initial_noise_role": "evaluator-only",
                "terminal_latent_MSE": latent_mse,
                "terminal_latent_corr": latent_corr,
                "terminal_latent_role": "evaluator-only",
                "sigma": math.sqrt(sigma2),
                "sigma_squared": sigma2,
                "runtime": seconds,
                "peak_memory": int(torch.cuda.max_memory_allocated()),
                "status": "ok",
                "wrong_key_detect": wrong_detect,
                "wrong_key_decode_non_none": wrong_decode,
                "sender_r1_exact": sender_r1_exact,
                **facts,
            }
            row.update({key: scored[stream][key] for key in SCORE_KEYS})
            rows.append(row)
            print(
                receiver_name, latent_name, "stream", stream,
                "detect", row["detect"], "decode", row["decode_non_none"],
                "exact", row["exact_message_match"], "sign", row["sign_agreement"],
                flush=True,
            )
    _write(cfg, config_path, case_dir, rows, sender_r1_exact, "RECORDED")
    if mode == "generate":
        stable_path = OUT / "images" / cfg["output_png_name"]
        if not stable_path.exists():
            stable_path.write_bytes(png_path_used.read_bytes())
        _quality(cfg, stable_path, slug)
        publish()
    print("RECORDED", slug, flush=True)
    return 0


def _key_parts(cfg):
    _enc, decoding_key, message, key_codeword = load_key(ROOT / cfg["key_path"])
    if key_hash(_enc) != cfg["expected_key_hash"]:
        raise SystemExit("key hash mismatch")
    return decoding_key, np.asarray(key_codeword).reshape(-1), message


def _write(cfg, config_path, case_dir, rows, sender_r1_exact, status) -> None:
    slug = cfg["experiment_id"]
    if rows:
        write_new_csv(PARTS / f"receiver_{slug}.csv", FIELDS, rows)
    write_json(PARTS / f"manifest_{slug}.json", {
        "experiment_id": slug,
        "git_rev": git_rev(),
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "mode": cfg["mode"],
        "guidance_text": float(cfg["guidance_scale"]),
        "guidance_image": float(cfg["image_guidance_scale"]),
        "sender_r1_exact": sender_r1_exact,
        "status": status,
        "sees_source_image": False,
        "config_hash": sha256_file(config_path),
    })
    (PARTS / f"status_{slug}.txt").write_text(f"{status} {slug} sender_r1_exact={sender_r1_exact}\n")
    (case_dir / "STATUS.txt").write_text(status + "\n")
    publish()


def _quality(cfg, image_path: Path, slug: str) -> None:
    try:
        from PIL import Image
        from experiments.image_utils import load_edict_image
        from experiments.instructpix2pix_prc.run_pair import clip_bundle, directional_clip
        from experiments.metrics import lpips_distance, pixel_mse

        source = load_edict_image(ROOT / cfg["source_png"], size=int(cfg.get("preprocessed_size", 512)))
        edit = Image.open(image_path).convert("RGB")
        if edit.size != source.size:
            raise ValueError(f"edit size {edit.size} does not match the {source.size} source crop")
        lpips, lpips_error = lpips_distance(edit, source, device="cuda")
        model, processor = clip_bundle("cuda")
        directional = directional_clip(
            edit, source, cfg["instruction"], cfg["source_caption"], model, processor, "cuda",
        )
        row = {
            "experiment_id": slug,
            "edit": cfg["edit"],
            "guidance_text": cfg["guidance_scale"],
            "guidance_image": cfg["image_guidance_scale"],
            "lpips_vs_source": lpips,
            "lpips_error": lpips_error or "",
            "pixel_mse_vs_source": pixel_mse(edit, source),
            "directional_clip": directional,
            "image_path": str(image_path.relative_to(ROOT)),
        }
        quality_fields = list(row.keys())
        write_new_csv(PARTS / f"quality_{slug}.csv", quality_fields, [row])
    except Exception as exc:
        print("quality failed", repr(exc), flush=True)


if __name__ == "__main__":
    sys.exit(main())
