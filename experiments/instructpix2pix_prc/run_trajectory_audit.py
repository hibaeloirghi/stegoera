"""Audit one saved DDIM trajectory.

The saved forward was float16. This job replays that trajectory with the
saved conditioning and then repeats the inverse tests in float32. It does
not sample new images or new PRC noise.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.instructpix2pix_prc.ddim_invert import invert_ddim_step, make_pair_ddim  # noqa: E402
from experiments.paths import RESULTS, WORKDIR  # noqa: E402

CONFIG_PATH = ROOT / "experiments" / "configs" / "instructpix2pix_prc" / "trajectory_audit.json"
OUT = RESULTS / "instructpix2pix_prc" / "trajectory_audit"

FIELDS = [
    "check", "dtype", "step", "chain_length",
    "mse", "max_abs", "correlation", "sign_agreement",
    "forward_residual_mse", "algebra_residual_mse",
    "solver_residual_first", "solver_residual_last", "solver_iters",
    "seconds", "note",
]


def compare(a, b):
    x = a.detach().float().flatten().cpu()
    y = b.detach().float().flatten().cpu()
    err = x - y
    corr = float(torch.corrcoef(torch.stack([x, y]))[0, 1].item()) if x.numel() > 1 else 1.0
    signs = torch.sign(x) == torch.sign(y)
    return {
        "mse": float(err.pow(2).mean().item()),
        "max_abs": float(err.abs().max().item()),
        "correlation": corr,
        "sign_agreement": float(signs.float().mean().item()),
    }


def load_unet(dtype):
    from diffusers import UNet2DConditionModel

    cfg = json.loads(CONFIG_PATH.read_text())
    unet = UNet2DConditionModel.from_pretrained(
        cfg["model_id"],
        subfolder="unet",
        revision=cfg["model_revision"],
        torch_dtype=dtype,
    )
    unet.to("cuda").eval()
    return unet


def predict_fn(unet, scheduler, image_latents, prompt_embeds, guidance, image_guidance):
    def predict(latents, timestep):
        with torch.no_grad():
            model_input = torch.cat([latents] * 3)
            model_input = scheduler.scale_model_input(model_input, timestep)
            model_input = torch.cat([model_input, image_latents], dim=1)
            noise = unet(model_input, timestep, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
            if hasattr(scheduler, "sigmas"):
                raise RuntimeError("DDIM audit scheduler should not expose sigmas")
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


def chain_inverse(predict, scheduler, end, timesteps):
    current = end
    for timestep in reversed(list(timesteps)):
        eps = predict(current, timestep)
        current = invert_ddim_step(scheduler, current, eps, timestep)
    return current


def solve_one_step(predict, scheduler, x_out, timestep, max_iter):
    eps = predict(x_out, timestep)
    current = invert_ddim_step(scheduler, x_out, eps, timestep)
    residuals = []
    for _ in range(max_iter):
        eps = predict(current, timestep)
        recon = scheduler.step(eps, timestep, current, eta=0.0, return_dict=False)[0]
        residuals.append(float((recon.float() - x_out.float()).pow(2).mean().item()))
        current = invert_ddim_step(scheduler, x_out, eps, timestep)
        if residuals[-1] < 1e-8:
            break
    return current, residuals


def row(check, dtype, **kwargs):
    base = {key: "" for key in FIELDS}
    base.update({"check": check, "dtype": dtype, **kwargs})
    return base


def audit_dtype(unet, scheduler, saved, timesteps, dtype, cfg, rows):
    device = "cuda"
    cast = dtype
    image_latents = saved["image_latents_cfg"].to(device=device, dtype=cast)
    prompt_embeds = saved["prompt_embeds"].to(device=device, dtype=cast)
    predict = predict_fn(
        unet, scheduler, image_latents, prompt_embeds,
        float(cfg["guidance_scale"]), float(cfg["image_guidance_scale"]),
    )
    start = saved["scaled_start"].to(device=device, dtype=cast)
    print("replay", str(dtype).replace("torch.", ""), flush=True)
    started = time.perf_counter()
    states = replay(predict, scheduler, start, timesteps)
    replay_seconds = time.perf_counter() - started
    label = "float16" if dtype == torch.float16 else "float32"
    saved_outputs = {
        0: saved["latent_after_step_0"],
        50: saved["latent_after_step_50"],
        99: saved["latent_after_step_99"],
    }
    for step, target in saved_outputs.items():
        stats = compare(states[step + 1], target)
        rows.append(row(
            "replay_vs_saved_output", label, step=step, seconds=replay_seconds, **stats,
            note="states[step+1] is the output of forward step `step`; saved tensors are float16 values",
        ))
        print(" replay", label, "step", step, "mse", stats["mse"], flush=True)

    # Step 0 uses the saved input and the saved output directly.
    x_in = saved["scaled_start"].to(device=device, dtype=cast)
    x_out = saved["latent_after_step_0"].to(device=device, dtype=cast)
    timestep = timesteps[0]
    eps_at_input = predict(x_in, timestep)
    recon = scheduler.step(eps_at_input, timestep, x_in, eta=0.0, return_dict=False)[0]
    recovered = invert_ddim_step(scheduler, x_out, eps_at_input, timestep)
    forward_mse = float((recon.float() - x_out.float()).pow(2).mean().item())
    algebra = compare(recovered, x_in)
    rows.append(row(
        "saved_step0_forward_and_algebra", label, step=0,
        forward_residual_mse=forward_mse, algebra_residual_mse=algebra["mse"], **algebra,
        note="noise prediction evaluated at the saved correct input",
    ))

    if dtype != torch.float32:
        return states

    tested = [int(s) for s in cfg["isolated_steps"]]
    for step in tested:
        x_in = states[step]
        x_out = states[step + 1]
        timestep = timesteps[step]
        eps_at_input = predict(x_in, timestep)
        recon = scheduler.step(eps_at_input, timestep, x_in, eta=0.0, return_dict=False)[0]
        forward_mse = float((recon.float() - x_out.float()).pow(2).mean().item())
        recovered_true_eps = invert_ddim_step(scheduler, x_out, eps_at_input, timestep)
        algebra = compare(recovered_true_eps, x_in)
        one = invert_ddim_step(scheduler, x_out, predict(x_out, timestep), timestep)
        one_stats = compare(one, x_in)
        solved, residuals = solve_one_step(
            predict, scheduler, x_out, timestep, int(cfg["fixed_point_max_iter"])
        )
        solved_stats = compare(solved, x_in)
        rows.append(row(
            "isolated_true_eps", "float32", step=step,
            forward_residual_mse=forward_mse, algebra_residual_mse=algebra["mse"], **algebra,
            note="inverse uses the noise prediction at the known input of this step",
        ))
        rows.append(row(
            "isolated_one_pass", "float32", step=step, **one_stats,
            note="inverse uses the noise prediction at the output of this step",
        ))
        rows.append(row(
            "isolated_fixed_point", "float32", step=step, **solved_stats,
            solver_residual_first=residuals[0] if residuals else "",
            solver_residual_last=residuals[-1] if residuals else "",
            solver_iters=len(residuals),
            note="solver residual is mse of step(eps(candidate), t, candidate) against the known output",
        ))
        print(
            " isolated", step,
            "one_pass_mse", one_stats["mse"],
            "solver_mse", solved_stats["mse"],
            "solver_residual", residuals[-1] if residuals else None,
            flush=True,
        )

    for length in [int(n) for n in cfg["chain_lengths"]]:
        end = states[length]
        started = time.perf_counter()
        hat = chain_inverse(predict, scheduler, end, timesteps[:length])
        seconds = time.perf_counter() - started
        stats = compare(hat, states[0])
        rows.append(row(
            "chained_one_pass", "float32", chain_length=length, seconds=seconds, **stats,
            note="start from the exact float32 latent after this many steps and invert those steps",
        ))
        print(" chain", length, "mse", stats["mse"], "corr", stats["correlation"], flush=True)

    full = next(item for item in rows if item["check"] == "chained_one_pass" and item["chain_length"] == 100)
    # The recovered start for the full chain was not kept on that row. Recompute once for regeneration.
    hat0 = chain_inverse(predict, scheduler, states[100], timesteps).to(dtype=cast)
    print("regenerate from chained recovery", flush=True)
    started = time.perf_counter()
    regenerated = replay(predict, scheduler, hat0, timesteps)
    seconds = time.perf_counter() - started
    final_stats = compare(regenerated[-1], states[-1])
    rows.append(row(
        "regenerate_from_recovered_start", "float32", chain_length=100, seconds=seconds, **final_stats,
        note="float32 forward from the chained one-pass recovery, compared with the float32 final latent",
    ))
    saved_final = compare(regenerated[-1], saved["latent_after_step_99"])
    rows.append(row(
        "regenerate_vs_saved_fp16_final", "float32", chain_length=100, **saved_final,
        note="same regeneration compared with the original float16 saved final latent",
    ))
    print(" regenerate mse", final_stats["mse"], flush=True)
    return states


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    saved = torch.load(WORKDIR / cfg["saved_state"], map_location="cpu")
    scheduler = make_pair_ddim()
    scheduler.set_timesteps(int(cfg["num_inference_steps"]), device="cuda")
    live = scheduler.timesteps.detach().cpu().to(torch.float32)
    if not torch.equal(live, saved["timesteps"].to(torch.float32)):
        (OUT / "STATUS.txt").write_text("FAIL\ntimesteps do not match the saved forward\n")
        return 2
    timesteps = list(scheduler.timesteps)
    rows = []
    print("gpu", torch.cuda.get_device_name(0), "mem_gib", torch.cuda.get_device_properties(0).total_memory / 2**30, flush=True)
    print("fp16 replay against saved outputs", flush=True)
    unet16 = load_unet(torch.float16)
    audit_dtype(unet16, scheduler, saved, timesteps, torch.float16, cfg, rows)
    del unet16
    torch.cuda.empty_cache()
    print("fp32 diagnostic", flush=True)
    unet32 = load_unet(torch.float32)
    audit_dtype(unet32, scheduler, saved, timesteps, torch.float32, cfg, rows)
    with (OUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "saved_forward_dtype": "float16",
        "diagnostic_dtype": "float32",
        "precision_note": (
            "Saved latents and conditioning are float32 files containing the float16 "
            "pipeline values. The float32 replay casts those tensors and the UNet to float32."
        ),
        "gpu": torch.cuda.get_device_name(0),
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "rows": rows,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "STATUS.txt").write_text("PASS\n")
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
