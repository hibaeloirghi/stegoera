"""DDIM inverse for the installed eta=0 epsilon update.

The forward step evaluates the denoiser at the noisier latent. This inverse
evaluates it at the cleaner latent that the step produced, then applies the
algebraic left inverse of that same update. The algebra is exact for a fixed
noise prediction. It is approximate for a state-dependent denoiser.
"""
from __future__ import annotations

import torch


def make_pair_ddim():
    from diffusers import DDIMScheduler

    return DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        timestep_spacing="leading",
    )


def invert_ddim_step(scheduler, x_prev, eps, timestep):
    """Left inverse of DDIMScheduler.step for eta=0 and prediction_type=epsilon.

    clip_sample and thresholding are off, matching the pair-job scheduler.
    """
    if scheduler.config.clip_sample or scheduler.config.thresholding:
        raise RuntimeError("inverse assumes clip_sample and thresholding are off")
    if scheduler.config.prediction_type != "epsilon":
        raise RuntimeError(f"inverse assumes epsilon, got {scheduler.config.prediction_type}")
    step_ratio = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    t = int(timestep)
    prev_t = t - step_ratio
    alpha = scheduler.alphas_cumprod[t].to(device=x_prev.device, dtype=torch.float32)
    if prev_t >= 0:
        alpha_prev = scheduler.alphas_cumprod[prev_t].to(device=x_prev.device, dtype=torch.float32)
    else:
        alpha_prev = scheduler.final_alpha_cumprod.to(device=x_prev.device, dtype=torch.float32)
    scale = torch.sqrt(alpha_prev / alpha)
    direction = torch.sqrt(1.0 - alpha_prev) - torch.sqrt(alpha_prev * (1.0 - alpha) / alpha)
    recovered = (x_prev.float() - direction * eps.float()) / scale
    return recovered.to(dtype=x_prev.dtype)


def ddim_invert(guided, scheduler, x_final, timesteps):
    """One conventional pass. guided(x, t) is the InstructPix2Pix noise prediction."""
    x = x_final
    for timestep in reversed(list(timesteps)):
        eps = guided(x, timestep)
        x = invert_ddim_step(scheduler, x, eps, timestep)
    return x


def ddim_invert_fixed_point(guided, scheduler, x_final, timesteps, max_iter=4, tol=1e-5):
    """Solve each actual eta=0 step, capped.

    At iteration k the candidate x is accepted only through the algebraic
    inverse of the noise prediction at that candidate. The logged residual is
    ||step(eps(x), t, x) - x_prev|| before the next update.
    """
    x = x_final
    residuals = []
    for timestep in reversed(list(timesteps)):
        x_prev = x
        eps = guided(x_prev, timestep)
        x = invert_ddim_step(scheduler, x_prev, eps, timestep)
        for _ in range(max_iter):
            eps = guided(x, timestep)
            recon = scheduler.step(eps, timestep, x, eta=0.0, return_dict=False)[0]
            residual = float((recon.float() - x_prev.float()).pow(2).mean().item())
            residuals.append(residual)
            x = invert_ddim_step(scheduler, x_prev, eps, timestep)
            if residual < tol:
                break
    return x, residuals


def assert_algebra_matches_scheduler():
    scheduler = make_pair_ddim()
    scheduler.set_timesteps(100)
    worst = 0.0
    generator = torch.Generator().manual_seed(0)
    for index in (0, 50, 99):
        timestep = scheduler.timesteps[index]
        x = torch.randn(1, 4, 8, 8, generator=generator)
        eps = torch.randn(1, 4, 8, 8, generator=generator)
        y = scheduler.step(eps, timestep, x, eta=0.0, return_dict=False)[0]
        x_hat = invert_ddim_step(scheduler, y, eps, timestep)
        worst = max(worst, float((x_hat - x).abs().max().item()))
    if worst > 1e-5:
        raise RuntimeError(f"DDIM inverse does not match scheduler.step, max abs {worst}")
    return worst
