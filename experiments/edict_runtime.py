"""EDICT invert/generate wrappers. Mirrors edict_functions.coupled_stablediffusion with extra logging.

Does not import edict_functions until load_models() so login-node py_compile stays GPU-free.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .image_utils import load_edict_image, pil_to_tensor, tensor_to_pil
from .paths import EDICT_ROOT, ensure_dirs

VAE_SCALE = 0.18215
LATENT_CH, LATENT_H, LATENT_W = 4, 64, 64


def _ensure_edict_on_path():
    root = str(EDICT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_models(device="cuda"):
    """Load SD 1.4 + CLIP in float64 via edict_functions.load_edict_models."""
    ensure_dirs()
    os.environ["EDICT_SKIP_MODEL_LOAD"] = "1"
    os.environ.setdefault("EDICT_HF_AUTH", str(EDICT_ROOT / "hf_auth"))
    _ensure_edict_on_path()
    import edict_functions as ef

    if ef.unet is None:
        ef.load_edict_models(device_arg=device)
    # float64 SD attention maps OOM on 24 GB; slice size 1 is still exact, just slower.
    if hasattr(ef.unet, "set_attention_slice"):
        try:
            ef.unet.set_attention_slice(1)
        except Exception as exc:
            print("attention slice not set:", exc)
    return ef


def make_ddim_scheduler(ef, steps=50):
    sched = ef.DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
        set_alpha_to_one=False,
    )
    sched.set_timesteps(steps)
    return sched


def alpha_bar_at(scheduler, t) -> float:
    """Cumulative alpha_bar_t = alphas_cumprod[t] as used by EDICT get_alpha_and_beta."""
    t_int = int(t.item() if torch.is_tensor(t) else t)
    val = scheduler.alphas_cumprod[t_int]
    if torch.is_tensor(val):
        return float(val.item())
    return float(val)


@torch.no_grad()
def encode_image_to_x0(ef, image, generator=None):
    """x0 = 0.18215 * VAE_enc.sample(I). Shape (1,4,64,64) float64.

    Encoder is sampled, matching edict_functions.image_to_latent (not the posterior mean).
    """
    if not torch.is_tensor(image):
        image = pil_to_tensor(image, device=ef.device, dtype=torch.float64)
    else:
        image = image.to(device=ef.device, dtype=torch.float64)
    x0 = ef.vae.encode(image).latent_dist.sample(generator=generator) * VAE_SCALE
    return x0


@torch.no_grad()
def decode_latent(ef, latent):
    image = ef.vae.decode(latent.to(ef.vae.dtype) / VAE_SCALE).sample
    return tensor_to_pil(image)


def clip_embeddings(ef, prompt: str, null_prompt: str = ""):
    tok = ef.clip_tokenizer
    uncond = tok(
        null_prompt,
        padding="max_length",
        max_length=tok.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    )
    cond = tok(
        prompt,
        padding="max_length",
        max_length=tok.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    )
    emb_u = ef.clip(uncond.input_ids.to(ef.device)).last_hidden_state
    if prompt == null_prompt:
        # Null-null guidance is exactly unconditional; avoid a duplicate CLIP/UNet pass.
        return emb_u, emb_u
    emb_c = ef.clip(cond.input_ids.to(ef.device)).last_hidden_state
    return emb_u, emb_c


def _noise_pred(ef, latent_model_input, t, emb_u, emb_c, guidance_scale):
    if emb_c is emb_u:
        return ef.unet(
            latent_model_input, t, encoder_hidden_states=emb_u
        ).sample
    if guidance_scale == 1:
        return ef.unet(
            latent_model_input, t, encoder_hidden_states=emb_c
        ).sample
    noise_u = ef.unet(latent_model_input, t, encoder_hidden_states=emb_u).sample
    noise_c = ef.unet(latent_model_input, t, encoder_hidden_states=emb_c).sample
    return noise_u + guidance_scale * (noise_c - noise_u)


@torch.no_grad()
def coupled_loop(
    ef,
    prompt: str,
    *,
    reverse: bool,
    steps: int = 50,
    mix_weight: float = 0.93,
    init_image_strength: float = 0.8,
    guidance_scale: float = 3.0,
    seed: int = 1,
    init_latent=None,
    fixed_starting_latent=None,
    null_prompt: str = "",
    leapfrog_steps: bool = True,
    run_baseline: bool = False,
    step_hook=None,
):
    """Invert or generate a coupled (x,y) pair. Returns dict with latents, timesteps, alpha, cosine trace.

    Timestep indexing (verified from my_diffusers DDIMScheduler.set_timesteps and edict_functions):
      S=50 -> timesteps_full = [980,960,...,0]
      t_limit = S - int(S * s); only timesteps_full[t_limit:] are used.
      reverse=True flips that slice so we walk from clean toward noisy.
      After reverse, the last discrete t is timesteps_full[t_limit] (e.g. s=0.8 -> t=780).
      alpha_t := alphas_cumprod[t] at that last reverse step.
    """
    if run_baseline:
        leapfrog_steps = False
    generator = torch.cuda.manual_seed(seed)
    scheduler = make_ddim_scheduler(ef, steps)
    t_limit = steps - int(steps * init_image_strength)
    timesteps = scheduler.timesteps[t_limit:]
    if reverse:
        timesteps = timesteps.flip(0)

    if fixed_starting_latent is not None:
        if isinstance(fixed_starting_latent, list):
            latent_pair = [l.clone() for l in fixed_starting_latent]
        else:
            latent_pair = [fixed_starting_latent.clone(), fixed_starting_latent.clone()]
    elif init_latent is not None:
        if isinstance(init_latent, list):
            latent_pair = [l.clone() for l in init_latent]
        else:
            latent_pair = [init_latent.clone(), init_latent.clone()]
    else:
        noise = torch.randn(
            (1, LATENT_CH, LATENT_H, LATENT_W),
            generator=generator,
            device=ef.device,
            dtype=torch.float64,
        )
        latent_pair = [noise.clone(), noise.clone()]

    emb_u, emb_c = clip_embeddings(ef, prompt, null_prompt=null_prompt)
    cosine = []
    last_t = None
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), leave=False):
        last_t = t
        affine_done = 0
        if reverse and (not run_baseline):
            if step_hook is not None:  # [EXP] match edict_functions reverse S4
                hooked = step_hook("S4", i, t, latent_pair, {"reverse": True, "phase": "before_unmix"})
                if hooked is not None:
                    latent_pair = hooked
            new_latents = [l.clone() for l in latent_pair]
            new_latents[1] = (new_latents[1].clone() - (1 - mix_weight) * new_latents[0].clone()) / mix_weight
            new_latents[0] = (new_latents[0].clone() - (1 - mix_weight) * new_latents[1].clone()) / mix_weight
            latent_pair = new_latents
            if step_hook is not None:  # [EXP]
                hooked = step_hook("S3", i, t, latent_pair, {"reverse": True, "phase": "after_unmix"})
                if hooked is not None:
                    latent_pair = hooked

        for latent_i in range(2):
            if run_baseline and latent_i == 1:
                continue
            li = latent_i
            if reverse and (not run_baseline):
                if leapfrog_steps:
                    orig_i = len(timesteps) - (i + 1)
                    offset = (orig_i + 1) % 2
                    li = (latent_i + offset) % 2
                else:
                    li = (latent_i + 1) % 2
            else:
                if leapfrog_steps:
                    offset = i % 2
                    li = (latent_i + offset) % 2
            lj = ((li + 1) % 2) if not run_baseline else li
            latent_model_input = latent_pair[lj]
            latent_base = latent_pair[li]
            noise_pred = _noise_pred(ef, latent_model_input, t, emb_u, emb_c, guidance_scale)
            step_call = ef.reverse_step if reverse else ef.forward_step
            new_latent = step_call(scheduler, noise_pred, t, latent_base)
            latent_pair[li] = new_latent.to(latent_base.dtype)
            affine_done += 1
            if step_hook is not None:  # [EXP]
                hooked = step_hook("S1" if affine_done == 1 else "S2", i, t, latent_pair, {"reverse": reverse, "updated_index": li})
                if hooked is not None:
                    latent_pair = hooked

        if (not reverse) and (not run_baseline):
            if step_hook is not None:  # [EXP]
                hooked = step_hook("S3", i, t, latent_pair, {"reverse": reverse})
                if hooked is not None:
                    latent_pair = hooked
            new_latents = [l.clone() for l in latent_pair]
            new_latents[0] = (mix_weight * new_latents[0] + (1 - mix_weight) * new_latents[1]).clone()
            new_latents[1] = ((1 - mix_weight) * new_latents[0] + (mix_weight) * new_latents[1]).clone()
            latent_pair = new_latents
            if step_hook is not None:  # [EXP]
                hooked = step_hook("S4", i, t, latent_pair, {"reverse": reverse})
                if hooked is not None:
                    latent_pair = hooked

        x, y = latent_pair[0].flatten(), latent_pair[1].flatten()
        cosine.append(
            float(F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item())
        )

    t_final = last_t
    alpha_t = alpha_bar_at(scheduler, t_final) if t_final is not None else None
    t_final_int = int(t_final.item() if torch.is_tensor(t_final) else t_final) if t_final is not None else None
    return {
        "latent_pair": latent_pair,
        "x": latent_pair[0],
        "y": latent_pair[1],
        "scheduler": scheduler,
        "timesteps": timesteps,
        "t_final": t_final_int,
        "alpha_t": alpha_t,
        "t_limit": t_limit,
        "cosine": cosine,
        "generator": generator,
    }


def invert_image(ef, image, prompt="", **kwargs):
    if not torch.is_tensor(image):
        if isinstance(image, (str, os.PathLike)):
            image = load_edict_image(image)
        x0 = encode_image_to_x0(ef, image, generator=torch.cuda.manual_seed(kwargs.get("seed", 1)))
    else:
        x0 = image
    out = coupled_loop(ef, prompt, reverse=True, init_latent=x0, **kwargs)
    out["x0"] = x0
    return out


def generate_from_pair(ef, latent_pair, prompt, **kwargs):
    return coupled_loop(
        ef, prompt, reverse=False, fixed_starting_latent=latent_pair, **kwargs
    )


def ddim_eps(x_t, x0, alpha_t):
    """eps = (x_t - sqrt(alpha_t) x0) / sqrt(1 - alpha_t)."""
    sa = alpha_t ** 0.5
    so = (1.0 - alpha_t) ** 0.5
    return (x_t - sa * x0) / so


def compose_xt(x0, eps, alpha_t):
    sa = alpha_t ** 0.5
    so = (1.0 - alpha_t) ** 0.5
    return sa * x0 + so * eps
