"""Differentiable EDICT suffix and receiver recovery. New file; production edict_runtime is untouched.

Replicates experiments/edict_runtime.coupled_loop step for step, but:
    * is NOT wrapped in no_grad, so autograd flows through the pair updates;
    * wraps each guided UNet evaluation in torch.utils.checkpoint (recompute in backward),
      so memory holds latent-size tensors, not UNet activations;
    * lets the UNet run in float64 (the fork's own model, same op as production) or in
      float32 (stock diffusers UNet, same fp16 SD-1.4 weights). The EDICT coupling arithmetic,
      mixing, and DDIM coefficients stay float64 in both cases.

Pair indexing used throughout (n = number of active steps, k = forward step index of the site):
    generate (forward) walk: i = 0..n-1, t_fwd[i] descending, leapfrog parity i % 2,
        affine S1, S2, then mix (S3 -> S4). The S4 state of step k is "site k".
    suffix Q_k  : forward steps i = k+1 .. n-1 applied to site k.
    receiver R_k: reverse steps j = 0 .. n-2-k ... see reverse_to_site: k_rev = n-1-k steps
        applied to the duplicated recovered pair, giving the state at "before unmix of reverse
        step k_rev", which equals site k when the pair is exact.

Frozen weights: callers must call freeze(). Gradients flow THROUGH computations to the
pair, not to the weights.
"""
from __future__ import annotations

import glob
import os
from typing import Callable, Optional

import torch
import torch.utils.checkpoint as ckpt

from experiments import edict_runtime as er

VAE_SCALE = er.VAE_SCALE


def load_unet32(device="cuda"):
    """Stock diffusers UNet, same SD-1.4 fp16 weights, run in float32. Not the fork's double-cast model."""
    from diffusers import UNet2DConditionModel as StockUNet

    hf = os.environ.get("HF_HOME", "/fs/clip-scratch/eloirghi/prc-in-edict/hf_cache")
    snap = sorted(glob.glob(f"{hf}/models--CompVis--stable-diffusion-v1-4/snapshots/*"))[0]
    unet = StockUNet.from_pretrained(snap, subfolder="unet", variant="fp16", torch_dtype=torch.float16)
    return unet.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


def freeze(ef, unet32=None):
    ef.unet.requires_grad_(False)
    ef.vae.requires_grad_(False)
    ef.clip.requires_grad_(False)
    ef.unet.eval(), ef.vae.eval()
    if unet32 is not None:
        unet32.requires_grad_(False)


class GuidedEps:
    """Classifier-free guided noise prediction eps_u + G (eps_c - eps_u), checkpointed."""

    def __init__(self, ef, prompt: str, guidance: float, unet_dtype: str = "float64", unet32=None, null_prompt: str = ""):
        self.ef = ef
        self.G = float(guidance)
        self.dtype = unet_dtype
        self.unet32 = unet32
        with torch.no_grad():
            self.eu, self.ec = er.clip_embeddings(ef, prompt, null_prompt=null_prompt)
        self.same = self.ec is self.eu
        if unet_dtype == "float32":
            assert unet32 is not None
            self.eu32 = self.eu.to(torch.float32)
            self.ec32 = self.ec.to(torch.float32)

    def _raw(self, lat64: torch.Tensor, t) -> torch.Tensor:
        if self.dtype == "float64":
            return er._noise_pred(self.ef, lat64, t, self.eu, self.ec, self.G)
        x = lat64.to(torch.float32)
        if self.same:
            return self.unet32(x, t, encoder_hidden_states=self.eu32).sample.to(torch.float64)
        both = self.unet32(torch.cat([x, x]), t, encoder_hidden_states=torch.cat([self.eu32, self.ec32])).sample
        u, c = both.chunk(2)
        return (u + self.G * (c - u)).to(torch.float64)

    def __call__(self, lat64: torch.Tensor, t) -> torch.Tensor:
        if torch.is_grad_enabled() and lat64.requires_grad:
            return ckpt.checkpoint(lambda l: self._raw(l, t), lat64, use_reentrant=False)
        return self._raw(lat64, t)


class Chain:
    """Pair-level EDICT forward/reverse walks over the same timestep slice as coupled_loop."""

    def __init__(self, ef, steps=50, strength=0.8, mix=0.93):
        self.ef = ef
        self.sched = er.make_ddim_scheduler(ef, steps)
        t_limit = steps - int(steps * strength)
        self.t_fwd = self.sched.timesteps[t_limit:]
        self.n = len(self.t_fwd)
        self.t_rev = self.t_fwd.flip(0)
        self.p = float(mix)

    def forward_steps(self, pair, i0: int, i1: int, eps_fn: Callable, record: Optional[dict] = None):
        """Generation steps i in [i0, i1). Matches coupled_loop(reverse=False)."""
        p = self.p
        pair = [pair[0], pair[1]]
        for i in range(i0, i1):
            t = self.t_fwd[i]
            for latent_i in range(2):
                li = (latent_i + (i % 2)) % 2
                lj = (li + 1) % 2
                e = eps_fn(pair[lj], t)
                new = self.ef.forward_step(self.sched, e, t, pair[li])
                pair[li] = new.to(pair[li].dtype)
            x, y = pair
            x2 = p * x + (1.0 - p) * y
            y2 = (1.0 - p) * x2 + p * y
            pair = [x2, y2]
            if record is not None:
                record[i] = [v.detach().cpu().clone() for v in pair]
        return pair

    def reverse_steps(self, pair, j0: int, j1: int, eps_fn: Callable, record: Optional[dict] = None):
        """Inversion steps j in [j0, j1). Matches coupled_loop(reverse=True): unmix then affine."""
        p = self.p
        n = self.n
        pair = [pair[0], pair[1]]
        for j in range(j0, j1):
            t = self.t_rev[j]
            x, y = pair
            y2 = (y - (1.0 - p) * x) / p
            x2 = (x - (1.0 - p) * y2) / p
            pair = [x2, y2]
            for latent_i in range(2):
                orig_i = n - (j + 1)
                li = (latent_i + (orig_i + 1) % 2) % 2
                lj = (li + 1) % 2
                e = eps_fn(pair[lj], t)
                new = self.ef.reverse_step(self.sched, e, t, pair[li])
                pair[li] = new.to(pair[li].dtype)
            if record is not None:
                record[j] = [v.detach().cpu().clone() for v in pair]
        return pair

    def k_rev(self, k: int) -> int:
        return self.n - 1 - k

    def reverse_to_site(self, pair, k: int, eps_fn: Callable):
        """Reverse steps 0..k_rev-1; result is the S4-equivalent state of forward step k."""
        return self.reverse_steps(pair, 0, self.k_rev(k), eps_fn)


def decode_image(ef, latent64: torch.Tensor) -> torch.Tensor:
    """VAE decode, float64, checkpointed when gradients are needed. Output in [-1,1] range (unclamped)."""
    def fn(z):
        return ef.vae.decode(z.to(ef.vae.dtype) / VAE_SCALE).sample
    if torch.is_grad_enabled() and latent64.requires_grad:
        return ckpt.checkpoint(fn, latent64, use_reentrant=False)
    return fn(latent64)


def encode_mean(ef, image: torch.Tensor) -> torch.Tensor:
    """Posterior MEAN encoder times 0.18215 (differentiable surrogate receiver front end)."""
    def fn(img):
        return ef.vae.encode(img.to(ef.vae.dtype)).latent_dist.mean * VAE_SCALE
    if torch.is_grad_enabled() and image.requires_grad:
        return ckpt.checkpoint(fn, image, use_reentrant=False)
    return fn(image)


class LpipsLoss:
    """Differentiable LPIPS(alex) on 256x256 bilinear-antialiased resizes.

    The reported metric experiments.metrics.lpips_distance resizes PIL images with torchvision
    Resize(256). This tensor path is for gradients only; reported numbers come from the PIL path.
    """

    def __init__(self, device="cuda"):
        import lpips

        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        def prep(x):
            x = x.clamp(-1.0, 1.0).to(torch.float32)
            return torch.nn.functional.interpolate(x, size=(256, 256), mode="bilinear", antialias=True, align_corners=False)
        return self.net(prep(a), prep(b)).mean()
