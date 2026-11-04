"""Qwen-Image-Edit calls for the PRC pilot.

The forward edit is ``QwenImageEditPipeline.__call__``. Velocity prediction
used by the inverse repeats the installed guidance formula and then calls
``FlowMatchEulerDiscreteScheduler.step`` arithmetic:

    previous = float32(sample) + (sigma_next - sigma) * velocity
    previous = previous.to(velocity.dtype)

when ``stochastic_sampling`` is false. The lagged inverse evaluates that
velocity at the later sample. It is an approximation of the implicit step,
and the pilot records the residual instead of calling the approximation exact.
"""
from __future__ import annotations

import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit as edit_module
import torch
from diffusers import QwenImageEditPipeline
from PIL import Image


def load_pipeline(model_id: str, revision: str) -> QwenImageEditPipeline:
    pipe = QwenImageEditPipeline.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if pipe.transformer.config.guidance_embeds:
        raise RuntimeError("checkpoint exposes guidance embeddings; guidance_scale would enter the transformer")
    if bool(pipe.scheduler.config.stochastic_sampling):
        raise RuntimeError("stochastic_sampling would draw noise inside scheduler.step")
    if int(pipe.vae_scale_factor) != 8 or int(pipe.latent_channels) != 16:
        raise RuntimeError(
            f"unexpected VAE scale {pipe.vae_scale_factor} channels {pipe.latent_channels}"
        )
    pipe.transformer.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.enable_model_cpu_offload()
    return pipe


def pack_noise(pipe: QwenImageEditPipeline, unpacked: torch.Tensor) -> torch.Tensor:
    """Pack ``(1, 1, 16, h, w)`` with the pipeline's own ``_pack_latents``."""
    batch, time, channels, height, width = unpacked.shape
    if (batch, time, channels) != (1, 1, 16):
        raise RuntimeError(f"unexpected unpacked noise shape {tuple(unpacked.shape)}")
    return pipe._pack_latents(unpacked, batch, channels, height, width)


def unpack_noise(pipe: QwenImageEditPipeline, packed: torch.Tensor, pixel_height: int, pixel_width: int) -> torch.Tensor:
    """Invert ``_pack_latents`` back to ``(1, 1, 16, h, w)``."""
    unpacked = pipe._unpack_latents(packed, pixel_height, pixel_width, pipe.vae_scale_factor)
    # Unpack returns (batch, channels, time, height, width). The sampler's
    # fresh noise is (batch, time, channels, height, width) with time = 1.
    return unpacked.permute(0, 2, 1, 3, 4).contiguous()


def assert_pack_roundtrip(pipe: QwenImageEditPipeline, pixel_height: int, pixel_width: int) -> None:
    from experiments.qwen_image_edit_prc.geometry import generation_shape

    shape = generation_shape(pixel_width, pixel_height)
    original = torch.randn(*shape, dtype=torch.float32)
    restored = unpack_noise(pipe, pack_noise(pipe, original), pixel_height, pixel_width)
    if restored.shape != original.shape or not torch.equal(restored, original):
        raise RuntimeError("pack/unpack does not restore the unpacked generation noise")


class _FreshNoiseGuard:
    """Raise if the pipeline samples noise while latents were supplied."""

    def __init__(self):
        self._original = edit_module.randn_tensor

    def __enter__(self):
        def blocked(*_args, **_kwargs):
            raise RuntimeError("pipeline drew fresh noise after latents were supplied")

        edit_module.randn_tensor = blocked
        return self

    def __exit__(self, *_exc):
        edit_module.randn_tensor = self._original


def _capture(pipe, box: dict):
    original_prepare = pipe.prepare_latents
    original_encode = pipe.encode_prompt

    def prepare(*args, **kwargs):
        latents, image_latents = original_prepare(*args, **kwargs)
        box["packed_latents"] = latents.detach().cpu()
        box["image_latents"] = image_latents.detach().cpu()
        return latents, image_latents

    def encode(*args, **kwargs):
        embeds, mask = original_encode(*args, **kwargs)
        kind = "negative" if box.get("_next_encode") == "negative" else "positive"
        box[f"{kind}_embeds"] = embeds.detach().cpu()
        box[f"{kind}_mask"] = mask.detach().cpu()
        box["_next_encode"] = "negative"
        return embeds, mask

    pipe.prepare_latents = prepare
    pipe.encode_prompt = encode
    return original_prepare, original_encode


def run_forward(
    pipe: QwenImageEditPipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    packed_latents: torch.Tensor,
    num_inference_steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
):
    """One pipeline edit. The returned trajectory starts with the supplied noise."""
    captured = {}
    trajectory = [packed_latents.detach().cpu()]
    original_prepare, original_encode = _capture(pipe, captured)

    def callback(_pipe, _step, timestep, callback_kwargs):
        trajectory.append(callback_kwargs["latents"].detach().cpu())
        captured.setdefault("timesteps_seen", []).append(timestep.detach().cpu())
        return callback_kwargs

    try:
        with _FreshNoiseGuard():
            output = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                true_cfg_scale=true_cfg_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                latents=packed_latents,
                output_type="pt",
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )
    finally:
        pipe.prepare_latents = original_prepare
        pipe.encode_prompt = original_encode
    if "packed_latents" in captured:
        # This is the noise after the pipeline casts it, which is the Euler state.
        trajectory[0] = captured["packed_latents"]
    scheduler = {
        "sigmas": pipe.scheduler.sigmas.detach().cpu(),
        "timesteps": pipe.scheduler.timesteps.detach().cpu(),
    }
    return output.images, trajectory, captured, scheduler


def predict_velocity(
    pipe: QwenImageEditPipeline,
    packed_latents: torch.Tensor,
    timestep: torch.Tensor,
    conditioning: dict,
    true_cfg_scale: float,
) -> torch.Tensor:
    """Guided velocity at one packed latent, using the pipeline's CFG formula."""
    device = pipe._execution_device
    latents = packed_latents.to(device=device, dtype=pipe.transformer.dtype)
    image_latents = conditioning["image_latents"].to(device=device, dtype=latents.dtype)
    prompt_embeds = conditioning["positive_embeds"].to(device=device, dtype=latents.dtype)
    prompt_mask = conditioning["positive_mask"].to(device=device)
    negative_embeds = conditioning["negative_embeds"].to(device=device, dtype=latents.dtype)
    negative_mask = conditioning["negative_mask"].to(device=device)
    model_input = torch.cat([latents, image_latents], dim=1)
    step = timestep.to(device=device).reshape(-1)
    if step.numel() != 1:
        raise RuntimeError("one timestep is required")
    step = step.expand(latents.shape[0]).to(dtype=latents.dtype)
    img_shapes = conditioning["img_shapes"]
    with pipe.transformer.cache_context("cond"):
        noise_pred = pipe.transformer(
            hidden_states=model_input,
            timestep=step / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=prompt_mask.sum(dim=1).tolist(),
            attention_kwargs={},
            return_dict=False,
        )[0]
    noise_pred = noise_pred[:, : latents.size(1)]
    with pipe.transformer.cache_context("uncond"):
        negative_pred = pipe.transformer(
            hidden_states=model_input,
            timestep=step / 1000,
            guidance=None,
            encoder_hidden_states_mask=negative_mask,
            encoder_hidden_states=negative_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=negative_mask.sum(dim=1).tolist(),
            attention_kwargs={},
            return_dict=False,
        )[0]
    negative_pred = negative_pred[:, : latents.size(1)]
    combined = negative_pred + true_cfg_scale * (noise_pred - negative_pred)
    conditional_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    combined_norm = torch.norm(combined, dim=-1, keepdim=True)
    return combined * (conditional_norm / combined_norm)


def euler_update(sample: torch.Tensor, velocity: torch.Tensor, sigma, sigma_next) -> torch.Tensor:
    """The installed non-stochastic Euler update, in the velocity's dtype."""
    dt = sigma_next - sigma
    return (sample.float() + dt * velocity).to(dtype=velocity.dtype)


def lagged_inverse(sample_next: torch.Tensor, velocity: torch.Tensor, sigma, sigma_next) -> torch.Tensor:
    """One rearrangement of the Euler update, with velocity taken at ``sample_next``."""
    dt = sigma_next - sigma
    return (sample_next.float() - dt * velocity).to(dtype=velocity.dtype)


def conditioning_from_capture(captured: dict, pixel_height: int, pixel_width: int, vae_scale_factor: int) -> dict:
    return {
        "positive_embeds": captured["positive_embeds"],
        "positive_mask": captured["positive_mask"],
        "negative_embeds": captured["negative_embeds"],
        "negative_mask": captured["negative_mask"],
        "image_latents": captured["image_latents"],
        "img_shapes": [[
            (1, pixel_height // vae_scale_factor // 2, pixel_width // vae_scale_factor // 2),
            (1, pixel_height // vae_scale_factor // 2, pixel_width // vae_scale_factor // 2),
        ]],
    }


def encode_png_conditioning(pipe, image: Image.Image, prompt: str, negative_prompt: str, true_cfg_scale: float):
    """Build conditioning from one image and one user string. Used by the receiver.

    The final latent estimate is the pipeline's VAE encode of this image, packed.
    That is not the sender's exact final latent.
    """
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import calculate_dimensions

    width, height, _ = calculate_dimensions(1024 * 1024, image.size[0] / image.size[1])
    multiple = pipe.vae_scale_factor * 2
    width = width // multiple * multiple
    height = height // multiple * multiple
    resized = pipe.image_processor.resize(image, height, width)
    processed = pipe.image_processor.preprocess(resized, height, width).unsqueeze(2)
    device = pipe._execution_device
    positive, positive_mask = pipe.encode_prompt(prompt=prompt, image=resized, device=device)
    negative, negative_mask = pipe.encode_prompt(prompt=negative_prompt, image=resized, device=device)
    _noise_shape = (
        1,
        1,
        pipe.latent_channels,
        2 * (height // multiple),
        2 * (width // multiple),
    )
    zeros = torch.zeros(_noise_shape, device=device, dtype=positive.dtype)
    _ignored, image_latents = pipe.prepare_latents(
        processed.to(device=device, dtype=positive.dtype),
        1,
        pipe.latent_channels,
        height,
        width,
        positive.dtype,
        device,
        None,
        zeros,
    )
    # prepare_latents packs the supplied zeros as well. The image latents are
    # the VAE source representation. Re-encode through the same normalize path
    # and pack that tensor: it is the receiver's estimate of the final latent.
    encoded = pipe._encode_vae_image(processed.to(device=device, dtype=positive.dtype), None)
    latent_h, latent_w = encoded.shape[3:]
    final_estimate = pipe._pack_latents(encoded, 1, pipe.latent_channels, latent_h, latent_w)
    conditioning = {
        "positive_embeds": positive.detach().cpu(),
        "positive_mask": positive_mask.detach().cpu(),
        "negative_embeds": negative.detach().cpu(),
        "negative_mask": negative_mask.detach().cpu(),
        "image_latents": image_latents.detach().cpu(),
        "img_shapes": [[
            (1, height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2),
            (1, height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2),
        ]],
    }
    if true_cfg_scale <= 1:
        raise RuntimeError("receiver settings do not enable true classifier-free guidance")
    return conditioning, final_estimate.detach().cpu(), height, width


def build_schedule(pipe, packed_latents: torch.Tensor, num_inference_steps: int) -> dict:
    """Rebuild the sigma grid from the checkpoint scheduler and the step count.

    The operational receiver calls this. It does not read a saved scheduler file.
    """
    import numpy as np
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import calculate_shift, retrieve_timesteps

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(
        int(packed_latents.shape[1]),
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device=packed_latents.device,
        sigmas=sigmas,
        mu=mu,
    )
    pipe.scheduler.set_begin_index(0)
    return {
        "sigmas": pipe.scheduler.sigmas.detach().cpu(),
        "timesteps": pipe.scheduler.timesteps.detach().cpu(),
        "mu": float(mu),
    }


def invert_packed(
    pipe,
    final_packed: torch.Tensor,
    conditioning: dict,
    scheduler_state: dict,
    true_cfg_scale: float,
) -> torch.Tensor:
    """Walk the saved sigma grid backward. One lagged velocity evaluation per step."""
    sigmas = scheduler_state["sigmas"].to(dtype=torch.float32)
    timesteps = scheduler_state["timesteps"]
    current = final_packed
    torch.set_grad_enabled(False)
    steps = int(timesteps.shape[0])
    for index in range(steps - 1, -1, -1):
        velocity = predict_velocity(
            pipe, current, timesteps[index], conditioning, true_cfg_scale
        )
        current = lagged_inverse(current, velocity.cpu(), sigmas[index], sigmas[index + 1])
    return current
