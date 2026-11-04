"""Canvas and generation-noise shape for Qwen-Image-Edit.

The pixel canvas follows ``calculate_dimensions`` in Diffusers 0.35.1
``pipeline_qwenimage_edit.py``: area 1024×1024, aspect ratio preserved,
each side rounded to a multiple of 32, then truncated to a multiple of
``vae_scale_factor * 2``. The fresh Gaussian is the unpacked tensor
``(batch, time, channels, latent_h, latent_w)`` created in ``prepare_latents``
before ``_pack_latents``. PRC length ``n`` is the number of entries in that
tensor. It is not the Stable Diffusion length 16384.
"""
from __future__ import annotations

import math

TARGET_AREA = 1024 * 1024
VAE_SCALE_FACTOR = 8
LATENT_CHANNELS = 16


def calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    """Return ``(width, height)`` rounded to multiples of 32.

    This is the same arithmetic as the pipeline helper of the same name.
    ``ratio`` is width / height.
    """
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def edit_canvas(width: int, height: int, target_area: int = TARGET_AREA) -> tuple[int, int]:
    """Pixel ``(width, height)`` of the edit, after the pipeline's second truncation."""
    calculated_width, calculated_height = calculate_dimensions(target_area, width / height)
    multiple = VAE_SCALE_FACTOR * 2
    calculated_width = calculated_width // multiple * multiple
    calculated_height = calculated_height // multiple * multiple
    return calculated_width, calculated_height


def generation_shape(pixel_width: int, pixel_height: int) -> tuple[int, int, int, int, int]:
    """Unpacked fresh-noise shape ``(1, 1, 16, latent_h, latent_w)``."""
    multiple = VAE_SCALE_FACTOR * 2
    latent_h = 2 * (int(pixel_height) // multiple)
    latent_w = 2 * (int(pixel_width) // multiple)
    return (1, 1, LATENT_CHANNELS, latent_h, latent_w)


def block_length(pixel_width: int, pixel_height: int) -> int:
    shape = generation_shape(pixel_width, pixel_height)
    return int(math.prod(shape))


def assert_dog1_canvas() -> None:
    """Lock the historical 500×375 photograph to the canvas this chapter uses."""
    width, height = edit_canvas(500, 375)
    if (width, height) != (1184, 896):
        raise RuntimeError(f"unexpected dog canvas {width}x{height}")
    if block_length(width, height) != 265216:
        raise RuntimeError(f"unexpected block length {block_length(width, height)}")
    receiver_width, receiver_height = edit_canvas(width, height)
    if (receiver_width, receiver_height) != (width, height):
        raise RuntimeError("received PNG canvas is not a fixed point of calculate_dimensions")
