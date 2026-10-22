"""Random differentiable stand-in for phone edits and platform re-encoding.

Training stages, matching the curriculum:
    clean      identity
    jpeg_photo brightness, contrast, power gamma, HSV saturation, then DiffJPEG
    full       jpeg_photo, plus scale, crop-and-resample, and either blur or sharpen
    eval       only the transforms named in `fixed`

`fixed` values are Python floats. Sampled values are constants with respect to the image,
so gradients flow through the pixels and stop at the sampled parameters, which is what a
straight-through distortion layer should do.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diff_jpeg import DiffJPEG


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = float(sigma)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()
    channels = image.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    x = F.pad(image, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, horizontal, groups=channels)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(x, vertical, groups=channels)


def rgb_to_hsv(rgb: torch.Tensor):
    maxc, idx = rgb.max(dim=1, keepdim=True)
    minc = rgb.min(dim=1, keepdim=True).values
    delta = maxc - minc
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    rc = (maxc - r) / delta.clamp(min=1e-8)
    gc = (maxc - g) / delta.clamp(min=1e-8)
    bc = (maxc - b) / delta.clamp(min=1e-8)
    hue = torch.where(idx == 0, bc - gc, torch.where(idx == 1, 2.0 + rc - bc, 4.0 + gc - rc))
    hue = torch.where(delta > 0, (hue / 6.0) % 1.0, torch.zeros_like(hue))
    sat = torch.where(maxc > 0, delta / maxc.clamp(min=1e-8), torch.zeros_like(maxc))
    return hue, sat, maxc


def hsv_to_rgb(hue: torch.Tensor, sat: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    sector_f = (hue % 1.0) * 6.0
    sector = torch.floor(sector_f)
    frac = sector_f - sector
    p = val * (1.0 - sat)
    q = val * (1.0 - sat * frac)
    t = val * (1.0 - sat * (1.0 - frac))
    columns = [
        torch.cat([val, t, p], dim=1),
        torch.cat([q, val, p], dim=1),
        torch.cat([p, val, t], dim=1),
        torch.cat([p, q, val], dim=1),
        torch.cat([t, p, val], dim=1),
        torch.cat([val, p, q], dim=1),
    ]
    out = torch.zeros_like(columns[0])
    for k, column in enumerate(columns):
        out = out + (sector == k).to(val.dtype) * column
    return out


def adjust_saturation(rgb: torch.Tensor, factor: float) -> torch.Tensor:
    hue, sat, val = rgb_to_hsv(rgb)
    return hsv_to_rgb(hue, (sat * float(factor)).clamp(0, 1), val)


def _draw(image: torch.Tensor, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * torch.rand((), device=image.device))


class MobileChannel(nn.Module):
    def __init__(self):
        super().__init__()
        self.jpeg = DiffJPEG()

    def photometric(self, image, brightness=0.0, contrast=1.0, gamma=1.0, saturation=1.0):
        x = image + float(brightness)
        x = (x - 0.5) * float(contrast) + 0.5
        x = x.clamp(min=1e-4).pow(float(gamma))
        return adjust_saturation(x, saturation)

    def sharpen(self, image, amount: float, sigma: float):
        return image + float(amount) * (image - gaussian_blur(image, sigma))

    def spatial(self, image, scale: float, crop_area: float, center: bool):
        n, _, height, width = image.shape
        scaled_h = max(8, int(round(height * float(scale))))
        scaled_w = max(8, int(round(width * float(scale))))
        scaled = F.interpolate(image, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False)
        area = float(crop_area) * scaled_h * scaled_w
        aspect = 1.0 if center else float(math.exp(_draw(image, math.log(0.75), math.log(1.33))))
        crop_h = min(scaled_h, max(1, int(round(math.sqrt(area / aspect)))))
        crop_w = min(scaled_w, max(1, int(round(area / max(crop_h, 1)))))
        if center:
            top = (scaled_h - crop_h) / 2.0
            left = (scaled_w - crop_w) / 2.0
        else:
            top = _draw(image, 0.0, float(scaled_h - crop_h)) if scaled_h > crop_h else 0.0
            left = _draw(image, 0.0, float(scaled_w - crop_w)) if scaled_w > crop_w else 0.0
        ys = top + (torch.arange(height, device=image.device, dtype=image.dtype) + 0.5) * (crop_h / height) - 0.5
        xs = left + (torch.arange(width, device=image.device, dtype=image.dtype) + 0.5) * (crop_w / width) - 0.5
        gy = (ys + 0.5) / scaled_h * 2.0 - 1.0
        gx = (xs + 0.5) / scaled_w * 2.0 - 1.0
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(n, -1, -1, -1)
        return F.grid_sample(scaled, grid, mode="bilinear", padding_mode="border", align_corners=False)

    def forward(self, image: torch.Tensor, stage: str = "clean", fixed: dict | None = None) -> torch.Tensor:
        fixed = dict(fixed or {})
        if stage == "clean" and not fixed:
            return image
        eval_mode = stage == "eval"
        photo = (not eval_mode and stage in ("jpeg_photo", "full")) or any(
            k in fixed for k in ("brightness", "contrast", "gamma", "saturation")
        )
        do_jpeg = (not eval_mode and stage in ("jpeg_photo", "full")) or "jpeg_quality" in fixed
        do_spatial = (not eval_mode and stage == "full") or any(k in fixed for k in ("scale", "crop_area"))
        do_sharp = (not eval_mode and stage == "full") or any(k in fixed for k in ("blur_sigma", "sharpen_amount"))

        x = image
        if photo:
            x = self.photometric(
                x,
                brightness=fixed.get("brightness", _draw(image, -0.15, 0.15)),
                contrast=fixed.get("contrast", _draw(image, 0.7, 1.3)),
                gamma=fixed.get("gamma", _draw(image, 0.8, 1.2)),
                saturation=fixed.get("saturation", _draw(image, 0.6, 1.4)),
            )
        if do_sharp:
            if "blur_sigma" in fixed and "sharpen_amount" not in fixed:
                x = gaussian_blur(x, fixed["blur_sigma"])
            elif "sharpen_amount" in fixed:
                x = self.sharpen(x, fixed["sharpen_amount"], fixed.get("blur_sigma", 1.0))
            elif torch.rand((), device=image.device) < 0.5:
                x = gaussian_blur(x, _draw(image, 0.5, 1.5))
            else:
                x = self.sharpen(x, _draw(image, 0.4, 1.2), _draw(image, 0.5, 1.5))
        if do_spatial:
            x = self.spatial(
                x,
                scale=fixed.get("scale", _draw(image, 0.5, 1.2) if not eval_mode else 1.0),
                crop_area=fixed.get("crop_area", _draw(image, 0.3, 0.9) if not eval_mode else 1.0),
                center=eval_mode,
            )
        if do_jpeg:
            x = self.jpeg(x, quality=float(fixed.get("jpeg_quality", _draw(image, 50, 95))))
        return x.clamp(0, 1)

    def benign_residual(self, image: torch.Tensor) -> torch.Tensor:
        """A sharpen-and-contrast edit with no payload, used as the discriminator's real class."""
        with torch.no_grad():
            edited = self.photometric(image, brightness=0.0, contrast=1.1, gamma=1.0, saturation=1.05)
            edited = self.sharpen(edited, amount=0.6, sigma=1.0).clamp(0, 1)
        return edited - image
