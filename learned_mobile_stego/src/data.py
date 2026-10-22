"""Cover images. A folder of photographs if `image_dir` is set, otherwise synthetic covers.

Synthetic covers are smooth on the left and textured on the right, so the perceptual mask has
something to suppress and something to use. They are a stand-in for a photo set, not one.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
import numpy as np


def synthetic_covers(n: int, size: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    image = torch.rand(n, 3, size, size, generator=g) * 0.15 + 0.45
    yy = torch.linspace(0, 1, size).view(1, 1, size, 1)
    xx = torch.linspace(0, 1, size).view(1, 1, 1, size)
    image = image + 0.08 * torch.sin(6.283 * (1.5 * xx + 0.7 * yy))
    noise = torch.randn(n, 3, size, size, generator=g) * 0.18
    image[:, :, :, size // 2 :] = (image[:, :, :, size // 2 :] + noise[:, :, :, size // 2 :]).clamp(0, 1)
    return image.clamp(0, 1)


def load_folder(path: str, size: int, limit: int | None = None) -> torch.Tensor:
    files = []
    root = Path(path)
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG"):
        files.extend(sorted(root.glob(ext)))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no images in {path}")
    batch = []
    for file in files:
        im = Image.open(file).convert("RGB")
        w, h = im.size
        side = min(w, h)
        im = im.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
        im = im.resize((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(im).astype("float32") / 255.0
        batch.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(batch)


def cover_batch(cfg: dict, n: int | None = None) -> torch.Tensor:
    size = int(cfg["image_size"])
    if cfg.get("image_dir"):
        return load_folder(cfg["image_dir"], size, limit=n)
    count = int(n if n is not None else cfg.get("num_synthetic", 4))
    return synthetic_covers(count, size, int(cfg.get("seed", 0)))
