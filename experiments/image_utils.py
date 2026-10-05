"""Image I/O matching EDICT (center-crop then 512 resize)."""
from pathlib import Path
from PIL import Image
import numpy as np
import torch


def load_edict_image(path, size=512):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    m = min(w, h)
    left = (w - m) / 2
    top = (h - m) / 2
    im = im.crop((left, top, left + m, top + m))
    return im.resize((size, size), resample=Image.Resampling.LANCZOS)


def pil_to_tensor(im, device, dtype=torch.float64):
    arr = np.array(im).astype(np.float64) / 255.0 * 2.0 - 1.0
    t = torch.from_numpy(arr[np.newaxis, ...].transpose(0, 3, 1, 2))
    return t.to(device=device, dtype=dtype)


def tensor_to_pil(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)


def jpeg_roundtrip(im, quality):
    import io
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize_roundtrip(im, scale=0.5):
    w, h = im.size
    small = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    return small.resize((w, h), Image.Resampling.LANCZOS)


def center_crop_roundtrip(im, crop_frac=0.875):
    w, h = im.size
    cw, ch = int(w * crop_frac), int(h * crop_frac)
    left = (w - cw) // 2
    top = (h - ch) // 2
    cropped = im.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.Resampling.LANCZOS)


def add_gaussian_noise_pil(im, sigma=0.02, rng=None):
    arr = np.array(im).astype(np.float64) / 255.0
    rng = np.random.default_rng(rng)
    noisy = np.clip(arr + rng.normal(0.0, sigma, arr.shape), 0.0, 1.0)
    return Image.fromarray((noisy * 255).round().astype("uint8"))


def list_photos(photo_dir, names=None):
    photo_dir = Path(photo_dir)
    if names:
        return [photo_dir / n for n in names]
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in photo_dir.iterdir() if p.suffix.lower() in exts)
