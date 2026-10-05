"""Pixel MSE, optional LPIPS / CLIP. Failures are recorded rather than invented."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def pixel_mse(a: Image.Image, b: Image.Image) -> float:
    """MSE on pixels mapped to [-1, 1], matching edict_functions.recon_test."""
    aa = np.array(a.convert("RGB")).astype(np.float64) / 255.0 * 2.0 - 1.0
    bb = np.array(b.convert("RGB")).astype(np.float64) / 255.0 * 2.0 - 1.0
    return float(np.square(aa - bb).mean())


def latent_mse(x, y) -> float:
    x = x.detach().to(torch.float64).flatten()
    y = y.detach().to(torch.float64).flatten()
    return float(torch.mean((x - y) ** 2).item())


_lpips_model = None


def lpips_distance(a: Image.Image, b: Image.Image, device="cuda"):
    global _lpips_model
    try:
        import lpips
        from torchvision import transforms
    except Exception as exc:
        return None, f"lpips_unavailable:{exc}"
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex").to(device)
        _lpips_model.eval()
    tfm = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )
    ta = tfm(a.convert("RGB")).unsqueeze(0).to(device) * 2 - 1
    tb = tfm(b.convert("RGB")).unsqueeze(0).to(device) * 2 - 1
    with torch.no_grad():
        d = _lpips_model(ta, tb)
    return float(d.item()), None


_clip_bundle = None


def clip_score(image: Image.Image, prompt: str, device="cuda"):
    global _clip_bundle
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:
        return None, f"clip_unavailable:{exc}"
    if _clip_bundle is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model.eval()
        _clip_bundle = (model, proc)
    model, proc = _clip_bundle
    inputs = proc(text=[prompt], images=image.convert("RGB"), return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        score = torch.cosine_similarity(out.image_embeds, out.text_embeds).item()
    return float(score), None
