"""Channel, mask, and encoder/decoder autograd.

    python tests/test_channel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.channel.mobile_channel import MobileChannel, adjust_saturation
from src.models.decoder import Decoder
from src.models.encoder import Encoder, perceptual_mask


def _image(seed=0, size=32):
    g = torch.Generator().manual_seed(seed)
    return (0.25 + 0.5 * torch.rand(2, 3, size, size, generator=g)).requires_grad_(True)


def _grad_reaches(name, output, image):
    if image.grad is not None:
        image.grad.zero_()
    output.sum().backward()
    total = float(image.grad.abs().sum())
    assert total > 0 and torch.isfinite(image.grad).all(), (name, total)
    print(f"grad {name}: {total:.4f}")


def test_each_transform_backward():
    channel = MobileChannel()
    image = _image()
    _grad_reaches("jpeg", channel.jpeg(image, quality=75), image)
    _grad_reaches("photo", channel.photometric(image, brightness=0.05, contrast=1.2, gamma=0.9, saturation=1.3), image)
    _grad_reaches("spatial", channel.spatial(image, scale=0.7, crop_area=0.5, center=True), image)
    _grad_reaches("sharpen", channel.sharpen(image, amount=0.8, sigma=1.0), image)
    _grad_reaches("jpeg_photo", channel(image, stage="jpeg_photo"), image)
    _grad_reaches("full", channel(image, stage="full"), image)
    fixed = channel(image, stage="eval", fixed={"jpeg_quality": 50, "crop_area": 0.4, "brightness": -0.05})
    assert fixed.shape == image.shape and float(fixed.min()) >= 0 and float(fixed.max()) <= 1
    _grad_reaches("eval_fixed", fixed, image)
    clean = channel(image, stage="clean")
    assert torch.equal(clean, image)
    sample = torch.rand(1, 3, 32, 32)
    def mse(q):
        return float((channel.jpeg(sample, q) - sample).square().mean())
    m30, m50, m75, m100 = mse(30), mse(50), mse(75), mse(100)
    print(f"jpeg mse q30={m30:.5f} q50={m50:.5f} q75={m75:.5f} q100={m100:.5f}")
    assert m30 > m50 > m75 > m100


def test_saturation_is_hsv():
    image = _image(seed=1)
    out = adjust_saturation(image, 1.5)
    _grad_reaches("saturation", out, image)
    unchanged = adjust_saturation(image.detach(), 1.0)
    assert torch.allclose(unchanged, image.detach(), atol=1e-5)


def test_mask_prefers_texture():
    flat = torch.full((1, 3, 32, 32), 0.5)
    assert float(perceptual_mask(flat).max()) <= 0.2 + 1e-5
    mixed = flat.clone()
    g = torch.Generator().manual_seed(2)
    mixed[:, :, :, 16:] = 0.15 + 0.7 * torch.rand(1, 3, 32, 16, generator=g)
    mask = perceptual_mask(mixed)
    assert float(mask[:, :, :, 16:].mean()) > float(mask[:, :, :, :16].mean())


def test_encoder_channel_decoder_backward():
    torch.manual_seed(0)
    encoder = Encoder(message_channels=8, base_channels=8, alpha=0.2)
    decoder = Decoder(8, base_channels=8)
    channel = MobileChannel()
    image = _image(size=32)
    bits = torch.randint(0, 2, (2, 8))
    stego, residual = encoder(image, bits)
    assert stego.shape == image.shape and residual.shape == image.shape
    received = channel(stego, stage="full")
    loss = decoder(received).square().mean() + (stego - image).abs().mean()
    loss.backward()
    assert float(encoder.head.weight.grad.abs().sum()) > 0
    assert float(decoder.head.weight.grad.abs().sum()) > 0
    assert torch.isfinite(encoder.head.weight.grad).all()


def test_memorizes_one_codeword():
    torch.manual_seed(0)
    bits = torch.randint(0, 2, (1, 32)).float()
    image = torch.rand(1, 3, 32, 32) * 0.5 + 0.25
    encoder = Encoder(32, base_channels=8, alpha=0.5)
    decoder = Decoder(32, base_channels=8)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    ber = 1.0
    for _ in range(80):
        stego, _ = encoder(image, bits)
        logits = decoder(stego)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, bits)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ber = float(((logits.detach() > 0).float() != bits).float().mean())
    print(f"memorize ber {ber:.3f}")
    assert ber == 0.0


if __name__ == "__main__":
    for test in (test_each_transform_backward, test_saturation_is_hsv, test_mask_prefers_texture, test_encoder_channel_decoder_backward, test_memorizes_one_codeword):
        test()
        print("PASS", test.__name__, flush=True)
