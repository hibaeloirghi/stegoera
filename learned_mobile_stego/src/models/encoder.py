"""U-Net residual encoder. The payload is tiled to B x H x W and concatenated with the cover.

I_stego = clip(I + alpha * tanh(r) * A, 0, 1), where A is high where local contrast is high
and near zero on flat regions. The last layer is zero-initialized, so the residual starts at 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..channel.mobile_channel import gaussian_blur


def tile_bits(bits: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(N, M) bits in {0,1} -> (N, M, H, W) in {-1,+1}."""
    mapped = bits.float() * 2.0 - 1.0
    return mapped[:, :, None, None].expand(-1, -1, height, width).contiguous()


def perceptual_mask(image: torch.Tensor, floor: float = 0.2) -> torch.Tensor:
    """(N, 1, H, W). High-pass plus local deviation, scaled per image.

    Smooth regions sit on `floor` rather than zero, so a flat wall carries less payload
    than foliage but is not a hard hole the code cannot write through.
    """
    gray = image.mean(dim=1, keepdim=True)
    high = (gray - gaussian_blur(gray, sigma=1.0)).abs()
    local = gaussian_blur(gray, sigma=2.0)
    deviation = gaussian_blur((gray - local).square(), sigma=2.0).clamp(min=0).sqrt()
    raw = high + deviation
    lo = raw.amin(dim=(2, 3), keepdim=True)
    hi = raw.amax(dim=(2, 3), keepdim=True)
    scaled = ((raw - lo) / (hi - lo + 1e-6)).clamp(0, 1).pow(1.5)
    return float(floor) + (1.0 - float(floor)) * scaled


def spatial_bit_image(bits: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(N, 1, H, W) in {-1,+1}. Pixel (y, x) displays bit (y*W+x) mod M.

    A 1x1 map of the channel-tiled codeword can only shift the whole image's color,
    which is 3 numbers and cannot hold the codeword. This plane gives each bit a location.
    """
    n, m = bits.shape
    values = bits.float() * 2.0 - 1.0
    index = (torch.arange(height * width, device=bits.device) % m).long()
    return values[:, index].reshape(n, 1, height, width)


def _norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class Block(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            _norm(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            _norm(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def _upsample_to(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)


class Encoder(nn.Module):
    def __init__(self, message_channels: int, base_channels: int = 16, alpha: float = 0.15):
        super().__init__()
        self.alpha = float(alpha)
        b = int(base_channels)
        self.in_conv = Block(3 + message_channels, b)
        self.down1 = Block(b, b * 2)
        self.down2 = Block(b * 2, b * 4)
        self.mid = Block(b * 4, b * 4)
        self.up2 = Block(b * 4 + b * 2, b * 2)
        self.up1 = Block(b * 2 + b, b)
        self.head = nn.Conv2d(b, 3, 1)
        self.local = nn.Sequential(
            nn.Conv2d(1, b, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, 3, 1),
        )
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)

    def forward(self, image: torch.Tensor, bits: torch.Tensor):
        height, width = image.shape[-2:]
        tiled = tile_bits(bits, height, width)
        x0 = self.in_conv(torch.cat([image, tiled], dim=1))
        x1 = self.down1(F.avg_pool2d(x0, 2))
        x2 = self.down2(F.avg_pool2d(x1, 2))
        x = self.mid(x2)
        x = self.up2(torch.cat([_upsample_to(x, x1), x1], dim=1))
        x = self.up1(torch.cat([_upsample_to(x, x0), x0], dim=1))
        raw = self.head(x) + self.local(spatial_bit_image(bits, height, width))
        residual = self.alpha * torch.tanh(raw) * perceptual_mask(image)
        return (image + residual).clamp(0, 1), residual
