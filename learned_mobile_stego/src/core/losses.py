"""L = λ_msg * BCE(logits, coded bits) + λ_perc * LPIPS + λ_l1 * |stego - cover| + λ_adv * L_disc.

BCE is on the permuted BCH codeword. LPIPS is computed in [-1, 1]. The adversarial term is the
generator side: the stego residual should be classified as a benign edit residual.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StegoLoss(nn.Module):
    def __init__(self, lambda_msg=1.0, lambda_perc=1.0, lambda_l1=1.0, lambda_adv=0.0, device="cpu"):
        super().__init__()
        self.lambda_msg = float(lambda_msg)
        self.lambda_perc = float(lambda_perc)
        self.lambda_l1 = float(lambda_l1)
        self.lambda_adv = float(lambda_adv)
        self.lpips = None
        if self.lambda_perc > 0:
            import lpips

            net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
            for param in net.parameters():
                param.requires_grad_(False)
            self.lpips = net

    def forward(self, cover, stego, logits, target_bits, disc_on_stego=None):
        bce = F.binary_cross_entropy_with_logits(logits, target_bits.float())
        l1 = (stego - cover).abs().mean()
        if self.lpips is None:
            perc = stego.new_zeros(())
        else:
            perc = self.lpips(stego * 2 - 1, cover * 2 - 1).mean()
        if disc_on_stego is None or self.lambda_adv == 0:
            adv = stego.new_zeros(())
        else:
            adv = F.binary_cross_entropy_with_logits(disc_on_stego, torch.ones_like(disc_on_stego))
        total = self.lambda_msg * bce + self.lambda_perc * perc + self.lambda_l1 * l1 + self.lambda_adv * adv
        return {"total": total, "bce": bce, "lpips": perc, "l1": l1, "adv": adv}


def discriminator_loss(real_logits, fake_logits):
    real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return real + fake
