#!/usr/bin/env python3
"""Train the encoder, decoder, and residual discriminator.

    python train.py --config configs/smoke_test.yaml

Curriculum (1-based epochs): clean, then JPEG + photometric, then crop/scale/sharpen.
The decoder is trained on the permuted BCH codeword. Message recovery is measured after
inverse-permutation and BCH decoding.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.channel.mobile_channel import MobileChannel
from src.core.losses import StegoLoss, discriminator_loss
from src.core.metrics import psnr, ssim
from src.core.payload import get_codec
from src.data import cover_batch
from src.models.decoder import Decoder
from src.models.discriminator import ResidualDiscriminator
from src.models.encoder import Encoder


def stage_for_epoch(epoch: int, cfg: dict) -> str:
    bounds = cfg["curriculum"]
    if epoch <= int(bounds["clean_until"]):
        return "clean"
    if epoch <= int(bounds["jpeg_until"]):
        return "jpeg_photo"
    return "full"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed_everything(int(cfg["seed"]))
    device_name = cfg.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    codec = get_codec(int(cfg["message_bits"]), float(cfg.get("target_ber", 0.10)))
    print(json.dumps({
        "device": device_name,
        "message_bits": codec.message_bits,
        "bch_n": codec.n,
        "bch_k": codec.k,
        "bch_d": codec.d,
        "bch_t": codec.t,
        "bch_radius": codec.radius,
        "target_ber": codec.target_ber,
    }), flush=True)

    encoder = Encoder(codec.n, int(cfg["base_channels"]), float(cfg["alpha"])).to(device)
    decoder = Decoder(codec.n, int(cfg["base_channels"])).to(device)
    disc = ResidualDiscriminator(int(cfg["base_channels"])).to(device)
    channel = MobileChannel().to(device)
    loss_fn = StegoLoss(
        cfg["lambda_msg"], cfg["lambda_perc"], cfg["lambda_l1"], cfg["lambda_adv"], device=device
    )
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=float(cfg["lr"]))
    opt_d = torch.optim.Adam(disc.parameters(), lr=float(cfg["lr"]))

    covers = cover_batch(cfg).to(device)
    key = cfg.get("key", 1)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    history = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        stage = stage_for_epoch(epoch, cfg)
        warm = int(cfg.get("image_warmup_epochs", 0))
        ramp = max(int(cfg.get("image_ramp_epochs", 1)), 1)
        image_scale = 0.0 if epoch <= warm else min(1.0, (epoch - warm) / ramp)
        loss_fn.lambda_l1 = float(cfg["lambda_l1"]) * image_scale
        loss_fn.lambda_perc = float(cfg["lambda_perc"]) * image_scale
        loss_fn.lambda_adv = float(cfg["lambda_adv"]) * image_scale
        order = torch.randperm(covers.shape[0], device=device)
        totals = []
        for step in range(int(cfg["steps_per_epoch"])):
            take = min(int(cfg["batch_size"]), covers.shape[0])
            start = (step * take) % covers.shape[0]
            idx = order[start : start + take]
            if idx.numel() < take:
                idx = torch.cat([idx, order[: take - idx.numel()]])
            batch = covers[idx]
            messages = torch.randint(0, 2, (batch.shape[0], codec.message_bits), device=device)
            coded = codec.encode(messages, key).to(device)

            with torch.no_grad():
                stego_det, _ = encoder(batch, coded)
            fake = (stego_det - batch).detach()
            real = channel.benign_residual(batch)
            opt_d.zero_grad(set_to_none=True)
            d_loss = discriminator_loss(disc(real), disc(fake))
            d_loss.backward()
            opt_d.step()

            opt.zero_grad(set_to_none=True)
            stego, _residual = encoder(batch, coded)
            received = channel(stego, stage=stage)
            logits = decoder(received)
            parts = loss_fn(batch, stego, logits, coded, disc(stego - batch))
            parts["total"].backward()
            opt.step()
            with torch.no_grad():
                coded_ber = float(((logits > 0).float() != coded).float().mean())
            totals.append({
                "loss": float(parts["total"]), "bce": float(parts["bce"]), "l1": float(parts["l1"]),
                "lpips": float(parts["lpips"]), "adv": float(parts["adv"]), "disc": float(d_loss),
                "coded_ber": coded_ber,
            })

        with torch.no_grad():
            probe_msg = torch.randint(0, 2, (covers.shape[0], codec.message_bits), device=device)
            probe_code = codec.encode(probe_msg, key).to(device)
            probe_stego, _ = encoder(covers, probe_code)
            probe_recv = channel(probe_stego, stage=stage)
            probe_logits = decoder(probe_recv)
            recovered = codec.decode(torch.sigmoid(probe_logits), key).to(device)
            exact = float((recovered == probe_msg).all(dim=1).float().mean())
            msg_ber = float((recovered != probe_msg).float().mean())
            row = {
                "epoch": epoch, "stage": stage, "image_loss_scale": image_scale,
                **{k: sum(t[k] for t in totals) / len(totals) for k in totals[0]},
                "message_ber": msg_ber, "exact_recovery": exact,
                "psnr_stego": psnr(probe_stego, covers), "ssim_stego": ssim(probe_stego, covers),
            }
        history.append(row)
        print(json.dumps(row), flush=True)

    elapsed = time.time() - started
    ckpt = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "discriminator": disc.state_dict(),
        "config": cfg,
        "code": {"message_bits": codec.message_bits, "n": codec.n, "k": codec.k, "d": codec.d, "t": codec.t},
        "key": key,
        "history": history,
        "seconds": elapsed,
    }
    ckpt_path = out / "checkpoint.pt"
    torch.save(ckpt, ckpt_path)
    (out / "history.json").write_text(json.dumps({"seconds": elapsed, "history": history}, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(ckpt_path), "seconds": elapsed, "epochs": int(cfg["epochs"])}), flush=True)


if __name__ == "__main__":
    main()
