#!/usr/bin/env python3
"""Sweep JPEG quality, crop area, and one combined phone-and-platform channel.

    python evaluate.py --config configs/smoke_test.yaml --checkpoint results/smoke/checkpoint.pt

LPIPS and PSNR compare the stego image with the cover. They do not include channel damage.
BER and exact recovery are measured on the B-bit message after inverse permutation and BCH.
Writes a CSV and a Markdown table.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.channel.mobile_channel import MobileChannel
from src.core.metrics import psnr, ssim
from src.core.payload import get_codec
from src.data import cover_batch
from src.models.decoder import Decoder
from src.models.encoder import Encoder


CONDITIONS = [
    {"name": "clean", "fixed": {}},
    *[{"name": f"jpeg_q{q}", "fixed": {"jpeg_quality": q}} for q in (30, 50, 75, 90, 100)],
    *[{"name": f"crop_{p:.1f}", "fixed": {"crop_area": p}} for p in (0.2, 0.4, 0.6, 0.8, 1.0)],
    {"name": "combined_tone_jpeg75_crop0.5", "fixed": {
        "brightness": 0.08, "contrast": 1.15, "saturation": 1.2, "gamma": 1.05,
        "jpeg_quality": 75, "crop_area": 0.5, "scale": 1.0,
    }},
]


def lpips_distance(net, a, b) -> float:
    return float(net(a * 2 - 1, b * 2 - 1).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    code = ckpt["code"]
    device_name = cfg.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    codec = get_codec(int(code["message_bits"]), n=int(code["n"]), d=int(code["d"]))
    encoder = Encoder(codec.n, int(cfg["base_channels"]), float(cfg["alpha"])).to(device).eval()
    decoder = Decoder(codec.n, int(cfg["base_channels"])).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    channel = MobileChannel().to(device).eval()
    import lpips

    lpips_net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    covers = cover_batch(cfg, n=int(cfg.get("eval_images", cfg.get("num_synthetic", 4)))).to(device)
    key = ckpt.get("key", cfg.get("key", 1))
    g = torch.Generator(device="cpu")
    g.manual_seed(int(cfg["seed"]) + 17)
    messages = torch.randint(0, 2, (covers.shape[0], codec.message_bits), generator=g)

    with torch.no_grad():
        coded = codec.encode(messages, key).to(device)
        stego, _ = encoder(covers, coded)
        quality = {
            "LPIPS": lpips_distance(lpips_net, stego, covers),
            "PSNR": psnr(stego, covers),
            "SSIM": ssim(stego, covers),
        }
        rows = []
        for cond in CONDITIONS:
            stage = "clean" if cond["name"] == "clean" else "eval"
            received = channel(stego, stage=stage, fixed=cond["fixed"])
            logits = decoder(received)
            recovered = codec.decode(decoder.probabilities(logits), key)
            exact = (recovered.cpu() == messages).all(dim=1).float().mean()
            ber = (recovered.cpu() != messages).float().mean()
            coded_ber = ((logits > 0).float().cpu() != coded.cpu()).float().mean()
            rows.append({
                "condition": cond["name"],
                "BER": float(ber),
                "Exact_Message_Recovery_Rate": float(exact),
                "LPIPS": quality["LPIPS"],
                "PSNR": quality["PSNR"],
                "SSIM": quality["SSIM"],
                "coded_BER": float(coded_ber),
                "PSNR_received_vs_cover": psnr(received, covers),
            })

    out = Path(args.out or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "eval_summary.csv"
    md_path = out / "eval_summary.md"
    fields = list(rows[0])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(f"{row[k]:.4f}" if isinstance(row[k], float) else str(row[k]) for k in fields) + " |" for row in rows]
    note = (
        "LPIPS, PSNR and SSIM compare the stego image to the cover, so they are the same on every row. "
        "BER is the fraction of message bits wrong after BCH decoding. "
        f"BCH n={codec.n}, k={codec.k}, t={codec.t}, radius={codec.radius:.4f}. "
        "PSNR_received_vs_cover includes the channel.\n"
    )
    md_path.write_text(note + "\n" + "\n".join([header, sep, *body]) + "\n")
    print(md_path.read_text())
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
