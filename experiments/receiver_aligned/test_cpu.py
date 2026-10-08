#!/usr/bin/env python3
"""CPU checks: mixing identities, Detect vs margin, exact message match.

No model load. Exits nonzero on the first failed check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.prc_runtime import encode_codeword, keygen, prc_modules, recover_soft_bits
from experiments.receiver_aligned.pair_diagnostics import (
    common_mode_residual,
    difference_ratio,
    mix_pair,
    unmix_pair,
)
from experiments.receiver_aligned.objectives import code_hinge
from experiments.receiver_aligned.prc_scores import detection_margin, exact_message_match
from experiments.receiver_aligned.state_io import code_hash, git_head


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise SystemExit(1)


def test_hinge():
    carrier = torch.tensor([1.5, -0.2], dtype=torch.float64)
    codeword = torch.tensor([1.0, 1.0], dtype=torch.float64)
    # First coordinate already past tau=1. Second is short by 1.2.
    value = code_hinge(carrier, codeword, 1.0)
    expected = (1.2 ** 2) / 2
    check("hinge", abs(float(value) - expected) < 1e-12, f"{float(value)} vs {expected}")


def test_mixing():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 8, generator=g, dtype=torch.float64)
    y = torch.randn(4, 8, generator=g, dtype=torch.float64)
    eta = torch.randn(4, 8, generator=g, dtype=torch.float64)
    p = 0.93
    rx, ry = common_mode_residual(x, y, eta, p)
    check("common_mode", torch.allclose(rx, torch.zeros_like(rx), atol=1e-12) and torch.allclose(ry, torch.zeros_like(ry), atol=1e-12), f"max {rx.abs().max().item():.3e}")
    after, predicted = difference_ratio(x, y, p)
    check("difference_p2", torch.allclose(after, predicted, atol=1e-12), f"max {(after-predicted).abs().max().item():.3e}")
    x2, y2 = mix_pair(x, y, p)
    xb, yb = unmix_pair(x2, y2, p)
    check("unmix_roundtrip", torch.allclose(xb, x, atol=1e-12) and torch.allclose(yb, y, atol=1e-12))


def test_prc():
    _, Encode, Detect, Decode, _ = prc_modules()
    message = [1, 0, 1, 1, 0, 0, 1, 0]
    enc, dec = keygen(n=512, message_length=32, fpr=1e-5, t=3, seed=2)
    np.random.seed(11)
    code = Encode(enc, message=message)
    uniq = set(torch.unique(code).tolist())
    check("signs_pm1", uniq <= {-1.0, 1.0}, str(sorted(uniq)))
    codeword, z = encode_codeword(enc, message=message, seed=11)
    soft = recover_soft_bits(z, variance=1.5)
    detected = bool(Detect(dec, soft))
    stat, thr, margin = detection_margin(dec, soft, clip=False)
    check("margin_matches_detect", detected == (margin >= 0.0), f"detect={detected} margin={margin:.6f} stat={stat:.4f} thr={thr:.4f}")
    clipped = detection_margin(dec, soft, clip=True)[2]
    check("clip_close_on_clean", abs(clipped - margin) < 1e-6, f"delta {clipped-margin:.3e}")
    decoded = Decode(dec, soft)
    check("exact_message", exact_message_match(decoded, message), f"decoded_none={decoded is None}")
    check("non_none_is_not_enough_alone", exact_message_match(decoded, [0] * len(message)) is False)
    # Random soft bits should usually miss. One draw is not an FPR estimate.
    rand = recover_soft_bits(torch.randn(512, dtype=torch.float64), variance=1.5)
    rand_det = bool(Detect(dec, rand))
    rand_margin = detection_margin(dec, rand, clip=False)[2]
    check("random_margin_matches", rand_det == (rand_margin >= 0.0), f"detect={rand_det} margin={rand_margin:.4f}")
    print("random_draw_detect", rand_det, "not an FPR measurement")


def main():
    print("code_hash", code_hash(Path(__file__).resolve().parent))
    print("experiment_git", git_head(ROOT))
    print("edict_git", git_head(ROOT / "EDICT"))
    print("prc_git", git_head(ROOT / "PRC-Watermark"))
    test_hinge()
    test_mixing()
    test_prc()
    print("ALL_PASS")


if __name__ == "__main__":
    main()
