#!/usr/bin/env python3
"""CPU: TorchDetector margin equals the numpy margin and the original Detect decision.

Cases (real n=16384 key, same seed as experiments): genuine codeword carriers at several
strengths, random carriers, zero/weak observations, observations bracketing the threshold.
Also checks autograd gives a finite gradient and that it matches a finite difference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments import prc_runtime as pr
from experiments.receiver_aligned.prc_scores import detection_margin
from experiments.receiver_aligned.prc_torch import TorchDetector
from experiments.receiver_aligned.state_io import code_hash, git_head


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    print("code_hash", code_hash(Path(__file__).resolve().parent), "git", git_head(ROOT), flush=True)
    enc, dec = pr.keygen(n=4 * 64 * 64, message_length=512, fpr=1e-5, t=3, seed=1)
    message = pr.bits_from_message_length(64, rng_seed=8)
    code, z = pr.encode_codeword(enc, message=message, seed=100)
    det = TorchDetector(dec, device="cpu")
    Detect = pr.prc_modules()[2]
    rng = np.random.default_rng(5)
    cases = []
    for a in (0.0, 0.2, 0.35, 0.38, 0.40, 0.42, 0.5, 0.8, 1.0):
        w = torch.from_numpy(rng.standard_normal(z.numel()))
        cases.append((f"carrier_a{a}", a * z + (1 - a ** 2) ** 0.5 * w))
    cases.append(("zeros", torch.zeros_like(z)))
    cases.append(("weak_1e-6", 1e-6 * torch.from_numpy(rng.standard_normal(z.numel()))))
    cases.append(("huge_signs", 50.0 * code.clone()))
    out = []
    for name, h in cases:
        for v in (0.5, 1.5):
            soft = pr.recover_soft_bits(h, variance=v)
            _s, _t, m_np = detection_margin(dec, soft, clip=False)
            _s2, _t2, m_np_clip = detection_margin(dec, soft, clip=True)
            m_t, st_t, th_t = det.margin(h, v)
            decision = bool(Detect(dec, soft))
            ok_margin = abs(float(m_t) - m_np_clip) <= 1e-6 * max(1.0, abs(m_np_clip))
            ok_dec = (float(m_t) >= 0) == decision
            out.append({"case": name, "sigma_squared": v, "margin_np": m_np, "margin_np_clip": m_np_clip,
                        "margin_torch": float(m_t), "detect": decision,
                        "margin_matches_numpy": ok_margin, "decision_matches_sign": ok_dec})
            check(f"{name} v={v}", ok_margin and ok_dec,
                  f"np={m_np:.6f} npclip={m_np_clip:.6f} torch={float(m_t):.6f} detect={decision}")
    # gradient: finite, and agrees with a central finite difference
    h0 = (0.4 * z + (1 - 0.16) ** 0.5 * torch.from_numpy(rng.standard_normal(z.numel()))).clone().requires_grad_(True)
    m, _, _ = det.margin(h0, 1.5)
    m.backward()
    g = h0.grad.clone()
    check("grad_finite", bool(torch.isfinite(g).all()), f"norm={float(g.norm()):.4f}")
    d = torch.from_numpy(rng.standard_normal(z.numel()))
    d = d / d.norm()
    eps = 1e-5
    mp = float(det.margin((h0.detach() + eps * d), 1.5)[0])
    mm = float(det.margin((h0.detach() - eps * d), 1.5)[0])
    fd = (mp - mm) / (2 * eps)
    ad = float((g * d).sum())
    check("grad_fd", abs(fd - ad) <= 1e-5 * max(1.0, abs(ad)), f"fd={fd:.8f} ad={ad:.8f} rel={abs(fd-ad)/max(1e-12,abs(ad)):.2e}")
    Path(ROOT / "results/receiver_aligned").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/receiver_aligned/test_cpu_torch.json").write_text(json.dumps(out, indent=1))
    print("ALL_PASS")


if __name__ == "__main__":
    main()
