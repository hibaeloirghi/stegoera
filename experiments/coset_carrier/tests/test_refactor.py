"""CPU tests for the coset/carrier/capped-loss refactor. Runs under pytest or as a plain script.

    python experiments/coset_carrier/tests/test_refactor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments import prc_runtime as pr
from experiments.receiver_aligned.prc_torch import TorchDetector
from experiments.receiver_aligned.prc_scores import exact_message_match
from experiments.coset_carrier.prc_coset import select_optimal_coset_codeword
from experiments.coset_carrier.carrier import get_subspace_carrier_mask, carrier_indices
from experiments.coset_carrier.optimization import (
    StegoLossConfig, capped_detector_loss, compute_stego_loss, edict_mix_for_steps,
)

N_SHORT = 2048
_KEY = {}


def short_key():
    if not _KEY:
        enc, dec = pr.keygen(n=N_SHORT, message_length=64, fpr=1e-5, t=3, seed=3)
        msg = pr.bits_from_message_length(64, rng_seed=5)
        _KEY.update(enc=enc, dec=dec, msg=msg)
    return _KEY["enc"], _KEY["dec"], _KEY["msg"]


def test_p_scaling():
    assert abs(edict_mix_for_steps(25) - 0.8649) < 1e-12
    assert abs(edict_mix_for_steps(50) - 0.93) < 1e-12


def test_coset_selector_valid_and_closer():
    enc, dec, msg = short_key()
    _, _, Detect, Decode, _ = pr.prc_modules()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 64, 64, generator=g, dtype=torch.float64)
    y = x + 0.1 * torch.randn(4, 64, 64, generator=g, dtype=torch.float64)
    idx = torch.randperm(4 * 64 * 64, generator=g)[:N_SHORT].sort().values
    sel = select_optimal_coset_codeword(enc, msg, x, y, n_candidates=32, carrier_idx=idx, seed=11)
    c = sel["codeword"]
    assert c.numel() == N_SHORT and set(c.unique().tolist()) <= {-1.0, 1.0}
    soft = pr.recover_soft_bits(3.0 * c, variance=1.5)
    assert bool(Detect(dec, soft))
    assert exact_message_match(Decode(dec, soft), msg)
    assert sel["cost"] <= sel["candidate0_cost"]
    assert sel["cost"] < sel["candidate_cost_mean"]
    assert sel["agreement"] >= sel["candidate_agreement_mean"] - 3 * sel["candidate_agreement_std"]
    print(f"coset R=32 n={N_SHORT}: selected agreement {sel['agreement']:.4f} weighted {sel['weighted_agreement']:.4f}; "
          f"candidates mean {sel['candidate_agreement_mean']:.4f} sd {sel['candidate_agreement_std']:.4f}")


def test_detector_loss_gradient_zero_when_saturated():
    enc, dec, msg = short_key()
    det = TorchDetector(dec)
    cw, _ = pr.encode_codeword(enc, message=msg, seed=1)
    h = (2.0 * cw.to(torch.float64)).clone().requires_grad_(True)
    margin, _, _ = det.margin(h, 1.5)
    assert float(margin) >= 20.0
    loss = capped_detector_loss(margin, 20.0)
    loss.backward()
    assert float(loss) == 0.0 and float(h.grad.abs().max()) == 0.0

    h2 = (0.05 * torch.randn(N_SHORT, dtype=torch.float64)).requires_grad_(True)
    m2, _, _ = det.margin(h2, 1.5)
    assert float(m2) < 20.0
    capped_detector_loss(m2, 20.0).backward()
    assert float(h2.grad.abs().max()) > 0.0


def test_dual_loss_only_hinge_gradient_when_saturated():
    enc, dec, msg = short_key()
    det = TorchDetector(dec)
    cw, _ = pr.encode_codeword(enc, message=msg, seed=2)
    idx = torch.arange(N_SHORT)
    h = (0.3 * cw.to(torch.float64) * 1.0)
    h = torch.where(torch.arange(N_SHORT) < 1500, 2.0 * cw.to(torch.float64), h).requires_grad_(True)
    cfg = StegoLossConfig(lambda_vis=0.0)
    out = compute_stego_loss(h, cw, idx, det, None, None, None, cfg)
    assert float(out["margin"]) >= 20.0 and float(out["l_det"]) == 0.0
    out["total"].backward()
    expected = -cfg.lambda_code * cw.to(torch.float64) * ((cfg.gamma - cw * h.detach()) > 0).to(torch.float64) / N_SHORT
    assert torch.allclose(h.grad, expected, atol=1e-15)


def test_carrier_mask_shape_count_deterministic():
    g = torch.Generator().manual_seed(1)
    h = torch.randn(1, 4, 64, 64, generator=g, dtype=torch.float64)
    h_rt = h + 0.2 * torch.randn(1, 4, 64, 64, generator=g, dtype=torch.float64)
    for n in (2048, 4096):
        m = get_subspace_carrier_mask(h, h_rt, n_carrier=n)
        assert m.shape == (4, 64, 64) and m.dtype == torch.bool and int(m.sum()) == n
        assert torch.equal(m, get_subspace_carrier_mask(h.clone(), h_rt.clone(), n_carrier=n))
        idx = carrier_indices(m)
        assert bool((torch.sign(h.flatten()[idx]) == torch.sign(h_rt.flatten()[idx])).all())


def report_short_key_tolerance():
    """Not an assertion: how many sign errors the n=2048 / 64-bit key tolerates (constant confidence)."""
    enc, dec, msg = short_key()
    _, _, Detect, Decode, _ = pr.prc_modules()
    rng = np.random.default_rng(0)
    for amp in (0.5, 1.0):
        for err in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            ok = 0
            for trial in range(3):
                cw, _ = pr.encode_codeword(enc, message=msg, seed=100 + trial)
                flip = torch.from_numpy(rng.random(N_SHORT) < err)
                z = amp * torch.where(flip, -cw, cw).to(torch.float64)
                soft = pr.recover_soft_bits(z, variance=1.5)
                ok += int(exact_message_match(Decode(dec, soft), msg))
            print(f"short-key tolerance amp={amp} err={err:.2f}: exact decode {ok}/3")


if __name__ == "__main__":
    tests = [test_p_scaling, test_coset_selector_valid_and_closer, test_detector_loss_gradient_zero_when_saturated,
             test_dual_loss_only_hinge_gradient_when_saturated, test_carrier_mask_shape_count_deterministic]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__, flush=True)
        except Exception as exc:
            failed += 1
            print("FAIL", t.__name__, repr(exc), flush=True)
    report_short_key_tolerance()
    sys.exit(1 if failed else 0)
