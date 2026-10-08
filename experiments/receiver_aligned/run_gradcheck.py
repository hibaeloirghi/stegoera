#!/usr/bin/env python3
"""Validation gate before any optimization.

T1  fp64 suffix Q_k from saved baseline site reproduces D0's saved baseline final pair.
T2  fp64 reverse from the saved final pair returns the saved site (exact pair-map self-check).
T3  fp32-UNet suffix vs the fp64 final pair (dedicated dtype correctness test).
T4  fp32-UNet surrogate receiver (encoder mean, duplicate, reverse) vs D0's fp64 C_encoder_mean
    site carrier, plus error against the true site.
T5  autograd vs central finite difference through the WHOLE fp32 surrogate pipeline (suffix,
    VAE decode, LPIPS, encoder, reverse, torch PRC margin), random direction on (eta_x, eta_y).
T6  autograd vs finite difference through a short fp64 coupled chain (3 steps).
Also records wall time and peak memory of one forward+backward.

Exit status is nonzero if any gated test fails; the smoke optimization depends on afterok.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.paths import RESULTS, ensure_dirs
from experiments.io_utils import append_csv, load_config, save_json, seed_everything
from experiments import edict_runtime as er
from experiments.receiver_aligned import diff_edict as de
from experiments.receiver_aligned.r1_pipeline import R1Config, R1Problem, build
from experiments.receiver_aligned.state_io import code_hash, git_head

CFG = ROOT / "experiments/configs/receiver_aligned/r1_smoke.json"
RESULTS_ALL = []


def record(job, ch, test, metric, value, passed=None, threshold=None, note=""):
    row = {"job_id": job, "code_hash": ch, "test": test, "metric": metric,
           "value": float(value) if value is not None else None, "passed": passed,
           "threshold": threshold, "note": note}
    RESULTS_ALL.append(row)
    append_csv(RESULTS / "receiver_aligned" / "gradcheck.csv", row)
    print(json.dumps(row), flush=True)
    return passed


def maxabs(a, b):
    return float(max((a[0] - b[0]).abs().max(), (a[1] - b[1]).abs().max()))


def main():
    cfg = load_config(CFG)
    ensure_dirs()
    seed_everything(int(cfg["seed"]))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    job = os.environ.get("SLURM_JOB_ID", "local")
    ch = code_hash(Path(__file__).resolve().parent)
    meta = {"job_id": job, "code_hash": ch, "experiment_git": git_head(ROOT), "edict_git": git_head(ROOT / "EDICT"),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "tf32": False, "config": cfg}
    print(json.dumps({k: v for k, v in meta.items() if k != "config"}), flush=True)
    ef = er.load_models(device=cfg["device"])
    u32 = de.load_unet32(cfg["device"])
    de.freeze(ef, u32)
    r1 = R1Config(**cfg["r1"])
    B = build(ef, u32, "float32", r1, cfg)
    chain, q, fin_ref, problem32 = B["chain"], B["q"], B["fin_ref"], B["problem"]
    eps32 = B["eps"]
    eps64 = de.GuidedEps(ef, cfg["edit_prompt"], float(cfg["guidance_scale"]), "float64")
    k, n = r1.k, chain.n
    ok = []

    # ---- T1 / T2: fp64 pair map reproduces D0 numbers
    with torch.no_grad():
        t0 = time.time()
        rec_fwd = {}
        fin64 = chain.forward_steps(q, k + 1, n, eps64, record=rec_fwd)
        t_t1 = time.time() - t0
        e1 = maxabs(fin64, fin_ref)
        ok.append(record(job, ch, "T1_fp64_suffix_vs_saved_final", "max_abs", e1, e1 < 1e-8, 1e-8, f"{t_t1:.0f}s"))
        t0 = time.time()
        rec_site = chain.reverse_to_site(fin_ref, k, eps64)
        e2 = maxabs(rec_site, q)
        ok.append(record(job, ch, "T2_fp64_reverse_vs_saved_site", "max_abs", e2, e2 < 1e-8, 1e-8, f"{time.time()-t0:.0f}s"))

        # ---- T3: fp32 UNet suffix
        torch.cuda.synchronize(); t0 = time.time()
        fin32 = chain.forward_steps(q, k + 1, n, eps32)
        torch.cuda.synchronize(); t_fp32 = time.time() - t0
        mse3 = float((fin32[0] - fin_ref[0]).square().mean())
        mx3 = maxabs(fin32, fin_ref)
        ok.append(record(job, ch, "T3_fp32_suffix_vs_fp64_final", "mse_x", mse3, mse3 < 1e-6, 1e-6, f"max_abs={mx3:.3e}; {t_fp32:.1f}s vs fp64 {t_t1:.0f}s"))
        record(job, ch, "T3_fp32_suffix_vs_fp64_final", "max_abs_both_streams", mx3)
        record(job, ch, "T3_fp32_suffix_vs_fp64_final", "speedup", t_t1 / t_fp32)

        # ---- T4: fp32 surrogate receiver vs D0 fp64 encoder-mean recovery
        img = de.decode_image(ef, fin_ref[0])
        z = de.encode_mean(ef, img)
        torch.cuda.synchronize(); t0 = time.time()
        site32 = chain.reverse_to_site([z, z], k, eps32)
        torch.cuda.synchronize(); t_rec32 = time.time() - t0
        d0vec = torch.load(RESULTS / "receiver_aligned/raw_observations/baseline_C_encoder_mean_s4_x.pt", map_location="cpu")["vector"].to(site32[0].device)
        mse4 = float((site32[0] - d0vec).square().mean())
        err_true = float((site32[0] - q[0]).square().mean())
        ok.append(record(job, ch, "T4_fp32_receiver_vs_D0_fp64_encoder_mean", "mse_site_x", mse4, mse4 < 1e-3, 1e-3, f"D0 oracle_site_x_mse(encoder_mean)=0.04791; reverse {t_rec32:.1f}s"))
        record(job, ch, "T4_fp32_receiver_vs_D0_fp64_encoder_mean", "mse_vs_true_site_fp32", err_true, None, None, "compare with D0 0.04791 (fp64)")
        # fp32 exactness of its own inverse pair map (A-stage analogue)
        rec32_inv = chain.reverse_to_site(fin32, k, eps32)
        record(job, ch, "T4b_fp32_roundtrip_site", "max_abs_site_pair", maxabs(rec32_inv, q), None, None, "fp32 forward then fp32 reverse")

    # ---- T5: autograd vs FD through the whole surrogate
    g = torch.Generator(device="cpu").manual_seed(123)
    rms = float(cfg["gradcheck"]["eta0_rms"])
    eta0 = [rms * torch.randn((1, 4, 64, 64), generator=g, dtype=torch.float64).to(ef.device) for _ in range(2)]
    d = [torch.randn((1, 4, 64, 64), generator=g, dtype=torch.float64).to(ef.device) for _ in range(2)]
    dn = float(torch.sqrt(sum(v.square().mean() for v in d) / 2))
    d = [v / dn for v in d]   # unit RMS per coordinate
    ex = eta0[0].clone().requires_grad_(True)
    ey = eta0[1].clone().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    out = problem32.evaluate(ex, ey)
    torch.cuda.synchronize(); t_fwd = time.time() - t0
    out["total"].backward()
    torch.cuda.synchronize(); t_all = time.time() - t0
    peak = torch.cuda.max_memory_allocated()
    gx, gy = ex.grad, ey.grad
    finite = bool(torch.isfinite(gx).all() and torch.isfinite(gy).all())
    ok.append(record(job, ch, "T5_grad_present_finite", "grad_norm_x", float(gx.norm()), finite and float(gx.norm()) > 0 and float(gy.norm()) > 0, None,
                     f"grad_norm_y={float(gy.norm()):.4e}"))
    record(job, ch, "T5_cost", "seconds_forward", t_fwd)
    record(job, ch, "T5_cost", "seconds_forward_backward", t_all)
    record(job, ch, "T5_cost", "peak_gpu_bytes", peak)
    for name in ("total", "l_code", "l_energy", "l_quality", "margin", "lpips", "pixel_mse", "pair_mse", "carrier_energy"):
        record(job, ch, "T5_terms_at_eta0", name, float(out[name]))
    ad = float((gx * d[0]).sum() + (gy * d[1]).sum())
    best = None
    del out
    torch.cuda.empty_cache()
    for e in cfg["gradcheck"]["fd_eps_pair"]:
        with torch.no_grad():
            lp_ = float(problem32.evaluate(eta0[0] + e * d[0], eta0[1] + e * d[1])["total"])
            lm_ = float(problem32.evaluate(eta0[0] - e * d[0], eta0[1] - e * d[1])["total"])
        fd = (lp_ - lm_) / (2 * e)
        rel = abs(fd - ad) / max(1e-12, abs(ad))
        record(job, ch, "T5_fd_vs_autograd", f"rel_err_eps_{e}", rel, None, None, f"fd={fd:.6e} autograd={ad:.6e}")
        best = rel if best is None else min(best, rel)
    ok.append(record(job, ch, "T5_fd_vs_autograd", "best_rel_err", best, best < 0.05, 0.05, "fp32 UNet; tolerance documented as 5%"))

    # ---- T6: fp64 chain-only FD (3 steps)
    s0 = int(cfg["gradcheck"]["short_chain_start_step"])
    q36 = [v.to(ef.device) for v in rec_fwd[s0]]
    w = [torch.randn((1, 4, 64, 64), generator=g, dtype=torch.float64).to(ef.device) for _ in range(2)]
    dd = [torch.randn((1, 4, 64, 64), generator=g, dtype=torch.float64).to(ef.device) for _ in range(2)]
    ddn = float(torch.sqrt(sum(v.square().mean() for v in dd) / 2))
    dd = [v / ddn for v in dd]

    def short_loss(p0, p1):
        fin = chain.forward_steps([p0, p1], s0 + 1, n, eps64)
        return (w[0] * fin[0]).sum() + (w[1] * fin[1]).sum()

    a0 = q36[0].clone().requires_grad_(True)
    a1 = q36[1].clone().requires_grad_(True)
    t0 = time.time()
    L = short_loss(a0, a1)
    L.backward()
    ad6 = float((a0.grad * dd[0]).sum() + (a1.grad * dd[1]).sum())
    t6 = time.time() - t0
    e6 = float(cfg["gradcheck"]["fd_eps_short"])
    with torch.no_grad():
        lp6 = float(short_loss(q36[0] + e6 * dd[0], q36[1] + e6 * dd[1]))
        lm6 = float(short_loss(q36[0] - e6 * dd[0], q36[1] - e6 * dd[1]))
    fd6 = (lp6 - lm6) / (2 * e6)
    rel6 = abs(fd6 - ad6) / max(1e-12, abs(ad6))
    ok.append(record(job, ch, "T6_fp64_short_chain_fd", "rel_err", rel6, rel6 < 1e-5, 1e-5, f"fd={fd6:.8e} autograd={ad6:.8e}; {n-1-s0} steps; fwd+bwd {t6:.0f}s"))

    summary = {"meta": {k_: v for k_, v in meta.items()}, "all_gated_passed": bool(all(x for x in ok if x is not None)),
               "rows": RESULTS_ALL}
    save_json(summary, RESULTS / "receiver_aligned" / "gradcheck_summary.json")
    print("GRADCHECK_ALL_PASSED" if summary["all_gated_passed"] else "GRADCHECK_FAILED", flush=True)
    sys.exit(0 if summary["all_gated_passed"] else 1)


if __name__ == "__main__":
    main()
