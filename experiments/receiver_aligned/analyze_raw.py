#!/usr/bin/env python3
"""CPU: score saved raw carrier vectors with the real key (seed 1), original Detect/Decode.

Reads results/receiver_aligned/raw_observations/*.pt files that hold {"vector": ...}.
Writes results/receiver_aligned/raw_scores_<job>.csv (never overwrites an older CSV).
Per vector and per soft-bit variance sigma^2 in {0.5, 1.5}:
    statistic, threshold, margin, detect, decode_non_none, exact_message_match,
    sign_agreement, corr(h, c) with the codeword, carrier RMS and the idealized-curve
    equivalent a_equiv = corr / E|g| is NOT claimed; corr is reported raw.
No vector is rescaled or selected by its key score.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.paths import RESULTS
from experiments import prc_runtime as pr
from experiments.io_utils import append_csv
from experiments.receiver_aligned.prc_scores import detection_margin, exact_message_match
from experiments.receiver_aligned.state_io import code_hash, git_head


def main():
    raw = RESULTS / "receiver_aligned" / "raw_observations"
    job = os.environ.get("SLURM_JOB_ID", str(int(time.time())))
    out = RESULTS / "receiver_aligned" / f"raw_scores_{job}.csv"
    ch = code_hash(Path(__file__).resolve().parent)
    # KeyGen is NOT reproducible from its seed (see key_io.py). A vector can only be scored with
    # the key that produced its codeword, so a saved key is mandatory. job 7788093 regenerated a key
    # and its watermarked-vector rows are INVALID; do not repeat that.
    if len(sys.argv) < 2:
        raise SystemExit("usage: analyze_raw.py <saved key .pkl>  (no key, no scoring)")
    from experiments.receiver_aligned.key_io import load_key
    enc, dec, message, code_np = load_key(Path(sys.argv[1]))
    code = torch.from_numpy(np.asarray(code_np)).double().flatten()
    Detect = pr.prc_modules()[2]
    Decode = pr.prc_modules()[3]
    files = sorted(p for p in raw.glob("*.pt") if "final_pair" not in p.name and (len(sys.argv) < 3 or sys.argv[2] in p.name))
    for f in files:
        obj = torch.load(f, map_location="cpu")
        if not isinstance(obj, dict) or "vector" not in obj:
            continue
        h = obj["vector"].double().flatten()
        case = f.name.split("_")[0] if not f.name.startswith("top_g") else "_".join(f.name.split("_")[:2])
        for v in (0.5, 1.5):
            soft = pr.recover_soft_bits(h, variance=v)
            stat, thr, margin = detection_margin(dec, soft, clip=False)
            det = bool(Detect(dec, soft))
            payload = Decode(dec, soft)
            row = {
                "job_id": job, "code_hash": ch, "file": f.name, "case": case, "sigma_squared": v,
                "sigma": float(np.sqrt(v)), "statistic": stat, "threshold": thr, "margin": margin,
                "detect": det, "decode_non_none": payload is not None,
                "exact_message_match": exact_message_match(payload, message),
                "sign_agreement": float((torch.sign(h) == torch.sign(code)).double().mean()),
                "corr_with_codeword": float((h * code).mean() / h.std()),
                "carrier_rms": float(h.square().mean().sqrt()),
                "carrier_mean": float(h.mean()), "carrier_std": float(h.std()),
            }
            append_csv(out, row)
            print({k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in row.items() if k in
                   ("file", "sigma_squared", "margin", "detect", "decode_non_none", "exact_message_match", "sign_agreement", "corr_with_codeword", "carrier_rms")}, flush=True)
    print("RAW_SCORES_DONE", out)


if __name__ == "__main__":
    main()
