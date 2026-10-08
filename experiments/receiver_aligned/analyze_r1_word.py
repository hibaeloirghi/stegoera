#!/usr/bin/env python3
"""CPU: which sign pattern did the R1 smoke make the PRC detector accept?

Uses the SAVED key of the same job (key_io). Compares the true-channel site carrier against:
  * the embedded codeword c (the real Encode output),
  * the one-time-pad pattern (sign pattern of the all-zero noiseless word),
  * the fraction of parity checks whose signed sign product is +1 (what Detect rewards),
for: the embedded codeword itself, the stego carrier, and the baseline-edit carrier (D0, key mismatch -> not used).
Output: results/receiver_aligned/r1_word_analysis_<job>.json (never overwrites).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.paths import RESULTS
from experiments.receiver_aligned.key_io import load_key


def main():
    key_path, vec_path = Path(sys.argv[1]), Path(sys.argv[2])
    enc, dec, message, code_np = load_key(key_path)
    (_g, H, otp, fpr, noise_rate, _tb, _gd, _bp, t) = dec
    r = H.shape[0]
    idx = H.indices.reshape(r, t)
    code = np.asarray(code_np, dtype=np.float64).flatten()
    h = torch.load(vec_path, map_location="cpu")["vector"].double().flatten().numpy()
    otp = np.asarray(otp, dtype=np.float64)
    otp_sign = 1 - 2 * otp  # signed multiplier used by Detect

    def parity_pass(vec_sign):
        v = vec_sign * otp_sign
        return float((np.prod(v[idx], axis=1) > 0).mean())

    sc, sh = np.sign(code), np.sign(h)
    sc[sc == 0] = 1
    sh[sh == 0] = 1
    out = {
        "key_vector_files": [str(key_path), str(vec_path)],
        "n": int(h.size), "r_checks": int(r), "t": int(t),
        "sign_agreement_stego_vs_codeword": float((sh == sc).mean()),
        "sign_agreement_stego_vs_otp_pattern": float((sh == otp_sign).mean()),
        "sign_agreement_stego_vs_minus_otp_pattern": float((sh == -otp_sign).mean()),
        "parity_pass_rate_stego": parity_pass(sh),
        "parity_pass_rate_embedded_codeword": parity_pass(sc),
        "parity_pass_rate_random_signs": float(np.mean([parity_pass(np.sign(np.random.randn(h.size))) for _ in range(5)])),
        "corr_h_codeword": float(np.corrcoef(h, code)[0, 1]),
        "carrier_rms": float(np.sqrt((h ** 2).mean())),
        "frac_coords_sign_matches_codeword_top10pct_abs": float((sh == sc)[np.abs(h) >= np.quantile(np.abs(h), 0.9)].mean()),
    }
    job = os.environ.get("SLURM_JOB_ID", str(int(time.time())))
    p = RESULTS / "receiver_aligned" / f"r1_word_analysis_{job}.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
