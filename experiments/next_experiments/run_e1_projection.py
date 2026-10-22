#!/usr/bin/env python3
"""Experiment E1: Direct-Carrier Projection Curve (CPU Oracle Coding Diagnostic).

This is an ORACLE CODING DIAGNOSTIC, not an operational image embedding.
Direct carrier manipulation is not directly realizable at the sender.

Procedure:
1. For saved recovered carriers h (jobs 7788181 and 7793436) and their target codewords c:
   Identify mismatches where c_i * h_i < 0.
   Target: h_i_new = c_i * tau.
   Delta_i = c_i * tau - h_i = c_i * (tau + |h_i|).
   Squared Euclidean carrier cost = (tau + |h_i|)^2.
2. Rank mismatches by correction cost (cheapest first).
3. Apply a sequence of correction budgets:
   fraction of mismatches corrected in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00].
4. For each budget:
   - Compute carrier RMS and max displacement.
   - Compute sign agreement with codeword.
   - Run original Detect and Decode at v = 1.5 (primary) and v = 0.5 (floor).
   - Evaluate exact_message_match (first 64 bits) and full_payload_match (512 bits).
5. Compare multiple target scales: tau in [0.3, 0.5, 0.8, 1.0].
6. Compare ranking strategies:
   - Cheapest first (minimum carrier cost).
   - Random mismatch correction (unranked control).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PRC-Watermark"))

from experiments.paths import RESULTS, ensure_dirs
from experiments.io_utils import save_json
from experiments.receiver_aligned.key_io import load_key, key_hash
from experiments.receiver_aligned.prc_scores import detection_margin, exact_message_match
from src.prc import Detect, Decode


def soft_bits_from_carrier(h, variance: float):
    scale = np.sqrt(2.0 * variance * (1.0 + variance))
    return torch.erf(torch.as_tensor(h, dtype=torch.float64) / scale)


def full_payload_match(decoded, sent_message, key_message_length: int = 512) -> bool:
    if decoded is None:
        return False
    dec = np.asarray(decoded).reshape(-1)
    if dec.size != key_message_length:
        return False
    msg = np.asarray(sent_message, dtype=int).reshape(-1)
    return bool(np.array_equal(dec[: msg.size].astype(int), msg) and np.all(dec[msg.size:] == 0))


def evaluate_carrier(dec_key, h, codeword, message, key_message_length=512, variance=1.5):
    h_np = np.asarray(h, dtype=float).flatten()
    c_np = np.asarray(codeword.detach().cpu().numpy() if hasattr(codeword, "detach") else codeword).flatten()
    agreement = float(np.mean(np.sign(h_np) == np.sign(c_np)))

    soft = soft_bits_from_carrier(h_np, variance=variance)
    stat, thr, margin = detection_margin(dec_key, soft, clip=False)
    det_result = Detect(dec_key, soft)

    t0 = time.time()
    decoded = Decode(dec_key, soft, print_progress=False)
    t_decode = time.time() - t0

    msg_match = exact_message_match(decoded, message)
    payload_match = full_payload_match(decoded, message, key_message_length=key_message_length)

    return {
        "detect": bool(det_result),
        "statistic": float(stat),
        "threshold": float(thr),
        "margin": float(margin),
        "decode_non_none": decoded is not None,
        "exact_message_match": msg_match,
        "full_payload_match": payload_match,
        "sign_agreement": agreement,
        "mean_confidence": float(np.mean(np.abs(soft.numpy()))),
        "decode_time_s": t_decode,
    }


def main():
    ensure_dirs()
    out_dir = RESULTS / "next_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print("EXPERIMENT E1: DIRECT-CARRIER PROJECTION CURVE")
    print("======================================================================")

    runs_to_test = [
        ("r1_codeword_7793436",
         RESULTS / "receiver_aligned/saved_states/r1_codeword_7793436_key.pkl",
         RESULTS / "receiver_aligned/raw_observations/r1_codeword_7793436_true_site_carrier.pt"),
        ("r1_smoke_7788181",
         RESULTS / "receiver_aligned/saved_states/r1_smoke_7788181_key.pkl",
         RESULTS / "receiver_aligned/raw_observations/r1_smoke_7788181_true_site_carrier.pt"),
    ]

    budgets = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    tau_list = [0.3, 0.5, 0.8, 1.0]

    all_records = []

    for run_tag, key_path, carrier_path in runs_to_test:
        print(f"\n==================================================")
        print(f"Testing carrier: {run_tag}")
        print(f"==================================================")

        enc_key, dec_key, message, codeword = load_key(key_path)
        kh = key_hash(enc_key)
        raw_data = torch.load(carrier_path, map_location="cpu")
        h_orig = raw_data["vector"].flatten().numpy()
        c = np.asarray(codeword.detach().cpu().numpy() if hasattr(codeword, "detach") else codeword).flatten()
        n = len(h_orig)

        # Baseline evaluation (0 budget)
        init_eval = evaluate_carrier(dec_key, h_orig, c, message, key_message_length=512, variance=1.5)
        print(f"Initial carrier: Agree={init_eval['sign_agreement']:.4f}, Margin={init_eval['margin']:.2f}, "
              f"Detect={init_eval['detect']}, Decode={init_eval['decode_non_none']}, ExactMsg={init_eval['exact_message_match']}")

        # Identify mismatches
        mismatch_mask = (c * h_orig < 0)
        mismatch_indices = np.where(mismatch_mask)[0]
        num_mismatches = len(mismatch_indices)
        print(f"Total coordinates: {n}, Mismatches: {num_mismatches} ({num_mismatches / n * 100:.2f}%)")

        for tau in tau_list:
            print(f"\n--- Strategy: Cheapest Mismatches First (tau = {tau}) ---")
            # Cost per mismatch: delta = c_i * tau - h_i.
            # cost_sq = (tau + |h_i|)^2
            costs = (tau + np.abs(h_orig[mismatch_indices])) ** 2
            sort_order = np.argsort(costs)  # ascending: cheapest first
            ranked_mismatches = mismatch_indices[sort_order]

            first_decode_agree = None
            first_decode_budget = None
            first_decode_rms = None

            for b in budgets:
                num_correct = int(round(b * num_mismatches))
                coords_to_fix = ranked_mismatches[:num_correct]

                h_proj = h_orig.copy()
                h_proj[coords_to_fix] = c[coords_to_fix] * tau

                disp = h_proj - h_orig
                disp_rms = float(np.sqrt(np.mean(disp ** 2)))
                disp_max = float(np.max(np.abs(disp)))
                carrier_energy = float(np.mean(h_proj ** 2))

                for v in [1.5, 0.5]:
                    ev = evaluate_carrier(dec_key, h_proj, c, message, key_message_length=512, variance=v)
                    rec = {
                        "experiment": "E1_projection",
                        "run_tag": run_tag,
                        "key_hash": kh,
                        "strategy": "cheapest_first",
                        "tau": tau,
                        "budget_frac_mismatches": b,
                        "coords_corrected": num_correct,
                        "variance": v,
                        "displacement_rms": disp_rms,
                        "displacement_max": disp_max,
                        "carrier_energy": carrier_energy,
                        **ev
                    }
                    all_records.append(rec)

                    if v == 1.5 and ev["exact_message_match"] and first_decode_agree is None:
                        first_decode_agree = ev["sign_agreement"]
                        first_decode_budget = b
                        first_decode_rms = disp_rms

                # Print summary for v=1.5
                ev15 = [r for r in all_records if r["run_tag"] == run_tag and r["strategy"] == "cheapest_first" and r["tau"] == tau and r["budget_frac_mismatches"] == b and r["variance"] == 1.5][-1]
                print(f"  b={b:4.2f} (n={num_correct:5d}): agree={ev15['sign_agreement']:.4f}, disp_rms={disp_rms:.4f}, "
                      f"margin={ev15['margin']:+6.1f}, Detect={ev15['detect']}, Decode={ev15['decode_non_none']}, ExactMsg={ev15['exact_message_match']}, FullPayload={ev15['full_payload_match']}")

            print(f"--> First decode for tau={tau}: budget={first_decode_budget}, agree={first_decode_agree}, disp_rms={first_decode_rms}")

        # Control: Random mismatch selection (tau = 0.5)
        print(f"\n--- Control: Random Mismatch Selection (tau = 0.5) ---")
        np.random.seed(12345)
        shuffled_mismatches = np.random.permutation(mismatch_indices)
        tau_ctrl = 0.5
        for b in budgets:
            num_correct = int(round(b * num_mismatches))
            coords_to_fix = shuffled_mismatches[:num_correct]

            h_proj = h_orig.copy()
            h_proj[coords_to_fix] = c[coords_to_fix] * tau_ctrl

            disp = h_proj - h_orig
            disp_rms = float(np.sqrt(np.mean(disp ** 2)))
            disp_max = float(np.max(np.abs(disp)))
            carrier_energy = float(np.mean(h_proj ** 2))

            ev = evaluate_carrier(dec_key, h_proj, c, message, key_message_length=512, variance=1.5)
            rec = {
                "experiment": "E1_projection",
                "run_tag": run_tag,
                "key_hash": kh,
                "strategy": "random_mismatches",
                "tau": tau_ctrl,
                "budget_frac_mismatches": b,
                "coords_corrected": num_correct,
                "variance": 1.5,
                "displacement_rms": disp_rms,
                "displacement_max": disp_max,
                "carrier_energy": carrier_energy,
                **ev
            }
            all_records.append(rec)
            print(f"  [Random] b={b:4.2f} (n={num_correct:5d}): agree={ev['sign_agreement']:.4f}, disp_rms={disp_rms:.4f}, "
                  f"margin={ev['margin']:+6.1f}, Detect={ev['detect']}, Decode={ev['decode_non_none']}, ExactMsg={ev['exact_message_match']}")

    # Save CSV
    csv_path = out_dir / "e1_projection_table.csv"
    fields = list(dict.fromkeys(k for r in all_records for k in r))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nWrote {len(all_records)} records to {csv_path}")

    # Save JSON summary
    summary_path = out_dir / "e1_projection_summary.json"
    save_json({"total_records": len(all_records), "records": all_records}, summary_path)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
