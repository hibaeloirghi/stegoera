#!/usr/bin/env python3
"""Experiment E0: Saved-Key / Payload / Decoder Audit (CPU).

Audits:
1. Exact dimensions and sign conventions for both saved keys (jobs 7788181 and 7793436).
2. Key/codeword pairing, noise rates, and parity constraints.
3. Payload padding structure (17 test bits || 39 randomizer bits || 64 message bits || 448 zero bits).
4. Slicing behavior of exact_message_match (first 64 bits) vs full 512-bit payload match.
5. True recovered carriers from actual PNGs (both jobs) under fixed soft-bit conversions.
6. Positive Controls:
   - PC1: Genuine codeword with high soft confidence.
   - PC2: Full PRC-signed Gaussian vector z = c * |g|.
   - PC3: Controlled noisy observations h = sqrt(1-s^2)*c + s*w across noise levels.
7. Synthetic Decoder Tests:
   - Controlled sign-error rates with constant confidence and realistic |h_true| confidence.
   - Confidence magnitude scaling with perfect signs.
   - Random vs structured error placement (least-confident vs most-confident coordinates).
8. Correctly Sized Key Control (message_length = 64, k = 120, no zero-padding).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PRC-Watermark"))

from experiments.paths import RESULTS, ensure_dirs
from experiments.io_utils import save_json
from experiments.receiver_aligned.key_io import load_key, key_hash
from experiments.receiver_aligned.prc_scores import detection_margin, exact_message_match
from src.prc import KeyGen, Encode, Detect, Decode
import src.pseudogaussians as pg


def soft_bits_from_carrier(h, variance: float):
    """Gunn App C.1 soft-bit conversion: erf(z / sqrt(2 v (1 + v))).

    v is sigma_squared (variance), not sigma.
    """
    scale = np.sqrt(2.0 * variance * (1.0 + variance))
    return torch.erf(torch.as_tensor(h, dtype=torch.float64) / scale)


def full_payload_match(decoded, sent_message, key_message_length: int = 512) -> bool:
    """Checks both the intended message (first 64 bits) and zero-padding (remaining bits)."""
    if decoded is None:
        return False
    dec = np.asarray(decoded).reshape(-1)
    if dec.size != key_message_length:
        return False
    msg = np.asarray(sent_message, dtype=int).reshape(-1)
    msg_ok = np.array_equal(dec[: msg.size].astype(int), msg)
    pad_ok = np.all(dec[msg.size:] == 0)
    return bool(msg_ok and pad_ok)


def evaluate_sample(dec_key, h, codeword, message, key_message_length=512, variance=1.5):
    """Evaluates Detect, Decode, margin, agreement, and payload equality."""
    if isinstance(h, torch.Tensor):
        h_np = h.detach().cpu().numpy().flatten()
    else:
        h_np = np.asarray(h, dtype=float).flatten()

    c_np = np.asarray(codeword.detach().cpu().numpy() if hasattr(codeword, "detach") else codeword).flatten()
    agreement = float(np.mean(np.sign(h_np) == np.sign(c_np)))

    # Compute soft bits
    soft = soft_bits_from_carrier(h_np, variance=variance)

    # Detect
    stat, thr, margin = detection_margin(dec_key, soft, clip=False)
    det_result = Detect(dec_key, soft)

    # Decode
    t0 = time.time()
    decoded = Decode(dec_key, soft, print_progress=False)
    decode_time = time.time() - t0

    dec_non_none = decoded is not None
    msg_match = exact_message_match(decoded, message)
    payload_match = full_payload_match(decoded, message, key_message_length=key_message_length)

    mean_conf = float(np.mean(np.abs(soft.numpy())))

    return {
        "detect": bool(det_result),
        "statistic": float(stat),
        "threshold": float(thr),
        "margin": float(margin),
        "decode_non_none": dec_non_none,
        "exact_message_match": msg_match,
        "full_payload_match": payload_match,
        "sign_agreement": agreement,
        "mean_confidence": mean_conf,
        "decode_time_s": decode_time,
        "decoded_length": len(decoded) if dec_non_none else 0,
    }


def main():
    ensure_dirs()
    out_dir = RESULTS / "next_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print("EXPERIMENT E0: SAVED-KEY / PAYLOAD / DECODER AUDIT")
    print("======================================================================")

    audit_records = []
    summary_report = {}

    keys_to_test = [
        ("r1_smoke_7788181",
         RESULTS / "receiver_aligned/saved_states/r1_smoke_7788181_key.pkl",
         RESULTS / "receiver_aligned/raw_observations/r1_smoke_7788181_true_site_carrier.pt"),
        ("r1_codeword_7793436",
         RESULTS / "receiver_aligned/saved_states/r1_codeword_7793436_key.pkl",
         RESULTS / "receiver_aligned/raw_observations/r1_codeword_7793436_true_site_carrier.pt"),
    ]

    for key_tag, key_path, carrier_path in keys_to_test:
        print(f"\n--- Auditing Key: {key_tag} ---")
        enc_key, dec_key, message, codeword = load_key(key_path)
        kh = key_hash(enc_key)
        print(f"Key hash: {kh}")

        G, otp, test_bits, g, noise_rate = enc_key
        (G_dec, H, otp_dec, fpr, noise_rate_dec, test_bits_dec, g_dec, max_bp_iter, t) = dec_key

        n, k = G.shape
        r = H.shape[0]
        num_test_bits = len(test_bits)
        key_capacity = k - num_test_bits - g
        msg_len = len(message)

        print(f"Dimensions: n={n}, k={k}, r={r}, g={g}, t={t}, test_bits={num_test_bits}, message={msg_len}")
        print(f"Key message capacity: {key_capacity} (padding: {key_capacity - msg_len} zeros)")
        print(f"Noise rate: {noise_rate:.6f}, max_bp_iter: {max_bp_iter}, FPR: {fpr}")

        # Check HG = 0 mod 2
        # Note: in KeyGen, generator_matrix and H are constructed with H G = 0 in GF(2)
        HG_sample = (H @ np.asarray(G[:n])) % 2
        hg_zeros = np.all(HG_sample == 0)
        print(f"Parity constraint H @ G = 0 (mod 2): {hg_zeros}")

        # Verify codeword pairing with key
        c_signs = np.asarray(codeword.detach().cpu().numpy() if hasattr(codeword, "detach") else codeword).flatten()
        c_bits = ((1 - c_signs) // 2).astype(int)
        c_clean = (c_bits ^ np.asarray(otp, dtype=int))
        syndrome = (H @ c_clean) % 2
        syndrome_pass_rate = float(np.mean(syndrome == 0))
        print(f"Genuine codeword syndrome pass rate: {syndrome_pass_rate:.4f} (noise rate {noise_rate:.4f})")

        # Load true recovered carrier
        raw_data = torch.load(carrier_path, map_location="cpu")
        h_true = raw_data["vector"].flatten().numpy()

        for v in [1.5, 0.5]:
            eval_true = evaluate_sample(dec_key, h_true, codeword, message, key_capacity, variance=v)
            print(f"True Recovered Carrier (v={v}): Detect={eval_true['detect']}, Margin={eval_true['margin']:.2f}, "
                  f"DecodeNonNone={eval_true['decode_non_none']}, ExactMsg={eval_true['exact_message_match']}, "
                  f"SignAgree={eval_true['sign_agreement']:.4f}, MeanConf={eval_true['mean_confidence']:.4f}")
            audit_records.append({
                "experiment": "E0_audit", "test_group": "true_recovered_carrier", "key_tag": key_tag,
                "variance": v, "case": f"{key_tag}_true_carrier", **eval_true
            })

        # ==============================================================
        # Positive Controls
        # ==============================================================
        print(f"\n--- Positive Controls for {key_tag} ---")

        # PC1: Pure genuine codeword soft bits (+- 0.999)
        pc1_soft = torch.as_tensor(c_signs * 0.999, dtype=torch.float64)
        for v in [1.5, 0.5]:
            eval_pc1 = evaluate_sample(dec_key, pc1_soft * np.sqrt(2 * v * (1 + v)), codeword, message, key_capacity, variance=v)
            print(f"PC1 (Codeword Direct, v={v}): Detect={eval_pc1['detect']}, Margin={eval_pc1['margin']:.2f}, "
                  f"Decode={eval_pc1['decode_non_none']}, ExactMsg={eval_pc1['exact_message_match']}, "
                  f"FullPayload={eval_pc1['full_payload_match']}")
            assert eval_pc1["exact_message_match"], f"FATAL: Positive Control 1 failed to decode for {key_tag} at v={v}!"
            assert eval_pc1["full_payload_match"], f"FATAL: Positive Control 1 failed full payload match for {key_tag} at v={v}!"
            audit_records.append({
                "experiment": "E0_audit", "test_group": "positive_control_1_codeword_direct", "key_tag": key_tag,
                "variance": v, "case": "pc1_codeword_direct", **eval_pc1
            })

        # PC2: Full PRC-signed Gaussian vector z = c * |g|
        np.random.seed(42)
        g_vec = np.random.randn(n)
        z_pc2 = c_signs * np.abs(g_vec)
        for v in [1.5, 0.5]:
            eval_pc2 = evaluate_sample(dec_key, z_pc2, codeword, message, key_capacity, variance=v)
            print(f"PC2 (PRC Gaussian Sample, v={v}): Detect={eval_pc2['detect']}, Margin={eval_pc2['margin']:.2f}, "
                  f"Decode={eval_pc2['decode_non_none']}, ExactMsg={eval_pc2['exact_message_match']}, "
                  f"FullPayload={eval_pc2['full_payload_match']}")
            assert eval_pc2["exact_message_match"], f"FATAL: Positive Control 2 failed to decode for {key_tag} at v={v}!"
            audit_records.append({
                "experiment": "E0_audit", "test_group": "positive_control_2_prc_gaussian", "key_tag": key_tag,
                "variance": v, "case": "pc2_prc_gaussian", **eval_pc2
            })

        # PC3: Controlled noisy observations of genuine codeword
        # h = sqrt(1 - sigma_c^2) * c + sigma_c * w
        print(f"PC3 (Noisy observations sweep):")
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
        for nl in noise_levels:
            np.random.seed(100 + int(nl * 100))
            w = np.random.randn(n)
            h_noisy = np.sqrt(max(0.0, 1.0 - nl ** 2)) * c_signs + nl * w
            for v in [1.5, 0.5]:
                eval_pc3 = evaluate_sample(dec_key, h_noisy, codeword, message, key_capacity, variance=v)
                audit_records.append({
                    "experiment": "E0_audit", "test_group": "positive_control_3_noisy_observations", "key_tag": key_tag,
                    "noise_level": nl, "variance": v, "case": f"noise_{nl:.2f}_v{v}", **eval_pc3
                })
            # print status for v=1.5
            ev15 = [r for r in audit_records if r["test_group"] == "positive_control_3_noisy_observations" and r["noise_level"] == nl and r["variance"] == 1.5][-1]
            print(f"  noise={nl:.2f}: sign_agree={ev15['sign_agreement']:.4f}, margin={ev15['margin']:+6.1f}, "
                  f"Detect={ev15['detect']}, Decode={ev15['decode_non_none']}, ExactMsg={ev15['exact_message_match']}, FullPayload={ev15['full_payload_match']}")

        # ==============================================================
        # Synthetic Decoder Tests
        # ==============================================================
        print(f"\n--- Synthetic Decoder Tests for {key_tag} ---")

        # Test A: Controlled sign error rate with constant confidence A=0.8
        error_rates = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]
        print("Test A: Sign error rate with constant confidence A=0.8:")
        for perr in error_rates:
            np.random.seed(200 + int(perr * 1000))
            flips = np.random.rand(n) < perr
            corrupted_signs = c_signs.copy()
            corrupted_signs[flips] *= -1
            h_syn = corrupted_signs * 0.8 * np.sqrt(2 * 1.5 * (1 + 1.5))  # calibrate to erf ~ 0.8
            eval_syn = evaluate_sample(dec_key, h_syn, codeword, message, key_capacity, variance=1.5)
            audit_records.append({
                "experiment": "E0_audit", "test_group": "synthetic_error_rate_constant_conf", "key_tag": key_tag,
                "error_rate": perr, "variance": 1.5, "case": f"perr_{perr:.3f}_const0.8", **eval_syn
            })
            print(f"  perr={perr:.3f}: sign_agree={eval_syn['sign_agreement']:.4f}, margin={eval_syn['margin']:+6.1f}, "
                  f"Detect={eval_syn['detect']}, Decode={eval_syn['decode_non_none']}, ExactMsg={eval_syn['exact_message_match']}")

        # Test B: Sign error rate with realistic carrier amplitudes |h_true|
        print("Test B: Sign error rate with realistic carrier amplitudes |h_true|:")
        for perr in error_rates:
            np.random.seed(300 + int(perr * 1000))
            flips = np.random.rand(n) < perr
            corrupted_signs = c_signs.copy()
            corrupted_signs[flips] *= -1
            h_syn_real = corrupted_signs * np.abs(h_true)
            eval_syn_real = evaluate_sample(dec_key, h_syn_real, codeword, message, key_capacity, variance=1.5)
            audit_records.append({
                "experiment": "E0_audit", "test_group": "synthetic_error_rate_real_amplitudes", "key_tag": key_tag,
                "error_rate": perr, "variance": 1.5, "case": f"perr_{perr:.3f}_real_amp", **eval_syn_real
            })
            print(f"  perr={perr:.3f}: sign_agree={eval_syn_real['sign_agreement']:.4f}, margin={eval_syn_real['margin']:+6.1f}, "
                  f"Detect={eval_syn_real['detect']}, Decode={eval_syn_real['decode_non_none']}, ExactMsg={eval_syn_real['exact_message_match']}")

        # Test C: Confidence magnitude sweep with perfect signs
        conf_magnitudes = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00, 1.50]
        print("Test C: Confidence magnitude sweep with 100% correct signs:")
        for mag in conf_magnitudes:
            h_mag = c_signs * mag
            eval_mag = evaluate_sample(dec_key, h_mag, codeword, message, key_capacity, variance=1.5)
            audit_records.append({
                "experiment": "E0_audit", "test_group": "synthetic_confidence_magnitude", "key_tag": key_tag,
                "confidence_mag": mag, "variance": 1.5, "case": f"mag_{mag:.2f}", **eval_mag
            })
            print(f"  mag={mag:.2f}: mean_conf={eval_mag['mean_confidence']:.4f}, margin={eval_mag['margin']:+6.1f}, "
                  f"Detect={eval_mag['detect']}, Decode={eval_mag['decode_non_none']}, ExactMsg={eval_mag['exact_message_match']}")

        # Test D: Random vs Structured error placement
        # Flip bits with smallest |h_true| vs largest |h_true| vs random
        print("Test D: Random vs Structured Error Placement (at perr=0.10 and 0.15):")
        for perr in [0.08, 0.10, 0.12, 0.15]:
            num_flips = int(round(perr * n))
            # 1. Random flips
            np.random.seed(400 + int(perr * 100))
            rand_idx = np.random.choice(n, size=num_flips, replace=False)
            h_rand = (c_signs * np.abs(h_true)).copy()
            h_rand[rand_idx] *= -1
            ev_rand = evaluate_sample(dec_key, h_rand, codeword, message, key_capacity, variance=1.5)

            # 2. Flips on smallest |h_true| (least confident coordinates)
            order_asc = np.argsort(np.abs(h_true))
            least_conf_idx = order_asc[:num_flips]
            h_least = (c_signs * np.abs(h_true)).copy()
            h_least[least_conf_idx] *= -1
            ev_least = evaluate_sample(dec_key, h_least, codeword, message, key_capacity, variance=1.5)

            # 3. Flips on largest |h_true| (most confident coordinates - adversarial)
            order_desc = np.argsort(-np.abs(h_true))
            most_conf_idx = order_desc[:num_flips]
            h_most = (c_signs * np.abs(h_true)).copy()
            h_most[most_conf_idx] *= -1
            ev_most = evaluate_sample(dec_key, h_most, codeword, message, key_capacity, variance=1.5)

            audit_records.append({"experiment": "E0_audit", "test_group": "error_placement", "key_tag": key_tag, "error_type": "random", "error_rate": perr, **ev_rand})
            audit_records.append({"experiment": "E0_audit", "test_group": "error_placement", "key_tag": key_tag, "error_type": "least_confident", "error_rate": perr, **ev_least})
            audit_records.append({"experiment": "E0_audit", "test_group": "error_placement", "key_tag": key_tag, "error_type": "most_confident", "error_rate": perr, **ev_most})

            print(f"  perr={perr:.2f}:")
            print(f"    Random:          agree={ev_rand['sign_agreement']:.4f}, margin={ev_rand['margin']:+6.1f}, Decode={ev_rand['decode_non_none']}, ExactMsg={ev_rand['exact_message_match']}")
            print(f"    Least Confident: agree={ev_least['sign_agreement']:.4f}, margin={ev_least['margin']:+6.1f}, Decode={ev_least['decode_non_none']}, ExactMsg={ev_least['exact_message_match']}")
            print(f"    Most Confident:  agree={ev_most['sign_agreement']:.4f}, margin={ev_most['margin']:+6.1f}, Decode={ev_most['decode_non_none']}, ExactMsg={ev_most['exact_message_match']}")

    # ==============================================================
    # Correctly Sized Shorter-Key Control (message_length = 64)
    # ==============================================================
    print(f"\n======================================================================")
    print("CONTROL: Correctly Sized Shorter Key (message_length = 64, k = 120)")
    print("======================================================================")

    np.random.seed(9999)
    short_enc_key, short_dec_key = KeyGen(n=16384, message_length=64, false_positive_rate=1e-5, t=3)
    short_kh = key_hash(short_enc_key)
    short_message = np.random.randint(0, 2, size=64)
    short_codeword = Encode(short_enc_key, message=short_message)
    short_c_signs = np.asarray(short_codeword.detach().cpu().numpy()).flatten()

    G_s, _, test_bits_s, g_s, _ = short_enc_key
    print(f"Shorter Key hash: {short_kh}")
    print(f"Shorter Key dimensions: G={G_s.shape}, test_bits={len(test_bits_s)}, g={g_s}, message_length=64")
    print(f"Total payload length in G: {len(test_bits_s) + g_s + 64} (NO trailing zero padding!)")

    # Positive control 1 on short key
    pc1_short = torch.as_tensor(short_c_signs * 0.999, dtype=torch.float64)
    eval_pc1_s = evaluate_sample(short_dec_key, pc1_short * np.sqrt(2 * 1.5 * (1 + 1.5)), short_codeword, short_message, 64, variance=1.5)
    print(f"Short Key PC1: Detect={eval_pc1_s['detect']}, Margin={eval_pc1_s['margin']:.2f}, Decode={eval_pc1_s['decode_non_none']}, ExactMsg={eval_pc1_s['exact_message_match']}, FullPayload={eval_pc1_s['full_payload_match']}")
    assert eval_pc1_s["exact_message_match"], "Short key PC1 failed exact message match!"
    assert eval_pc1_s["full_payload_match"], "Short key PC1 failed full payload match!"

    # Positive control 2 on short key
    z_short = short_c_signs * np.abs(np.random.randn(n))
    eval_pc2_s = evaluate_sample(short_dec_key, z_short, short_codeword, short_message, 64, variance=1.5)
    print(f"Short Key PC2: Detect={eval_pc2_s['detect']}, Margin={eval_pc2_s['margin']:.2f}, Decode={eval_pc2_s['decode_non_none']}, ExactMsg={eval_pc2_s['exact_message_match']}, FullPayload={eval_pc2_s['full_payload_match']}")
    assert eval_pc2_s["exact_message_match"], "Short key PC2 failed exact message match!"

    # Error rate test on short key
    print("Short Key error rate tolerance (constant conf A=0.8):")
    for perr in [0.0, 0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25]:
        flips = np.random.rand(n) < perr
        s_signs = short_c_signs.copy()
        s_signs[flips] *= -1
        h_s = s_signs * 0.8 * np.sqrt(2 * 1.5 * (1 + 1.5))
        ev_s = evaluate_sample(short_dec_key, h_s, short_codeword, short_message, 64, variance=1.5)
        audit_records.append({
            "experiment": "E0_audit", "test_group": "short_key_error_rate", "key_tag": "short_key_64",
            "error_rate": perr, "variance": 1.5, **ev_s
        })
        print(f"  perr={perr:.3f}: agree={ev_s['sign_agreement']:.4f}, margin={ev_s['margin']:+6.1f}, Detect={ev_s['detect']}, Decode={ev_s['decode_non_none']}, ExactMsg={ev_s['exact_message_match']}, FullPayload={ev_s['full_payload_match']}")

    # Save CSV
    csv_path = out_dir / "e0_audit_table.csv"
    fields = list(dict.fromkeys(k for r in audit_records for k in r))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(audit_records)
    print(f"\nWrote {len(audit_records)} records to {csv_path}")

    # Save JSON summary
    summary_path = out_dir / "e0_audit_summary.json"
    save_json({"total_records": len(audit_records), "records": audit_records}, summary_path)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
