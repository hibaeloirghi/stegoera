"""CPU check of the four short-payload PRC keys. No image model.

Saves the key objects and the three trial noises. Prints hashes and
pass/fail only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.paths import RESULTS  # noqa: E402
from experiments.prc_runtime import (  # noqa: E402
    detect_decode,
    encode_codeword,
    keygen,
    recover_soft_bits,
)
from experiments.receiver_aligned.key_io import key_hash, load_key, save_key  # noqa: E402

OUT = RESULTS / "ip2p_edict_prc" / "short_payload_study"
KEYS = OUT / "keys"
TRIALS = OUT / "trials"
N = 4 * 64 * 64  # fixed latent length
T_SPARSE = 3  # fixed parity-check weight
FPR = 1e-5  # fixed false-positive target
LENGTHS = (8, 16, 32, 64)


def digest(array) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def main():
    KEYS.mkdir(parents=True, exist_ok=True)
    TRIALS.mkdir(parents=True, exist_ok=True)
    rows = []
    magnitudes = []
    for trial in range(3):  # three predeclared trials
        rng = np.random.default_rng(50_000 + trial)  # magnitude source, independent of KeyGen
        magnitudes.append(rng.standard_normal(N).astype(np.float64))
    for index, length in enumerate(LENGTHS):
        enc, dec = keygen(n=N, message_length=length, fpr=FPR, t=T_SPARSE, seed=8_000 + length)
        generator, _otp, test_bits, g, noise_rate = enc
        _n, k = generator.shape
        if k != length + int(g) + len(test_bits):  # capacity must be the requested payload plus the derived redundancy
            raise RuntimeError(f"unexpected k={k} for length {length}")
        try:
            encode_codeword(enc, message=[0] * (length + 1), seed=1)  # one bit over capacity must be rejected
            too_long_rejected = False
        except AssertionError:
            too_long_rejected = True
        probe = [int(bit) for bit in np.random.default_rng(90_000 + length).integers(0, 2, size=length)]
        codeword, z = encode_codeword(enc, message=probe, seed=91_000 + length)  # throwaway probe, not a study trial
        soft = recover_soft_bits(z, variance=1.5)  # direct initial-noise soft observations
        detected, decode_ok, decoded = detect_decode(dec, soft)
        exact = decoded is not None and list(np.asarray(decoded, dtype=int).reshape(-1)) == probe
        code_soft = recover_soft_bits(codeword.to(dtype=torch.float64), variance=1.5)  # direct codeword channel
        code_detect, code_ok, code_decoded = detect_decode(dec, code_soft)
        code_exact = code_decoded is not None and list(np.asarray(code_decoded, dtype=int).reshape(-1)) == probe
        wrong_enc, wrong_dec = keygen(n=N, message_length=length, fpr=FPR, t=T_SPARSE, seed=70_000 + length)
        wrong_detect, wrong_ok, _wrong = detect_decode(wrong_dec, soft)
        gaussian = torch.from_numpy(np.random.default_rng(60_000 + length).standard_normal(N))
        gauss_detect, gauss_ok, _gauss = detect_decode(dec, recover_soft_bits(gaussian, variance=1.5))
        key_path = KEYS / f"length_{length}.pkl"
        ident = save_key(key_path, enc, dec, probe, codeword)  # the object Stage 1 must load
        reloaded_enc, reloaded_dec, _message, _code = load_key(key_path)
        reload_ok = key_hash(reloaded_enc) == ident and key_hash(reloaded_enc) == key_hash(enc)
        reloaded_detect, _reloaded_ok, reloaded_decoded = detect_decode(reloaded_dec, soft)
        reload_exact = reloaded_decoded is not None and list(np.asarray(reloaded_decoded, dtype=int).reshape(-1)) == probe
        for trial, magnitude in enumerate(magnitudes):
            payload = [int(bit) for bit in np.random.default_rng(100_000 + length * 10 + trial).integers(0, 2, size=length)]
            trial_code, _ignored = encode_codeword(enc, message=payload, seed=110_000 + length * 10 + trial)
            # sample() would draw a new Gaussian. The study multiplies this trial's saved magnitudes.
            signed = torch.as_tensor(trial_code, dtype=torch.float64) * torch.from_numpy(np.abs(magnitude))
            np.savez(
                TRIALS / f"length_{length}_trial_{trial}.npz",
                message=np.asarray(payload, dtype=np.uint8),
                codeword=np.asarray(trial_code, dtype=np.float64),
                z_prc=signed.numpy().reshape(1, 4, 64, 64),
                gaussian=magnitude.reshape(1, 4, 64, 64),
            )
        rows.append({
            "message_length": length,
            "key_id": ident,
            "n": N,
            "k": int(k),
            "g": int(g),
            "num_test_bits": int(len(test_bits)),
            "t": T_SPARSE,
            "F": FPR,
            "noise_rate": float(noise_rate),
            "sigma_squared": 1.5,
            "capacity_rejects_extra_bit": too_long_rejected,
            "codeword_detect": code_detect,
            "codeword_exact": code_exact,
            "noise_detect": detected,
            "noise_decode_non_none": decode_ok,
            "noise_exact": exact,
            "wrong_key_detect": wrong_detect,
            "wrong_key_decode_non_none": wrong_ok,
            "gaussian_detect": gauss_detect,
            "gaussian_decode_non_none": gauss_ok,
            "reload_ok": reload_ok and reloaded_detect and reload_exact,
            "probe_message_sha256": digest(probe),
        })
        print("length", length, "key", ident, "exact", exact, "reload", reload_ok, flush=True)
    ok = all(
        row["capacity_rejects_extra_bit"] and row["codeword_exact"] and row["noise_exact"]
        and row["reload_ok"] and (not row["wrong_key_detect"]) and (not row["gaussian_detect"])
        for row in rows
    )
    with (OUT / "coding_controls.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "coding_controls_status.json").write_text(json.dumps({"ok": ok, "rows": rows}, indent=2) + "\n")
    (OUT / "STATUS_CODING.txt").write_text("PASS\n" if ok else "FAIL\n")
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
