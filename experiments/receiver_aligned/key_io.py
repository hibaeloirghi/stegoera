"""Save/load PRC keys. Needed because KeyGen is NOT reproducible from its seed.

PRC-Watermark `KeyGen` draws the generator matrix, one-time pad and test bits with
`galois.GF.Random(...)` (fresh entropy). `np.random.seed` only fixes the parity-check
matrix and the permutation. Verified: two `keygen(seed=1)` calls in one process give different
otp/test-bit/generator hashes and identical H (`key_repro_check.py`).

Rule used from here on: a vector can only be re-scored offline together with the key that
produced the codeword it was embedded with. Every job that embeds or scores saves its key.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np


def _h(a) -> str:
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()[:12]


def key_hash(enc_key) -> str:
    G, otp, test_bits, _g, _noise = enc_key
    return f"otp{_h(otp)}_tb{_h(test_bits)}_G{_h(np.asarray(G))}"


def save_key(path: Path, enc_key, dec_key, message, codeword) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"enc_key": enc_key, "dec_key": dec_key, "message": list(message),
                     "codeword": np.asarray(codeword.detach().cpu().numpy() if hasattr(codeword, "detach") else codeword)}, f)
    return key_hash(enc_key)


def load_key(path: Path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["enc_key"], d["dec_key"], d["message"], d["codeword"]
