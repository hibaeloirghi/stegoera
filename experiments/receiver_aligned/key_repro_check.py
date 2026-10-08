#!/usr/bin/env python3
"""Is the PRC key reproducible from its seed? Prints hashes of the key parts, twice per process."""
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments import prc_runtime as pr


def h(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()[:12]


for rep in range(2):
    enc, dec = pr.keygen(n=4 * 64 * 64, message_length=512, fpr=1e-5, t=3, seed=1)
    G, otp, tb, g, nr = enc
    print(f"rep{rep} otp={h(otp)} testbits={h(tb)} G={h(np.asarray(G))} H={h(dec[1].indices)}", flush=True)
    code, _ = pr.encode_codeword(enc, message=pr.bits_from_message_length(64, 8), seed=100)
    print(f"rep{rep} codeword={h(code.numpy())}", flush=True)
