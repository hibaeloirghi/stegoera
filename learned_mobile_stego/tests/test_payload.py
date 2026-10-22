"""Payload shapes, key permutation, and BCH correction up to the designed radius.

    python tests/test_payload.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.payload import get_codec, key_seed


def test_shapes_and_key():
    codec = get_codec(64, 0.10)
    print(f"B=64 n={codec.n} k={codec.k} d={codec.d} t={codec.t} radius={codec.radius:.4f}")
    assert codec.radius >= 0.10
    message = torch.randint(0, 2, (4, 64))
    a = codec.encode(message, key=7)
    b = codec.encode(message, key=7)
    c = codec.encode(message, key=8)
    assert a.shape == (4, codec.n)
    assert set(a.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert key_seed(7) != key_seed(8)
    recovered = codec.decode(a, key=7)
    assert recovered.shape == (4, 64)
    assert torch.equal(recovered, message)
    assert not torch.equal(codec.decode(a, key=8), message)


def test_corrects_designed_errors():
    codec = get_codec(64, 0.10)
    rng = np.random.default_rng(0)
    for trial in range(5):
        message = torch.randint(0, 2, (1, 64))
        coded = codec.encode(message, key=11)
        positions = rng.choice(codec.n, size=codec.t, replace=False)
        noisy = coded.clone()
        noisy[0, positions] = 1.0 - noisy[0, positions]
        assert torch.equal(codec.decode(noisy, key=11), message), trial
    # A fraction of the codeword equal to the designed radius, on several random patterns.
    flips = int(codec.t)
    assert flips / codec.n + 1e-9 >= min(0.10, codec.radius)


def test_other_lengths_roundtrip():
    for bits in (32, 128):
        codec = get_codec(bits, 0.10)
        message = torch.randint(0, 2, (2, bits))
        print(f"B={bits} n={codec.n} k={codec.k} t={codec.t} radius={codec.radius:.4f}")
        assert codec.radius >= 0.10
        assert torch.equal(codec.decode(codec.encode(message, key=3), key=3), message)


if __name__ == "__main__":
    for test in (test_shapes_and_key, test_corrects_designed_errors, test_other_lengths_roundtrip):
        test()
        print("PASS", test.__name__, flush=True)
