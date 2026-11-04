"""PRC adapter for Qwen-Image-Edit.

Key generation, Encode, Detect, and Decode are the existing PRC-Watermark
functions, reached through ``experiments.prc_runtime`` and
``experiments.receiver_aligned.key_io``. This module does not change those
files. ``pseudogaussians.sample`` is not used for a trial: the ordinary edit
and the PRC edit share one saved standard-Gaussian draw.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from experiments.prc_runtime import (
    detection_margin,
    detect_decode,
    keygen,
    prc_modules,
    recover_soft_bits,
)
from experiments.receiver_aligned.key_io import key_hash, load_key, save_key

SOFT_VARIANCE = 1.5
PARITY_WEIGHT = 3
FALSE_POSITIVE_RATE = 1e-5


def generate_key(n: int, message_length: int, seed: int):
    """Sample one key. The seed does not reproduce the galois one-time pad.

    The returned objects have to be saved. A later job must load them rather
    than call this function again.
    """
    if message_length != 8:
        raise RuntimeError("this chapter's first key is an 8-bit payload, not a padded longer key")
    return keygen(
        n=n,
        message_length=message_length,
        fpr=FALSE_POSITIVE_RATE,
        t=PARITY_WEIGHT,
        seed=seed,
    )


def encode_message(encoding_key, message: list[int], seed: int) -> torch.Tensor:
    """Return the ±1 codeword. The seed fixes Encode's internal coins only."""
    _keygen, encode, *_rest = prc_modules()
    np.random.seed(seed)
    codeword = encode(encoding_key, message=list(message))
    values = set(torch.unique(codeword.detach().cpu()).tolist())
    if not values.issubset({-1.0, 1.0}):
        raise RuntimeError(f"Encode returned values other than ±1: {values}")
    return codeword


def sign_magnitudes(codeword: torch.Tensor, gaussian: np.ndarray) -> torch.Tensor:
    """``z_i = c_i * abs(g_i)`` in the flattened unpacked generation order."""
    flat = np.asarray(gaussian, dtype=np.float64).reshape(-1)
    signs = codeword.detach().to(dtype=torch.float64).cpu().reshape(-1).numpy()
    if flat.shape != signs.shape:
        raise RuntimeError(f"gaussian length {flat.shape} does not match codeword {signs.shape}")
    return torch.from_numpy(signs * np.abs(flat))


def soft_bits(z_flat: torch.Tensor):
    return recover_soft_bits(z_flat, variance=SOFT_VARIANCE)


def score_latent(decoding_key, z_flat: torch.Tensor, message: list[int] | None):
    """Detect and Decode at the fixed variance. ``message`` is only for equality.

    Pass ``message=None`` in the operational receiver. Equality is an evaluator
    comparison and is left as ``None`` in that case.
    """
    posterior = soft_bits(z_flat)
    detected, decode_ok, decoded = detect_decode(decoding_key, posterior)
    statistic, threshold, margin = detection_margin(decoding_key, posterior)
    bits = None if decoded is None else [int(bit) for bit in np.asarray(decoded, dtype=int).reshape(-1)]
    exact = None
    ber = None
    if message is not None:
        # Decode=None is a failed recovery. BER is recorded only when bits come back.
        if bits is None or len(bits) != len(message):
            exact = False
        else:
            exact = bits == [int(bit) for bit in message]
            ber = float(np.mean(np.asarray(bits) != np.asarray(message, dtype=int)))
    return {
        "detect": bool(detected),
        "decode_non_none": bool(decode_ok),
        "decoded_bits": bits,
        "exact_message": exact,
        "payload_ber": ber,
        "detector_statistic": statistic,
        "detector_threshold": threshold,
        "detector_margin": margin,
        "soft_variance": SOFT_VARIANCE,
    }


def save_sender_key(path: Path, encoding_key, decoding_key, message, codeword) -> str:
    return save_key(path, encoding_key, decoding_key, message, codeword)


def load_sender_key(path: Path):
    return load_key(path)


def save_decoding_key_only(path: Path, decoding_key) -> None:
    """Operational key file: decoding key only, no message and no codeword."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump({"dec_key": decoding_key}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_decoding_key_only(path: Path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if "message" in payload or "codeword" in payload or "enc_key" in payload:
        raise RuntimeError(f"{path} carries sender material; the receiver key must contain only dec_key")
    if "dec_key" not in payload:
        raise RuntimeError(f"{path} has no dec_key")
    return payload["dec_key"]


def describe_key(encoding_key) -> dict:
    generator, _otp, test_bits, redundancy, noise_rate = encoding_key
    n, k = generator.shape
    return {
        "key_id": key_hash(encoding_key),
        "n": int(n),
        "k": int(k),
        "g": int(redundancy),
        "num_test_bits": int(len(test_bits)),
        "t": PARITY_WEIGHT,
        "F": FALSE_POSITIVE_RATE,
        "noise_rate": float(noise_rate),
        "soft_variance": SOFT_VARIANCE,
        "message_capacity": int(k - len(test_bits) - int(redundancy)),
    }
