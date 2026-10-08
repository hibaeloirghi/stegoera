"""Detector margin and message check.

The unclipped margin is the same arithmetic as PRC Detect:
    statistic = sum log((1+Pi)/2)
    threshold = sqrt(2 C log(1/F)) + 0.5 sum log(1-Pi^2)
    margin = statistic - threshold
Detect returns statistic >= threshold.

Clipping is a separate, documented path. Pi is clamped to
(-1+1e-12, 1-1e-12) before the logs. It is not the operational
Detect call. Tests record when the two disagree.
"""
from __future__ import annotations

import numpy as np
import torch


def _parity_terms(decoding_key, posteriors, clip: bool):
    (
        _generator_matrix,
        parity_check_matrix,
        one_time_pad,
        false_positive_rate_key,
        noise_rate,
        _test_bits,
        _g,
        _max_bp_iter,
        t,
    ) = decoding_key
    signed = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float))
    values = signed * posteriors.detach().to(torch.float64).flatten().cpu().numpy()
    r = parity_check_matrix.shape[0]
    pi = np.prod(values[parity_check_matrix.indices.reshape(r, t)], axis=1)
    if clip:
        pi = np.clip(pi, -1.0 + 1e-12, 1.0 - 1e-12)
    log_plus = np.log((1.0 + pi) / 2.0)
    log_minus = np.log((1.0 - pi) / 2.0)
    log_prod = log_plus + log_minus
    const = 0.5 * np.sum(np.square(log_plus) + np.square(log_minus) - 0.5 * np.square(log_prod))
    threshold = float(np.sqrt(2.0 * const * np.log(1.0 / false_positive_rate_key)) + 0.5 * log_prod.sum())
    statistic = float(log_plus.sum())
    return statistic, threshold, statistic - threshold


def detection_margin(decoding_key, posteriors, clip: bool = False):
    return _parity_terms(decoding_key, posteriors, clip=clip)


def exact_message_match(decoded, message) -> bool:
    """True only when the decoded payload's first bits equal the sent message."""
    if decoded is None:
        return False
    sent = np.asarray(message, dtype=int).reshape(-1)
    got = np.asarray(decoded).reshape(-1)
    if got.size < sent.size:
        return False
    return bool(np.array_equal(got[: sent.size].astype(int), sent))
