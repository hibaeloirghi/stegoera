"""Algebra check for a DDIM step and its inverse, with a fixed predictor.

This is not InstructPix2Pix and not an exact inversion of a UNet. It checks
the scalar update we will later wrap around the real denoiser.
"""
from __future__ import annotations

import numpy as np


def ddim_step(x, eps, alpha_t, alpha_prev):
    """One eta=0 DDIM update from noise level alpha_t toward alpha_prev."""
    x = np.asarray(x, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)
    pred_x0 = (x - np.sqrt(1.0 - alpha_t) * eps) / np.sqrt(alpha_t)
    return np.sqrt(alpha_prev) * pred_x0 + np.sqrt(1.0 - alpha_prev) * eps


def ddim_invert_at_available_state(x, eps_fn, alpha_current, alpha_noisier):
    """Reverse one step using the prediction at the state we already have.

    The forward step evaluated eps at the noisier state. This inverse evaluates
    eps at the cleaner state. Those coincide only when eps does not depend on x.
    """
    eps = eps_fn(x)
    pred_x0 = (x - np.sqrt(1.0 - alpha_current) * eps) / np.sqrt(alpha_current)
    return np.sqrt(alpha_noisier) * pred_x0 + np.sqrt(1.0 - alpha_noisier) * eps


def _forward(x, eps_fn, alphas):
    states = [np.asarray(x, dtype=np.float64)]
    for alpha_t, alpha_prev in zip(alphas[:-1], alphas[1:]):
        states.append(ddim_step(states[-1], eps_fn(states[-1]), alpha_t, alpha_prev))
    return states


def _inverse(x, eps_fn, alphas):
    state = np.asarray(x, dtype=np.float64)
    for alpha_current, alpha_noisier in zip(alphas[:0:-1], alphas[-2::-1]):
        state = ddim_invert_at_available_state(state, eps_fn, alpha_current, alpha_noisier)
    return state


def _solve_one_step(x_prev, eps_fn, alpha_t, alpha_prev, n_iter=25):
    """Find x such that ddim_step(x, eps_fn(x), ...) equals x_prev.

    Damped least squares on one 4-vector. Capped. Not a UNet inversion.
    """
    target = np.asarray(x_prev, dtype=np.float64)
    x = target.copy()
    residuals = []
    for _ in range(n_iter):
        f = ddim_step(x, eps_fn(x), alpha_t, alpha_prev) - target
        residuals.append(float(np.linalg.norm(f)))
        if residuals[-1] < 1e-12:
            break
        jac = np.zeros((x.size, x.size), dtype=np.float64)
        scale = 1e-6
        for i in range(x.size):
            xp = x.copy()
            xp[i] += scale
            fp = ddim_step(xp, eps_fn(xp), alpha_t, alpha_prev) - target
            jac[:, i] = (fp - f) / scale
        delta, *_ = np.linalg.lstsq(jac, f, rcond=None)
        x = x - delta
    final = ddim_step(x, eps_fn(x), alpha_t, alpha_prev) - target
    return {
        "n_iter_logged": len(residuals),
        "residual_norm_first": residuals[0],
        "residual_norm_last": float(np.linalg.norm(final)),
        "converged": bool(np.linalg.norm(final) < 1e-8),
    }


def run_toy_ddim():
    alphas = np.array([0.05, 0.2, 0.45, 0.75, 0.95], dtype=np.float64)
    x_t = np.array([0.4, -1.2, 0.2, 0.7], dtype=np.float64)
    constant = np.array([0.3, -0.1, 0.5, -0.4], dtype=np.float64)

    def eps_constant(_x):
        return constant

    def eps_linear(x):
        return 0.25 * x

    forward_constant = _forward(x_t, eps_constant, alphas)
    recovered_constant = _inverse(forward_constant[-1], eps_constant, alphas)
    constant_err = float(np.max(np.abs(recovered_constant - x_t)))

    forward_linear = _forward(x_t, eps_linear, alphas)
    recovered_linear = _inverse(forward_linear[-1], eps_linear, alphas)
    approx_err = float(np.max(np.abs(recovered_linear - x_t)))

    alpha_t, alpha_prev = float(alphas[0]), float(alphas[1])
    x_prev = ddim_step(x_t, eps_linear(x_t), alpha_t, alpha_prev)
    solved = _solve_one_step(x_prev, eps_linear, alpha_t, alpha_prev)

    return {
        "approximation": (
            "Inverse evaluates the predictor at the cleaner state. "
            "The forward step evaluated it at the noisier state."
        ),
        "constant_predictor_max_abs_error": constant_err,
        "constant_predictor_exact": bool(constant_err < 1e-8),
        "state_dependent_predictor_max_abs_error": approx_err,
        "one_step_fixed_point": solved,
    }
