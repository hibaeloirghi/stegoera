"""Coupling-layer hooks for E5–E7. No-op is step_hook=None (do not pass this module)."""
from __future__ import annotations

import torch


def _add_pair(pair, eta, site, extra):
    out = [pair[0].clone(), pair[1].clone()]
    if site in ("S3", "S4"):
        out[0] = out[0] + eta
        out[1] = out[1] + eta
    else:
        idx = extra.get("updated_index", 0)
        out[idx] = out[idx] + eta
    return out


def make_additive_hook(site, k_forward, eta, informed=False, n_forward=None):
    """Add eta at (site, k) on the generate path. If informed, subtract on the inverse path.

    Inverse index mapping: forward loop i=0 is early denoise (high t); reverse loop
    i=0 is early invert (low t). The matching reverse index is (n_forward-1-k_forward)
    when both walks use the same timestep slice length.
    """

    def hook(site_now, i, t, pair, extra):
        reverse = bool(extra.get("reverse", False))
        if site_now != site:
            return None
        if not reverse:
            if i != k_forward:
                return None
            return _add_pair(pair, eta, site, extra)
        if not informed:
            return None
        if n_forward is None:
            return None
        k_rev = (n_forward - 1) - k_forward
        if i != k_rev:
            return None
        # Inverse of add is subtract. S4-forward matches reverse before_unmix.
        return _add_pair(pair, -eta, site, extra)

    return hook


def make_window_prc_hook(site, ks, etas, informed=False, n_forward=None):
    """V2/V3: different eta tensors at a set of forward step indices, same site."""
    table = {int(k): e for k, e in zip(ks, etas)}

    def hook(site_now, i, t, pair, extra):
        reverse = bool(extra.get("reverse", False))
        if site_now != site:
            return None
        if not reverse:
            if i not in table:
                return None
            return _add_pair(pair, table[i], site, extra)
        if not informed:
            return None
        if n_forward is None:
            return None
        k_fwd = (n_forward - 1) - i
        if k_fwd not in table:
            return None
        return _add_pair(pair, -table[k_fwd], site, extra)

    return hook
