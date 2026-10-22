"""Payload bits, a binary BCH wrapper, and a key-derived permutation.

c = Perm_k(BCH(m || pad)). The network predicts the permuted codeword (length n), not the
raw B message bits: BCH adds parity, so those positions have to be embedded or the code
cannot correct anything. `decode` inverts the permutation and returns the B message bits.

`galois.BCH` is the ECC. A primitive binary BCH code with room for B message bits does not
always reach a worst-case correction radius of 0.10*n (the designed-distance bound spends
about m*t parity bits). The constructor takes the shortest code, among n in {127, 255, 511,
1023}, whose relative radius t/n is largest and at least `target_ber` when such a code
exists. The achieved t is stored on the codec and checked by the unit test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

_CACHE_PATH = Path(__file__).with_name("bch_cache.json")
_BUILT: dict[tuple, "PayloadCodec"] = {}
# Measured with galois.BCH: shortest n in {127, 255, 511, 1023} with t/n >= 0.10 and k >= B.
_KNOWN = {
    (32, 0.10): {"n": 127, "k": 36, "d": 31, "t": 15, "radius": 15 / 127},
    (64, 0.10): {"n": 255, "k": 71, "d": 59, "t": 29, "radius": 29 / 255},
    (128, 0.10): {"n": 511, "k": 130, "d": 111, "t": 55, "radius": 55 / 511},
}


def key_seed(key) -> int:
    if isinstance(key, int):
        raw = int(key).to_bytes(16, "little", signed=False)
    else:
        raw = str(key).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def _load_cache() -> dict:
    if _CACHE_PATH.is_file():
        return json.loads(_CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def _search_bch(message_bits: int, target_ber: float):
    import galois

    best = None
    for n in (127, 255, 511, 1023):
        if n <= message_bits + 8:
            continue
        lo, hi = 2, n - message_bits
        feasible = None
        while lo <= hi:
            d = (lo + hi) // 2
            try:
                code = galois.BCH(n, d=d)
            except Exception:
                hi = d - 1
                continue
            if code.k >= message_bits:
                feasible = code
                lo = int(code.d) + 1
            else:
                hi = d - 1
        if feasible is None:
            continue
        t = (int(feasible.d) - 1) // 2
        radius = t / n
        if best is None or radius > best[0] or (radius == best[0] and n < best[1]):
            best = (radius, n, int(feasible.k), int(feasible.d), t)
        print(f"bch n={n} k={feasible.k} d={feasible.d} t={t} radius={radius:.4f}", flush=True)
        if radius + 1e-12 >= target_ber:
            break
    if best is None:
        raise RuntimeError(f"no binary BCH code can carry {message_bits} bits")
    return {"n": best[1], "k": best[2], "d": best[3], "t": best[4], "radius": best[0]}


class PayloadCodec:
    def __init__(self, message_bits: int = 64, target_ber: float = 0.10, n: int | None = None, d: int | None = None):
        if message_bits < 1:
            raise ValueError("message_bits must be positive")
        import galois

        self.message_bits = int(message_bits)
        self.target_ber = float(target_ber)
        if n is None or d is None:
            cache = _load_cache()
            slot = _KNOWN.get((self.message_bits, self.target_ber)) or cache.get(f"{self.message_bits}:{self.target_ber}")
            if slot is None:
                slot = _search_bch(self.message_bits, self.target_ber)
                cache[f"{self.message_bits}:{self.target_ber}"] = slot
                _save_cache(cache)
            n, d = int(slot["n"]), int(slot["d"])
        self.bch = galois.BCH(int(n), d=int(d))
        if self.bch.k < self.message_bits:
            raise RuntimeError(f"BCH n={n} d={d} has k={self.bch.k} < {self.message_bits}")
        self.n = int(self.bch.n)
        self.k = int(self.bch.k)
        self.d = int(self.bch.d)
        self.t = (self.d - 1) // 2
        self.radius = self.t / self.n

    def permutation(self, key) -> np.ndarray:
        return np.random.default_rng(key_seed(key)).permutation(self.n)

    def encode(self, message: torch.Tensor, key) -> torch.Tensor:
        """message (..., B) in {0,1} -> permuted codeword (..., n) float32."""
        shape = message.shape[:-1]
        bits = message.detach().round().clamp(0, 1).to(torch.int64).reshape(-1, self.message_bits).cpu().numpy()
        padded = np.zeros((bits.shape[0], self.k), dtype=np.int64)
        padded[:, : self.message_bits] = bits
        import galois

        coded = np.asarray(self.bch.encode(galois.GF2(padded)), dtype=np.int64)
        coded = coded[:, self.permutation(key)]
        out = torch.from_numpy(coded.copy()).to(device=message.device, dtype=torch.float32)
        return out.reshape(*shape, self.n)

    def decode(self, received: torch.Tensor, key) -> torch.Tensor:
        """received (..., n) probabilities, bits, or logits -> message (..., B) int64.

        Values inside [0, 1] are treated as probabilities (threshold 0.5). Anything else is
        treated as a logit (threshold 0).
        """
        import galois

        shape = received.shape[:-1]
        flat = received.detach().reshape(-1, self.n)
        if torch.all((flat >= 0) & (flat <= 1)):
            hard = flat >= 0.5
        else:
            hard = flat > 0
        perm = torch.as_tensor(self.permutation(key), device=flat.device)
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(self.n, device=flat.device)
        ordered = hard[:, inverse].cpu().numpy().astype(np.int64)
        rows = []
        for row in ordered:
            try:
                decoded = np.asarray(self.bch.decode(galois.GF2(row)), dtype=np.int64).reshape(-1)
            except Exception:
                decoded = row[: self.k]
            rows.append(decoded[: self.message_bits])
        out = torch.from_numpy(np.stack(rows)).to(device=received.device)
        return out.reshape(*shape, self.message_bits)


def get_codec(message_bits: int = 64, target_ber: float = 0.10, n: int | None = None, d: int | None = None) -> PayloadCodec:
    key = (message_bits, target_ber, n, d)
    if key not in _BUILT:
        _BUILT[key] = PayloadCodec(message_bits, target_ber, n=n, d=d)
    return _BUILT[key]
