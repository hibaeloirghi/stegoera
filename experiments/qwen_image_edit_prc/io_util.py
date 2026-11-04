"""Small file helpers for the Qwen chapter.

A path that already exists is an error. Partial writes use a sibling temporary
file and ``os.replace`` only after an exclusive create of the destination.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def create_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def _exclusive(path: Path):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)


def write_bytes(path: Path, payload: bytes) -> None:
    fd = _exclusive(path)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)


def write_text(path: Path, text: str) -> None:
    write_bytes(path, text.encode("utf-8"))


def write_json(path: Path, payload: dict) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_numpy(path: Path, array: np.ndarray) -> None:
    fd = _exclusive(path)
    os.close(fd)
    # ``np.save`` cannot take the fd we just created. The exclusive create
    # above is the lock; ``np.save`` replaces the empty file in place.
    np.save(path, np.asarray(array))


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    fd = _exclusive(path)
    os.close(fd)
    torch.save(tensor.detach().to("cpu"), path)


def read_png(path: Path) -> tuple[bytes, str]:
    """Return the file bytes and their SHA-256. Callers decode those bytes."""
    payload = Path(path).read_bytes()
    return payload, sha256_bytes(payload)
