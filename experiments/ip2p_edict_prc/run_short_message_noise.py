"""Copy the existing 8-bit and 16-bit keys and sample this study's noises.

No image model. Messages are not printed. Seeds do not replace the saved arrays:
Encode draws codeword bits with galois, so the npz files are the record.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.paths import RESULTS, WORKDIR  # noqa: E402
from experiments.prc_runtime import detect_decode, encode_codeword, recover_soft_bits  # noqa: E402
from experiments.receiver_aligned.key_io import key_hash, load_key  # noqa: E402

CONFIG_PATH = ROOT / "experiments" / "configs" / "ip2p_edict_prc" / "short_message_recovery.json"


def digest(array) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def claim(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite {path}") from exc
    os.close(fd)


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    out = RESULTS / "ip2p_edict_prc" / cfg["output_dir"]
    key_dir = out / "keys"
    trial_dir = out / "trials"
    control_dir = out / "controls"
    for folder in (key_dir, trial_dir, control_dir):
        folder.mkdir(parents=True, exist_ok=True)
    frozen = out / "frozen_config.json"
    claim(frozen)
    shutil.copyfile(CONFIG_PATH, frozen)
    n = int(cfg["n"])
    lengths = [int(item) for item in cfg["message_lengths"]]
    keys = {}
    for length in lengths:
        source = WORKDIR / cfg["existing_key_dir"] / f"length_{length}.pkl"
        dest = key_dir / f"length_{length}.pkl"
        claim(dest)
        shutil.copyfile(source, dest)
        enc, dec, _probe, _probe_code = load_key(dest)
        ident = key_hash(enc)
        if ident != cfg["expected_key_id"][str(length)]:
            raise RuntimeError(f"length {length} key id {ident} does not match the frozen id")
        generator, _otp, test_bits, g, noise_rate = enc
        _rows, k = generator.shape
        if int(k) != length + int(g) + len(test_bits):
            raise RuntimeError(f"length {length} capacity k={k} is not payload plus redundancy")
        try:
            encode_codeword(enc, message=[0] * (length + 1), seed=1)
        except AssertionError:
            rejected = True
        else:
            rejected = False
        if not rejected:
            raise RuntimeError(f"length {length} accepted a message one bit over capacity")
        keys[length] = (enc, dec, float(noise_rate), int(g), len(test_bits))
        print("key", length, ident, "k", int(k), "g", int(g), "test_bits", len(test_bits), flush=True)

    magnitudes = []
    for trial in cfg["trials"]:
        magnitudes.append(
            np.random.default_rng(int(cfg["magnitude_seed"]) + int(trial)).standard_normal(n).astype(np.float64)
        )
    for length in lengths:
        enc, dec, _noise_rate, _g, _tests = keys[length]
        for trial, magnitude in zip(cfg["trials"], magnitudes):
            payload = [
                int(bit)
                for bit in np.random.default_rng(int(cfg["message_seed"]) + length * 10 + int(trial)).integers(0, 2, size=length)
            ]
            codeword, _ignored = encode_codeword(
                enc, message=payload, seed=int(cfg["encode_seed"]) + length * 10 + int(trial)
            )
            code = np.asarray(codeword, dtype=np.float64).reshape(-1)
            if code.shape != (n,) or not np.all(np.abs(code) == 1):
                raise RuntimeError(f"length {length} trial {trial} codeword is not a ±1 vector of length {n}")
            signed = code * np.abs(magnitude)
            soft = recover_soft_bits(torch.from_numpy(signed), variance=1.5)
            detected, _ok, decoded = detect_decode(dec, soft)
            exact = decoded is not None and list(np.asarray(decoded, dtype=int).reshape(-1)) == payload
            if not (detected and exact):
                raise RuntimeError(f"direct noise decode failed for length {length} trial {trial}")
            z_path = trial_dir / f"length_{length}_trial_{trial}_z.npy"
            secret_path = trial_dir / f"length_{length}_trial_{trial}_secret.npz"
            claim(z_path)
            np.save(z_path, signed.reshape(1, 4, 64, 64))
            claim(secret_path)
            np.savez(secret_path, message=np.asarray(payload, dtype=np.uint8), codeword=code)
            print(
                "trial", length, trial,
                "message_sha256", digest(payload),
                "codeword_sha256", digest(code),
                "direct_exact", exact,
                flush=True,
            )
    for trial, magnitude in zip(cfg["trials"], magnitudes):
        z8 = np.load(trial_dir / f"length_8_trial_{trial}_z.npy")
        z16 = np.load(trial_dir / f"length_16_trial_{trial}_z.npy")
        if not np.array_equal(np.abs(z8).reshape(-1), np.abs(magnitude)):
            raise RuntimeError(f"trial {trial} 8-bit magnitudes do not match the shared Gaussian")
        if not np.array_equal(np.abs(z16).reshape(-1), np.abs(magnitude)):
            raise RuntimeError(f"trial {trial} 16-bit magnitudes do not match the shared Gaussian")
        if np.array_equal(z8, z16):
            raise RuntimeError(f"trial {trial} codeword signs did not differ across lengths")

    images = [(image_id, instruction_id) for image_id, _path in cfg["images"] for instruction_id in cfg["instructions"]]
    for index, (image_id, instruction_id) in enumerate(images):
        draw = np.random.default_rng(int(cfg["control_seed"]) + index).standard_normal(n).astype(np.float64)
        path = control_dir / f"{image_id}__{instruction_id}__lencontrol__trial0_z.npy"
        claim(path)
        np.save(path, draw.reshape(1, 4, 64, 64))
        print("control", image_id, instruction_id, "sha256", digest(draw), flush=True)
    (out / "STATUS_NOISE.txt").write_text("PASS\n")
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
