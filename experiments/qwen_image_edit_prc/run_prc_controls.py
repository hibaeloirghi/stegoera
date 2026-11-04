"""CPU checks for one 8-bit PRC key sized to the Qwen generation noise.

The block length comes from the photograph's edit canvas. The key objects are
saved. A seed is not a substitute for those files. This script does not load
Qwen weights.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.qwen_image_edit_prc.geometry import assert_dog1_canvas, block_length, edit_canvas
from experiments.qwen_image_edit_prc.io_util import sha256_file, write_json
from experiments.qwen_image_edit_prc.prc_adapter import (
    describe_key,
    encode_message,
    generate_key,
    key_hash,
    load_sender_key,
    save_decoding_key_only,
    save_sender_key,
    score_latent,
    sign_magnitudes,
)


def main() -> None:
    config_path = Path(sys.argv[1])
    cfg = json.loads(config_path.read_text())
    assert_dog1_canvas()
    photo = ROOT / cfg["source_photo"]
    photo_hash = sha256_file(photo)
    if photo_hash != cfg["source_sha256"]:
        raise RuntimeError("source photograph hash does not match the config")
    with Image.open(photo) as image:
        width, height = image.size
    canvas_w, canvas_h = edit_canvas(width, height)
    n = block_length(canvas_w, canvas_h)
    message = [int(bit) for bit in cfg["message_bits"]]
    if len(message) != int(cfg["payload_length"]):
        raise RuntimeError("payload length does not match message_bits")
    keys_dir = ROOT / cfg["keys_dir"]
    keys_dir.mkdir(parents=True, exist_ok=True)
    sender_path = keys_dir / "len8_sender.pkl"
    receiver_path = keys_dir / "len8_receiver.pkl"
    wrong_path = keys_dir / "len8_wrong.pkl"
    for path in (sender_path, receiver_path, wrong_path):
        if path.exists():
            raise FileExistsError(path)

    encoding_key, decoding_key = generate_key(n, len(message), int(cfg["keygen_seed"]))
    summary = describe_key(encoding_key)
    if summary["message_capacity"] != len(message):
        raise RuntimeError(
            f"key capacity {summary['message_capacity']} is not the 8-bit payload"
        )
    if summary["n"] != n or summary["t"] != int(cfg["t"]):
        raise RuntimeError("key parameters do not match the config")
    try:
        encode_message(encoding_key, message + [0], int(cfg["encode_seed"]))
    except AssertionError:
        rejected_extra_bit = True
    else:
        rejected_extra_bit = False
    codeword = encode_message(encoding_key, message, int(cfg["encode_seed"]))
    code_score = score_latent(decoding_key, codeword, message)
    gaussian = np.random.default_rng(int(cfg["gaussian_seed"])).standard_normal(n)
    prc_noise = sign_magnitudes(codeword, gaussian)
    noise_score = score_latent(decoding_key, prc_noise, message)
    ordinary = torch.from_numpy(gaussian.astype(np.float64))
    gaussian_score = score_latent(decoding_key, ordinary, message)
    wrong_enc, wrong_dec = generate_key(n, len(message), int(cfg["wrong_key_seed"]))
    wrong_score = score_latent(wrong_dec, prc_noise, message)
    ident = save_sender_key(sender_path, encoding_key, decoding_key, message, codeword)
    save_decoding_key_only(receiver_path, decoding_key)
    save_sender_key(wrong_path, wrong_enc, wrong_dec, message, torch.zeros(1))
    reloaded_enc, reloaded_dec, reloaded_message, reloaded_code = load_sender_key(sender_path)
    reload_same = key_hash(reloaded_enc) == ident and list(reloaded_message) == message
    reloaded_score = score_latent(
        reloaded_dec, sign_magnitudes(torch.as_tensor(reloaded_code), gaussian), message
    )
    passed = bool(
        rejected_extra_bit
        and code_score["exact_message"]
        and noise_score["exact_message"]
        and reload_same
        and reloaded_score["exact_message"]
        and not wrong_score["detect"]
        and not wrong_score["decode_non_none"]
        and not gaussian_score["detect"]
        and not gaussian_score["decode_non_none"]
    )
    report = {
        "passed": passed,
        "photo_sha256": photo_hash,
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "key": summary,
        "rejected_extra_bit": rejected_extra_bit,
        "codeword": code_score,
        "prc_noise": noise_score,
        "gaussian": gaussian_score,
        "wrong_key": wrong_score,
        "reload_exact": reloaded_score["exact_message"],
        "sender_key": str(sender_path.relative_to(ROOT)),
        "receiver_key": str(receiver_path.relative_to(ROOT)),
        "note": "One Gaussian negative control is not an empirical false-positive rate.",
    }
    write_json(ROOT / cfg["controls_json"], report)
    print(json.dumps({"passed": passed, "n": n, "key_id": summary["key_id"]}), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
