"""CPU key for the ImgEdit-Bench style split.

Every style photograph is 1024×1024, so the generation noise has length
262144. The keys already checked for the iPhone canvases do not match that
length. KeyGen is not reproducible from its seed; a later job loads the
pickle written here.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.qwen_image_edit_prc.geometry import block_length, edit_canvas  # noqa: E402
from experiments.qwen_image_edit_prc.io_util import sha256_file, write_bytes, write_json  # noqa: E402
from experiments.qwen_image_edit_prc.prc_adapter import (  # noqa: E402
    describe_key,
    encode_message,
    generate_key,
    key_basename,
    load_decoding_key_only,
    load_sender_key,
    save_decoding_key_only,
    save_sender_key,
    score_latent,
    sign_magnitudes,
)


def bits_to_text(bits) -> str:
    if len(bits) % 8:
        raise RuntimeError("message bit length is not a whole number of bytes")
    raw = bytearray()
    for start in range(0, len(bits), 8):
        value = 0
        for bit in bits[start : start + 8]:
            value = (value << 1) | int(bit)
        raw.append(value)
    return raw.decode("utf-8")


def write_or_match(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"{path} already exists with different contents")
        return
    write_json(path, payload)


def load_message(cfg: dict) -> list[int]:
    pilot = json.loads((ROOT / cfg["message_config"]).read_text())
    if pilot["message_text"] != cfg["message_text"]:
        raise RuntimeError("message text does not match the 120-bit pilot config")
    message = [int(bit) for bit in pilot["message_bits"]]
    if len(message) != int(cfg["payload_length"]) or bits_to_text(message) != cfg["message_text"]:
        raise RuntimeError("pilot message_bits do not decode to the configured text")
    if int(cfg["t"]) != 3 or float(cfg["false_positive_rate"]) != 1e-5:
        raise RuntimeError("this chapter's PRC adapter is fixed at t=3 and F=1e-5")
    return message


def build_catalog(cfg: dict) -> dict:
    tar_path = Path(cfg["benchmark_tar"])
    if sha256_file(tar_path) != cfg["benchmark_sha256"]:
        raise RuntimeError("ImgEdit Benchmark.tar hash does not match the config")
    with tarfile.open(tar_path) as archive:
        payload = archive.extractfile("Benchmark/singleturn/singleturn.json").read()
        raw = json.loads(payload)
        style = []
        for key, item in raw.items():
            if item.get("edit_type") != cfg["edit_type"]:
                continue
            image_id = item["id"]
            if not image_id.startswith("style/") or ".." in image_id:
                raise RuntimeError(f"unexpected style image id {image_id}")
            style.append({"key": str(key), "id": image_id, "prompt": item["prompt"]})
        style.sort(key=lambda case: int(case["key"]))
        if len(style) != int(cfg["expected_cases"]):
            raise RuntimeError(f"expected {cfg['expected_cases']} style cases, found {len(style)}")
        if any(not case["key"].isdigit() for case in style):
            raise RuntimeError("style case keys are not decimal ids")
        counts: dict[str, int] = {}
        for case in style:
            counts[case["id"]] = counts.get(case["id"], 0) + 1
            if not case["prompt"].startswith("Transfer the image into "):
                raise RuntimeError(f"case {case['key']} is not an ImgEdit style instruction")
        if len(counts) != int(cfg["expected_images"]) or any(count != 20 for count in counts.values()):
            raise RuntimeError(f"expected 5 images with 20 prompts each, found {counts}")
        image_root = Path(cfg["image_root"])
        images = {}
        expected_w, expected_h = (int(v) for v in cfg["expected_canvas"])
        for image_id in sorted(counts):
            member = archive.extractfile(f"Benchmark/singleturn/{image_id}")
            blob = member.read()
            digest = hashlib.sha256(blob).hexdigest()
            out = image_root / image_id
            if out.exists():
                if sha256_file(out) != digest:
                    raise RuntimeError(f"{out} does not match Benchmark.tar")
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                write_bytes(out, blob)
            with Image.open(out) as image:
                width, height = image.size
            if (width, height) != (expected_w, expected_h):
                raise RuntimeError(f"{image_id} is {width}x{height}, not {expected_w}x{expected_h}")
            canvas_w, canvas_h = edit_canvas(width, height)
            length = block_length(canvas_w, canvas_h)
            if (canvas_w, canvas_h) != (expected_w, expected_h) or length != int(cfg["expected_n"]):
                raise RuntimeError(f"{image_id} canvas gives n={length}")
            images[image_id] = {"sha256": digest, "width": width, "height": height}
    return {
        "edit_type": cfg["edit_type"],
        "n": int(cfg["expected_n"]),
        "canvas": [expected_w, expected_h],
        "prompt_source": (
            "ImgEdit-Bench singleturn.json edit_type=style. "
            "Each case keeps that file's instruction. "
            "The Qwen-Image report scores this split and does not publish a replacement prompt. "
            "Figure 15's 'Turn into realistic style' is a system-prompt example."
        ),
        "images": images,
        "cases": [
            {
                "key": case["key"],
                "id": case["id"],
                "prompt": case["prompt"],
                "sha256": images[case["id"]]["sha256"],
                "gaussian_seed": int(cfg["gaussian_seed_base"]) + index,
            }
            for index, case in enumerate(style)
        ],
    }


def ensure_keys(cfg: dict, message: list[int], n: int) -> None:
    keys_dir = ROOT / cfg["keys_dir"]
    keys_dir.mkdir(parents=True, exist_ok=True)
    stem = key_basename(cfg["payload_length"])
    sender_path = keys_dir / f"{stem}_sender.pkl"
    receiver_path = keys_dir / f"{stem}_receiver.pkl"
    wrong_path = keys_dir / f"{stem}_wrong.pkl"
    controls_path = ROOT / cfg["controls_json"]
    if controls_path.exists():
        report = json.loads(controls_path.read_text())
        if not report.get("passed") or int(report["key"]["n"]) != n:
            raise RuntimeError(f"existing controls did not pass for n={n}")
        print(json.dumps({"passed": True, "n": n, "reused": True}), flush=True)
        return
    if not sender_path.exists():
        if receiver_path.exists() or wrong_path.exists():
            raise RuntimeError(f"partial key directory {keys_dir}")
        encoding_key, decoding_key = generate_key(n, len(message), int(cfg["keygen_seed"]))
        wrong_enc, wrong_dec = generate_key(n, len(message), int(cfg["wrong_key_seed"]))
        codeword = encode_message(encoding_key, message, int(cfg["encode_seed"]))
        save_sender_key(sender_path, encoding_key, decoding_key, message, codeword)
        save_decoding_key_only(receiver_path, decoding_key)
        save_sender_key(wrong_path, wrong_enc, wrong_dec, message, torch.zeros(1))
    if not sender_path.exists() or not receiver_path.exists() or not wrong_path.exists():
        raise RuntimeError(f"partial key directory {keys_dir}")

    encoding_key, decoding_key, saved_message, saved_code = load_sender_key(sender_path)
    saved_message = [int(bit) for bit in saved_message]
    if saved_message != message:
        raise RuntimeError("saved key carries a different message")
    summary = describe_key(encoding_key)
    if summary["n"] != n or summary["t"] != int(cfg["t"]) or summary["message_capacity"] != len(message):
        raise RuntimeError(f"saved key does not match n={n}: {summary}")
    try:
        encode_message(encoding_key, message + [0], int(cfg["encode_seed"]))
    except AssertionError:
        rejected_extra_bit = True
    else:
        rejected_extra_bit = False
    # Encode draws fresh redundancy bits, so a second call is a different codeword.
    # The saved codeword is the one the GPU job will embed.
    codeword = torch.as_tensor(saved_code).detach().cpu().reshape(-1).to(dtype=torch.float64)
    if int(codeword.numel()) != n or not set(torch.unique(codeword).tolist()).issubset({-1.0, 1.0}):
        raise RuntimeError("saved codeword is not a ±1 vector of length n")
    code_score = score_latent(decoding_key, codeword, message)
    gaussian = np.random.default_rng(int(cfg["gaussian_seed"])).standard_normal(n)
    prc_noise = sign_magnitudes(codeword, gaussian)
    noise_score = score_latent(decoding_key, prc_noise, message)
    gaussian_score = score_latent(decoding_key, torch.from_numpy(gaussian.astype(np.float64)), message)
    _wrong_enc, wrong_dec, _wrong_message, _wrong_code = load_sender_key(wrong_path)
    wrong_score = score_latent(wrong_dec, prc_noise, message)
    receiver_dec = load_decoding_key_only(receiver_path)
    receiver_score = score_latent(receiver_dec, prc_noise, message)
    passed = bool(
        rejected_extra_bit
        and code_score["exact_message"]
        and noise_score["exact_message"]
        and receiver_score["exact_message"]
        and not wrong_score["detect"]
        and not wrong_score["decode_non_none"]
        and not gaussian_score["detect"]
        and not gaussian_score["decode_non_none"]
    )
    report = {
        "passed": passed,
        "canvas_width": int(cfg["expected_canvas"][0]),
        "canvas_height": int(cfg["expected_canvas"][1]),
        "key": summary,
        "rejected_extra_bit": rejected_extra_bit,
        "codeword": code_score,
        "prc_noise": noise_score,
        "gaussian": gaussian_score,
        "wrong_key": wrong_score,
        "receiver_key_exact": receiver_score["exact_message"],
        "sender_key": str(sender_path.relative_to(ROOT)),
        "receiver_key": str(receiver_path.relative_to(ROOT)),
        "note": "One Gaussian negative control is not an empirical false-positive rate.",
    }
    write_json(controls_path, report)
    print(json.dumps({"passed": passed, "n": n, "key_id": summary["key_id"]}), flush=True)
    if not passed:
        raise SystemExit(1)


def main() -> None:
    cfg = json.loads(Path(sys.argv[1]).read_text())
    message = load_message(cfg)
    catalog = build_catalog(cfg)
    write_or_match(ROOT / cfg["cases_json"], catalog)
    ensure_keys(cfg, message, int(catalog["n"]))


if __name__ == "__main__":
    main()
