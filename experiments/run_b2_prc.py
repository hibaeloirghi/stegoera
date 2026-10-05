#!/usr/bin/env python3
"""B2: PRC's own encode.py then decode.py with repo defaults (SD 2.1-base)."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from experiments.paths import PRC_ROOT, RESULTS, HF_HOME, ensure_dirs
from experiments.io_utils import load_config, save_json, append_csv


def run(cmd, cwd, env):
    print("RUN", " ".join(cmd), "cwd=", cwd)
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=False)
    return proc.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure_dirs()
    outdir = RESULTS / "b2"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, outdir / "config.json")

    env = os.environ.copy()
    env["HF_HOME"] = str(HF_HOME)
    env["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME)
    env["TRANSFORMERS_CACHE"] = str(HF_HOME)
    test_num = int(cfg.get("test_num", 3))
    (PRC_ROOT / "keys").mkdir(exist_ok=True)

    py = sys.executable
    rc_enc = run(
        [py, "encode.py", "--test_num", str(test_num), "--inf_steps", str(cfg.get("inf_steps", 50))],
        cwd=str(PRC_ROOT),
        env=env,
    )
    rc_dec = 1
    if rc_enc == 0:
        rc_dec = run(
            [py, "decode.py", "--test_num", str(test_num), "--inf_steps", str(cfg.get("inf_steps", 50))],
            cwd=str(PRC_ROOT),
            env=env,
        )

    decoded_txt = PRC_ROOT / "decoded.txt"
    results_flags = []
    if decoded_txt.is_file():
        results_flags = [line.strip() for line in decoded_txt.read_text().splitlines() if line.strip()]
        (outdir / "decoded.txt").write_text(decoded_txt.read_text())

    n_true = sum(1 for x in results_flags if x in ("True", "true", "1"))
    row = {
        "status": "ok" if rc_enc == 0 and rc_dec == 0 else "fail",
        "encode_rc": rc_enc,
        "decode_rc": rc_dec,
        "test_num": test_num,
        "n_decoded_lines": len(results_flags),
        "n_true": n_true,
        "flags": "|".join(results_flags),
        "decoded_txt": str(decoded_txt),
        "model_id": cfg.get("model_id", "stabilityai/stable-diffusion-2-1-base"),
    }
    append_csv(outdir / "b2.csv", row)
    save_json(row, outdir / "summary.json")
    print(row)
    if row["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
