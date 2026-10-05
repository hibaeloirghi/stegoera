"""Filesystem layout for PRC-in-EDICT experiments."""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("PRC_EDICT_WORKDIR", "/nfshomes/eloirghi/prc-in-edict")).resolve()
SCRATCH = Path(os.environ.get("PRC_EDICT_SCRATCH", "/fs/clip-scratch/eloirghi/prc-in-edict")).resolve()
EDICT_ROOT = WORKDIR / "EDICT"
PRC_ROOT = WORKDIR / "PRC-Watermark"
EXPERIMENTS = WORKDIR / "experiments"
PHOTOS = EXPERIMENTS / "photos"
RESULTS = Path(os.environ.get("PRC_EDICT_RESULTS", str(WORKDIR / "results")))
LOGS = Path(os.environ.get("PRC_EDICT_LOGS", str(WORKDIR / "logs")))
HF_HOME = Path(os.environ.get("HF_HOME", str(SCRATCH / "hf_cache")))
CONDA_PREFIX = Path(os.environ.get("PRC_EDICT_CONDA", str(SCRATCH / "conda/envs/edict-prc")))


def ensure_dirs():
    for p in (RESULTS, LOGS, HF_HOME, PHOTOS, SCRATCH / "keys"):
        p.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME))
