#!/bin/bash
# Recorded commands to build the shared edict-prc environment on a compute node.
# Do not run the heavy pip install on the login node; submit slurm/setup_env.sh instead.
set -euo pipefail

WORKDIR="${PRC_EDICT_WORKDIR:-/nfshomes/eloirghi/prc-in-edict}"
SCRATCH="${PRC_EDICT_SCRATCH:-/fs/clip-scratch/eloirghi/prc-in-edict}"
ENV_PREFIX="${PRC_EDICT_CONDA:-${SCRATCH}/conda/envs/edict-prc}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export CONDA_PKGS_DIRS="${SCRATCH}/conda/pkgs"
export PIP_CACHE_DIR="${SCRATCH}/pip_cache"
mkdir -p "$HF_HOME" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$ENV_PREFIX" "${SCRATCH}/results"

module load miniforge3/24.3.0
eval "$(conda shell.bash hook)"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.11
fi
conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip
# PRC pins torch 2.1.0; EDICT's vendored my_diffusers is imported as a local package (no pip diffusers==0.6).
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  transformers==4.29.2 \
  diffusers==0.21.4 \
  numpy==1.26.0 \
  huggingface_hub==0.25.0 \
  datasets \
  compressai \
  scikit-image \
  opencv-python \
  pycryptodome \
  accelerate \
  galois==0.4.1 \
  ldpc==0.1.51 \
  scipy \
  matplotlib \
  pandas \
  omegaconf \
  ftfy \
  regex \
  tqdm \
  Pillow \
  lpips

python - <<'PY'
from huggingface_hub import snapshot_download
import os
cache = os.environ["HF_HOME"]
repos = [
    "openai/clip-vit-large-patch14",
    "CompVis/stable-diffusion-v1-4",
    "stabilityai/stable-diffusion-2-1-base",
]
for repo in repos:
    print("download", repo)
    try:
        snapshot_download(repo, cache_dir=cache)
        print("ok", repo)
    except Exception as e:
        print("FAIL", repo, type(e).__name__, e)
PY

python -c "import torch,diffusers,transformers,galois,ldpc; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('ok')"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "HF_HOME=$HF_HOME"
