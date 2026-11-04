#!/bin/bash
#SBATCH --job-name=qwen-edit-setup
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.err
# Install a new prefix. The edict-prc environment is not modified.
set -euo pipefail
WORKDIR=/nfshomes/eloirghi/prc-in-edict
SCRATCH=/fs/clip-scratch/eloirghi/prc-in-edict
ENV_PREFIX="${SCRATCH}/conda/envs/qwen-image-edit-prc"
export CONDA_PKGS_DIRS="${SCRATCH}/conda/pkgs"
export PIP_CACHE_DIR="${SCRATCH}/pip_cache"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "${SCRATCH}/hf_cache" "$WORKDIR/logs"
module load miniforge3/24.3.0
eval "$(conda shell.bash hook)"
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.11
fi
# shellcheck disable=SC1091
source "$WORKDIR/experiments/slurm/qwen_image_edit_prc/env.sh"
python -m pip install --upgrade pip
# cu121 stops at torch 2.5.1. That build supports bf16 on the A6000 and matches this cluster's existing CUDA 12.1 wheels.
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "$WORKDIR/experiments/qwen_image_edit_prc/requirements.txt"
python - <<'PY'
import hashlib
import inspect
from diffusers import QwenImageEditPipeline
from diffusers.image_processor import VaeImageProcessor
import diffusers
import torch
import transformers
path = inspect.getfile(QwenImageEditPipeline)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("torch", torch.__version__)
print("pipeline_file", path)
print("pipeline_sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())
print("resize", inspect.signature(VaeImageProcessor.resize))
print("preprocess", inspect.signature(VaeImageProcessor.preprocess))
print("postprocess", inspect.signature(VaeImageProcessor.postprocess))
PY
echo INSTALL_DONE
