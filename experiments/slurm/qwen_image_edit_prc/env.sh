#!/bin/bash
# Runtime paths for the Qwen chapter. Do not source experiments/slurm/common.sh:
# that script selects the historical edict-prc environment.
WORKDIR="${PRC_EDICT_WORKDIR:-/nfshomes/eloirghi/prc-in-edict}"
SCRATCH="${PRC_EDICT_SCRATCH:-/fs/clip-scratch/eloirghi/prc-in-edict}"
export QWEN_ENV="${QWEN_ENV:-${SCRATCH}/conda/envs/qwen-image-edit-prc}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export PIP_CACHE_DIR="${SCRATCH}/pip_cache"
export XDG_CACHE_HOME="${SCRATCH}/xdg_cache"
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ ! -x "${QWEN_ENV}/bin/python" ]]; then
  echo "ERROR: missing ${QWEN_ENV}/bin/python" >&2
  exit 1
fi
export PATH="${QWEN_ENV}/bin:${PATH}"
hash -r
if [[ -f "${WORKDIR}/.hf_token" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '[:space:]' < "${WORKDIR}/.hf_token")"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
cd "$WORKDIR"
echo "python=$(command -v python) $(python -c 'import sys; print(sys.version.split()[0])')"
