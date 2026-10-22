#!/bin/bash
# Shared paths. Source from SLURM scripts after SBATCH headers.
WORKDIR="${PRC_EDICT_WORKDIR:-/nfshomes/eloirghi/prc-in-edict}"
SCRATCH="${PRC_EDICT_SCRATCH:-/fs/clip-scratch/eloirghi/prc-in-edict}"
export PRC_EDICT_WORKDIR="$WORKDIR"
export PRC_EDICT_SCRATCH="$SCRATCH"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export PRC_EDICT_RESULTS="${PRC_EDICT_RESULTS:-${WORKDIR}/results}"
export PRC_EDICT_CONDA="${PRC_EDICT_CONDA:-${SCRATCH}/conda/envs/edict-prc}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$WORKDIR/logs" "$PRC_EDICT_RESULTS" "$HF_HOME"
# Prefix env is a conda -p install; `bin/activate` is not always present. Put it on PATH instead.
if [[ -x "${PRC_EDICT_CONDA}/bin/python" ]]; then
  export PATH="${PRC_EDICT_CONDA}/bin:${PATH}"
else
  echo "ERROR: missing ${PRC_EDICT_CONDA}/bin/python" >&2
  exit 1
fi
hash -r
echo "python=$(command -v python) $(python -c 'import sys; print(sys.version.split()[0])')"
# Hugging Face token lives in a gitignored file; never echo it.
if [[ -f "${WORKDIR}/.hf_token" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '[:space:]' < "${WORKDIR}/.hf_token")"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
cd "$WORKDIR"
