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
export PYTHONUNBUFFERED=1
mkdir -p "$WORKDIR/logs" "$PRC_EDICT_RESULTS" "$HF_HOME"
if [[ -f "${PRC_EDICT_CONDA}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PRC_EDICT_CONDA}/bin/activate"
fi
cd "$WORKDIR"
