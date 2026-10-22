#!/bin/bash
#SBATCH --job-name=ip2p-ed-cpu
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.err
# Algebra of the coupled sampler. No model weights.
set -euo pipefail
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/common.sh
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
srun python -u experiments/ip2p_edict_prc/run_cpu_algebra.py
