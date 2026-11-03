#!/bin/bash
#SBATCH --job-name=ip2p-sp
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-11%8
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%A_%a.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%A_%a.err
# Short-payload study: 4 lengths x 3 trials, all five photos and frozen instructions.
set -euo pipefail
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/common.sh
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
srun python -u experiments/ip2p_edict_prc/run_short_payload.py
