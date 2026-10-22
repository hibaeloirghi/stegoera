#!/bin/bash
#SBATCH --job-name=ip2p-ed-pair
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.err
# One Gaussian pair and one PRC pair, then the matched inverse of each final pair.
set -euo pipefail
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/common.sh
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
srun python -u experiments/ip2p_edict_prc/run_milestone.py
