#!/bin/bash
#SBATCH --job-name=edit-stego
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=1-11%1
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%A_%a.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%A_%a.err
set -euo pipefail
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/common.sh
srun python experiments/run_edit_stego.py \
  --config experiments/configs/edit_stego_screen.json \
  --task-id "${SLURM_ARRAY_TASK_ID}"
