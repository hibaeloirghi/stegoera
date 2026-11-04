#!/bin/bash
#SBATCH --job-name=qwen-prc-controls
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.err
# CPU-only PRC checks. No image-model weights are loaded.
set -euo pipefail
# shellcheck disable=SC1091
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/qwen_image_edit_prc/env.sh
python experiments/qwen_image_edit_prc/run_prc_controls.py \
  experiments/configs/qwen_image_edit_prc/pilot.json
