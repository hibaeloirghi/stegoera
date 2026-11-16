#!/bin/bash
#SBATCH --job-name=qwen-prc-style
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=high
#SBATCH --array=0-4
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%A_%a.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%A_%a.err
# Five A6000 tasks, 20 style edits each. Submit after the key job exits 0.
# QoS high allows one day, and 100 edits do not fit in one day on one GPU.
set -euo pipefail
# shellcheck disable=SC1091
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/qwen_image_edit_prc/env.sh
python experiments/qwen_image_edit_prc/run_imgedit_style.py \
  experiments/configs/qwen_image_edit_prc/imgedit_style.json
