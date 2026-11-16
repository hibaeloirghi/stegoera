#!/bin/bash
#SBATCH --job-name=qwen-prc-iphone10
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.err
# One A6000, model loaded once. Ten ordinary edits, ten PRC edits, PNG recovery,
# and a panel per photo. Submit after the photo-9 key controls exit 0.
set -euo pipefail
# shellcheck disable=SC1091
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/qwen_image_edit_prc/env.sh
python experiments/qwen_image_edit_prc/run_iphone_batch.py \
  experiments/configs/qwen_image_edit_prc/iphone10.json
