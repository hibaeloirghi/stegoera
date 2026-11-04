#!/bin/bash
#SBATCH --job-name=qwen-prc-pilot
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.err
# qos=high is required because the default QoS caps memory at 32G and the
# bf16 checkpoint is 54 GiB. One A6000 holds the transformer; the text encoder
# is offloaded. Do not submit this until the CPU controls have passed.
set -euo pipefail
# shellcheck disable=SC1091
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/qwen_image_edit_prc/env.sh
python experiments/qwen_image_edit_prc/run_pilot.py \
  experiments/configs/qwen_image_edit_prc/pilot.json
