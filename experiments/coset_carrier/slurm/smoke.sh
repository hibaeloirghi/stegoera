#!/bin/bash
#SBATCH --job-name=cc-smoke
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.err
# Usage: sbatch smoke.sh [config.yaml] [image]
set -euo pipefail
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/common.sh

CONFIG="${1:-experiments/coset_carrier/configs/embedding.yaml}"
IMAGE="${2:-experiments/photos/imagenet_dog_1.jpg}"
srun python experiments/coset_carrier/run_experiment.py --config "$CONFIG" --image_path "$IMAGE"
