#!/bin/bash
#SBATCH --job-name=lms-train
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.err
set -euo pipefail
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/common.sh
cd /nfshomes/eloirghi/prc-in-edict/learned_mobile_stego
srun python train.py --config configs/train.yaml
srun python evaluate.py --config configs/train.yaml --checkpoint results/train/checkpoint.pt
