#!/bin/bash
#SBATCH --job-name=lms-smoke
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.err
set -euo pipefail
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/common.sh
cd /nfshomes/eloirghi/prc-in-edict/learned_mobile_stego
srun python train.py --config configs/smoke_test.yaml
srun python evaluate.py --config configs/smoke_test.yaml --checkpoint results/smoke/checkpoint.pt
