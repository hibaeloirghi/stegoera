#!/bin/bash
#SBATCH --job-name=ne-e0-e1
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.err
set -euo pipefail
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/common.sh

echo "=========================================================="
echo "Starting E0: Saved-Key / Payload / Decoder Audit"
echo "=========================================================="
srun python experiments/next_experiments/run_e0_audit.py

echo "=========================================================="
echo "Starting E1: Direct-Carrier Projection Curve"
echo "=========================================================="
srun python experiments/next_experiments/run_e1_projection.py

echo "E0 and E1 completed successfully."
