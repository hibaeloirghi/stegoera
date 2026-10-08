#!/bin/bash
#SBATCH --job-name=ra-r1word
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
srun python experiments/receiver_aligned/analyze_r1_word.py results/receiver_aligned/saved_states/r1_smoke_7788181_key.pkl results/receiver_aligned/raw_observations/r1_smoke_7788181_true_site_carrier.pt
