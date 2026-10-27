#!/bin/bash
#SBATCH --job-name=ip2p-gdiag-old
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.err
# Float64 SD 1.4, the same GPU type as experiments/slurm/full_prc_sign.sh.
# One historical dog1 row. The config path is the first argument.
set -euo pipefail
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/common.sh
srun python -u experiments/ip2p_edict_prc/guidance_diag/run_old_trace.py "$1"
