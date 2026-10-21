#!/bin/bash
#SBATCH --job-name=ip2p-cpu-ctrl
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.err
# CPU only: PRC key controls and checkpoint config. No GPU, no image generation.
set -euo pipefail
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/common.sh
# Public checkpoint. Do not send the local Hugging Face token.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
srun python -u experiments/instructpix2pix_prc/run_cpu_controls.py
