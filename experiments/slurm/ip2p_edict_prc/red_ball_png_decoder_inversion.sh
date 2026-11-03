#!/bin/bash
#SBATCH --job-name=ip2p-ed-ballpng
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.err
# Same PNG receiver on the saved red-ball image. Does not rewrite earlier results.
set -euo pipefail
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/common.sh
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
srun python -u experiments/ip2p_edict_prc/run_png_decoder_inversion.py experiments/configs/ip2p_edict_prc/red_ball_png_decoder_inversion.json
