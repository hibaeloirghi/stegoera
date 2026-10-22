#!/bin/bash
#SBATCH --job-name=dl-sd21
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.err
set -euo pipefail
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/common.sh
python - <<'PY'
import os
from huggingface_hub import snapshot_download
assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing"
print("download stabilityai/stable-diffusion-2-1-base")
snapshot_download("stabilityai/stable-diffusion-2-1-base", cache_dir=os.environ["HF_HOME"])
print("ok")
PY
