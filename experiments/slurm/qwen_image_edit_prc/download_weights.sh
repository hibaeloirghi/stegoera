#!/bin/bash
#SBATCH --job-name=qwen-edit-download
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.out
#SBATCH --error=/nfshomes/eloirghi/prc-in-edict/logs/%x_%j.err
# Download Qwen/Qwen-Image-Edit at the pinned revision. No other Qwen checkpoint.
set -euo pipefail
# shellcheck disable=SC1091
source /nfshomes/eloirghi/prc-in-edict/experiments/slurm/qwen_image_edit_prc/env.sh
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

revision = "ac7f9318f633fc4b5778c59367c8128225f1e3de"
repo = "Qwen/Qwen-Image-Edit"
cache = os.environ["HF_HOME"]
print("download", repo, revision, "cache", cache, flush=True)
path = Path(snapshot_download(repo, revision=revision, cache_dir=cache))
print("snapshot", path, flush=True)
if "2509" in path.as_posix() or "2511" in path.as_posix():
    raise SystemExit("refusing a later Qwen-Image-Edit checkpoint")
index = json.loads((path / "model_index.json").read_text())
if index.get("_class_name") != "QwenImageEditPipeline":
    raise SystemExit(f"unexpected pipeline class {index.get('_class_name')}")
hashes = {}
for file in sorted(path.rglob("*")):
    if not file.is_file():
        continue
    if file.stat().st_size < 5_000_000 and "safetensors" not in file.name:
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
    else:
        hasher = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    hashes[str(file.relative_to(path))] = {"sha256": digest, "bytes": file.stat().st_size}
out = Path("results/qwen_image_edit_prc")
out.mkdir(parents=True, exist_ok=True)
manifest = out / "weight_hashes.json"
if manifest.exists():
    raise SystemExit(f"refusing to overwrite {manifest}")
manifest.write_text(json.dumps({
    "repo": repo,
    "revision": revision,
    "snapshot": str(path),
    "pipeline_class": index["_class_name"],
    "files": hashes,
}, indent=2) + "\n")
print("files", len(hashes), "manifest", manifest, flush=True)
PY
echo DOWNLOAD_DONE
