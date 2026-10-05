# Cluster discovery (UMIACS Nexus) — 2026-10-05

Login node: `nexusmc201.umiacs.umd.edu`. User: `eloirghi`. No `SLURM_JOB_ID` on login.

## Accounts / QOS / default partition

`sacctmgr show user eloirghi`: **Def Acct = nexus**.

Associations (partition field empty; partitions are selected by account):

| account   | QOS                                      | Default QOS |
|-----------|------------------------------------------|-------------|
| nexus     | default, high, medium                    | default     |
| scavenger | scavenger, scavenger-aarch64             | scavenger   |
| mc2       | default, high, highmem, huge-long, medium| default     |

QOS walltimes of interest: `default` MaxWall=3 days; `medium` 2 days; `high` 1 day; `scavenger` 3 days.

**Default partition** is `tron` (`Default=YES`). `AllowAccounts=nexus`, `AllowQos=default,medium,high`. Partition QOS `tron` caps `cpu=32,gres/gpu=4,mem=256G`. DefaultTime=1h, MaxTime unlimited.

`mc2` partition is CPU-only (legacy nodes). `scavenger` is preemptible and requires `--account=scavenger`.

**Choice used in scripts (please confirm):** `--partition=tron --account=nexus --qos=default`. GPU type `--gres=gpu:rtxa5000:1` (24 GB) because a 1-minute `srun --gres=gpu:1` landed on an **RTX 3070 8 GB**, which is too small for SD 1.4 in float64.

## GPU / CPU partitions (abbrev.)

- GPU (nexus-usable): `tron` (rtxa4000/5000/6000, rtx2080ti, rtx3070).
- GPU (other labs, not our accounts): clip, cml-*, vulcan-*, gamma, class, cbcb, …
- Preemptible GPU: `scavenger` (many GPU types, including A100/H100/H200).
- CPU: tron nodes without requesting GRES; `mc2` / `oasis` / `vulcan-cpu` are other labs.

`squeue -u $USER` was empty at discovery time.

## Filesystems / quota

- Home `/nfshomes/eloirghi`: **30 GB, 17 GB used, 14 GB free** (Isilon). Too small for HF weights.
- `/fs/nexus-scratch`: 150 T, **not writable** (`mkdir /fs/nexus-scratch/eloirghi` Permission denied on login *and* on tron64). `environments.txt` still mentions a deleted path `/fs/nexus-scratch/eloirghi/caches/conda/envs/prcsem`.
- `/fs/nexus-projects`: not writable.
- `/fs/clip-scratch`: world-writable sticky; **created `/fs/clip-scratch/eloirghi`** (20 T, 15 T free). Used as scratch because nexus-scratch has no user directory.
- Login `/scratch0` is node-local (not for jobs).

**HF_HOME / conda / results large files:** `/fs/clip-scratch/eloirghi/prc-in-edict/`. Confirm this is acceptable (CLIP lab scratch, not nexus-scratch).

## Modules / Python

- `miniforge3/24.3.0` (no `conda` on PATH until `module load`).
- CUDA modules 11.8–13.3 available; driver on a tron GPU node was **595.71 / CUDA 13.2**.
- Login Python 3.9.25. No project conda env currently present.

## Compute-node probes (1-minute `srun`)

1. CPU job `cluster-disc` on **tron64**: compute nodes **can reach** `https://huggingface.co` and `https://github.com` (HTTP/2 200). Failed to create nexus-scratch dir. No GPU (GRES not requested).
2. GPU job `gpu-disc` on **tron64**: `NVIDIA GeForce RTX 3070`, 8192 MiB, `CUDA_VISIBLE_DEVICES=0`.

## Ambiguities (asked before long GPU jobs)

1. Account: nexus vs scavenger vs mc2 — default **nexus**.
2. Partition: default **tron**.
3. GPU type: unspecified `gpu:1` can yield 8 GB cards; scripts request **rtxa5000**.
4. Scratch: clip-scratch vs missing nexus-scratch directory.
