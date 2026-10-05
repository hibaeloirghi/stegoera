# REPORT: PRC-in-EDICT

Date: 2026-10-05. Workdir: `/nfshomes/eloirghi/prc-in-edict`. Cluster: UMIACS Nexus (`nexusmc201`).

## 1. What ran and job IDs

Discovery-only (login + 1-minute `srun`):

| job (srun) | partition | node | notes |
|---|---|---|---|
| cluster-disc | tron / nexus / default | tron64 | HF and GitHub HTTP 200; cannot mkdir `/fs/nexus-scratch/eloirghi` |
| gpu-disc | tron / nexus / default, `--gres=gpu:1` | tron64 | **RTX 3070 8 GB** — too small for SD 1.4 float64 |

Environment / GPU experiment jobs: see later updates in this file (submitted after `experiments/slurm/setup_env.sh`).

**Not yet run at the time of the first draft of this report:** B1–B2, E1–E4. They wait on the conda env and on SD 1.4 download (may need a Hugging Face token).

## 2. Upstream-file changes (`# [EXP]` lines)

Documentation-only commit in each clone: `docs: explanatory comments only` (no `# [EXP]`).

Behavior changes (second commit, `exp: cluster import/cache hooks`):

**EDICT `edict_functions.py`**

- Placeholders + `load_edict_models()` so `EDICT_SKIP_MODEL_LOAD=1` can import without GPU.
- `EDICT_HF_AUTH` / `HF_TOKEN` / ignore the `<insert your token here>` placeholder.
- `use_auth_token` only if a real token exists.
- Default import path still calls `load_edict_models()` (notebook-compatible).

**PRC-Watermark**

- `encode.py` / `decode.py`: `hf_cache_dir = os.environ.get("HF_HOME", "/home/xuandong/mnt/hf_models")`
- `inversion.py`: `import os`; `cache_dir=os.environ.get("HF_HOME", ...)` on `stable_diffusion_pipe`.

No other upstream executable edits. All experiment logic is under `experiments/`.

## 3. Results tables

None yet (jobs not completed). CSVs will land in `results/{b1,b2,e1,e2,e3,e4}/`.

Planned controls for E3 (`experiments/configs/e3_controls.json`): `correct`, `wrong_key`, `unwatermarked`, `mismatch_G`. Gamma=0 is also in the main/smoke sweeps.

## 4. Failures and diagnosis

| item | diagnosis |
|---|---|
| `/fs/nexus-scratch/eloirghi` missing | Permission denied on login and compute. Scratch used: `/fs/clip-scratch/eloirghi/prc-in-edict` (please confirm). |
| Unspecified `--gres=gpu:1` | Drew an 8 GB RTX 3070. Scripts request `rtxa5000`. |
| `hf_auth` | Placeholder only. SD 1.4 download may fail without a HF token. |
| Photo count | Only two user photos + three EDICT demo images. Not 20–50. |

## 5. Assumptions and unverified items

- **alpha_t:** `DDIMScheduler.alphas_cumprod[t]` with `t` the last reverse timestep. For S=50, s=0.8, `t_limit=10`, full timesteps `[980,960,…,0]`, reverse slice ends at **t=780**. Logged per run; not assumed in analysis without the log.
- **Encoder:** EDICT uses `vae.encode(...).latent_dist.sample()` (stochastic, seeded), **not** the mean/mode.
- **run_baseline docstring** in `EDICT_editing` is inverted vs code (True = DDIM).
- **Multi-bit messages:** `Encode(encoding_key, message=...)` exists in `src/prc.py`. `encode.py` CLI does not pass a message. README says a longer-message CLI “will be added later”.
- **Decode:** `ldpc.bp_decoder` then a confident-row linear solve, not a packaged “BP-OSD” call.
- **Receiver u:** implemented as `x_t_hat / sqrt(1-alpha_t)` as specified (not the DDIM eps formula used on the sender).
- **Stream:** edited image decoded from the **x** latent (index 0).
- **Joint env:** one Python 3.11 + torch 2.1 env; EDICT uses vendored `my_diffusers`, PRC uses pip `diffusers==0.21.4`.
- **float64:** EDICT UNet/VAE/CLIP `.double()` as README/paper. PRC B2 baseline stays float32 as in that repo.
- **Photos:** user JPEGs (pesto, salad) + EDICT `experiment_images` copies for extra sanity. Confirm a 20–50 photo folder if you have one.
- **CLIP-scratch:** used because nexus-scratch is not provisioned for this account.

## 6. Recommended next experiments

1. Confirm partition/account (`tron`/`nexus`/`default`) and GPU type (`rtxa5000`).
2. Provide a Hugging Face token if SD 1.4 is gated.
3. After B1 recon MSE looks like EDICT paper (near-lossless at s=1), run E3 smoke (8 tasks) before the 360-task main array.
4. Try receiver using DDIM `eps_hat` instead of `x_t/sqrt(1-alpha_t)`.
5. Compare decoder-inv init `sample` vs `mode`.
6. SD 2.1 port only after SD 1.4 numbers exist.

## Git

Original HEADs: EDICT `c35a81f8e3f35a3ee0f93c25af87b6d3826ac5af`, PRC `3f372c1d9cafcd0732331c0d00f0430ee822a389`. Branch `exp/prc-in-edict` in both. No push.
