# REPORT: PRC-in-EDICT

Date: 2026-10-05. Workdir: `/nfshomes/eloirghi/prc-in-edict`. Cluster: UMIACS Nexus (`nexusmc201`).

**Current protocol (user correction 2026-10-07).** The only target pipeline is a real photo inverted with a null base prompt, then denoised with a nonempty **editing prompt** while PRC is embedded. Success requires both detection/decoding and an image that preserves the unwatermarked EDICT edit. Decoded stream is **x** (`latent_pair[0]`). The earlier null-null E3 smoke is only a channel diagnostic, not the target application.

## 1. What ran and job IDs

Discovery-only (login + 1-minute `srun`):

| job (srun) | partition | node | notes |
|---|---|---|---|
| cluster-disc | tron / nexus / default | tron64 | HF and GitHub HTTP 200; cannot mkdir `/fs/nexus-scratch/eloirghi` |
| gpu-disc | tron / nexus / default, `--gres=gpu:1` | tron64 | **RTX 3070 8 GB** — too small for SD 1.4 float64 |

| job id | name | state | notes |
|---|---|---|---|
| 7756897 | edict-prc-setup | CANCELLED | 8 CPUs hit `QOSMaxCpuPerJobLimit` (default QOS max is 4 CPU / 1 GPU / 32G) |
| 7756898 | edict-prc-setup | COMPLETED 12:40 | Python 3.11 + torch 2.1.0+cu121. CLIP and **SD 1.4 downloaded without a token**. `stabilityai/stable-diffusion-2-1-base` failed **401** (B2 needs a HF token). CUDA false because this job was CPU-only. |
| 7757394 | b1-edict | COMPLETED 00:28:57 | Null recon + pesto-in-snow edit. Stream **x**. See §3. |
| 7757586 | b3-hook | CANCELLED by user-priority redirect | Superseded by direct E3 test before completion. |
| 7757587, 7757621–7757626 | earlier E1–E7 chain | CANCELLED | Replaced by the requested edited-stego pipeline. |
| 7757762 | e3-smoke | COMPLETED | Null-null diagnostic: gamma 0 rejected; gamma 1 detected/decoded but destroyed image identity. |
| 7780544 | edit-stego-base | SUBMITTED | Null inversion → “A golden retriever” edit, no watermark. |
| 7780545 | edit-stego array 1–11 | CANCELLED during task 1 | Stopped after finding repeated receiver work. No result row produced. |
| 7781538 | optimized edit-stego array 1–11 | SUBMITTED | Reuses sender inversion; one decoder inversion per image; null guidance uses one UNet call; float32/frozen-VAE decoder optimization; matched and null receivers. |

## 2. Upstream-file changes (`# [EXP]` lines)

Documentation-only commit in each clone: `docs: explanatory comments only` (no `# [EXP]`).

### EDICT `edict_functions.py`

| line | text |
|---|---|
| 49 | `# [EXP] Placeholders so EDICT_SKIP_MODEL_LOAD=1 can import helpers without GPU/HF at import time.` |
| 60 | `def load_edict_models(...):` docstring starts `[EXP] Load CLIP + SD 1.4...` |
| 71 | `auth_path = os.environ.get("EDICT_HF_AUTH", "hf_auth")  # [EXP] allow non-cwd token file` |
| 76 | `auth_token = None  # [EXP] ignore the repo placeholder; try anonymous HF access` |
| 81 | `# [EXP] Pass use_auth_token only when a real token exists (gated SD 1.4).` |
| 96 | `# [EXP] Preserve upstream import-time load unless experiments opt out.` |
| 817 | `step_hook=None):  # [EXP] optional; None is a no-op and must not change math` |
| 948 | `if step_hook is not None:  # [EXP] S4 on reverse: pair is the mixed state; subtract here to undo a forward S4 add` |
| 959 | `if step_hook is not None:  # [EXP] S3 on reverse: after unmix` |
| 1035 | `if step_hook is not None:  # [EXP] S1=first affine of this timestep, S2=second; paper x_inter / y_inter (leapfrog may swap which stream)` |
| 1041 | `if step_hook is not None:  # [EXP] S3: (x_inter, y_inter) before mixing` |
| 1052 | `if step_hook is not None:  # [EXP] S4: after mixing (x_{t-1}, y_{t-1})` |

Hook mapping (paper Eq. 14 → code). Leapfrog may swap which stream is updated first; S1/S2 follow the code's first/second affine of that timestep, not a fixed x/y label.

| paper | code |
|---|---|
| `x_inter = a_t x_t + b_t eps(y_t,t)` | first `forward_step` / `reverse_step` of the pair |
| `y_inter = a_t y_t + b_t eps(x_inter,t)` | second affine of the pair |
| mix `x_{t-1} = p x_inter + (1-p) y_inter` | `new_latents[0] = mix_weight*x + (1-mix_weight)*y` |
| mix `y_{t-1} = p y_inter + (1-p) x_{t-1}` | `new_latents[1]` uses already-updated x |
| S1 | additive on the stream just written after the first affine (`updated_index`) |
| S2 | same after the second affine |
| S3 | both streams after both affines, before mix (forward) / after unmix (reverse) |
| S4 | both streams after mix (forward) / before unmix (reverse) |

`experiments/edict_runtime.py` `coupled_loop` mirrors the same five hook sites (`# [EXP]` at the S4/S3 reverse and S1–S4 forward calls). Experiment logic lives under `experiments/`; those mirror lines exist so E5–E7 can run without calling `coupled_stablediffusion` for the hooked path.

### PRC-Watermark

| file | text |
|---|---|
| `encode.py` | `hf_cache_dir = os.environ.get("HF_HOME", ...)  # [EXP] honor HF_HOME on the cluster` |
| `decode.py` | same |
| `inversion.py` | `import os  # [EXP] needed to read HF_HOME for cache_dir` and `cache_dir=os.environ.get("HF_HOME", ...)  # [EXP] honor HF_HOME on the cluster` |

No other upstream executable edits.

## 3. Results tables

CSVs land in `results/{b1,b3,e1,e2,e3,e4,e5,e6,e7}/`. Every experiment row has `condition` (`null_null` in Phase 1). E3/E4 also log `receiver_mode` and `stream`. Empty cells are not zeros.

### B1 (job 7757394) — null recon identity, then a prompted edit

Null prompt invert+regen at s=0.8, S=50, p=0.93, float64, stream **x**, photo `photo_pesto.jpeg`. The recon equals the VAE round-trip; remaining error vs the photo is the VAE, not EDICT.

| metric | value |
|---|---|
| t_final | 780 |
| alpha_t | 0.04382697233721093 |
| pixel_mse recon vs orig | 0.007359958772367199 |
| pixel_mse recon vs VAE | **0.0** |
| pixel_mse VAE vs orig | 0.007359958772367199 |
| latent_mse x0 vs recon x | 2.3745384775898023e-30 |
| cosine(x,y) after recon invert | 0.9874967709454548 |

Edit images (`edit_x.png`, `edit_y.png`) used a nonempty snow prompt; they are not Phase 1.

### Phase 1 (`null_null`)

| exp | job | detect / metric | notes |
|---|---|---|---|
| B3 hook=None | 7757586 | max abs diff | must be 0 vs omitted kwarg |
| E1 VAE PRC | 7757587 | detect/decode | autoencoder only |
| E2 null invert+regen | 7757621 | pixel_mse vs VAE | identity control at s=1 |
| E3 smoke | 7757622 | detect/decode, LPIPS | gamma 0 and 1, mlen 64 |
| E4 attacks | 7757623 | detect after jpeg/resize/noise/crop | uses E3 outputs |
| E5 coupling eta | 7757624 | ignorant vs informed offset | H1, H3 |
| E6 PRC in hook | 7757625 | detect/LPIPS by variant | H2; V1 at S1–S4, V2/V3 at S4 |
| E7 p_i mix | 7757626 | proxy bit acc, x/y div | H4 |

### Target edited-stego pipeline (`null_target`) — submitted

| exp | matched receiver | null-receiver | mismatch loss |
|---|---|---|---|
| edited baseline + top/coupling screen | job 7781538 | job 7781538 | pending |

Sender: null base-prompt inversion of `imagenet_dog_1.jpg`, followed by denoising with edit prompt `A golden retriever`, S=50, s=0.8, p=0.93, G=3. Screen: top gamma 0.05/0.1/0.2/0.35; coupling single-step S1–S3 at lambda 0.03 and S4 at 0.01/0.03/0.1; coherent three-step S4 at total-energy-scaled lambda 0.03. Every output is received in matched-edit-prompt and null-receiver modes. Quality is measured against both the original and the unwatermarked edited baseline (pixel MSE and LPIPS), plus CLIP alignment to the edit prompt.

### Hypotheses vs outcomes (H1–H4)

Predictions recorded in `results/predictions.md` before E5–E7. Outcomes are **not filled** until those jobs write CSVs.

| id | prediction | outcome | numbers |
|---|---|---|---|
| H1 | ignorant coupling eta is an additive offset on recovered top latents; no richer than E3 top injection | pending | |
| H2 | V2 detectable; V3 not | pending | |
| H3 | informed subtract of eta recovers the pre-injection pair up to fp error (`offset_informed_latent_max`) | pending | |
| H4 | measurable x/y divergence, low per-coordinate bit acc | pending | |

H2 caveat (not a result): V2 drops the formal undetectability argument because noise across steps is not independent.

## 4. Failures and diagnosis

| item | diagnosis |
|---|---|
| `/fs/nexus-scratch/eloirghi` missing | Permission denied on login and compute. Scratch used: `/fs/clip-scratch/eloirghi/prc-in-edict` (user confirmed). |
| Unspecified `--gres=gpu:1` | Drew an 8 GB RTX 3070. Scripts request `rtxa6000`. |
| QOS | default MaxTRESPerJob cpu=4, gres/gpu=1, mem=32G. One GPU at a time. |
| B1 CUDA OOM (7757364/7757375) | `coupled_loop` lacked `@torch.no_grad()`. Fixed; attention slice 1; A6000. |
| B1 PATH (7757358) | conda not on PATH. `common.sh` prepends env `bin/`. |
| SD 2.1 401 | gated; B2 not started. Token is in gitignored `.hf_token`, not committed. |

## 5. Assumptions and unverified items

- **alpha_t:** `DDIMScheduler.alphas_cumprod[t]` with `t` the last reverse timestep. For S=50, s=0.8, `t_limit=10`, full timesteps `[980,960,…,0]`, reverse slice ends at **t=780**. Logged per run.
- **Encoder:** EDICT uses `vae.encode(...).latent_dist.sample()` (stochastic, seeded), **not** the mean/mode.
- **Null-null:** CLIP still tokenizes `""` as the conditional embedding. With C empty that equals the unconditional embedding, so `Phi = Theta(x_t,t,null)` and G drops out.
- **Receiver u:** `x_t_hat / sqrt(1-alpha_t)` as specified.
- **Stream:** decoded from the **x** latent (index 0), including E5–E7.
- **Joint env:** one Python 3.11 + torch 2.1 env; EDICT uses vendored `my_diffusers`, PRC uses pip `diffusers==0.21.4`.
- **float64:** EDICT UNet/VAE/CLIP `.double()`. PRC B2 baseline (not run) would stay float32.
- **Photos:** EDICT `experiment_images` copies in `experiments/photos/`. Phase 1 uses `imagenet_cat.jpg` (E3 smoke also gamma 0/1).
- **E7 mix_weight:** per-coordinate `p_i` is applied only in `experiments.edict_runtime.coupled_loop`, not in upstream `coupled_stablediffusion`. Elementwise affine mixing remains invertible for `p_i in (0,1)`.
- **E5 H3:** judged on latent-pair inverse (`offset_informed_latent_*`), not the VAE round-trip columns.

## 6. Recommended next experiments

1. After B3 max-abs is 0, read Phase 1 CSVs in order E1–E7.
2. If Phase 1 E3 gamma=0 does not roughly match E2 VAE recon, stop and diagnose before any prompt.
3. Only then submit Phase 2 `null_target` with matched vs null-receiver.
4. B2 / SD 2.1 after Phase 1 numbers exist (download script ready).
5. Do not expand E3 to a 360-task grid.

## Git

Original HEADs: EDICT `c35a81f8e3f35a3ee0f93c25af87b6d3826ac5af`, PRC `3f372c1d9cafcd0732331c0d00f0430ee822a389`. Branch `exp/prc-in-edict` in both. No push.
