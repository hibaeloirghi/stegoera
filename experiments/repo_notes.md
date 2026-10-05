# Upstream repo notes

Workdir: `/nfshomes/eloirghi/prc-in-edict/`
Clones (2026-10-05):

- EDICT `https://github.com/salesforce/EDICT` HEAD `c35a81f8e3f35a3ee0f93c25af87b6d3826ac5af` (branch `exp/prc-in-edict`)
- PRC-Watermark `https://github.com/XuandongZhao/PRC-Watermark` HEAD `3f372c1d9cafcd0732331c0d00f0430ee822a389` (branch `exp/prc-in-edict`)

## EDICT (salesforce/EDICT)

**Entry points**

- `EDICT.ipynb` / `EDICT_no_images.ipynb` — documented usage.
- `edict_functions.py` — actual algorithm. **Loads CLIP + SD 1.4 onto CUDA at import time** (after `[EXP]`, skippable via `EDICT_SKIP_MODEL_LOAD=1`).
- No `encode.py`-style CLI. `experiment_images/` holds demo photos.

**Checkpoint:** `CompVis/stable-diffusion-v1-4` (`revision=fp16` then `.double()`). CLIP: `openai/clip-vit-large-patch14`. VAE latent **4×64×64** at 512². Scaling **0.18215**. Encoder uses **`latent_dist.sample()`** (not the mean/mode).

**Requirements:** `environment.yaml` — **Python 3.8.5**, **PyTorch 1.11.0 + CUDA 11.3**, pip `diffusers==0.6.0`, `transformers==4.19.2`. Runtime code imports **`my_diffusers`** (vendored diffusers 0.3.0 fork with float64 timestep casts), not pip diffusers. README: double precision for inversion/reconstruction.

**Auth:** README says paste a Hugging Face token into `hf_auth` with no newline. The cloned file is the placeholder `<insert your token here>`. `use_auth_token` is passed to `from_pretrained`.

**Editing API:** `EDICT_editing(im, base_prompt, edit_prompt, steps=50, mix_weight=0.93, init_image_strength=0.8, guidance_scale=3, run_baseline=False)`.
**Docstring bug:** `EDICT_editing` says `run_baseline=True` is EDICT; **code treats True as DDIM** (no mixing, no leapfrog) and False as EDICT.

**Scheduler:** `DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False, set_alpha_to_one=False)`. `set_timesteps(50)` → `arange(0,1000,20)[::-1]` = `[980,960,…,0]`. `alpha_t` in experiments is `scheduler.alphas_cumprod[t]` at the last inverted timestep.

**Mixing:** forward `x ← p x + (1-p) y` then `y ← (1-p) x' + p y` with p=`mix_weight`. Inverse mixing is applied *before* reverse affine steps. Leapfrog flips which stream is updated first.

**Bug:** `image_to_latent` tensor branch does not `return` (PIL path is fine).

## PRC-Watermark (XuandongZhao/PRC-Watermark)

**Entry points**

- `encode.py` — KeyGen + Encode + generate images.
- `decode.py` — exact_inversion + Detect/Decode.
- `inversion.py` — `stable_diffusion_pipe`, `generate`, `exact_inversion`.
- `src/prc.py` — KeyGen/Encode/Detect/Decode.
- `src/pseudogaussians.py` — `z_i = c_i |g_i|`, `recover_posteriors` via `erf`.
- `src/inverse_stable_diffusion.py` — `decoder_inv` (100-step Adam on `||D(z)-x||`) and DPMSolver `forward_diffusion`.
- `src/modified_stable_diffusion.py`, `src/optim_utils.py`.
- `src/baseline/` — Tree-Ring / Gaussian Shading; **not** used by the PRC default path.

**Checkpoint:** default `stabilityai/stable-diffusion-2-1-base`. Latent **4×64×64** (same n as SD 1.4 at 512). Pipeline dtype **float32**. Scheduler **DPMSolverMultistep** (not EDICT's DDIM).

**Requirements:** Python **3.11.9**, `torch==2.1.0`, `transformers==4.29.2`, `diffusers==0.21.4`, `galois==0.4.1`, `ldpc==0.1.51`. Conflicts with EDICT's conda pin (torch 1.11 / py3.8). Joint experiments use **one modern env** and EDICT's vendored `my_diffusers`.

**Auth:** no token file. `hf_cache_dir` was hardcoded to `/home/xuandong/mnt/hf_models` (`[EXP]` now honors `HF_HOME`). README: change that path.

**Messages:** README says longer-message CLI “will be added later”. **`Encode(encoding_key, message=...)` already exists**; `encode.py` calls `Encode(encoding_key)` with no message (random payload including test bits). Default `KeyGen(..., message_length=512)`. Decode is BP (`ldpc.bp_decoder`) then a confident-row linear solve (OSD-like, not a named BP-OSD library). `decode.py` uses posterior variance **1.5** hardcoded.

## Joint-experiment implications

- Run EDICT + PRC **on SD 1.4** first, as requested. PRC wrappers in `experiments/` use EDICT's VAE (4×64×64) and a decoder-inv loop copied from PRC but not bound to `InversableStableDiffusionPipeline`.
- Do **not** import `edict_functions` on the login node without `EDICT_SKIP_MODEL_LOAD=1`.
- Receiver in E3 uses EDICT inversion with the **target** prompt, not PRC's DPMSolver inversion.
