# Audit (receiver-aligned PRC in EDICT)

Written before any new model job. Running array `7783940` was left in place. Its scripts were not edited.

Hashes at audit time:

- EDICT `a11504ae7902763655f4247335454be2f3b37499` (branch work sits on top of this checkout; `edict_functions.py` contains `# [EXP]` hook lines)
- PRC-Watermark `4f8e09bc0590e4e8d89cc23888b68e559417748a`
- experiment repo `4dbaf6afc0e9ceaa9b01b4b7a58d62104b573be0` plus uncommitted experiment files
- model: CompVis/stable-diffusion-v1-4, float64, DDIM `beta_start=0.00085`, `beta_end=0.012`, `scaled_linear`, `set_alpha_to_one=False`, 1000 train steps
- VAE scale `0.18215`

The earlier chat export contained a Hugging Face token. Rotate that token. This audit does not read or print it.

## Confirmed facts

### EDICT runtime (`experiments/edict_runtime.py`)

`coupled_loop` follows the same control flow as `EDICT/edict_functions.py:coupled_stablediffusion` (about lines 943–1055):

- S1 is the first affine update of the step, S2 the second. `updated_index` is whichever stream leapfrog selected. S1 is not hard-wired to `x`.
- Leapfrog: forward offset `i % 2`; reverse offset `(n-1-i+1) % 2`, matching upstream lines 970–982 and runtime lines 208–218.
- Forward mixing is `x <- p x + (1-p) y`, then `y <- (1-p) x_new + p y` (runtime 237–239, upstream 1049–1050).
- Reverse unmix runs before the reverse affine steps, y then x (runtime 195–197, upstream 956–957).
- The loop returns both streams (`latent_pair`, `x`, `y`).
- An explicit pair is accepted: `init_latent=[x,y]` on inversion and `fixed_starting_latent=[x,y]` on generation.
- Timestep slice: `t_limit = S - int(S*s)`, then `scheduler.timesteps[t_limit:]`, flipped when inverting. For S=50, s=0.8 the last inversion timestep is 780 and `alpha_t = alphas_cumprod[780]`. Completed rows all report `t_final=780`.
- `forward_step` / `reverse_step` only read `alphas_cumprod`. Upstream builds two schedulers; the runtime builds one. That does not change the coefficients.
- Encoder is `latent_dist.sample * 0.18215`, not the posterior mean (`encode_image_to_x0`, and upstream `image_to_latent`).
- `decode_latent` divides by 0.18215 and `tensor_to_pil` clamps, rounds to uint8, and builds a PNG. There is no saved float-image path.
- The loop does not record the pair at an interior site. The only cached sender state is the null-inversion top pair `results/edit_stego/sender_inversion_s0.8_S50_p0.93_seed1.pt`. Final `(x,y)` after the edit was not saved.
- `hook=None` regression job B3 has a config and no result file. It was not completed. Equivalence to upstream is unread from a numeric test.

Deviations that are real, not guessed:

- When the prompt equals the null prompt, `_noise_pred` runs one UNet call. Upstream always runs unconditional and conditional calls. For an empty prompt those two embeddings match, so the guided residual is zero either way. Bitwise equality is not tested.
- The runtime does not call upstream `init_attention_func`. Edit jobs do not use prompt-to-prompt. Effect on the golden-retriever trajectory is unverified.
- Only the `x` stream is decoded. EDICT §4.4 also decodes `y` and reports the two images look alike. That is not a latent-equality claim, and this code never checks the `y` image.

### PRC (`PRC-Watermark/src/prc.py`, `src/pseudogaussians.py`)

Experiment key, from the edit and shaping configs: `n=16384`, `message_length=512` passed to `KeyGen`, payload actually encoded is 64 bits, `F=1e-5`, `t=3`.

`KeyGen` then sets `num_test_bits = ceil(log2(1/F)) = 17`, `g = secpar = int(log2(C(n,3))) = 39`, `k = 512+39+17 = 568`, `noise_rate = 1 - 2^(-secpar/g^2)`. `Encode` draws `g` randomizer bits and a Bernoulli(`noise_rate`) error. Codeword values are `1-2*bits`, so bit 0 is `+1` and bit 1 is `-1`.

`Detect` multiplies soft bits by `(1-2*noise_rate)*(1-2*otp)`, forms the product on each weight-`t` check, and thresholds `sum log((1+Pi)/2)`. `Decode` runs belief propagation and returns `None` when the test bits do not match. A non-`None` decode is not by itself proof of the sent message; the shaping script only then compares the first 64 bits. Rows with `decode=False` leave `bit_accuracy` empty.

`sample` builds `z_i = c_i |g_i|`. `recover_posteriors` is `erf(z / sqrt(2 v (1+v)))`. The file's default `v` is 1.5. Gunn Appendix C.1 derives that `erf` under a Gaussian observation model. Intermediate EDICT latents were not shown to be that observation.

### Calibration

`calibrate_variance` tries `v in {0.5, 1.0, 1.5, 2.0, 3.0}` and keeps the maximum of `(detect + decode, mean |soft bit|)`. Smaller `v` makes `|erf|` larger. When nothing detects, `v=0.5` wins on the magnitude tie-break. When `v=0.5` does detect, it also wins, because every larger `v` has a smaller mean magnitude. Every completed edit-screen and shaping row has `sigma=0.5`, including the old null-null `gamma=1` cat that did detect. That value is the bottom of the grid, not an estimate of recovery noise.

`detection_margin` in `prc_runtime.py` copies `Detect` without clipping. A one-off login check on one `n=512` random vector agreed with `Detect`. That is not a suite, and `|Pi|=1` would make `log(0)`.

### What the finished jobs actually show

Edit screen `7782006` finished. All 11 embeddings: `detect=False`, `decode=False`. Strongest additive points: top `gamma=0.35` sign agreement 0.613, LPIPS vs the unwatermarked edit 0.176; S4 `lambda=0.10` agreement 0.545, LPIPS 0.016. The coherent three-step S4 window matched at agreement 0.515, LPIPS 0.0039. No margin was stored for that screen, so those rows do not say how far the parity statistic sat under the threshold.

Shaping array `7783940` is not a failure of the job. Task 0 finished. Task 1 has a matched row and was still the running task at audit time. Tasks 2–8 were queued.

| task | what | sender detect | sender margin | receiver margin | LPIPS vs edit |
|---|---|---|---:|---:|---:|
| 0 | top shape `lambda=0.05`, `tau=1` | false | -102.0 | -137.2 | 0.183 |
| 1 | top shape `lambda=0.15` | false | -104.8 | -117.6 | 0.754 |

Best-of-1024 selected candidate 302 with unweighted agreement 0.510 and weighted agreement 0.515. `fraction_changed=0.864` because `tau=1` treats most unit-scale coordinates as short. The sender-side margin is already about -100 on the shaped noise, before VAE, PNG, or the loss of `y`. For this shaping rule, the PRC constraint is not present on the carrier the sender writes. Increasing `lambda` from 0.05 to 0.15 did not close that gap and did wreck the edit. Tasks at `lambda=0.5` and `1.0` have not run; they are not failures yet.

The null-null cat at `gamma=1` did detect and decode, with sign agreement 0.918 and LPIPS 0.591 against the photo. That is full replacement, not an edit of the same picture. It is not used as evidence that PRC editing works.

## Likely issues

1. Top-latent reliability shaping, as implemented, does not put the detector over threshold even as an oracle on the sender noise. Image damage at `lambda=0.15` is already far past the 0.03 LPIPS screen.
2. The operational receiver initializes `y` as a copy of a recovered `x`. EDICT's exact inverse needs the pair that was actually produced. No measurement yet separates that loss from VAE/PNG loss, because the final pairs were not saved.
3. `x_t/sqrt(1-alpha_t)` still contains host content. The residual that subtracts `sqrt(alpha) * x0_hat` uses the edited latent as `x0_hat`, so it is not the original photo's noise. Shaping task 0's residual margin (-118) was less negative than the raw statistic (-137) and still a clear miss.
4. `sigma=0.5` cannot be read as a calibrated channel variance.

## Unverified

- Numeric `hook=None` match to upstream.
- Bitwise match of the one-UNet null path to upstream's two identical UNet calls.
- Whether a regenerated `gamma=0.35` or S4 `lambda=0.10` PNG matches the historical file. D0 will hash them and label a mismatch as a rerun.
- A/B/C/D drops. Not claimed.
- Pair-consistency optimization, adjoint gradients, and any security or FPR claim. A single negative baseline does not establish `F=1e-5`.

## Smallest next job

CPU tests of mixing algebra and of `Detect` versus the margin, with no model load.

Then one GPU ladder, D0, on three already-chosen outputs: the unwatermarked golden-retriever edit, top `gamma=0.35`, and S4 `lambda=0.10`. It reuses the cached null inversion, regenerates each trajectory once, and compares stages A (true final pair), B (`y` replaced by `x`), C (float image, encoder sample and decoder inversion), and D (PNG). Fixed soft-bit variances 0.5 and 1.5 are both reported. Neither is chosen after seeing the key score.

---

# Audit update — handoff session (2026-10-07, evening)

Sources: local code, `results/receiver_aligned/audit_cpu.json` (job 7787904), `audit_carrier_requirement.csv`, `test_cpu_torch.json` (job 7787969), `d0_ladder.csv` (job 7786379, still running), SLURM accounting. Nothing here edits a script used by a running job.

## D. Calibration: parameter naming (corrects earlier text)

`PRC-Watermark/src/pseudogaussians.py:28-40`: `recover_posteriors(z, variances=v)` is `erf(z / sqrt(2 v (1+v)))`. Gunn et al. Appendix C.1, Fact 1 (printed p. 23) is `erf(z' / sqrt(2 σ² (1+σ²)))` with `z' ~ N(z, σ²)`. So the code's `v` is **σ² (variance)**, not σ. Verified numerically in `audit_cpu.py` (max |diff| = 0 for v = 0.5, 1.0, 1.5).

Consequences for every earlier table and CSV:

| column / text | what it actually is | σ (std) |
|---|---|---|
| `sigma = 0.5` | σ² = 0.5, the bottom of the `calibrate_variance` grid | 0.707 |
| `sigma = 1.5` (D0 primary, repo default) | σ² = 1.5 | 1.225 |

New outputs use `sigma_squared` and `sigma = sqrt(sigma_squared)`. Old files are not rewritten. The soft bit that `Detect` sees is `(1-2·noise_rate)·(1-2·otp_i)·erf(z_i / sqrt(2 v (1+v)))`.

`calibrate_variance` (`prc_runtime.py:171-188`) confirmed: grid {0.5, 1.0, 1.5, 2.0, 3.0}; key `(int(detect)+int(decode), mean |soft|)`; smaller v gives larger |soft|, so v = 0.5 wins unless a larger v changes detect/decode. It is **not** a channel-error estimate, and picking v by each sample's key score is outside Theorem 2's independence condition.

Margins at different σ² are not comparable. On a random carrier the margin is about -22 at σ² = 1.5 and about -110 to -126 at σ² = 0.5 (`audit_carrier_requirement.csv`, a = 0 rows). The recorded shaping "sender margins" (-102, -105) were at σ² = 0.5 and are therefore close to what a random carrier gives, not "100 below a threshold that a stronger λ would close".

## B. PRC additions

- `Detect` returns **True** for an all-zero or near-zero soft-bit vector: `Pi = 0` gives `stat == thr` exactly, and Detect uses `>=`. `test_cpu_torch.json` rows `zeros` and `weak_1e-6` show margin `0.000000` and detect `True` at both σ². Decode returns None on those inputs (no confident rows), so exact message match stays False. This is a degenerate input, not a measured FPR, but it matters for any optimizer: random carriers sit near -20, so *shrinking* the carrier raises the margin toward 0 without adding any code. The R1 objective therefore carries a carrier-energy anchor (`l_energy`), and rows report carrier energy.
- Torch margin (`prc_torch.py`) equals the numpy `detection_margin` (clip=True) to ~1e-9 and matches `Detect`'s decision on genuine, random, weak, zero and near-threshold carriers (n = 16384 real key). Autograd gradient equals a central finite difference to 1.8e-7 relative. Pi is clamped to [-1+1e-12, 1-1e-12]; clip vs no-clip are identical on every tested input.
- Message equality is `exact_message_match` (`prc_scores.py`), not `decode is not None`.

### Idealized-carrier requirement curve (not an EDICT channel)

Real key (n = 16384, t = 3, F = 1e-5, seed 1), carrier `h = a·z_PRC + sqrt(1-a²)·w + 0.13·noise`, `w` iid N(0,1), `z_PRC = c|g|`, 2 seeds:

| a | corr(h, c) | sign agreement | margin σ²=1.5 | detect | exact message (σ²=1.5, 1 seed) |
|---|---|---|---|---|---|
| 0.30 | 0.22–0.24 | 0.59–0.60 | -12 / -11 | no | no |
| 0.35 | 0.26–0.28 | 0.61–0.62 | -6.5 / -5.8 | no | — |
| 0.40 | 0.30–0.32 | 0.62–0.63 | +1.0 / +0.6 | yes | no (decode None) |
| 0.50 | 0.38–0.40 | 0.66–0.67 | +22 / +19 | yes | **yes** |
| 0.60 | 0.46–0.48 | 0.69–0.70 | +54 / +48 | yes | yes |
| 1.00 | 0.79 | 0.96 | +330 | yes | yes |

At σ² = 0.5 and a = 0.5 the detector is True (margin +112) but `Decode` returns None. Detect and decode are different events. On an iid unit-variance carrier, detection needs carrier–codeword correlation of roughly 0.3 and exact decoding roughly 0.4. These are properties of this carrier model and key, not universal thresholds, and they are not claimed for EDICT latents (non-Gaussian, host-dependent).

## E. Shaping anomaly: quantified

`audit_cpu.json`. Shared: `alpha_t = 0.04383` (t = 780), `sqrt(1-alpha) = 0.9778`, host eps RMS 0.8927 (variance 0.797, not 1), max |eps| 4.16, selected codeword hash `54f2ecbd8f736f7e` (candidate 302 of 1024), codeword/sign(eps) agreement 0.5103, rebuilt pair at λ = 0: max |x_marked − x_t| = 4.4e-16 (rebuild path is clean), LPIPS reference `baseline.png` hash `ee8d944a…` (byte-matches the regenerated baseline in D0).

| | λ = 0.05 | λ = 0.15 |
|---|---|---|
| τ | 1 | 1 |
| modified-coordinate fraction | 0.864 | 0.864 |
| correction RMS / max | 0.0653 / 0.258 | 0.1958 / 0.775 |
| pair shift RMS (x_marked − x_t) | 0.0638 | 0.1915 |
| cos(correction, eps) | **-0.570** | **-0.570** |
| var(eps_shaped)/var(eps) | 0.922 | 0.798 |
| sign flips of eps | 2.5 % | 7.9 % |
| sign agreement with codeword, before → after | 0.510 → 0.535 | 0.510 → 0.589 |
| blend γ with the same correction RMS | 0.066 | 0.197 |
| cos(blend correction, eps) / variance ratio | -0.015 / 1.003 | -0.072 / 1.017 |
| recorded LPIPS vs unwatermarked edit | 0.183 | 0.754 |
| additive-screen LPIPS at the matching γ | γ=0.1: 0.026 (γ=0.066 not run) | γ=0.2: 0.087 |

Reading: where active, the correction is `λ(c − eps)`, a shrink of the noise toward ±1. It is strongly anti-aligned with eps (cos −0.57) and lowers the noise variance by 8 % / 20 %, while a random-direction blend of the same RMS preserves variance (cos ≈ 0). Nominal λ is not the perturbation amplitude (RMS = λ·1.30 at τ = 1). Flipping sign needs |eps| < λ/(1−λ), which is why agreement barely moves. The large LPIPS per unit RMS is **consistent with** the diffusion model reacting to a coherent noise-amplitude reduction, but that mechanism is a hypothesis: no GPU control that scales eps by (1−λ) without any code has been run. The `eps` energy shift is the only quantified difference between shaping and blending here.

**Validity caveat (finding F):** for λ = 0.15 the stored `sender_margin`, `margin`, `detect`, `decode` and `margin_residual` were computed against a decoding key that does not belong to the codeword, so they carry no information about shaping. Only the image-side numbers (LPIPS, pixel MSE, energy) and the key-independent geometry in the table above are usable for λ = 0.15. For λ = 0.05 the scores are valid.

Not yet recorded for these tasks: `Detect` on the sender-side carrier at σ² = 1.5 (the stored sender margins are at σ² = 0.5). Also: tasks 2 (λ = 0.50, PNG exists, **no scores**) and 3–8 of array 7783940 were cancelled by uid 17696 (this account) at 20:44:31; no record of user approval was found in the local files. They are *not evaluated*, not failed.

## A. EDICT additions

- Hook/step/stream-order audit (unchanged from above) holds. `diff_edict.py` re-implements the same step functions without `no_grad`; `run_gradcheck.py` T1/T2 compare it against D0's saved pairs from `coupled_loop`.
- Upstream regression: B3 (job 7757586) was **cancelled at 17:30 on 2026-10-05 after 8:43 of run time and wrote no result** (stderr shows it was mid-run). Replaced by `run_upstream_regression.py` (job 7787914): S = 10, s = 0.8, p = 0.93, G = 3, edit prompt, custom runs before the first upstream call (upstream's `init_attention_func` monkey-patches CrossAttention for the whole process), per-event comparison of both streams. Status: see `DIAGNOSTIC_REPORT.md`.
- Upstream always evaluates two UNet branches; custom fuses them when the prompt equals the null prompt. The new regression uses an edit prompt for generation and a null prompt for inversion, so both branches are exercised.

## C. VAE additions

- Sender encoder: posterior **sample** (`encode_image_to_x0`, seed 1). Receiver stage C reports sample, mean, and decoder inversion separately. `decoder_inv_vae` default is `n_steps=100`; D0 and R1 pass 50 from config, so the repo default and the job value differ.
- VAE and UNet weights are frozen (`requires_grad_(False)`) in the optimization path. Decoder inversion optimizes only z. Float64 throughout the baseline; the fork hard-codes `.double()` inside GroupNorm (`my_diffusers/models/resnet.py:336,354`) and softmax (`attention.py:77`), which is why `unet.float()` cannot be used. The fp32 fast path loads **stock diffusers 0.21.4** with the same fp16 SD-1.4 weights and is gated by the dtype test in `run_gradcheck.py` before it is used.
- Convergence of decoder inversion is not logged by the repo (final loss not recorded). Not audited further.

## F. PRC key is not reproducible from its seed (new, affects earlier results)

`PRC-Watermark/src/prc.py:KeyGen` draws the generator matrix, one-time pad and test bits with `galois.GF.Random(...)`, which uses fresh entropy; `np.random.seed(seed)` in `prc_runtime.keygen` only fixes the parity-check matrix and permutation. `key_repro_check.py` (two separate srun processes, two calls each, `seed=1`): `H` hash identical (`d762e1bef354`) in all four calls, but otp / test bits / G hashes differ in all four. `Encode` also draws its `g` randomizer bits with `GF.Random`, so the codeword for a given key and message is not reproducible either.

Effects, by what can be verified:

1. **D0 and every job whose sender and receiver run in one process are internally consistent** (same key, same codeword). Their saved CSVs are valid. Their saved raw carrier vectors **cannot** be re-scored in another process because the key was not saved.
2. **Job 7788093 (`analyze_raw.py`) used a regenerated key. Its watermarked-vector rows are invalid** and are flagged in `results/receiver_aligned/NOTE_raw_scores_7788093_INVALID.md`. It also showed the symptom first: baseline `A s4_x` margin −30.2 (σ²=1.5) vs −19.4 in D0's CSV for the same vector.
3. **Shaping array 7783940, task 1 (λ = 0.15): its PRC scores are invalid.** `selected_codeword_n1024_seed1.pt` was written at 19:34:42 by task 0 (task started 19:33:59). Task 1 started 20:03 and loaded that cached codeword while drawing its own fresh key, so its codeword did not belong to its decoding key and detection could not succeed regardless of the embedding. Its image-only numbers (LPIPS 0.754, pixel MSE 0.146) are unaffected. Task 0 (λ = 0.05) generated key and codeword in the same process and is valid. `sign_agreement` against the codeword is key-independent and stays meaningful.
4. The edit-screen array (7782006) generates the codeword in each task's own process; unaffected.
5. The idealized-carrier curve (`audit_cpu.py` part 3) and the CPU margin test generate key, codeword and carrier in one process; unaffected.

Fix (new code only; `prc_runtime.py` is untouched because queued jobs import it): `experiments/receiver_aligned/key_io.py` saves the key and a hash `otp…_tb…_G…` per job; `run_r1_smoke_v2.py` saves it and records it as `key_id`; `analyze_raw.py` now refuses to run without a saved key. The earlier CSV field `key_id = seed…` in drafts is replaced by that hash.

## Corrections to earlier reasoning (this session)

- D0 stage A shows the custom pair map is self-consistent (top error 4.9e-29). It is not an upstream regression.
- Stage B/C/D numbers for the unwatermarked baseline say nothing about the watermarked channel.
- The idealized curve above says the *evidence* a carrier must carry; it does not say whether a perturbation with that evidence can be hidden. That is the R1 question.
