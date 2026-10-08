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
