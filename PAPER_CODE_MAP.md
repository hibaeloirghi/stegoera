# Paper to code map

Line ranges are the files as of this audit. Equation numbers are the arXiv EDICT text (`EDICT-paper.pdf`, Wallace, Gokul, Naik) and the ICLR 2025 PRC text (`gunn PRC image watermark.pdf`, Gunn, Zhao, Song). Implementations that are not in those papers are marked **OUR PROPOSAL**.

## Wallace, Gokul, Naik — EDICT

| Paper | Local code | What the code does | Deviation |
|---|---|---|---|
| Sec. 3.1, Eqs. 1–4. `x_t = sqrt(alpha_t) x + sqrt(1-alpha_t) eps`, DDIM step `x_{t-1} = a_t x_t + b_t eps`, with `a_t = sqrt(alpha_{t-1}/alpha_t)` and the `b_t` in Eq. 4. Classifier-free `Phi` is Eq. 3. | `EDICT/edict_functions.py` `forward_step` 698–727, `reverse_step` 733–764. `experiments/edict_runtime.py` `_noise_pred` 115–126, `ddim_eps` 286–290, `compose_xt` 293–296. | `forward_step` is the DDIM update written with `alpha_quotient`. Guidance is `eps_u + G (eps_c - eps_u)`. | `ddim_eps` / `compose_xt` are the algebraic rearrangement of Eq. 1. They do not establish that the recovered `eps` is iid standard normal. |
| Sec. 3.2. DDIM inversion approximates `eps(x_t,t) ≈ eps(x_{t-1},t)` (Eq. 5). Unconditional error is smaller than guided error. | Null base prompt in every edit job. `clip_embeddings` 89–112 skips the second UNet call when the prompt is empty. | Removes the guidance residual on the source inversion only. | Upstream always evaluates both branches. The runtime fuses them when they are identical. Not bitwise-tested. |
| Sec. 4.1, Eqs. 10–11. Coupled step `x <- a x + b eps(y)`, `y <- a y + b eps(x_new)`, and the two-line inverse. | `coupled_stablediffusion` 964–1033 and `coupled_loop` 204–230, via `forward_step` / `reverse_step`. | One affine update per stream per timestep, leapfrogged. | Leapfrog is the paper's "in practice we alternate order" sentence in Sec. 4.3, not a separate numbered equation. |
| Sec. 4.2, Eqs. 12–13, Fig. 5. Mixing `x = p x + (1-p) y`, inverse dilation `x = (x - (1-p) y) / p`. `p=0.93` for 50 steps. Fig. 5: small `p` is unstable in float. | Forward mix, runtime 237–239 and upstream 1049–1050. Reverse unmix, runtime 195–197 and upstream 956–957. Jobs use `p=0.93`, float64. | `y` is mixed using the already updated `x`, which is what those lines do. Common-mode addition commutes with this mix. `x_after - y_after = p^2 (x_before - y_before)` for the mix alone. | Eq. 13 in the extracted text is the one-line inverse written for a single average. The code inverts the two-line mix actually implemented. CPU tests check the code's mix, not a rewritten one. |
| Sec. 4.3, Eqs. 14–15. Full pair denoise and full pair invert. Double precision. Alternate `x`/`y` order. | The whole `coupled_loop` body, 187–244. | Eq. 14 is the forward body; Eq. 15 is the reverse body, including unmix. | Exactness needs the pair that was produced. Decoding only `x` and setting `y_hat = x_hat` is not Eq. 15. |
| Sec. 4.4. `x0 = VAE(I)`, `y0 = x0`, noise `s*S` steps with `C_base`, denoise with `C_target`, `G=3`. `x` and `y` images look alike. | `invert_image` 268–277, `generate_from_pair` 280–283, edit configs. Photo `imagenet_dog_1.jpg`, `C_base=""`, `C_edit="A golden retriever"`, `S=50`, `s=0.8`, `G=3`. | This is the sender skeleton. | Only `x` is decoded. Table 1's autoencoder floor is not remeasured here. `y0=x0` is the start of inversion, not the end of generation. |
| Sec. 4.4, linearization and mixing dilation make `(x_t, y_t)` diverge slightly. | Cosine trace stored on the loop result, not used as a watermark. | Logged only if a caller keeps it. | — |

## Gunn, Zhao, Song — PRC watermark

| Paper | Local code | What the code does | Deviation |
|---|---|---|---|
| Sec. 3.2, `Generate` / `Sample`. Diffusion from a Gaussian latent, then `D`. | Not what the edit jobs do. EDICT starts from a photo. | — | Our sender is an edit, not `Sample`. |
| Sec. 3.2, footnote 5. Decoder inversion: gradient descent on `z` so `D(z)≈x`, init `E(x)`. | `experiments/prc_runtime.py` `decoder_inv_vae` 78–106. Adam, 50 steps, lr 0.1, cosine warmup 10, MSE sum, VAE weights frozen, float64. | Matches the footnote's objective. | Init is `latent_dist.sample`, not the mean. Step count and optimizer are the repo's, not a number in the footnote. |
| Sec. 3.2, `Recover`. Prompt-free inversion in the paper's main scheme. | Edit jobs invert with the edit prompt for the matched receiver. | Matched receiver is a feasibility test. | A null receiver was also run on the old screen. It is not the main criterion. |
| Sec. 3.2 and Algorithm `PRCWat.Sample`, step 3. `z_i = c_i \|y_i\|`. | `src/pseudogaussians.py` `sample` 15–22. Top-blend jobs call this, then mix with `gamma`. | Full pseudogaussian only when `gamma=1`. | `gamma<1` blends and coupling adds are **OUR PROPOSAL**. The undetectability theorem is for the full pseudogaussian sample. |
| Appendix B, Algorithm 1 `PRC.KeyGen`. Sparse checks, generator, OTP, test bits, permutation. | `src/prc.py` `KeyGen` 71–115. | Same pieces. `F=1e-5`, `t=3`, `n=16384`, key message length 512. | Parameter formulas (`g=secpar`, `noise_rate=1-2^(-secpar/g^2)`) are the repo defaults used by the jobs. |
| Appendix B, Algorithm 2 `PRC.Encode`. `c = G y ⊕ e ⊕ otp` with Bernoulli noise `e`. | `Encode` 126–138. | Same, then mapped to ±1 by `1-2*bits`. | Paper's Algorithm 2 prints bits in `F_2`. The tensor sign convention is bit 0 → +1. |
| Appendix B, Algorithm 3 `PRC.Detect`. OTP and `(1-2η)` attenuation, parity products, threshold with `log((1+s)/(1-s))` and `C log(1/F)`. | `Detect` 151–170. | Same test, written as `sum log((1+Pi)/2) ≥ threshold`. | `experiments/prc_runtime.py` `detection_margin` is a reimplementation for logging. It is not a new detector. |
| Appendix B, Algorithm 4 `PRC.Decode`. BP-OSD, reject on bad test bits. | `Decode` 183–220. `bp_decoder`, then a linear solve on confident rows. | Test-bit failure returns `None`. | The solve is the repo's OSD-style step, not a claim of a specific library OSD. |
| Appendix C.1. `E[sign] = erf(z / sqrt(2 σ^2 (1+σ^2)))` under a Gaussian channel. | `recover_posteriors` 28–40. | That formula. | Using it on an EDICT latent assumes the Gaussian channel. D0 does not treat that as proved. Default `v=1.5` in the file; the sweep's selector returned 0.5 for a different reason (see AUDIT.md). |
| Appendix C.2, Theorem 2. FPR ≤ `F` for an image independent of the key. | `F=1e-5` stored on the key and used by `Detect`. | The code implements the threshold from that proof. | Host-adaptive embedding, best-of-N selection, and picking `σ` by the key's own score are outside the theorem's independence condition. Not claimed. |
| Appendix C.3. Practical undetectability has limits the proof does not cover. | — | — | No security claim is made for any edit in this repo. |

## OptMark

Not implemented. Sec. 3.3–3.5 (stage-wise embedding, quality constraints, adjoint checkpointing) are motivation for a later receiver-aligned optimization. OptMark's extractor is not PRC, and its continuous adjoint is not the gradient of discrete EDICT. No adjoint is in the code.

## Our proposals (not paper methods)

| Construction | Code | Status at audit |
|---|---|---|
| Top-latent blend `sqrt(1-γ^2) eps_inv + γ z_PRC` | `experiments/run_edit_stego.py` `inject_top` 28–35 | Ran. No detection at γ≤0.35. |
| Additive coupling `η = λ z_PRC` at S1–S4 | `experiments/hooks.py`, `make_embedding` 49–77 | Ran. No detection. S4 λ=0.10 was the strongest weak hit (agreement 0.545, LPIPS 0.016). |
| Coherent same-sign window with `λ/sqrt(K)` | `make_embedding` 60–77 | Ran. `λ/sqrt(K)` equalizes the sum of injection energies, not recovered energy or LPIPS. |
| Best-of-N codeword plus margin push `eps + λ c max(0, τ - c eps)` | `experiments/run_shape_stego.py` `shape_toward` 34–39, `select_codeword` | Task 0–1 finished and do not detect on the sender noise. Tasks 2–8 still queued. |
| Receiver-aligned site objective (code margin, detector margin, LPIPS to the unwatermarked edit, pair penalty) | not written yet | Waits on the D0 ladder. |
