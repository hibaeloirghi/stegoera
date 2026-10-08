# Optimization report

2026-10-08. Single-S4 receiver-aligned gradient smoke, one run (job 7788181), principal case, one photo/prompt/key. This is a smoke test: 8 Adam steps, lr 0.02, no seeds, no repeats. It is not a method and nothing here is promised to generalize.

## Setup

Free perturbation `(eta_x, eta_y)` of the true baseline S4 pair at step 20, followed by the exact EDICT forward suffix (30 steps, guided UNet) and VAE decode; surrogate receiver = encoder mean → duplicate → reverse to the S4 site → `x`. Loss = `0.1·hinge(0.3 − c·h)²` + `20·softplus(−margin/20)` (differentiable PRC margin at v = 1.5) + carrier-energy anchor (50) + 20·LPIPS + 5·pixel-MSE vs the unwatermarked edit; `beta_pair = 0`. UNet fp32 for optimization (stock diffusers 0.21.4, same fp16 weights), pair/VAE fp64. The final image is re-evaluated on the true channel: fp64 UNet suffix, saved PNG, reloaded, decoder inversion (50 steps), duplicate, fp64 reverse, original `Detect`/`Decode`. Final iterate reported; no best-iterate picking.

## Gradient gate (7788045, all 6 gated tests passed)

| test | result | threshold |
|---|---|---|
| T1 fp64 suffix reproduces saved final pair | 0.0 | 1e-8 |
| T2 fp64 reverse reproduces saved site | 3e-14 | 1e-8 |
| T3 fp32-UNet suffix vs fp64 final (MSE x) | 2.2e-13 (max abs 4.1e-6), 16.4× faster | 1e-6 |
| T4 fp32 receiver vs D0 fp64 encoder-mean site | 1.5e-12 (site error 0.04791 both) | 1e-3 |
| T5 autograd vs finite difference through the whole pipeline | rel. err 0.13 % (eps 0.002; 2.5 % at eps 0.01) | 5 % |
| T6 fp64 3-step chain, autograd vs FD | 5.9e-8 | 1e-5 |

fp32 forward+backward ≈ 90 s, peak 24.3 GB. The terms at η = 0: margin −18.7, LPIPS 5e-4 (small non-zero; I attribute it to the fp32 UNet and the in-graph decode vs the historical fp64 PNG but did not isolate it).

## Smoke result (7788181; key `otp655ddd0a246a_tb6408441c0324_G3e417b827611`)

Surrogate curve (iteration → margin): 0 → −22.9, 1 → +8.5, 2 → +34.4, 4 → +70.0, 8 → +109.1. LPIPS vs edit rose 0 → 0.0154 and pixel MSE 0 → 0.017 over the 8 steps; both were still rising when it stopped. Sign agreement with the codeword did not move (0.496 → 0.495).

True channel, saved/reloaded PNG (aligned from `r1_smoke_7788181_true_rows_aligned.csv`; the original `r1_smoke.csv` has misaligned columns in its last three rows, see below):

| | margin (v = 1.5) | detect | decode_non_none | exact_message_match | sign agreement | LPIPS vs edit | LPIPS vs original | edit alignment |
|---|---|---|---|---|---|---|---|---|
| stego PNG, true key | **+67.7** | **True** | False | False | 0.493 | 0.0154 | 0.239 | 0.252 |
| baseline edit PNG, same key (control) | −25.3 | False | False | False | 0.495 | 0 | 0.237 (D0 value) | 0.250 (D0 value) |
| stego PNG, independent wrong key | −24.5 | False | False | False | – | – | – | – |

overall_success (detect and exact message and LPIPS ≤ target) = **False**.

Reading, with the limits stated:

1. The edited stego photo looks like the baseline edit (triplet `images/r1_smoke_7788181_triplet_original_baseline_stego.png`); LPIPS to the unwatermarked edit is 0.0154 and the edit alignment (CLIP 0.252 vs 0.250 for the baseline) and LPIPS to the original (0.239 vs 0.237 for the baseline) are essentially unchanged. Visual check of the triplet by me only, on one image.
2. `Detect` returned True at margin +67.7 at v = 1.5, through the PNG and decoder inversion, with both controls negative. The detection survives the true channel; the surrogate margin (109) overstated it by ~40.
3. The message was **not** transmitted. `Decode` returned no payload, sign agreement with the embedded codeword is 0.493 (chance), correlation −0.015.
4. Why detection succeeded without the codeword (`r1_word_analysis_7793363.json`, same saved key): the fraction of parity checks with positive signed product is 0.545 for the stego carrier vs 0.496 for random signs and 0.951 for the embedded codeword. Detect's statistic pools 15,777 checks, so a 5-point excess in pass-rate is ~12 standard errors and exceeds the F = 1e-5 threshold. The optimizer found a carrier that raises the detector statistic by a pattern that is not the codeword (agreement with the one-time-pad pattern 0.493, also chance). This matches the earlier worry about gaming the margin and shows my loss did not constrain what the margin does not: the codeword hinge weight (0.1·mean-square) was too small next to the margin term to pull toward the true codeword. I have not shown which sign pattern was found, only that it is not c.
5. Therefore this is a detection-only result at LPIPS 0.015 vs the baseline edit, on one photo and one key. It is not evidence that an exact 64-bit message can be embedded; a detector-satisfying carrier is not a PRC codeword carrier.
6. D0 gave per-case upper context: the additive carriers needed LPIPS 0.155 (γ = 0.35) for agreement 0.60 and still failed Detect. The smoke gets Detect at a tenth of that LPIPS, but with the wrong word.

What I do not know: whether the same optimizer reaches the codeword if the loss pushes the correct word (e.g. a stronger/penalized agreement or a decode-oriented term), what LPIPS that would cost, or whether the wrong-word solution is robust across keys. The smoke does not answer that. R2 (pair-consistency β), R3, R4, random-sign controls and extra trials were **not** submitted.

## Logging defect (fixed in analysis, not in the original CSV)

`io_utils.append_csv` writes the first row's header for later rows; the three rows appended after the optimization curve (true-channel row and the two controls) have different keys and landed under wrong headers in `results/receiver_aligned/r1_smoke.csv` (e.g. `detect` showing −10856). The values are positional in their own key order; `repair_r1_csv.py` re-aligned them into a NEW file and the values match the TRUE_CHANNEL / CONTROL lines of the job log. The original file is untouched. Any future writer must use a fixed schema.

## Provenance

Job 7788181, code hash `5a5a241890b3df94`, experiment git `efddc0c`, EDICT `a11504a`, PRC `4f8e09b`. Key saved: `saved_states/r1_smoke_7788181_key.pkl`. Optimization 670 s; true evaluation 1009 s.

---

# Run 2: codeword-sign term (job 7793436, 2026-10-08)

Change vs run 1 (everything else identical, same photo/prompt, new key `otp0bc2fe711d81_tbf2f9c57f6ecb_G6cdd1c80818b`): added `a_sign=30 · mean softplus(−sign(c)·h / 0.5)` to `L_code`; 16 Adam steps, lr 0.03 (run 1: 8 steps, lr 0.02). Per-job CSV `r1_codeword_7793436.csv` (union schema, no misalignment). Final iterate reported. ~54 min wall (75 s per step, ~17 min true evaluation).

Surrogate curve: margin −19.1 → +137.2; sign agreement with the codeword 0.5015 → 0.5451 (still rising ≈ 0.004 per step at the end, not saturated); LPIPS vs edit 0 → 0.0264 (flat from step 7); pixel MSE 0 → 0.041 (still rising).

True channel, saved/reloaded PNG, decoder inversion, fp64:

| | margin (v = 1.5) | detect | decode_non_none | exact_message_match | sign agreement | LPIPS vs edit | edit alignment |
|---|---|---|---|---|---|---|---|
| stego PNG, true key | +112.3 | True | False | False | 0.535 | 0.0265 | 0.236 |
| baseline edit PNG, same key | −18.2 | False | False | False | 0.500 | 0 | – |
| stego PNG, independent wrong key | −17.7 | False | False | False | – | – | – |

overall_success = False.

Reading:
1. The sign term does what it was added for, but slowly: agreement with the real codeword rose to 0.535 on the true PNG (run 1: 0.493), i.e. the carrier now carries some codeword content. It is far below the ≈ 0.63 / ≈ 0.67 that Detect-on-codeword and Decode needed on the idealized carrier, so decoding still returns nothing. Run 1 got Detect without the codeword; here Detect is again satisfied, now partly by codeword content but I have not re-run the parity analysis on this vector, so how much of the +112 comes from the codeword vs another pattern is not established.
2. The surrogate overstates the true agreement (0.545 vs 0.535) and the true margin is lower than the surrogate's (112 vs 137), but the effect is much smaller than in run 1. Site error to the stego site 0.048 (encoder-mean level).
3. Distortion: LPIPS to the unwatermarked edit is 0.0265 (run 1 0.0154), edit alignment 0.236 vs 0.250 baseline (a drop I have not interpreted; one image). Pixel MSE 0.041 is rising and not converged.
4. No claim that more steps would reach decoding. Extrapolating a rate of 0.004/step to 0.63 would need ~ 20+ more steps and rising distortion; the rate may slow, and I have no evidence either way. The distortion budget at that point is unknown.
5. Still one photo, one key, one seed. No R2/R3/R4 run.
