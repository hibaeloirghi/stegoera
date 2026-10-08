# Diagnostic report

Updated 2026-10-08 after jobs 7786379 (D0), 7787914 (upstream regression), 7788045 (gradient gate). Principal case: `imagenet_dog_1.jpg`, null inversion, "A golden retriever", S=50, s=0.8, p=0.93, G=3, n=16384, 64-bit message, t=3, F=1e-5. One photo, one prompt, one key per job. No statement below is a general claim.

`sigma_squared` means the soft-bit variance v (see AUDIT.md). Primary v = 1.5; v = 0.5 is the lower edge of the calibration grid, reported as a second column and not chosen after the fact.

## 1. Upstream regression (7787914, S=10, float64)

Custom EDICT loop vs the upstream `EDICT/edict_functions.py` code with the `[EXP]` hooks, same photo, prompt, seed:

| comparison | events compared | max abs difference |
|---|---|---|
| inversion, custom vs upstream (all hook events, both streams) | 32 / 32, keys equal | 0.0 |
| generation from the same pair, custom vs upstream | 32 / 32, keys equal | 0.0 |
| custom generation, recorder hook vs `hook=None` (final pair) | – | 0.0 |

Exact equality on this configuration means the hooks do not change the computation and the custom loop reproduces upstream. Scope: S=10, one photo, one seed. The S=50 principal case is validated indirectly (D0 stage A / gradient gate T1 reproduce the historical S=50 final pair with max abs 0.0).

## 2. D0 ladder (7786379, 3h26, float64, three cases)

Rows are single measurements with one key per case. Margin = Detect statistic minus threshold at v = 1.5 (≥ 0 is Detect True). `agree` = sign agreement with the embedded codeword. The "site" is the S4 x-state at step 20 (timestep 380).

| case | LPIPS vs edit | sender site margin / agree | after pair-only recovery (B, x duplicated) | encoder mean (C) | decoder inversion (C) | saved PNG (D) |
|---|---|---|---|---|---|---|
| baseline (no embedding) | 0 | −19.4 / 0.498 | −19.6 / 0.4985 | −21.4 / 0.4995 | −20.5 / 0.499 | −20.6 / 0.4996 |
| top γ = 0.35 (additive, top latent) | 0.155 | −8.7 / 0.604 | −8.8 / 0.605 | −10.1 / 0.599 | −8.2 / 0.603 | −8.3 / 0.603 |
| S4 λ = 0.10 (additive shaping at S4) | 0.017 | −18.0 / 0.534 | −18.1 / 0.533 | −18.0 / 0.532 | −17.4 / 0.534 | −17.6 / 0.533 |

Detect, decode_non_none and exact_message_match are False in every row (all 41). Site errors (MSE, x): B 0.0009–0.0011; encoder mean 0.048; decoder inversion 0.0057–0.0085; PNG 0.0060–0.0087. The same-pair, true-y site check (A) is exact.

What this does and does not say:

- For these two fixed carriers the received sign agreement and margin move by less than ~0.01 in agreement and ~2 in margin from the sender site down to the saved PNG. In this experiment the pair map, the VAE and PNG save/reload are not where the signal disappears; the carrier is already below the detection level at the sender site (−8.7 and −18.0 vs ≥ 0 needed).
- The additive carriers are weak at the source. γ = 0.35 reaches agreement 0.60 at LPIPS 0.155; λ = 0.10 at S4 reaches 0.53 at LPIPS 0.017. Idealized-carrier analysis (AUDIT.md B) puts detection at agreement ≈ 0.63 and exact decoding ≈ 0.67 for a unit-variance Gaussian carrier with the real key. That analysis is a requirement for that model, not a limit of EDICT.
- Encoder-sample and encoder-mean sites are equally good, so the VAE sampling noise is not the issue here. The 0.048 site error of the encoder-mean surrogate is the receiver-model error the optimizer sees; it did not change these two carriers' scores.
- No row says anything about whether a *different* perturbation, found by optimization, survives. That is the next report.

## 3. Shaping array 7783940 (corrected)

Validity of earlier numbers (AUDIT.md F): λ = 0.05 scores are valid; the λ = 0.15 PRC scores (sender −105, receiver −118) are **invalid** (cached codeword + freshly drawn key). Image-side values are valid: λ = 0.05 LPIPS 0.183, λ = 0.15 LPIPS 0.754. The previous claim "shaping does not impose a detectable PRC constraint" for λ = 0.15 is therefore not supported. For λ = 0.05 at σ² = 0.5 (−102 / −137) the scores are near random-carrier values. The geometry (`audit_cpu.json`): the correction is mostly an anti-aligned shrink of the noise amplitude; only 2.5 % (λ = 0.05) and 7.9 % (λ = 0.15) of signs flip.

## 4. Gradient gate (7788045) — see OPTIMIZATION_REPORT

All six gated tests passed.

## 5. Not measured / open

- The additive screen (7782006) never stored a margin; its agreement values (γ = 0.35: 0.613, λ = 0.10: 0.545) are consistent with D0.
- Shaping tasks 2–8 (λ ≥ 0.5) were cancelled and never evaluated.
- Nothing here tests a different photo, prompt, or key. No security claim is made from LPIPS or decoding.
