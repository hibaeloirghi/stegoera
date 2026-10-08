# Handoff status

Written 2026-10-07 22:50 EDT by the second agent. Everything below was read from local files, SLURM, or a job that ran this session. Previous summaries were treated as claims, not facts.

## Workspace

- Project `/nfshomes/eloirghi/prc-in-edict` (branch `master`, HEAD `25e4fc6`; many files uncommitted, including all `experiments/*` edits from earlier sessions and everything from this one). Nothing pushed.
- Scratch/env `/fs/clip-scratch/eloirghi/prc-in-edict/` (conda `edict-prc`, HF cache with SD-1.4 fp16, CLIP). Valid.
- Papers on disk: `/nfshomes/eloirghi/EDICT-paper.pdf`, `/nfshomes/eloirghi/gunn PRC image watermark.pdf`, chat export `/nfshomes/eloirghi/cursor_edict_and_prc_experiments.md`. **The OptMark PDF and `cursor_agent_chat_review_plan.md` were not found.** OptMark section numbers are therefore unverified (see `PAPER_CODE_MAP.md`). I did not invent any OptMark-to-code mapping. Please send the PDF if you want one.
- Cluster: `tron / nexus / qos=default`, 4 CPU / 32 GB, one `rtxa6000` per job (48 GB). A second GPU job does not start while one runs (queue shows Priority/Resources behind `ra-d0`).

## Queue at write time

| job | name | state | note |
|---|---|---|---|
| 7786379 | ra-d0 | RUNNING (2h02, limit 6h) | baseline case done (69 min/case); `top_g0.35` case in stage D; `s4_l0.10` still to run. Not touched. |
| 7787914 | ra-upreg | PENDING | upstream-vs-custom regression, S=10, edit prompt, per-event compare |
| 7788045 | ra-gradchk | PENDING | validation gate: pair-map reproduction, fp32-UNet dtype test, autograd vs FD (6 tests) |
| 7788181 | ra-r1smoke2 | PENDING, `afterok:7788045` | one R1 gradient smoke + true PNG evaluation + 2 controls |

Finished this session: 7787904 `ra-audit` (CPU, 1:47), 7787969 `ra-cpu2` (CPU, 0:20, passed), 7788093 `ra-rawscore` (CPU, **invalid output**, see below). Finished earlier: 7786378 `ra-cpu` (passed, coverage reviewed: mixing identities, signs, margin vs Detect on n=512, exact-message check).

## Findings that change how earlier numbers should be read

1. **`sigma` in all earlier tables is σ² (variance).** `recover_posteriors(v) = erf(z / sqrt(2 v (1+v)))` and Gunn App. C.1 Fact 1 uses σ² in that slot. `0.5` and `1.5` mean σ² = 0.5, 1.5 (σ = 0.707, 1.225). `0.5` is the bottom of the calibration grid, not a measured error. Margins at different σ² are not comparable (random carrier: ≈ −22 at 1.5, ≈ −110…−126 at 0.5).
2. **The PRC key is not reproducible from its seed.** One-time pad, test bits and generator matrix use unseeded `galois.GF.Random`. Consequences: D0's raw vectors cannot be re-scored offline (key not saved); my own re-score job 7788093 is invalid; **shaping task 1 (λ = 0.15) paired a cached codeword with a different key, so its PRC scores are invalid** (its LPIPS 0.754 is valid). New code saves the key. Details: `AUDIT.md` F.
3. **Shaping anomaly is geometric, not a rebuild bug.** λ = 0 rebuild error 4e-16. The correction is `λ(c − eps)` on 86 % of coordinates: cos(correction, eps) = −0.57, noise variance reduced 8 % (λ = 0.05) and 20 % (λ = 0.15), only 2.5 % / 7.9 % of signs flip, agreement with codeword 0.510 → 0.535 / 0.589. A random-direction blend of equal RMS has cos ≈ 0 and keeps variance. That the LPIPS jump is caused by the variance shrink is a hypothesis; no control has been run.
4. **What the unchanged PRC detector needs** (idealized unit-variance carrier + 0.13 noise, real key): detection at carrier–codeword correlation ≈ 0.3 (sign agreement ≈ 0.63), exact decoding at ≈ 0.4 (≈ 0.67). Top γ = 0.35 is below that on an ideal carrier already. This is about carrier model and key, not a universal threshold.
5. **`Detect` returns True for all-zero / tiny soft bits** (margin exactly 0, `>=`). Random carriers sit at −20, so an optimizer can raise the margin by shrinking the carrier. The R1 loss anchors carrier energy for this reason.
6. B3 (`7757586 b3-hook`) never produced a result: sacct shows it cancelled by this account after 8:43 of runtime. `ra-upreg` is its replacement.
7. 7783940 tasks 2–8 were cancelled by uid 17696 (this account) at 20:44:31 (sacct). I found no record of your approval in the files. Task 2 (λ = 0.50) ran 12 min and left no PNG or scores in `results/shape_stego`. They are **not evaluated**, not failed. I did not restart them.

## What is validated and what is not (honest)

Validated by a run: torch PRC margin equals `Detect`'s decision and numpy margin (n = 16384 key, 24 carrier cases) and autograd matches finite differences; idealized-carrier curve; shaping geometry; soft-bit convention.
Written, **not yet run on GPU**: upstream regression, fp32-UNet dtype test, autograd FD through the model, R1 smoke. No claim is made about any of them until their jobs finish.

## Security / hygiene

- **Please rotate the Hugging Face token that appeared in the earlier exported chat** if not already done. I did not read or print any credential. `.hf_token` is gitignored and was not opened.
- No tokens were added to code, logs or reports.

## Next, in order

1. Let `ra-d0` finish (≈ 00:10 EDT). It is the only owner of the GPU slot.
2. `ra-upreg` and `ra-gradchk` run when the slot frees; read `results/receiver_aligned/upstream_regression/summary.json` and `gradcheck.csv`.
3. `ra-r1smoke2` starts only if the gradcheck gate passes.
4. Update `DIAGNOSTIC_REPORT.md` and `OPTIMIZATION_REPORT.md` with whatever those produce. Do not submit R2/R3/R4 or a new strength grid before that.
