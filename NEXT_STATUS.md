# NEXT_STATUS: Message Recovery Through EDICT Coupling-Layer PRC Embedding

Written 2026-10-08 by the taking-over agent upon workspace audit.

## 1. Actual SLURM Queue

- Verified with `squeue -u eloirghi` on 2026-10-08:
- **Queue status: EMPTY.** No running, pending, or suspended jobs.
- All historical jobs (7786379, 7787914, 7788045, 7788181, 7793363, 7793436) are in terminal state `COMPLETED`.

## 2. Latest Complete Outputs (Verified Artifacts)

| Job ID | Script / Experiment | Key Artifacts | Outcome Summary |
|---|---|---|---|
| 7786379 | `d0.sh` (D0 ladder) | `results/receiver_aligned/d0_ladder.csv` (41 rows) | Baseline, top $\gamma=0.35$, S4 $\lambda=0.10$. Pair-only MSE $4.9 \times 10^{-29}$, PNG decoder inversion MSE $\approx 0.0167$. Carrier evidence already deficient before pixel recovery. |
| 7787914 | `upstream_regression.sh` | `results/receiver_aligned/upstream_regression/` | Custom EDICT loop vs upstream EDICT: max absolute difference = 0.0 across all 32 inversion and 32 generation events ($S=10$). |
| 7788045 | `gradcheck.sh` | `results/receiver_aligned/gradcheck_summary.json` | All 6 gradient gate tests passed. fp32 UNet within $4.1 \times 10^{-6}$ max abs diff of fp64, 16.4x faster. Autograd vs FD relative error 0.13%. |
| 7788181 | `r1_smoke_v2.sh` (R1 smoke) | `r1_smoke_meta_7788181.json`, `r1_smoke_7788181_true_rows_aligned.csv`, `saved_states/r1_smoke_7788181_key.pkl`, `raw_observations/r1_smoke_7788181_true_site_carrier.pt` | True saved PNG: Detect=True (margin +67.7 at $v=1.5$), Decode=None, exact message=False, sign agreement 0.493, LPIPS vs unwatermarked edit 0.0154, CLIP 0.252 vs 0.250. Wrong-key and baseline controls rejected. |
| 7793363 | `analyze_r1_word.sh` | `r1_word_analysis_7793363.json` | Shows R1 smoke stego carrier raised parity pass-rate from 0.496 to 0.545 across 15,777 checks (12 SE above chance), achieving Detect without matching the embedded codeword. |
| 7793436 | `r1_codeword_v3.sh` | `r1_codeword_meta_7793436.json`, `r1_codeword_7793436.csv`, `saved_states/r1_codeword_7793436_key.pkl`, `raw_observations/r1_codeword_7793436_true_site_carrier.pt` | True saved PNG: Detect=True (margin +112.3 at $v=1.5$), Decode=None, exact message=False, sign agreement 0.535, LPIPS vs edit 0.0265, CLIP 0.236 vs 0.250. Controls rejected. Union-schema CSV verified clean. |

## 3. Saved-Key Availability

Both receiver-aligned optimization runs have complete, verified saved keys in `results/receiver_aligned/saved_states/`:
- `r1_smoke_7788181_key.pkl`: key hash `otp655ddd0a246a_tb6408441c0324_G3e417b827611`, contains `enc_key`, `dec_key`, `message` (64 bits), `codeword` (shape 16384).
- `r1_codeword_7793436_key.pkl`: key hash `otp0bc2fe711d81_tbf2f9c57f6ecb_G6cdd1c80818b`, contains `enc_key`, `dec_key`, `message` (64 bits), `codeword` (shape 16384).
- Respective raw true-carrier observations: `r1_smoke_7788181_true_site_carrier.pt` and `r1_codeword_7793436_true_site_carrier.pt` in `results/receiver_aligned/raw_observations/`.

## 4. Current Source / Config Hashes

- **EDICT git:** `a11504ae7902763655f4247335454be2f3b37499`
- **PRC-Watermark git:** `4f8e09bc0590e4e8d89cc23888b68e559417748a`
- **Experiment repo HEAD:** `5ac51f4`
- **Config hashes:**
  - `r1_smoke`: `31e19faf2fa6ef12`
  - `r1_codeword`: `855183c16b1e4861`
- **Source photo:** `experiments/photos/imagenet_dog_1.jpg`
- **Baseline unwatermarked edit PNG:** `results/edit_stego/baseline.png` (SHA-256 starts `ee8d944a...`)

## 5. Known Defects, Clarifications, and Hygiene

1. **PRC KeyGen Entropy:** `KeyGen` in `PRC-Watermark/src/prc.py` samples $G$, OTP, and test bits via unseeded `galois.GF.Random`. A random seed does not reproduce the key. Saved serialized key files (`.pkl`) with SHA-256 integrity hashes are mandatory and used throughout all subsequent tests.
2. **Payload Capacity vs Message Slicing:**
   - In `KeyGen`, `key_message_length = 512` is passed, giving code dimension $k = 512 + 39 + 17 = 568$.
   - In `Encode`, payload = `test_bits` (17) || `Random(g)` (39) || `message` (64) || `zeros` (448).
   - In `Decode`, the solver returns all 512 bits following `test_bits` and randomizer bits.
   - In `exact_message_match`, only the first 64 bits are compared (`decoded[:64] == message`).
   - E0 explicitly separates full 512-bit payload equality from 64-bit intended-message equality, and includes a correctly sized 64-bit key control (`key_message_length = 64`).
3. **Detection Margin vs Codeword Alignment:**
   - Both optimized runs achieved large positive detector margins (+67.7 and +112.3) by optimizing the parity statistic without sufficient codeword alignment (agreement 0.493 and 0.535).
   - BP-OSD decoding requires substantial sign agreement ($\approx 0.67$ on idealized carriers).
   - Optimization must penalize/constrain codeword mismatch directly rather than rewarding unbounded detector margin.
4. **Credential Hygiene Reminder:**
   - A Hugging Face token appeared in an earlier chat export. Please ensure this token is rotated if it has not already been rotated.
   - The token in `.hf_token` is never printed, logged, or read into conversation context.

## 6. Staged Next Plan

- **E0 (CPU):** Saved-key, payload, and decoder audit. Load true keys and carriers, test positive controls (genuine codeword, PRC-signed Gaussian, controlled noisy observations), synthetic error tolerance, full vs sliced payload equality, and correctly sized key control.
- **E1 (CPU):** Direct-carrier projection curve. Rank mismatches by carrier correction cost, compute empirical decoding threshold as a function of correction budget and carrier MSE.
- **E2 (GPU):** Codeword-only vs saturated-detector objectives at S4 site.
- **E3 (GPU, conditional):** Sender-site minimum-cost projection control.
- **E4 (GPU, conditional):** Reduced-carrier trial if full 16,384-dim carrier remains unmanageable.
