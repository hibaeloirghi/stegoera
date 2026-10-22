# DECODER_AUDIT: Saved-Key, Payload, Decoder, and Projection Audit

Completed 2026-10-08 via SLURM job `7794039` on `tron62`.
Artifacts:
- `results/next_experiments/e0_audit_table.csv` (164 records)
- `results/next_experiments/e0_audit_summary.json`
- `results/next_experiments/e1_projection_table.csv` (324 records)
- `results/next_experiments/e1_projection_summary.json`

---

## 1. Executive Summary

1. **Both saved keys and genuine codewords are 100% valid and decodable:**
   All positive controls—direct codeword signs, PRC-signed Gaussian samples, and noisy observations up to noise $\sigma_c = 0.70$ (sign agreement $\ge 0.846$)—achieved **100% exact message match and full payload match** on original `Detect` and `Decode`.
2. **Decoding failure is NOT a bug in keys, padding, slicing, or carrier scale:**
   - Perfect signs decode even with microscopic carrier confidence ($\text{mean } |\text{soft}| = 0.0041$, margin $+0.0$).
   - Full 512-bit payload match and 64-bit intended message match are completely aligned.
   - Sizing the key to exactly 64 bits ($k=120$, no trailing zero padding) yields identical decoding thresholds.
3. **The true root cause of decoding failure in jobs 7788181 and 7793436 is insufficient codeword agreement:**
   - Synthetic error sweeps and oracle projection curves establish that **BP-OSD decoding requires sign agreement $\ge 0.75 - 0.81$**.
   - Job 7788181 (R1 smoke) had sign agreement **0.4927** (pure chance).
   - Job 7793436 (R1 codeword) raised sign agreement to **0.5352**, which is still far below the $\approx 0.75$ threshold.
4. **Detector margin was severely over-rewarded and decoupled from decoding:**
   - Both runs achieved large positive detector margins (+67.7 and +112.3) because the detector loss drove the parity product statistic without aligning individual coordinate signs with the codeword.
   - In E1 direct-carrier projection, decoding succeeds at margin $\approx +80$ once agreement reaches $0.814$, whereas the previous optimizer reached margin $+112$ at agreement $0.535$.

---

## 2. Key Architecture, Dimensions, and Payload Verification

| Parameter | Saved Key 7788181 | Saved Key 7793436 | Correctly Sized Control |
|---|---|---|---|
| Key Hash | `otp655ddd0a246a_tb6408441c0324_G3e417b827611` | `otpc3515754f545_tb9667a8058dba_Gcda3c4902d1b` | `otp4f4a1065b1b9_tb5f75e9162ae1_Gf12b32f594b4` |
| Carrier dimension $n$ | 16384 | 16384 | 16384 |
| Generator cols $k$ | 568 | 568 | 120 |
| Parity checks $r$ | 15777 | 15777 | 16225 |
| Check weight $t$ | 3 | 3 | 3 |
| Test bits ($\lceil \log_2(1/F) \rceil$) | 17 | 17 | 17 |
| Randomizer bits $g$ | 39 | 39 | 39 |
| Sent message length | 64 | 64 | 64 |
| Key message capacity | 512 | 512 | 64 |
| Zero padding | 448 zeros | 448 zeros | **0 zeros** |
| Noise rate | 0.017616 | 0.017616 | 0.017616 |
| Parity constraint $H G = 0 \pmod 2$ | **True** | **True** | **True** |
| Codeword syndrome pass rate | 0.9506 | 0.9506 | 0.9506 |

### Payload Slicing Analysis
In `PRC.Encode`:
$$\text{payload} = \text{test\_bits (17)} \,\|\, \text{Random}(g=39) \,\|\, \text{message (64)} \,\|\, \text{zeros (448)}$$
In `PRC.Decode`:
The decoder solves for $\text{recovered\_string}$, verifies $\text{recovered\_string}[:17] == \text{test\_bits}$, and returns $\text{recovered\_string}[17+39:]$, which has length 512.
In `prc_scores.exact_message_match`, `decoded[:64] == message` is checked.
In all positive control tests where `Decode` returned non-None:
- `exact_message_match`: **True** (first 64 bits match).
- `full_payload_match`: **True** (all 512 bits match, including the 448 zero bits).
Hence, the padding structure does not cause any decoding anomaly.

---

## 3. Evaluation of True Recovered PNG Carriers

Evaluating the actual recovered carriers saved from the PNG decoder-inversion pipeline:

| Run | Variance $v=\sigma^2$ | Detect | Margin | Decode Non-None | Exact Message Match | Full Payload Match | Sign Agreement | Mean Confidence |
|---|---|---|---|---|---|---|---|---|
| `r1_smoke_7788181` | 1.5 | **True** | **+67.73** | False | False | False | **0.4927** | 0.2859 |
| `r1_smoke_7788181` | 0.5 | **True** | **+382.94** | False | False | False | **0.4927** | 0.5247 |
| `r1_codeword_7793436` | 1.5 | **True** | **+112.33** | False | False | False | **0.5352** | 0.2812 |
| `r1_codeword_7793436` | 0.5 | **True** | **+639.16** | False | False | False | **0.5352** | 0.5179 |

Both carriers easily pass `Detect` (positive margin), but fail `Decode` because sign agreement with the target codeword is essentially chance (0.493 and 0.535).

---

## 4. Positive Controls (Validation of Original Code)

| Control Case | Key | Variance $v$ | Detect | Margin | Decode | Exact Msg | Full Payload | Sign Agree |
|---|---|---|---|---|---|---|---|---|
| PC1: Codeword Direct | 7788181 | 1.5 | True | +8161.1 | **True** | **True** | **True** | 1.0000 |
| PC1: Codeword Direct | 7788181 | 0.5 | True | +8161.1 | **True** | **True** | **True** | 1.0000 |
| PC2: PRC Gaussian Sample | 7788181 | 1.5 | True | +331.0 | **True** | **True** | **True** | 1.0000 |
| PC2: PRC Gaussian Sample | 7788181 | 0.5 | True | +2048.1 | **True** | **True** | **True** | 1.0000 |
| PC1: Codeword Direct | 7793436 | 1.5 | True | +8161.1 | **True** | **True** | **True** | 1.0000 |
| PC2: PRC Gaussian Sample | 7793436 | 1.5 | True | +331.0 | **True** | **True** | **True** | 1.0000 |
| Short Key PC1 ($k=120$) | short | 1.5 | True | +8323.8 | **True** | **True** | **True** | 1.0000 |
| Short Key PC2 ($k=120$) | short | 1.5 | True | +345.1 | **True** | **True** | **True** | 1.0000 |

### Noisy Observations Sweep (PC3 on Key 7793436, $v=1.5$)
Carrier: $h = \sqrt{1 - \sigma_c^2} \cdot c + \sigma_c \cdot w$.

| Noise Level $\sigma_c$ | Sign Agreement | Margin | Detect | Decode | Exact Message | Full Payload |
|---|---|---|---|---|---|---|
| 0.00 | 1.0000 | +751.0 | True | True | True | True |
| 0.20 | 1.0000 | +696.6 | True | True | True | True |
| 0.35 | 0.9958 | +576.1 | True | True | True | True |
| 0.45 | 0.9762 | +492.9 | True | True | True | True |
| 0.50 | 0.9555 | +449.4 | True | True | True | True |
| 0.60 | 0.9092 | +342.9 | True | True | True | True |
| 0.70 | 0.8511 | +242.1 | True | True | True | True |

**Verdict on Positive Controls:** 100% of positive controls decode flawlessly. The decoder and keys are verified sound.

---

## 5. Synthetic Decoder Diagnostic Tests

### Test A: Controlled Sign Error Rate (Key 7793436, Constant Confidence $A=0.8$)
| Bit Error Rate $p_{\text{err}}$ | Sign Agreement | Margin | Detect | Decode | Exact Message |
|---|---|---|---|---|---|
| 0.000 | 1.0000 | +5241.5 | True | True | True |
| 0.050 | 0.9522 | +3781.0 | True | True | True |
| 0.100 | 0.9007 | +2556.1 | True | True | True |
| 0.140 | 0.8651 | +1935.6 | True | True | True |
| 0.180 | 0.8204 | +1198.8 | True | True | True |
| 0.200 | 0.8014 | +882.4 | True | True | True |
| **0.250** | **0.7497** | **+502.8** | **True** | **True** | **True** |
| 0.300 | 0.6947 | +37.8 | True | **False** | **False** |
| 0.350 | 0.6542 | -136.9 | False | False | False |

### Test B: Sign Error Rate with Realistic Carrier Amplitudes $|h_{\text{true}}|$
| Bit Error Rate $p_{\text{err}}$ | Sign Agreement | Margin | Detect | Decode | Exact Message |
|---|---|---|---|---|---|
| 0.000 | 1.0000 | +275.7 | True | True | True |
| 0.100 | 0.9011 | +130.4 | True | True | True |
| 0.140 | 0.8613 | +98.1 | True | True | True |
| 0.180 | 0.8228 | +50.0 | True | True | True |
| 0.200 | 0.7949 | +29.9 | True | True | True |
| **0.250** | **0.7518** | **+21.7** | **True** | **False** | **False** |
| 0.300 | 0.6942 | -0.4 | False | False | False |

### Test C: Confidence Magnitude Sweep with 100% Perfect Signs
| Magnitude $A$ | Mean Soft Confidence | Margin | Detect | Decode | Exact Message |
|---|---|---|---|---|---|
| 0.01 | 0.0041 | +0.0 | True | **True** | **True** |
| 0.05 | 0.0206 | +0.1 | True | **True** | **True** |
| 0.10 | 0.0412 | +0.9 | True | **True** | **True** |
| 0.30 | 0.1231 | +22.8 | True | **True** | **True** |
| 0.50 | 0.2037 | +103.4 | True | **True** | **True** |
| 1.00 | 0.3944 | +751.0 | True | **True** | **True** |

**Crucial Finding:** Carrier magnitude is NOT a bottleneck. If signs are aligned, the decoder succeeds even at microscopic confidence ($A=0.01$).

### Test D: Structured Error Placement (Random vs Least Confident vs Most Confident)
At $p_{\text{err}} = 0.10$ (90% agreement):
- **Least Confident bits flipped:** Margin = +261.4, **Decode = True, Exact Message = True**.
- **Random bits flipped:** Margin = +127.3, **Decode = True, Exact Message = True**.
- **Most Confident bits flipped:** Margin = +24.1, **Decode = False, Exact Message = False**.

This shows BP-OSD is resilient when small errors happen on low-confidence coordinates, but brittle if high-confidence coordinates have wrong signs.

---

## 6. Experiment E1: Direct-Carrier Projection Curve

On the recovered carrier of `r1_codeword_7793436` (initial agreement 0.5352, margin +112.3), we rank mismatches $c_i \cdot h_i < 0$ by squared correction cost $(\tau + |h_i|)^2$ and apply target values $h_i' = c_i \cdot \tau$:

### Target Scale $\tau = 1.0$ (Cheapest Mismatches First)
| Fraction Corrected $b$ | Coords Corrected | Sign Agreement | Disp RMS | Margin | Detect | Decode | Exact Msg | Full Payload |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0 | 0.5352 | 0.0000 | +112.3 | True | False | False | False |
| 0.10 | 762 | 0.5817 | 0.2282 | +109.9 | True | False | False | False |
| 0.20 | 1523 | 0.6281 | 0.3394 | +103.7 | True | False | False | False |
| 0.30 | 2285 | 0.6746 | 0.4373 | +96.9 | True | False | False | False |
| 0.40 | 3046 | 0.7211 | 0.5313 | +84.9 | True | False | False | False |
| 0.50 | 3808 | 0.7676 | 0.6245 | +78.2 | True | False | False | False |
| **0.60** | **4570** | **0.8141** | **0.7195** | **+83.9** | **True** | **True** | **True** | **True** |
| 0.70 | 5331 | 0.8605 | 0.8189 | +109.6 | True | True | True | True |
| 0.80 | 6093 | 0.9070 | 0.9263 | +175.3 | True | True | True | True |
| 1.00 | 7616 | 1.0000 | 1.2160 | +480.7 | True | True | True | True |

### Target Scale $\tau = 0.8$ (Cheapest Mismatches First)
- First decode at $b = 0.70$, agreement **0.8605**, displacement RMS **0.7069**, margin **+88.2**.

### Target Scale $\tau = 0.5$ (Cheapest Mismatches First)
- First decode at $b = 0.80$, agreement **0.9070**, displacement RMS **0.6308**, margin **+83.0**.

### Comparison Across Strategies at First Decode
| Target Scale $\tau$ | Strategy | First Decode Budget $b$ | Agreement | Disp RMS | Margin |
|---|---|---|---|---|---|
| 1.0 | Cheapest First | 0.60 | 0.8141 | 0.7195 | +83.9 |
| 0.8 | Cheapest First | 0.70 | 0.8605 | 0.7069 | +88.2 |
| 0.5 | Cheapest First | 0.80 | 0.9070 | 0.6308 | +83.0 |
| 0.5 | Random Mismatches | 0.60 | 0.8141 | 0.6897 | +61.9 |

---

## 7. Implications for GPU Optimization (E2)

1. **Stop pushing detector margin past $+20$:**
   In E1, decoding succeeds with detector margins of $+60 \dots +85$. The previous optimizer drove margin to $+112$ with only $0.535$ agreement. Margins $> +30$ do not help decoding.
2. **Focus optimization power on Codeword Alignment ($L_{\text{cw}}$):**
   In previous jobs, $a_{\text{hinge}} = 0.1$ gave negligible gradient signal compared to $L_{\text{margin}}$. In E2, Trial A will completely drop the detector term ($b=0$) and allocate all gradient force to $L_{\text{cw}}$. Trial B will saturate the detector term at $m_{\text{target}} = 20.0$.
3. **Target Agreement Threshold:**
   To cross into decodability, the optimization must move sign agreement from $0.535$ past $\approx 0.75 - 0.80$.
