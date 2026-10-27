# Guidance diagnostic plan

Stage 0 only. No sender was rerun and no result file was overwritten. This note records the two pipelines as they are. It does not change either one.

Branch `exp/ip2p-edict-prc-guidance-diag` was created locally from `exp/ip2p-edict-prc` at `f51dcdb`. It was not pushed. Unrelated dirty files, including the files imported by running job `8031477` (`ip2p-smrec`, short-message recovery), stay unstaged and are not edited.

The frozen PNG receiver is `recover_latent` in `experiments/ip2p_edict_prc/run_png_decoder_inversion.py` as of `f51dcdb`: Adam, learning rate 0.1, 100 iterations, cosine warmup 10, sum of squared error, posterior-mode initialization, checkpoints 0/25/50/100, iteration 100 primary, `sigma^2 = 1.5`. The uncommitted edit to that file only adds an optional config path and a `require_mode_failure` flag. It does not change `recover_latent`. This diagnostic calls `recover_latent` and does not go through `main()`, so that flag is not used. The running job imports the file, so the file is not edited.

## Verified facts

Checked against local artifacts on 27 Oct 2026.

- Old pipeline, `experiments/run_full_prc_sign.py`, table `results/full_prc_sign/all_experiments.csv` (120 data rows, sha256 `bd4aa61d…`). All 40 watermarked PNG rows have Detect, Decode, and exact message true. All 10 unwatermarked PNG rows have Detect and exact message false. Those 10 are the `nowm` rows the script still inverts (`guide == null` and `target in {null, style}`). Oracle latent rows for watermarked edits have sign agreement 1 on both streams.
- The old PNG receiver VAE-encodes the saved PNG with `latent_dist.sample` (seed 1), multiplies by 0.18215, copies that one latent into both streams, and runs the reverse coupled loop at strength 1. It does not use the posterior mode.
- Hybrid true final pair, text 7.5 and image 1.5: both streams Detect, Decode, and match the message. Golden-retriever job `7951673`, red-ball job `7960674`. Noise MSE is about `3e-28`.
- Exact final `x` duplicated as `(x, x)`: golden-retriever job `7960756` returns the exact message on both streams. Detect is false on stream 0 (margin −2.03) and true on stream 1 (margin +2.20). Red-ball job `8020824` fails Detect, Decode, and exact message on both streams.
- Golden-retriever PNG, 100-step decoder inversion, job `8020189`: exact message true on both streams, Detect false. Red-ball PNG, job `8020596`: Detect, Decode, and exact message false at every predeclared checkpoint. VAE-mode rows fail on both images.
- Source conditioning was the saved three-branch conditioning in both hybrid tests. A missing source latent does not explain the red-ball failure. Duplicating `x` is already enough to lose the red-ball message, while the true pair still recovers.
- Saved forward terminal `x−y` MSE is 0.006421 on the golden-retriever edit and 0.00758 on the red-ball edit. That is the 0.0064 to 0.0076 range named in the prompt.

Reference artifacts that this diagnostic reuses, and does not regenerate:

| Artifact | Path | sha256 |
| --- | --- | --- |
| Source photo | `EDICT/experiment_images/imagenet_dog_1.jpg` | `ece278b0e08239c2cf8d6876d534a20e0e56c593fc7d70e2d828c0fec8151ca0` |
| Hybrid key | `results/instructpix2pix_prc/pair/pair_encoding.pkl` | `2e53acc14b9605a8fbfb09027435c4f8f7b96e04c0770178bab38008e973d8a2` |
| Hybrid noise | `results/instructpix2pix_prc/pair/raw_noise.npz` | `063cadb3e4f247c720d92d86d84777b3ef8b8c16c0b5ff00d7d3aa0ef094848e` |
| Wrong key | `results/instructpix2pix_prc/raw_keys/cpu_controls_wrong_key.pkl` | `b78f6e8584986143191ba452b1aa3db2a13d474f5283441fa92fcd01e35022a1` |
| Golden pair | `results/ip2p_edict_prc/milestone/pairs.pt` | `3d35a455853a5640b6f8e7fab2d9135844c18832dc30d5b6b7acfe735e392ead` |
| Golden PRC PNG | `results/ip2p_edict_prc/milestone/images/prc_stream0.png` | `762f41188780613635e266f97659b64db6bae2167ed6b81c16e4928d92bd25b2` |
| Golden Gaussian PNG | `results/ip2p_edict_prc/milestone/images/gaussian_stream0.png` | `c1ecb11d1dfec14985dba92b1faa6bd5a45295664c02ec672cb65bfc731936c9` |
| Red-ball pair | `results/ip2p_edict_prc/red_ball/pairs.pt` | `45eac3ae8fee58f4aa342ba7f3c2d2a781a4306a0e490c75b5b0c42d3f16f74e` |
| Red-ball PRC PNG | `results/ip2p_edict_prc/red_ball/images/prc_stream0.png` | `65ed30b0a077cde3369b081004a7e18d7e3dac3fe9f0df3dc87139d3df06076c` |
| Red-ball Gaussian PNG | `results/ip2p_edict_prc/red_ball/images/gaussian_stream0.png` | `ba2e119e8c983832a478481457e529691fc64f446ef96ada6f5b7ffdee479906` |
| Old key | `results/full_prc_sign/key.pkl` | `057d4b298cfa19b7bd877a4cda8bb27a1a5127216b3f684a2ffd454be3237c61` |
| Old dog1 PNG | `results/full_prc_sign/images/imagenet_dog_1__caption__small__prc.png` | `afa8f2fe6262241c0ce6b7e07ecc658166206946ab6b7b360e1bd4c95d75a3ee` |

Key id for the hybrid, from the encoding key and not from a new `KeyGen`: `otpbc4dafb803c7_tba80a78948303_Gba71308104b6`. The old key is a different key. Its dog1 codeword sha256 is `0d60f396382d6acadfdeabfe38094577502b8908ea1c1721658140a9a7a4c1e4`. Message length is 64 in both, but the message hashes differ. Nothing here prints a message, a token, or a key.

Cluster check: account `nexus`, QOS `default`, partition `tron`. RTX 2080 Ti and A6000 are both eligible. Job `8031477` is running on one 2080 Ti and is not cancelled. Scratch `/fs/clip-scratch/eloirghi/prc-in-edict` and conda env `edict-prc` exist.

## Old pipeline: SD 1.4 + EDICT + PRC

Code: `experiments/edict_runtime.py` and `experiments/run_full_prc_sign.py`. The arithmetic below was read from those files and from `EDICT/edict_functions.py`.

Guidance. Two UNet calls, then

```text
noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
```

`experiments/edict_runtime.py` lines 124–126. The same lines in the original EDICT function are `EDICT/edict_functions.py` lines 1021–1023: `grad = noise_pred_cond - noise_pred_uncond`, then `noise_pred_uncond + guidance_scale * grad`. This is uncond + G·(cond − uncond). The conditional branch is the text embedding. The unconditional branch is the empty prompt. There is no image-conditioning branch. The historical scale is the constant `GUIDANCE = 3.0` (`run_full_prc_sign.py` line 37), passed as `guidance_scale` for the photo inversion, the generation, and the PNG inversion (`edict_kwargs`, lines 147–156).

Two shortcuts do not change that formula when it is used. If the prompt equals the empty prompt, only the unconditional call is made (lines 116–119). If `guidance_scale == 1`, only the conditional call is made (lines 120–123). That second shortcut equals the formula, because uncond + 1·(cond − uncond) = cond. The dog1 row chosen below has a non-empty invert prompt and a non-empty generate prompt, so both passes use the two-call formula.

Steps and grid. `STEPS = 50`, `STRENGTH = 1.0`, `MIX = 0.93` (lines 34–36). `t_limit = steps - int(steps * init_image_strength)` is 0 (`edict_runtime.py` line 160), so the walk is the full scheduler grid. `EDICT/my_diffusers/schedulers/scheduling_ddim.py` lines 161–163 build `arange(0, 1000, 1000 // 50)[::-1]`, which is 980, 960, …, 0. Generation walks that order. The inverse flips it (line 162–163). The CSV row used below has `t_final = 980` and `n_steps = 50`.

Mixing and alternation. Forward, after both affine updates: `x <- p x + (1-p) y`, then `y <- (1-p) x_new + p y` (lines 237–239). Inverse, before the affine updates: recover `y`, then recover `x` (lines 196–197). Leapfrog uses the same index rule as EDICT: forward offset `i % 2` (lines 216–218); inverse `orig_i = len(timesteps) - (i+1)`, `offset = (orig_i+1) % 2` (lines 209–212). Noise is predicted on the partner stream.

Dtypes and VAE. UNet, VAE, and CLIP text are loaded as fp16 and then `.double()` (`EDICT/edict_functions.py` lines 82–91), so the denoiser runs in float64. Latents are float64 (line 180 of `edict_runtime.py`). `VAE_SCALE = 0.18215` (line 18). Encode is `vae.encode(image).latent_dist.sample(generator) * 0.18215` (lines 70–79), not the posterior mode. Decode divides by 0.18215 (lines 84–86). The generator is `torch.cuda.manual_seed(seed)` with `SEED = 1`.

How `y` starts. The source photo is center-cropped and Lanczos-resized to 512 (`experiments/image_utils.py` lines 8–15). `invert_image` samples one VAE latent, copies it into both streams (`edict_runtime.py` lines 268–275 and 170–173), and runs the reverse loop. The two resulting streams are then signed separately: `c * |z|` on each (`run_full_prc_sign.py` lines 103–106 and 222–223). Generation starts from that signed pair (`generate_from_pair`, lines 280–283). The streams share codeword signs and do not share magnitudes. The source of the inversion is `EDICT/experiment_images/imagenet_dog_1.jpg`, not a watermarked PNG.

Key and noise. The historical process wrote `results/full_prc_sign/key.pkl` (encoding key, decoding key, wrong decoding key, 64-bit message, one codeword per photo, variance 1.5, steps 50, strength 1, mix 0.93). This diagnostic loads that file. It does not call `KeyGen` again. `KeyGen` is not reproducible from its seed (`experiments/receiver_aligned/key_io.py`).

Receiver conditioning. The PNG receiver opens the saved PNG and calls `invert_image` with the generation prompt (`run_full_prc_sign.py` lines 269–274). For the row below, that prompt is the “small” caption, not the invert caption and not an InstructPix2Pix instruction.

Historical row, predeclared before any new run. Image `imagenet_dog_1`, guide `caption`, target `small`, watermark `prc`, channel `png`. Invert prompt: “a brown and white spaniel standing on a dirt path in front of green bushes”. Generate prompt: “a brown and white spaniel wearing a red collar, standing on a dirt path in front of green bushes”. Guidance 3, 50 steps, strength 1, mix 0.93. Detect true, Decode true, exact message true, sign agreement x = 0.9219970703125, y = 0.924560546875, margin 313.22562123516946. This row is the regression because its prompts are non-empty, so scale 3 enters the formula. It was not chosen by trying new configurations. The CSV’s Detect and Decode are computed from stream x only (`score_latent`, lines 128–140); both sign agreements are stored. All eight dog1 watermarked PNG rows recovered. This one is the replay target.

The historical job requested an A6000 (`experiments/slurm/full_prc_sign.sh`) because the UNet is float64. Stage 1’s old-pipeline replay and Stage 3 use one A6000 for that reason. Hybrid jobs stay on one RTX 2080 Ti, which is the card the hybrid reference used. A float64 replay on a 2080 Ti would be a different numerical setup from the row it has to match.

## Hybrid: InstructPix2Pix denoiser + EDICT coupling

Code: `experiments/ip2p_edict_prc/sampler.py` and `make_denoise` in `experiments/ip2p_edict_prc/run_milestone.py`. The installed Diffusers pipeline is `diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py` in the `edict-prc` env.

Guidance. One UNet call on three concatenated branches, then the installed formula (pipeline lines 357–363), copied in `make_denoise` lines 104–108:

```text
noise = noise_uncond
        + guidance_scale * (noise_text - noise_image)
        + image_guidance_scale * (noise_image - noise_uncond)
```

Expanded, the branch weights are `s_T` on the text-and-image branch, `s_I - s_T` on the image-only branch, and `1 - s_I` on the empty branch. Reference values are `s_T = 7.5` and `s_I = 1.5` (`milestone.json`). That is 7.5·text − 6·image − 0.5·uncond. It is not uncond + 7.5·(cond − uncond), and it is not “guidance 3”.

Branch order. Text embeddings are `[prompt, negative, negative]` (pipeline line 547). Image latents are `[source, source, zeros]` (pipeline line 699). `noise.chunk(3)` is therefore text, image, uncond. The negative prompt in the saved runs is the empty string. The source latent in that concat is the VAE posterior mode and is not multiplied by 0.18215 (pipeline line 677). The diffusion latent that is decoded is divided by `scaling_factor` 0.18215 (`run_milestone.py` lines 114–117). Those are different conventions inside one pipeline. This diagnostic does not change them.

The pipeline turns three-branch guidance on only when `guidance_scale > 1.0` and `image_guidance_scale >= 1.0` (line 265). `__call__` accepts the two scales as floats (lines 144–145). G1 through G4 all satisfy the gate. `make_denoise` does not consult the gate. It always evaluates three branches and always applies the formula, including a zero empty-branch coefficient when `s_I = 1`.

Steps and grid. 50 steps. `sampler.py` lines 48–49 build `range(0, 1000, 20)[::-1]`, the same 980…0 grid. `init_image_strength` is present in the JSON and is not an argument of `coupled_loop`. The hybrid always walks the full grid. At strength 1 the old pipeline also walks the full grid, by a different mechanism.

Mixing and alternation. `mix` and `unmix` (`sampler.py` lines 117–143) match the old in-place order, including the second mix reading the updated first stream. Leapfrog forward and inverse (`sampler.py` lines 165–175) match the old index rule. Affine math is float64. The partner-stream noise is whatever `denoise` returns.

Dtypes. Latent pair float64 (`sampler.py` line 159). UNet and VAE are loaded `torch.float16` (`run_milestone.py` lines 174–176). The partner is cast to fp16 only for the UNet call (line 96) and the affine step casts back to float64.

How `y` starts. Both streams are clones of one fresh noise tensor (`run_milestone.py` line 122). For the PRC row that noise is `z_prc` from `raw_noise.npz`, the saved codeword times Gaussian magnitudes. The source photograph is image conditioning only. It is not the start of either stream.

Key and receiver conditioning. `load_key` on `pair_encoding.pkl`, checked against the expected hash. The inverse is given the saved `image_latents` and `prompt_embeds` from `pairs.pt` and the instruction those embeddings were built from. It is not given `y` or the initial noise when the test is a duplicated-`x` or PNG test. The true-pair inverse is given the saved final pair.

## Differences

| Item | Old EDICT+PRC | Hybrid |
| --- | --- | --- |
| Denoiser | SD 1.4, 4 input channels | InstructPix2Pix, 8 input channels |
| Guidance | uncond + G·(cond − uncond) | uncond + s_T·(text − image) + s_I·(image − uncond) |
| Scale | one scale, 3.0 | text 7.5 and image 1.5 |
| Branches that receive a weight | empty text, and the text prompt | empty, image-only, text-and-image |
| Steps | 50 | 50 |
| Grid | 980, 960, …, 0 | 980, 960, …, 0 |
| Mix `p` | 0.93, y uses the new x | 0.93, y uses the new x |
| Leapfrog | EDICT index rule | same index rule |
| Latent dtype | float64 | float64 |
| UNet dtype | float64 after `.double()` | float16 |
| VAE of the diffusion latent | sample · 0.18215, decode divides by 0.18215 | decode divides by 0.18215 |
| Image conditioning | none inside the UNet | posterior mode, not multiplied by 0.18215, concatenated on the channel axis |
| Start of `(x, y)` | inverted photo, then each stream signed `c·\|z\|` | both streams cloned from one fresh `z_prc` |
| What the edit changes | the generation prompt | an instruction, with the photo held as conditioning |
| PNG to latent | posterior sample · 0.18215, seed 1 | posterior mode · 0.18215, then optional frozen Adam inversion |
| Key | `results/full_prc_sign/key.pkl` | `pair_encoding.pkl` |

Scale 3 in the old formula and text scale 3 in the hybrid formula are not the same number in two units. Stage 3 says this again next to the 7.5 result.

## What is measured

Trace columns, predeclared: `pipeline`, `edit`, `guidance_config`, `path`, `step`, `t`, `mse_xy`, `maxabs_xy`, `norm_ratio`, `state_err_x`, `state_err_y`, `state_corr_x`, `state_corr_y`.

`path` is `F` (forward), `I` (inverse of the true final pair), or `D` (inverse of exact final `x` copied into both streams). Step 0 is the snapshot before any update of that pass. `t` is blank at step 0 and is the timestep just applied otherwise. `norm_ratio` is `||x|| / ||y||`.

State error is mean squared error against the forward snapshot at the same diffusion index. Inverse step `k` is compared with forward snapshot `50-k`. For `F`, the reference is the snapshot itself. The forward trajectory is written only for that comparison. The function that runs path `D` does not take it as an argument.

First departure, predeclared, not tuned after the plot: the smallest step `k >= 1` on path `D` whose `state_err_x` exceeds `1e-4`. Step 0 of path `D` already has `state_err_y` equal to the terminal `x−y` gap, because `y` was replaced by `x` before any inverse step. That step-0 gap is reported separately and is not called a departure of the inverse.

Hybrid reference replay. For each saved edit, rerun the forward from the saved `z_prc` and the saved conditioning at text 7.5 and image 1.5. The endpoint must match `pairs.pt` within max abs `1e-3`, or the job stops. Paths `I` and `D` then start from the saved final tensors. This is a replay of the existing sender, not a new guidance config.

Old regression. Guidance 3, the dog1 caption→small row, same photo, same saved codeword. The historical PNG is inverted with the unchanged receiver. The job fails if stream x does not Detect, Decode, and match the message, or if either sign agreement differs from the CSV by more than 0.02. The oracle inverse of the replay should have sign agreement at least 0.999 on both streams, because that inverse is the exact inverse of the pair it just produced. A hash difference of the replay PNG is recorded and is not itself the failure condition.

Stage 3. The same dog1 case with the single scale set to 7.5. Invert, generate, and PNG receive all see 7.5, because the historical script has one `GUIDANCE` constant. This 7.5 is uncond + 7.5·(cond − uncond) on SD 1.4. It is not the hybrid’s text 7.5.

Stage 2, not submitted in Stage 0. Reference text 7.5 / image 1.5 is not rerun. New sender configs, same photo, instruction, key, `z_prc`, 50 steps, `p = 0.93`, grid, float64 latents, fp16 UNet:

| Id | Text scale | Image scale |
| --- | ---: | ---: |
| G1 | 3.0 | 1.5 |
| G2 | 3.0 | 1.0 |
| G3 | 7.5 | 1.0 |
| G4 | 1.5 | 1.0 |

Eight jobs: two instructions times G1–G4. No other values. Receiver order per job: R1 true final pair; stop that job if either stream loses Detect, Decode, or the exact message. Then R2 exact `x` as `(x, x)`, R3 saved PNG through posterior mode times scale, duplicated, R4 the frozen 100-step inversion of that same PNG, duplicated. One decoder inversion per job, shared by both streams. Iteration 100 is the only inverted checkpoint.

Edit quality uses the existing definitions. Pixel MSE is `experiments/metrics.py` `pixel_mse` on `[-1, 1]`. LPIPS is AlexNet LPIPS at 256. Directional CLIP is `directional_clip` in `experiments/instructpix2pix_prc/run_pair.py` lines 125–134: cosine of (image embedding of the edit minus image embedding of the source) with (text embedding of the instruction minus text embedding of the source caption). Source caption, for both instructions, is the caption already used for this photograph: “a brown and white spaniel standing on a dirt path in front of green bushes”. The text instruction is the edit instruction itself. Panel order: source, reference `prc_stream0.png`, G1, G2, G3, G4.

Controls. One wrong-key score on the recovered noise of each run. The saved Gaussian hybrid outputs, which already exist for the reference config, are scored with the correct key. They are not regenerated. Noise MSE and correlation are marked evaluator-only.

Reading, fixed here before the runs. Guidance is supported if some lower-guidance config recovers the message at R2 and at R3 or R4 on red-ball, still recovers it on the golden retriever, the edit is still the requested edit, and the terminal `x−y` MSE and the first-departure step move in the same direction. Guidance is not supported if every lower-guidance red-ball R2 still fails, or if the old pipeline at 7.5 still recovers. Otherwise the result is mixed. A trace that moves with the failure is evidence, not a cause. D2 and D3 stay unresolved by a guidance grid. One key, one message, one photo, and two instructions are not a rate.

Jobs that will be submitted after this note:

1. Hybrid reference traces, one RTX 2080 Ti. Two edits, paths F, I, and D. No new guidance.
2. Old pipeline guidance 3, one A6000. Regression plus paths F, I, and D.
3. Old pipeline guidance 7.5, one A6000, only if job 2 reaches the regression.

Stage 2 stays unsubmitted until confirmed. Eight 2080 Ti jobs, each exclusive of the GPU because the decoder inversion peaked at 9,206,097,408 bytes (9.2 GB). About two minutes per image for the iteration-100 inverse, plus three coupled passes of about ten seconds and model load. Wall-time request 40 minutes per job. Storage is the new PNGs, latent pairs, and CSVs, on the order of tens of megabytes, under `results/ip2p_edict_prc/guidance_diag/`.
