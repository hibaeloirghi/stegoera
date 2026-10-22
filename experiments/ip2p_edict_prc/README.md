# InstructPix2Pix denoiser + EDICT coupling + PRC noise

This folder is one sender and one receiver. The sender edits a photograph. The receiver is given the edited latent pair, the same source conditioning, and the same instruction, and it tries to recover the message that was written into the initial noise.

The arithmetic is unchanged by the comments in the Python files. Each executable line in `sampler.py`, `run_milestone.py`, and `run_cpu_algebra.py` has a comment saying what that line does and which original source it follows. This file is the map of those files. The only control-flow addition is an optional config path, described at the bottom. With no argument, `run_milestone.py` still reads `milestone.json` and still writes `results/ip2p_edict_prc/milestone`.

## What is combined

Three pieces, each left in its own role.

| Piece | Role here | What it is not |
| --- | --- | --- |
| InstructPix2Pix | The denoiser. It sees the source photo and the instruction. | Not the sampler, and not the inverse. |
| EDICT | The two-stream sampler and its algebraic inverse. | Not a fixed-point search, and not the old single-stream DDIM inverse. |
| PRC | The signs of the fresh initial noise, and the test that reads those signs back. | Not a change to the denoiser or the sampler. |

Generation starts from fresh noise. The source photograph is conditioning. It is not the latent the sampler starts from.

## Papers and the code they correspond to

**InstructPix2Pix.** Brooks, Holynski, and Efros, CVPR 2023. Local PDF: `InstructPix2Pix.pdf` at the repository root. Section 3.2 trains a denoiser on a noisy latent concatenated with the source-image latent, conditioned on an edit instruction. Section 3.2.1, “Classifier-free Guidance for Two Conditionings,” defines the two scales `s_I` and `s_T`.

The weights used here are `timbrooks/instruct-pix2pix`, revision `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`. The class is `StableDiffusionInstructPix2PixPipeline` in the installed Diffusers 0.21.4 file

`.../diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py`.

The lines this experiment follows:

| Installed line | What it does | Where we call it |
| --- | --- | --- |
| 281 | Preprocess the photograph. | `pipe.image_processor.preprocess` |
| 336 | Repeat the noisy latent three times. | `torch.cat([latent] * 3)` in `make_denoise` |
| 338–340 | Concatenate image conditioning on the channel axis. | `torch.cat([model_input, image_latents], dim=1)` |
| 343–345 | One UNet call. | `unet(...)` in `make_denoise` |
| 358 and 547 | Split and stack as text, image, unconditional. The prompt batch is `[instruction, negative, negative]`. | `noise.chunk(3)` |
| 357–363 | `uncond + s_T (text - image) + s_I (image - uncond)`. | the `guided = ...` expression |
| 652–699 | Encode the source with the VAE mode and build `[source, source, zeros]`. | `prepare_image_latents` |

The image is resized by `experiments/image_utils.py` `load_edict_image` (lines 8–15): center crop to a square, then Lanczos to 512. That is the EDICT photograph preprocessing, applied before the pipeline sees the image.

**EDICT.** Wallace, Gokul, and Ermon, CVPR 2023, arXiv:2211.12446. Local PDF: `EDICT_Exact_Diffusion_Inversion_via_Coupled_Transformations.pdf`. Original implementation: `EDICT/edict_functions.py` in this repository (the Salesforce EDICT code). The comments in that file map the generative affine step to Eq. (14) and the inverse affine step to Eq. (15).

| Original lines | What they do | Our function |
| --- | --- | --- |
| `edict_functions.py` 673–691 | `get_alpha_and_beta` | `_coeff` |
| `edict_functions.py` 698–727 | `forward_step` | `forward_step` |
| `edict_functions.py` 733–764 | `reverse_step` | `reverse_step` |
| `edict_functions.py` 905–910 | DDIM betas and `set_timesteps` | `EdictSchedule` |
| `my_diffusers/.../scheduling_ddim.py` 161–163 | timestep grid `arange(0, 1000, stride)[::-1]` | `EdictSchedule.timesteps` |
| `edict_functions.py` 888–891 | copy one noise tensor into both streams | `run_case`: `[noise.clone(), noise.clone()]` |
| `edict_functions.py` 941 | flip the timestep list on the inverse | `coupled_loop` when `reverse` is true |
| `edict_functions.py` 956–957 | undo the mixing layer, y then x | `unmix` |
| `edict_functions.py` 969–982 | leapfrog index on the inverse and on the forward | the `li = ...` block in `coupled_loop` |
| `edict_functions.py` 984 | epsilon is computed on the other stream | `lj = (li + 1) % 2` and `denoise(pair[lj], ...)` |
| `edict_functions.py` 1026–1033 | apply `forward_step` or `reverse_step` to one stream | `pair[li] = step(...)` |
| `edict_functions.py` 1048–1050 | forward mix; the second line reads the updated first stream | `mix` |

The mixing weight `p = 0.93` is the default in `coupled_stablediffusion` (`edict_functions.py` line 816). Fifty steps is the `steps=50` default on line 806. The grid at 50 steps is 980, 960, …, 0. This is not the leading grid 991, 981, … used by the earlier single-stream DDIM experiment.

**PRC.** Gunn, Zhao, and Song, “An Undetectable Watermark for Generative Image Models.” Local PDF: `gunn_image_watermark.pdf`. Original code: `PRC-Watermark/src/prc.py` and `PRC-Watermark/src/pseudogaussians.py`.

| Paper | Code |
| --- | --- |
| Appendix B, Algorithm 1, KeyGen | `prc.py` `KeyGen`, line 71 |
| Appendix B, Algorithm 2, Encode | `prc.py` `Encode`, line 126. The return on line 138 is `1 - 2 * bits`, so the codeword is in `{+1, -1}`. |
| Appendix B, Algorithm 3, Detect | `prc.py` `Detect`, line 151 |
| Appendix B, Algorithm 4, Decode | `prc.py` `Decode`, line 183 |
| Algorithm 6, the pseudogaussian sample `z_i = c_i * abs(g_i)` | `pseudogaussians.py` `sample`, lines 15–22 |
| Algorithm 7, soft bits then Detect | `pseudogaussians.py` `recover_posteriors`, line 28, then `Detect` |
| Algorithm 8, Decode on those soft bits | `Decode` |

The saved noise in `results/instructpix2pix_prc/pair/raw_noise.npz` was built as `codeword * abs(gaussian)`. This experiment does not call `Encode` again. It reuses that vector and the key in `pair_encoding.pkl`. The soft-bit variance is the predeclared `sigma^2 = 1.5` (`recover_posteriors` default). It is not fit to the recovery error.

## Files in this folder

| File | Runs where | Purpose |
| --- | --- | --- |
| `sampler.py` | CPU and GPU | Schedule, affine steps, mixing, and the coupled loop. No weights. |
| `run_cpu_algebra.py` | CPU job `experiments/slurm/ip2p_edict_prc/cpu_algebra.sh` | Proves the loop inverts when epsilon depends on the partner. |
| `run_milestone.py` | GPU | Loads InstructPix2Pix, edits, inverts, scores. |
| `../configs/ip2p_edict_prc/milestone.json` | the golden-retriever run | Instruction “Turn the dog into a golden retriever.” Output folder `milestone`. |
| `../configs/ip2p_edict_prc/red_ball.json` | the red-ball run | Instruction “Add a red ball next to the dog.” Output folder `red_ball`. |
| `../slurm/ip2p_edict_prc/milestone.sh` | GPU | Calls `run_milestone.py` with no argument. |
| `../slurm/ip2p_edict_prc/red_ball.sh` | GPU | Calls `run_milestone.py` with the red-ball config. |

Results are not code. `results/ip2p_edict_prc/milestone/` is the first instruction. `results/ip2p_edict_prc/red_ball/` is the second. Neither job reads the other’s latents.

## Order of one GPU run

1. Load the key and the saved noise. Check the key hash, the codeword, the message, and the shape `(1, 4, 64, 64)`.
2. Load InstructPix2Pix in float16. Refuse to continue unless the UNet has 8 input channels and the VAE has 4 latent channels.
3. Center-crop the dog photograph to 512. Encode the instruction and the source image with the pipeline’s own functions, so the three-branch batches match the installed code.
4. Build the EDICT schedule and the denoiser closure.
5. Gaussian case. Copy the saved ordinary Gaussian into both streams. Run `coupled_loop(..., reverse=False)`. Run `coupled_loop` again with `reverse=True` on that final pair only. Compare each recovered stream with the Gaussian.
6. PRC case. Same calls, starting from `z_prc`. After the geometry comparison, run Detect, Decode, and exact message equality on each recovered stream, plus the wrong key.
7. Decode each final latent with the InstructPix2Pix VAE and save the PNG. Save the pairs, the conditioning, and the timestep grid.

The receiver in step 5 and step 6 does not read the forward trace. `trace` is returned and then ignored. Saved velocities are not an input.

## How a single coupled step is invertible

On a forward step the loop does this, in order:

1. For each of the two streams, compute epsilon at the other stream, then call `forward_step` on this stream. The epsilon does not depend on the stream being updated, so `reverse_step` with that same epsilon is an algebraic inverse.
2. Mix, with the second assignment reading the updated first stream.

The inverse does the opposite:

1. Unmix, y then x.
2. Apply `reverse_step` in the leapfrog order from `edict_functions.py` lines 969–975, again computing epsilon at the partner.

A small residual of one affine step is not the success test. The test is whether the recovered pair equals the initial noise, and, for PRC, whether Decode returns the message.

## What this receiver is

The GPU script’s receiver has the true source-image latents and the true instruction embeddings. It is given the exact final latent pair produced by the forward pass. That is source-known latent-pair recovery.

It is not recovery from a PNG. It is not recovery from one image when the source photograph is unknown. Those are later receivers. Invertibility of the pair does not establish them.

## The optional config argument

`run_milestone.py` with no arguments reads `milestone.json` and writes `results/ip2p_edict_prc/milestone`, as before. A path argument selects another JSON file. If that file contains `output_dir`, results go to `results/ip2p_edict_prc/<output_dir>`. The red-ball job uses this so it does not overwrite the golden-retriever outputs. No sampling formula, guidance formula, mix, or PRC score is different between the two configs except the instruction string and the output folder.
