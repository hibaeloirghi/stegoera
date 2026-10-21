# InstructPix2Pix implementation audit

Audit of the installed pipeline before any PRC edit job. Checkpoint class
names and the model revision are filled by
`results/instructpix2pix_prc/cpu_controls/checkpoint/checkpoint_summary.json`
when the CPU control job finishes. This file records the source that job is
allowed to call.

## Environment

Inspected on the cluster with
`/fs/clip-scratch/eloirghi/prc-in-edict/conda/envs/edict-prc/bin/python`.

| package | version |
| --- | --- |
| diffusers | 0.21.4 |
| torch | 2.1.0+cu121 |
| transformers | 4.29.2 |

Pipeline class: `StableDiffusionInstructPix2PixPipeline`.

Local file:

`/fs/clip-scratch/eloirghi/prc-in-edict/conda/envs/edict-prc/lib/python3.11/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py`

The class is not the current Diffusers main branch. `__call__` in this file
has no `height` or `width` argument.

## `__call__` arguments

Signature defaults, lines 139–156:

| argument | default |
| --- | --- |
| `num_inference_steps` | 100 |
| `guidance_scale` | 7.5 |
| `image_guidance_scale` | 1.5 |
| `eta` | 0.0 |
| `latents` | None |
| `output_type` | `"pil"` |
| `callback` | None |
| `callback_steps` | 1 |

`eta` is passed only when `scheduler.step` accepts it
(`prepare_extra_step_kwargs`, lines 568–583). The callback receives
`(step_index, timestep, latents)` after a scheduler step (lines 377–381).
There is no separate intermediate-state API. `output_type="latent"` returns
the final denoised latent and skips the VAE decode (lines 383–388).

The first ordinary edit will use these defaults and the scheduler named by
the checkpoint `model_index.json`. Recovery experiments will switch to DDIM
with `eta=0` only after that edit is inspected. Fifty steps is the recovery
target, not the installed quality-check default.

## Generation noise is not the source latent

`prepare_latents` (lines 635–650) builds

```
(batch, vae.config.latent_channels, height / vae_scale_factor, width / vae_scale_factor)
```

`num_channels_latents` is `vae.config.latent_channels` (line 303), not
`unet.config.in_channels`.

`prepare_image_latents` (lines 652–701) encodes the source image. With
classifier-free guidance it concatenates three copies on the batch axis:
source, source, zeros (lines 697–699).

The denoising loop concatenates those image latents onto the noisy latent on
the channel axis (lines 336–340). The installed check (lines 316–324) expects

```
unet.in_channels == latent_channels + image_latent_channels
```

PRC noise is created with `latent_channels` only. The CPU control job refuses
to continue unless `in_channels == 2 * latent_channels`.

## Supplied `latents` are scaled once

Lines 643–649: a caller-supplied tensor is moved to the device and then
multiplied by `scheduler.init_noise_sigma`. The same multiply is applied to
noise the pipeline samples itself.

`DDIMScheduler.__init__` sets `init_noise_sigma = 1.0`
(`scheduling_ddim.py` line 234). Other schedulers do not. The edit job must
pass raw `z_prc` and let `prepare_latents` scale it, then save both the raw
tensor and the tensor returned by `prepare_latents`.

## Source-image encoding and normalization

`__call__` preprocesses pixels with `VaeImageProcessor.preprocess` (line 281)
before `prepare_image_latents`.

In this version the processor defaults (`image_processor.py` lines 62–70) are
`do_resize=True`, `do_normalize=True`, `do_binarize=False`,
`do_convert_rgb=False`. Resize rounds each side down to a multiple of
`vae_scale_factor` (`get_default_height_width`, lines 185–198). Normalize maps
`[0, 1]` to `[-1, 1]` (`normalize`, lines 129–133). A 4-channel tensor is
returned unchanged (lines 298–301), so a precomputed latent can be passed, but
the first edit will not do that.

`prepare_image_latents` uses `vae.encode(image).latent_dist.mode()` (lines
674–677). It does not multiply by `vae.config.scaling_factor`. The decode path
does divide by that factor (line 384). The original InstructPix2Pix latent
diffusion code scales the encoder output. This installed 0.21.4 path does not.
The first ordinary edit uses the installed path unchanged. If that edit does
not follow the instruction, stop and report this discrepancy before any PRC
sweep. Do not upgrade Diffusers for this experiment.

## Guidance branches

Guidance is active when `guidance_scale > 1` and `image_guidance_scale >= 1`
(line 265).

Text embeddings are ordered `[prompt, negative, negative]` (line 547). Image
latents are ordered `[source, source, zeros]` (line 699). The UNet batch is
therefore:

1. instruction text, source image latent
2. unconditional text, source image latent
3. unconditional text, zero image latent

The combination (lines 357–363) is

```
uncond
+ guidance_scale * (text - image)
+ image_guidance_scale * (image - uncond)
```

For schedulers with a `sigmas` attribute, lines 351–354 and 371–372 convert
the prediction around this combination. `DDIMScheduler` in this install has no
`sigmas` attribute; that hack does not run for DDIM. Whether the checkpoint's
default scheduler has `sigmas` is recorded from the loaded object, not assumed.

## Image shape

`EDICT/experiment_images/imagenet_dog_1.jpg` is 500×375 RGB. This `__call__`
cannot be asked for 512×512. Passed through unchanged, resize floors it to
496×368, and the generation noise is not `[1, 4, 64, 64]`.

Prior EDICT runs use `load_edict_image`: center crop, then Lanczos resize to
512 (`experiments/image_utils.py` lines 8–15). This experiment uses that same
preprocessed image and saves it. The generation shape is then determined by
the VAE scale factor. The CPU job computes it from `vae/config.json` and
refuses a key whose block length is anything else. A 512 input and the usual
SD scale of 8 give `[1, 4, 64, 64]` only if the checkpoint config says so.

## Scheduler defaults still owned by the checkpoint

The class docstring (lines 82–84) lists DDIM, LMS, and PNDM as examples. The
running scheduler is whatever `model_index.json` names. Its
`scheduler_config.json` owns `prediction_type`, `clip_sample`,
`timestep_spacing`, and `beta_schedule`. Class defaults in `scheduling_ddim.py`
(`prediction_type="epsilon"`, `clip_sample=True`, `timestep_spacing="leading"`,
lines 163–176 and 187–198) are not the checkpoint until the config agrees.

## PRC code used for the noise

`PRC-Watermark/src/prc.py`:

- `KeyGen` lines 71–115. `message_length` is the payload capacity, not the
  generator width. Generator width is `message_length + g + num_test_bits`.
- `Encode` lines 126–138. Returns `1 - 2 * bits`, so GF(2) bit 0 is `+1` and
  bit 1 is `-1`. Randomness inside Encode (the `g` random bits and the
  Bernoulli noise) is not returned separately. The saved codeword contains it.
- `np.random.seed` does not reproduce `GF.Random` parts of the key. See
  `experiments/receiver_aligned/key_io.py`. The experiment saves the key object.

`PRC-Watermark/src/pseudogaussians.py` `sample` (lines 15–22) is
`z = c * abs(standard normal)`. `recover_posteriors` (lines 27–38) takes
`variances`. In the paper, `sigma` is the standard deviation of
`z_recovered - z`. The formula uses `sigma_squared`. The code argument is that
variance. The predeclared value is `1.5`, so `sigma = sqrt(1.5)`.

`t=3` is the requested sparsity. Appendix C.3 of the Gunn paper says this
concrete setting is not cryptographically undetectable. The CPU controls do
not change `t`.

## Checkpoint record from job 7948380

`timbrooks/instruct-pix2pix`, revision `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`.
Public, not gated. Hub license field: `mit`. Pipeline class matches.

| field | value |
| --- | --- |
| scheduler class | `EulerAncestralDiscreteScheduler` |
| UNet `in_channels` | 8 |
| VAE `latent_channels` | 4 |
| UNet `sample_size` | 64 |
| generation shape at 512 | `[1, 4, 64, 64]`, `n=16384` |
| VAE `scaling_factor` in the json | absent |
| installed `AutoencoderKL` default | 0.18215 (`autoencoder_kl.py` line 96) |

The scheduler json was written by Diffusers 0.12.0.dev0 and still contains
DPM-solver fields (`algorithm_type`, `solver_order`, `solver_type`).
`EulerAncestralDiscreteScheduler` does not use those fields. It does use
`beta_start=0.00085`, `beta_end=0.012`, `beta_schedule=scaled_linear`,
`prediction_type=epsilon`, `num_train_timesteps=1000`. `timestep_spacing` is
absent, so the class default `linspace` applies (`scheduling_euler_ancestral_discrete.py`
line 139). With that spacing, `init_noise_sigma` is `sigmas.max()` (lines
173–176), not 1. The pair job records the value after `set_timesteps`.

`step` draws a fresh Gaussian on every update (line 356) and adds it (line 358).
This scheduler is the quality check. It is not the recovery sampler.

`DDIMScheduler.step` adds noise only when `eta > 0` (`scheduling_ddim.py` lines
455–468). Its previous-timestep update subtracts
`num_train_timesteps // num_inference_steps` (line 407), which matches
`timestep_spacing="leading"`. The DDIM probe uses that spacing, `eta=0`,
`clip_sample=false`, `steps_offset=1`, `set_alpha_to_one=false`, and the
checkpoint beta schedule. `clip_sample` on the DDIM class defaults to true;
the probe does not use that default.

## L1 inverse

`experiments/instructpix2pix_prc/ddim_invert.py` inverts
`DDIMScheduler.step` (`scheduling_ddim.py` lines 407–453) at `eta=0`,
`prediction_type=epsilon`, with `clip_sample` and `thresholding` off.
For a fixed noise prediction the left inverse matches `step` to about `1e-6`
on a CPU tensor. The denoiser is not fixed. Each inverse step evaluates the
same three-branch InstructPix2Pix prediction used in the forward loop
(pipeline lines 336–363) at the cleaner latent and at the forward timestep.
That is the approximation. A capped fixed-point pass solves the actual
`step` update only if the one-pass result misses the message or the
ordinary-noise inversion correlation is below 0.99. Euler ancestral edits
are not inverted, because that scheduler adds new noise on every step.
