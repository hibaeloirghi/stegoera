# Paper to code

Gunn et al. was read from `gunn_image_watermark.pdf` in this repository
(ICLR 2025, 26 pages). The InstructPix2Pix PDF is not in the repository. The
section numbers below are the ones named in the experiment brief, matched to
the installed Diffusers 0.21.4 pipeline. They are not quotations from a local
Brooks PDF. Nothing here says the Gunn Stable Diffusion 2.1 results transfer
to InstructPix2Pix.

## InstructPix2Pix, installed pipeline

Source file: `pipeline_stable_diffusion_instruct_pix2pix.py` in the conda env
(diffusers 0.21.4). Line numbers are that file.

The training objective in the brief (Sec. 3.2, Eq. 1) is a denoising loss on
a latent conditioned by the source image and the instruction text. This
experiment does not train. The sampling counterpart is the UNet call at lines
343–345: noisy latent, timestep, and `encoder_hidden_states`.

Sec. 3.2 concatenation is lines 338–340. Image conditioning is appended on the
channel axis after `scale_model_input`. It is not the tensor whose signs PRC
replaces.

Sec. 3.2.1, Eq. 3, separate text and image guidance, is lines 357–363:

```
uncond + s_T * (text - image) + s_I * (image - uncond)
```

with `s_T = guidance_scale` and `s_I = image_guidance_scale`. The three
branches are built at lines 547 and 697–699. This is not EDICT's mixing weight
or edit strength.

Fig. 12 in the brief is output variation under different initial noise. The
generation screen compares ordinary Gaussian noise with PRC sign noise. That
comparison is an experiment, not a result already shown for this editor.

`image_guidance_scale` defaults to 1.5 and `guidance_scale` to 7.5 (lines
144–145). `__call__` defaults to 100 steps. The checkpoint scheduler may still
differ from DDIM.

## Gunn PRC, local PDF

| paper | code |
| --- | --- |
| Appendix B, Algorithm 1 KeyGen | `PRC-Watermark/src/prc.py` `KeyGen`, lines 71–115 |
| Appendix B, Algorithm 2 Encode | `Encode`, lines 126–138 |
| Appendix B, Algorithm 3 Detect | `Detect`, lines 151–170 |
| Appendix B, Algorithm 4 Decode | `Decode`, lines 183–220 |
| Appendix C, Algorithm 5 | same `KeyGen` |
| Appendix C, Algorithm 6, and the main-text sampler | `pseudogaussians.sample`, lines 15–22: `z_i = c_i * abs(y_i)` |
| Algorithm 7 and Appendix C.1 | `recover_posteriors`, lines 27–38, then `Detect` |
| Algorithm 8 | `Decode` on those soft values |
| Footnote 5 | decoder inversion: gradient steps on `\|\|D(z)-x\|\|`, init at the encoder output. Our later L2 receiver may reuse `experiments/prc_runtime.py` `decoder_inv_vae`. It is not part of the CPU controls. |
| Appendix C.3 | `t=3`, `n=16384` is a practical setting. The paper says undetectability is not guaranteed for these concrete parameters, and the scheme should not be used for steganography. |

Algorithm 6 writes the initial noise and then calls `Generate`. In this
experiment `Generate` is one InstructPix2Pix `__call__`. Source-image
conditioning and the instruction stay outside that noise vector.

Algorithm 7 sets `sigma` to the standard deviation of the recovery error
`z' - z`. The soft bit is

```
erf( z_i / sqrt(2 * sigma^2 * (1 + sigma^2)) )
```

The code argument `variances` is `sigma^2`. The paper's default is `sigma =
sqrt(3/2)`, which is `sigma^2 = 1.5`. The prose in the PDF extracts with a
broken radical; the formula and the installed default `1.5` agree with
`sigma^2 = 3/2`. Controls use that predeclared value. They do not search over
it.

`Encode` returns `1 - 2 * bits`. That is the `{-1,+1}` codeword used by
Algorithm 6, not the raw GF(2) vector printed in Algorithm 2.

The paper's `Recover` for Stable Diffusion 2.1 is not a receiver for
InstructPix2Pix. DDIM inversion is an approximation even with a known prompt
(main text around the Recover algorithm). Footnote 5 is Hong et al. decoder
inversion, not a proof of exact latent recovery. New inversion adapters in
this experiment are ours.
