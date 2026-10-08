# Diagnostic report

CPU mixing and PRC checks, and the GPU ladder, are submitted separately. This file records what is already measured. It will be updated when those jobs finish. Nothing below is an optimization result.

## Already measured, from finished jobs

Top-latent shaping at `tau=1` does not impose a detectable PRC constraint on the noise it writes.

- `lambda=0.05`: sender margin -102, receiver margin -137, detect false, LPIPS versus the unwatermarked edit 0.183. Best-of-1024 agreement 0.510.
- `lambda=0.15`: sender margin -105, receiver margin -118, detect false, LPIPS 0.754.

The sender margin is the score on the blended noise before pixels. The gap is not explained by the VAE. `lambda=0.5` and `lambda=1` are still queued inside job 7783940 and are not scored here.

The additive screen never stored a parity margin. Its best weak points remain top `gamma=0.35` (sign agreement 0.613, LPIPS 0.176) and S4 `lambda=0.10` (agreement 0.545, LPIPS 0.016). Both failed `Detect`. Sign agreement is not the decision rule.

`sigma=0.5` on every old row is the bottom of `calibrate_variance`'s grid. See AUDIT.md. D0 reports variances 0.5 and 1.5 and does not pick one after the fact.

## Not yet measured

Stages A–D (true pair, duplicated `x`, float VAE, PNG) are the GPU job `ra-d0`. Until it returns, there is no number for how much of a real code is lost by dropping `y` or by the autoencoder.
