# Image-only receivers

Stage 2 inverted the saved PNG with the original photo latent and the instruction embeddings. These jobs do not. The sender, where a new image is made, still uses that photo and the instruction. The receivers below start after the PNG exists.

Both receivers get:

- the edited PNG
- the PRC key
- the InstructPix2Pix weights at revision `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`
- the sampler that made the image: 50 steps, mix 0.93, leapfrog, grid 980…0, and that image's text and image guidance scales
- the empty negative prompt
- one latent taken from the PNG, then copied into both EDICT streams

The latent is obtained twice, and both are scored: VAE posterior mode times the scaling factor, and the frozen 100-step Adam decoder inversion (lr 0.1, warmup 10, sigma^2 = 1.5). One inversion is shared by both receivers. Iteration 100 is the one that is inverted.

`image_key_model` encodes the empty string as its prompt and uses an all-zero image latent for every branch. `image_key_model_prompt` encodes the editing instruction and still uses an all-zero image latent. Neither receives the source photo.

With those inputs the three UNet branches are not independent. The empty-image receiver's three branches match, so both guidance scales cancel and the inverse is unconditional. The prompt receiver's image-only branch matches its empty branch, so the image guidance scale cancels and the text scale remains. The grid is still the right test: each cell is a different PNG, and the text scale still changes the prompt receiver.

## Grid

Text scales 1.5, 3.0, 5.0, 7.5. Image scales 1.0, 1.25, 1.5. Both instructions. The ten images already on disk are not regenerated:

| Text | Image | Golden retriever and red ball |
| ---: | ---: | --- |
| 7.5 | 1.5 | existing reference PNG |
| 3.0 | 1.5 | existing G1 PNG |
| 3.0 | 1.0 | existing G2 PNG |
| 7.5 | 1.0 | existing G3 PNG |
| 1.5 | 1.0 | existing G4 PNG |
| 1.5 | 1.25 | new |
| 1.5 | 1.5 | new |
| 3.0 | 1.25 | new |
| 5.0 | 1.0 | new |
| 5.0 | 1.25 | new |
| 5.0 | 1.5 | new |
| 7.5 | 1.25 | new |

Same photo, key, and PRC noise as Stage 2. A new sender also inverts its true final pair with the source conditioning. That row is named `sender_check_true_pair`. If Detect, Decode, or the exact message fails on either stream, the image-only receivers are not run.

Noise MSE and correlation are evaluator-only. Detect, Decode, and exact message are separate. This is one photo, one key, and two instructions.
