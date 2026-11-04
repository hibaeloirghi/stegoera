# Qwen-Image-Edit + PRC

This chapter embeds an 8-bit PRC payload in the fresh generation noise of `Qwen/Qwen-Image-Edit` revision `ac7f9318f633fc4b5778c59367c8128225f1e3de`. Earlier EDICT and InstructPix2Pix results are left unchanged. The image model runs in the separate scratch environment `qwen-image-edit-prc`.

The checkpoint is the August 2025 edit model. Diffusers is pinned to 0.35.1, the release that contains the Qwen-Image-Edit resize fix. No quantization is used. `tron`'s default QoS allows 32G of RAM, which cannot hold the 54 GiB of bf16 weights, so the pilot asks for QoS `high` (120G) and one RTX A6000. The text encoder is offloaded; the transformer stays on the GPU.

The operational receiver reads the saved PNG, the decoding key, and `experiments/configs/qwen_image_edit_prc/receiver_settings.json`. That settings file has an empty user instruction. It does not contain the sender instruction. Saved latents, the source photograph, and the codeword are diagnostic inputs.

Jobs:

```bash
sbatch experiments/slurm/qwen_image_edit_prc/setup_env.sh
sbatch --dependency=afterok:<setup> experiments/slurm/qwen_image_edit_prc/download_weights.sh
sbatch --dependency=afterok:<setup> experiments/slurm/qwen_image_edit_prc/prc_controls.sh
```

The pilot script is not submitted until the controls pass and the installed pipeline audit is checked.
