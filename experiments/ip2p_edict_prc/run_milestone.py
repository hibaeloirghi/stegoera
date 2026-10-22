"""One InstructPix2Pix denoiser inside the EDICT coupled sampler.

Both streams start from the same fresh noise. The source photograph is image
conditioning only. The inverse is the reverse pass of this same loop, given
the final latent pair and the same conditioning. It does not read intermediate
velocities.

An optional config path may be passed as the first argument. With no argument
this file still reads milestone.json and writes results/ip2p_edict_prc/milestone.
That is the only control-flow addition. The sampler calls are unchanged.

Comments describe the lines. They do not change the arithmetic.
"""
from __future__ import annotations  # annotations stay strings

import csv  # the per-stream score table
import json  # read the config and write the manifest
import os  # the SLURM job id
import sys  # import path and the optional config argument
import time  # forward and inverse durations
from pathlib import Path  # filesystem paths

import numpy as np  # the saved PRC arrays
import torch  # latents and the UNet

ROOT = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(ROOT))  # make `experiments` importable

from experiments.image_utils import load_edict_image, tensor_to_pil  # noqa: E402  # center-crop to 512, and latent-to-PNG
from experiments.ip2p_edict_prc.sampler import EdictSchedule, coupled_loop  # noqa: E402  # the coupling
from experiments.paths import RESULTS, WORKDIR  # noqa: E402  # repo root and results root
from experiments.prc_runtime import detect_decode, detection_margin, recover_soft_bits  # noqa: E402  # Gunn Algorithms 7 and 8 wrappers
from experiments.receiver_aligned.key_io import key_hash, load_key  # noqa: E402  # the key object that actually encoded this noise

CONFIG_PATH = ROOT / "experiments" / "configs" / "ip2p_edict_prc" / "milestone.json"  # default config when no argument is given
OUT = RESULTS / "ip2p_edict_prc" / "milestone"  # default output; main() may point this at another folder

FIELDS = [  # column order of results.csv
    "run", "stream", "mse", "max_abs", "correlation", "sign_agreement", "sign_errors",  # noise recovery
    "detect", "margin", "decode_non_none", "exact_message_match",  # PRC decisions
    "wrong_key_detect", "seconds",  # negative control and inverse duration
]


def fail(reason: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)  # the failure still leaves a status file
    (OUT / "STATUS.txt").write_text("FAIL\n" + reason + "\n")  # why the job stopped
    print("FAIL:", reason, flush=True)  # same reason in the SLURM log
    return 2  # nonzero process status


def compare(candidate, target):
    x = candidate.detach().to(torch.float64).flatten().cpu()  # recovered latent as one vector
    y = target.detach().to(torch.float64).flatten().cpu()  # the noise this run started from
    err = x - y  # coordinate-wise recovery error
    corr = float(torch.corrcoef(torch.stack([x, y]))[0, 1].item())  # correlation of the two vectors
    signs = torch.sign(x) == torch.sign(y)  # True where the sign matches, which is the PRC bit
    return {  # one row's geometry fields
        "mse": float(err.pow(2).mean().item()),  # mean squared error
        "max_abs": float(err.abs().max().item()),  # largest absolute coordinate error
        "correlation": corr,  # 1 when the vector is recovered
        "sign_agreement": float(signs.float().mean().item()),  # fraction of matching signs
        "sign_errors": int((~signs).sum().item()),  # count of sign mismatches, including zeros
    }


def score_prc(decoding_key, wrong_key, recovered, codeword, message, sigma_squared):
    flat = recovered.detach().to(torch.float64).flatten().cpu()  # the vector Detect and Decode see
    soft = recover_soft_bits(flat, variance=sigma_squared)  # Gunn posterior, pseudogaussians.py recover_posteriors, variances=sigma^2
    detected, decode_ok, decoded = detect_decode(decoding_key, soft)  # PRC-Watermark Detect then Decode, prc.py lines 151 and 183
    _statistic, _threshold, margin = detection_margin(decoding_key, soft)  # signed distance of the detector score above its threshold
    sent = np.asarray(message, dtype=int).reshape(-1)  # the 64 bits that were encoded
    if decoded is None:  # Decode returns None when the codeword is not recovered
        exact = False  # the message was not returned
    else:  # Decode returned a bit vector
        got = np.asarray(decoded, dtype=int).reshape(-1)  # those bits
        exact = bool(got.size == sent.size and np.array_equal(got, sent))  # exact equality of the whole message, not a prefix
    wrong_detect, _wrong_decode, _wrong = detect_decode(  # the same soft bits under a key that did not encode this noise
        wrong_key, recover_soft_bits(flat, variance=sigma_squared)  # a second posterior call with the same variance
    )
    code = torch.as_tensor(np.asarray(codeword), dtype=torch.float64).flatten()  # the {+1,-1} codeword from Encode
    return {  # PRC fields for this stream
        "detect": detected,  # Algorithm 3 / the operational detector
        "margin": margin,  # positive means Detect is true at this false-positive setting
        "decode_non_none": decode_ok,  # Decode returned a vector
        "exact_message_match": exact,  # that vector is the encoded message
        "wrong_key_detect": wrong_detect,  # should be false
        "sign_agreement": float((torch.sign(flat) == code).to(torch.float64).mean().item()),  # signs against the codeword, not against z
        "sign_errors": int((torch.sign(flat) != code).sum().item()),  # codeword sign mismatches
    }


def make_denoise(unet, image_latents, prompt_embeds, guidance, image_guidance):
    def denoise(partner, timestep):
        with torch.no_grad():  # the inverse must not store a gradient graph; the forward is not trained
            latent = partner.to(device="cuda", dtype=torch.float16)  # the partner stream, cast to the UNet dtype
            t = timestep.detach().to(device="cuda").round().to(dtype=torch.long)  # EDICT time 980, 960, ... as an integer index
            model_input = torch.cat([latent] * 3)  # three copies: text, image, unconditional. Installed pipeline line 336
            # DDIM scale_model_input is the identity. The checkpoint's Euler
            # scheduler is not used, so its sigma scaling is not applied.
            model_input = torch.cat([model_input, image_latents], dim=1)  # append the trained image channels. Installed pipeline lines 338-340. Brooks et al. Sec. 3.2
            noise = unet(model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]  # one UNet call for all three branches. Installed pipeline lines 343-345
            noise_text, noise_image, noise_uncond = noise.chunk(3)  # order is prompt, negative, negative. Installed pipeline line 547 and chunk at line 358
            guided = (  # Brooks et al. Sec. 3.2.1, two-conditioning classifier-free guidance. Installed pipeline lines 357-363
                noise_uncond  # epsilon(z, empty image, empty text)
                + guidance * (noise_text - noise_image)  # s_T times (text-and-image branch minus image-only branch)
                + image_guidance * (noise_image - noise_uncond)  # s_I times (image-only branch minus empty branch)
            )
            return guided  # this epsilon is what EDICT's affine step treats as frozen

    return denoise  # the closure the coupled loop calls on the partner stream


def decode(pipe, latent):
    scale = float(pipe.vae.config.scaling_factor)  # InstructPix2Pix VAE scale, class default 0.18215
    z = latent.to(device="cuda", dtype=pipe.vae.dtype) / scale  # diffusion latent back to VAE units
    image = pipe.vae.decode(z).sample  # RGB tensor in approximately [-1, 1]
    return tensor_to_pil(image)  # map that tensor to an 8-bit PNG


def run_case(name, denoise, schedule, noise, cfg):
    start = [noise.clone(), noise.clone()]  # edict_functions.py lines 888-891: one noise tensor copied into both streams
    print(name, "forward", flush=True)  # mark the generation pass in the log
    started = time.perf_counter()  # start of generation
    final, trace = coupled_loop(  # generation. The trace is not passed to the inverse
        denoise, schedule, start, reverse=False,  # forward direction
        mix_weight=float(cfg["mix_weight"]), leapfrog=bool(cfg["leapfrog"]),  # p and the leapfrog flag from the config
    )
    forward_seconds = time.perf_counter() - started  # generation time
    print(name, "inverse", flush=True)  # mark the inverse pass
    started = time.perf_counter()  # start of the inverse
    recovered, _back = coupled_loop(  # matched inverse. Its only latent input is the final pair
        denoise, schedule, final, reverse=True,  # reverse direction of the same function
        mix_weight=float(cfg["mix_weight"]), leapfrog=bool(cfg["leapfrog"]),  # same p and leapfrog as the forward
    )
    inverse_seconds = time.perf_counter() - started  # inverse time
    return {  # everything the caller scores or saves
        "start": start,  # the two copies of the fresh noise
        "final": final,  # the latent pair after the last mixing layer
        "recovered": recovered,  # the pair the inverse returned
        "forward_seconds": forward_seconds,  # recorded in the manifest indirectly through the run
        "inverse_seconds": inverse_seconds,  # written on each results row
        "steps": len(trace) - 1,  # number of timesteps actually taken
    }


def main():
    global OUT  # allow this function to retarget the output folder
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG_PATH  # optional config; the default remains the golden-retriever milestone
    cfg = json.loads(config_path.read_text())  # one JSON object of strings and numbers
    OUT = RESULTS / "ip2p_edict_prc" / cfg.get("output_dir", "milestone")  # a new instruction writes elsewhere; the default folder is unchanged
    OUT.mkdir(parents=True, exist_ok=True)  # create that folder
    (OUT / "images").mkdir(exist_ok=True)  # PNGs live in a subfolder
    torch.backends.cudnn.deterministic = True  # same UNet kernels on the forward and the inverse
    torch.backends.cudnn.benchmark = False  # do not pick a different kernel per shape

    _enc, decoding_key, message, key_codeword = load_key(WORKDIR / cfg["key_path"])  # the encoding that produced the saved codeword
    if key_hash(_enc) != cfg["expected_key_hash"]:  # refuse a different key than the one named in the config
        return fail(f"key hash {key_hash(_enc)}")  # stop before any image
    noise_file = np.load(WORKDIR / cfg["saved_noise"])  # gaussian, z_prc, codeword, message
    codeword = np.asarray(noise_file["codeword"]).reshape(-1)  # {+1,-1} vector, n = 16384
    if not np.array_equal(np.asarray(key_codeword).reshape(-1), codeword):  # the pickle and the npz must be the same encoding
        return fail("saved codeword does not match the PRC noise")  # stop rather than score the wrong word
    if not np.array_equal(np.asarray(message, dtype=int).reshape(-1), np.asarray(noise_file["message"], dtype=int)):  # message agreement
        return fail("saved message does not match the key")  # stop
    z_prc = torch.from_numpy(np.asarray(noise_file["z_prc"])).to(torch.float64)  # c * |g|, Gunn Algorithm 6, pseudogaussians.sample
    gaussian = torch.from_numpy(np.asarray(noise_file["gaussian"])).to(torch.float64)  # the |g| magnitudes' underlying g, without the code signs
    if z_prc.shape != (1, 4, 64, 64):  # InstructPix2Pix latent shape at 512 pixels
        return fail(f"unexpected noise shape {tuple(z_prc.shape)}")  # stop
    _wrong_enc, wrong_dec, _wm, _wc = load_key(WORKDIR / cfg["wrong_key_path"])  # a key that did not encode z_prc

    from diffusers import StableDiffusionInstructPix2PixPipeline  # the trained editor, imported here so the CPU algebra file never loads it

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(  # weights of timbrooks/instruct-pix2pix at the pinned revision
        cfg["model_id"], revision=cfg["model_revision"], torch_dtype=torch.float16  # float16 UNet; the latent arithmetic stays float64
    ).to("cuda")  # one GPU
    if pipe.unet.config.in_channels != 8 or pipe.vae.config.latent_channels != 4:  # 4 noise channels plus 4 trained image channels
        return fail("unexpected InstructPix2Pix channel counts")  # do not run a different architecture under this name
    source = load_edict_image(WORKDIR / cfg["source_photo"], size=int(cfg["preprocessed_size"]))  # center crop, then Lanczos to 512. image_utils.py lines 8-15
    source.save(OUT / "images" / "00_source.png")  # the photograph the edit was conditioned on
    image = pipe.image_processor.preprocess(source)  # the pipeline's own tensor conversion, installed __call__ line 281
    prompt_embeds = pipe._encode_prompt(  # CLIP embeddings of the instruction. Installed pipeline lines 535-548
        cfg["instruction"], "cuda", 1, True, negative_prompt=cfg["negative_prompt"]  # three copies because classifier-free guidance is on
    )
    image_latents = pipe.prepare_image_latents(  # VAE mode of the source, then [source, source, zeros]. Installed lines 652-699
        image, 1, 1, prompt_embeds.dtype, "cuda", True  # batch 1, one image, guidance on
    )
    if prompt_embeds.shape[0] != 3 or image_latents.shape[0] != 3:  # both conditionings must have the three branches
        return fail(f"conditioning batches {tuple(prompt_embeds.shape)} {tuple(image_latents.shape)}")  # stop if guidance was not built

    schedule = EdictSchedule(int(cfg["num_inference_steps"]), device="cpu", dtype=torch.float64)  # alpha table and the 980..0 grid
    denoise = make_denoise(  # bind this UNet and this conditioning into the partner-stream function
        pipe.unet, image_latents, prompt_embeds,  # weights and the two saved conditionings
        float(cfg["guidance_scale"]), float(cfg["image_guidance_scale"]),  # s_T and s_I
    )
    print("gpu", torch.cuda.get_device_name(0), flush=True)  # which card ran the edit
    rows = []  # one dict per stream
    cases = {}  # name -> forward/inverse bundle
    for name, raw in (("gaussian", gaussian), ("prc", z_prc)):  # ordinary noise first, then the same magnitudes with PRC signs
        cases[name] = run_case(name, denoise, schedule, raw.to(device="cuda"), cfg)  # forward and matched inverse
        for stream, recovered in enumerate(cases[name]["recovered"]):  # score x and y separately
            stats = compare(recovered, raw.to(device="cuda"))  # distance to the fresh noise, not to a saved velocity
            row = {  # geometry first; PRC fields stay empty for the Gaussian run
                "run": name,  # gaussian or prc
                "stream": stream,  # 0 is x, 1 is y
                "seconds": cases[name]["inverse_seconds"],  # inverse duration
                **stats,  # mse, correlation, signs
                "detect": "",  # filled only for PRC
                "margin": "",  # filled only for PRC
                "decode_non_none": "",  # filled only for PRC
                "exact_message_match": "",  # filled only for PRC
                "wrong_key_detect": "",  # filled only for PRC
            }
            if name == "prc":  # the Gaussian run has no codeword to decode
                prc = score_prc(  # Detect, Decode, and exact message equality
                    decoding_key, wrong_dec, recovered, codeword, message, float(cfg["sigma_squared"])  # predeclared sigma^2, not a searched value
                )
                row.update(prc)  # overwrite the sign fields with the codeword comparison and fill the decisions
            rows.append(row)  # keep the row
            print(  # one log line per stream
                name, "stream", stream,  # which pair and which member
                "corr", round(stats["correlation"], 4),  # recovery correlation
                "max_abs", stats["max_abs"],  # recovery max error
                "exact", row["exact_message_match"],  # blank for Gaussian, true or false for PRC
                flush=True,
            )
        for stream, latent in enumerate(cases[name]["final"]):  # decode the edited pair, not the recovered noise
            decode(pipe, latent).save(OUT / "images" / f"{name}_stream{stream}.png")  # one PNG per stream

    torch.save(  # the tensors a later check can reload; the inverse in this process did not read them back
        {
            "gaussian_final": [item.cpu() for item in cases["gaussian"]["final"]],  # ordinary final pair
            "prc_final": [item.cpu() for item in cases["prc"]["final"]],  # PRC final pair
            "prc_recovered": [item.cpu() for item in cases["prc"]["recovered"]],  # PRC inverse output
            "gaussian_recovered": [item.cpu() for item in cases["gaussian"]["recovered"]],  # Gaussian inverse output
            "image_latents": image_latents.detach().to(torch.float32).cpu(),  # the three image-conditioning latents
            "prompt_embeds": prompt_embeds.detach().to(torch.float32).cpu(),  # the three text embeddings
            "timesteps": schedule.timesteps.cpu(),  # the grid that was actually walked
        },
        OUT / "pairs.pt",  # one file
    )
    with (OUT / "results.csv").open("w", newline="") as handle:  # the score table
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")  # drop keys that are not columns
        writer.writeheader()  # column names
        writer.writerows(rows)  # four rows: two runs, two streams
    manifest = {  # what this run was, next to the numbers
        "receiver_level": "source-known exact final latent pair",  # the source latent and the instruction were the ones used to edit
        "forward": "InstructPix2Pix three-branch denoiser inside the EDICT leapfrog and mixing layer",  # the hybrid
        "inverse": "reverse=True of the same loop; intermediate velocities are not inputs",  # no saved epsilon is fed back
        "initialization": "both streams cloned from one fresh noise tensor; source latent is not the start",  # not an inverted photograph
        "steps": int(cfg["num_inference_steps"]),  # S
        "mix_weight": float(cfg["mix_weight"]),  # p
        "timestep_grid": "EDICT arange: 980, 960, ..., 0",  # not Diffusers leading 991, 981, ...
        "latent_dtype": "float64",  # affine arithmetic
        "unet_dtype": "float16",  # denoiser weights and activations
        "vae_scaling_factor": float(pipe.vae.config.scaling_factor),  # the number decode divided by
        "instruction": cfg["instruction"],  # the edit text
        "job_id": os.environ.get("SLURM_JOB_ID", ""),  # empty if the script is not under SLURM
        "gpu": torch.cuda.get_device_name(0),  # card name
        "rows": rows,  # a copy of the table
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")  # save that description
    (OUT / "STATUS.txt").write_text("PASS\n")  # the job recorded both runs; PASS is not a claim about the message
    print("PASS", flush=True)  # last log line
    return 0  # success


if __name__ == "__main__":
    sys.exit(main())  # run when this file is the program
