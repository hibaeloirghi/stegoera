#!/usr/bin/env python3
"""Custom coupled_loop versus UPSTREAM coupled_stablediffusion, state by state.

Replaces the B3 hook=None job that was cancelled before it wrote a result.
Same initial pair, model/embeddings, scheduler timesteps, guidance, mixing p,
stream order, float64. Prompted edit (G=3), not null-null.

Order matters. Upstream's coupled_stablediffusion calls init_attention_func(),
which monkey-patches CrossAttention._attention for the whole process. Every
custom run is therefore done BEFORE the first upstream call, so the custom path
is the one production jobs actually used.

Runs (S=10, s=0.8 -> 8 active steps, so even/odd leapfrog parity are both hit):
    1 custom inversion, null base, hook = recorder
    2 custom generation with edit prompt from the custom inverted pair, recorder
    3 custom generation, step_hook=None  (bit-equality with run 2 final pair)
    4 upstream inversion from the same initial pair, recorder
    5 upstream generation from the SAME custom inverted pair, recorder

Hook events are compared one by one: (site, step, timestep, direction, updated stream,
phase) must match and both streams are compared.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.paths import PHOTOS, RESULTS, ensure_dirs
from experiments.io_utils import append_csv, save_json, seed_everything
from experiments.image_utils import load_edict_image
from experiments import edict_runtime as er
from experiments.receiver_aligned.state_io import code_hash, git_head, save_state

S, STRENGTH, P, G, SEED = 10, 0.8, 0.93, 3.0, 1
EDIT = "A golden retriever"


def recorder(bucket):
    def hook(site, i, t, pair, extra):
        bucket.append({
            "key": (site, int(i), int(t.item() if torch.is_tensor(t) else t), bool(extra.get("reverse")),
                    extra.get("updated_index"), extra.get("phase")),
            "x": pair[0].detach().cpu().clone(),
            "y": pair[1].detach().cpu().clone(),
        })
        return None
    return hook


def compare(name, a, b, rows):
    out = {"name": name, "n_events_a": len(a), "n_events_b": len(b), "keys_equal": None,
           "max_abs": 0.0, "max_abs_x": 0.0, "max_abs_y": 0.0, "first_nonzero_event": None,
           "scale_max_abs_state": 0.0}
    if len(a) != len(b):
        out["keys_equal"] = False
        return out
    ok = True
    for j, (ea, eb) in enumerate(zip(a, b)):
        if ea["key"] != eb["key"]:
            ok = False
            out["first_key_mismatch"] = [j, list(map(str, ea["key"])), list(map(str, eb["key"]))]
            break
        dx = float((ea["x"] - eb["x"]).abs().max())
        dy = float((ea["y"] - eb["y"]).abs().max())
        scale = float(max(ea["x"].abs().max(), ea["y"].abs().max()))
        out["scale_max_abs_state"] = max(out["scale_max_abs_state"], scale)
        if max(dx, dy) > 0 and out["first_nonzero_event"] is None:
            out["first_nonzero_event"] = j
        out["max_abs_x"] = max(out["max_abs_x"], dx)
        out["max_abs_y"] = max(out["max_abs_y"], dy)
        rows.append({"comparison": name, "event": j, "site": ea["key"][0], "step": ea["key"][1],
                     "timestep": ea["key"][2], "reverse": ea["key"][3], "updated_index": ea["key"][4],
                     "phase": ea["key"][5], "max_abs_x": dx, "max_abs_y": dy})
    out["keys_equal"] = ok
    out["max_abs"] = max(out["max_abs_x"], out["max_abs_y"])
    return out


def main():
    ensure_dirs()
    seed_everything(SEED)
    outdir = RESULTS / "receiver_aligned" / "upstream_regression"
    outdir.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "local")
    meta = {
        "job_id": job, "code_hash": code_hash(Path(__file__).resolve().parent),
        "experiment_git": git_head(ROOT), "edict_git": git_head(ROOT / "EDICT"),
        "S": S, "s": STRENGTH, "p": P, "G": G, "seed": SEED, "edit_prompt": EDIT,
        "dtype": "float64", "model": "CompVis/stable-diffusion-v1-4",
        "custom_attention": "diffusers default, set_attention_slice(1)",
        "upstream_attention": "EDICT init_attention_func patched CrossAttention._attention (slice size from module)",
    }
    print(json.dumps(meta), flush=True)
    ef = er.load_models(device="cuda")
    img = load_edict_image(PHOTOS / "imagenet_dog_1.jpg")
    x0 = er.encode_image_to_x0(ef, img, generator=torch.cuda.manual_seed(SEED))
    pair0 = [x0.clone(), x0.clone()]
    kw = dict(steps=S, mix_weight=P, init_image_strength=STRENGTH, guidance_scale=G, seed=SEED)
    timings = {}

    t0 = time.time()
    ev_ci = []
    inv_c = er.coupled_loop(ef, "", reverse=True, init_latent=[p.clone() for p in pair0], step_hook=recorder(ev_ci), **kw)
    timings["custom_inversion"] = time.time() - t0
    inv_pair = [p.detach().clone() for p in inv_c["latent_pair"]]

    t0 = time.time()
    ev_cg = []
    gen_c = er.coupled_loop(ef, EDIT, reverse=False, fixed_starting_latent=[p.clone() for p in inv_pair], step_hook=recorder(ev_cg), **kw)
    timings["custom_generation_recorder"] = time.time() - t0

    t0 = time.time()
    gen_c_none = er.coupled_loop(ef, EDIT, reverse=False, fixed_starting_latent=[p.clone() for p in inv_pair], step_hook=None, **kw)
    timings["custom_generation_hook_none"] = time.time() - t0

    t0 = time.time()
    ev_ui = []
    inv_u = ef.coupled_stablediffusion(
        "", reverse=True, init_image=[p.clone() for p in pair0], return_latents=True,
        step_hook=recorder(ev_ui), **kw)
    timings["upstream_inversion"] = time.time() - t0

    t0 = time.time()
    ev_ug = []
    gen_u = ef.coupled_stablediffusion(
        EDIT, reverse=False, fixed_starting_latent=[p.clone() for p in inv_pair], return_latents=True,
        step_hook=recorder(ev_ug), **kw)
    timings["upstream_generation"] = time.time() - t0

    rows = []
    summary = {
        "meta": meta, "timings_s": timings,
        "hook_none_vs_recorder_final_pair_max_abs": float(max(
            (gen_c["latent_pair"][0] - gen_c_none["latent_pair"][0]).abs().max(),
            (gen_c["latent_pair"][1] - gen_c_none["latent_pair"][1]).abs().max())),
        "inversion_events": compare("inversion_custom_vs_upstream", ev_ci, ev_ui, rows),
        "generation_events": compare("generation_custom_vs_upstream", ev_cg, ev_ug, rows),
        "inversion_final_pair_max_abs": float(max(
            (inv_c["latent_pair"][0] - inv_u[0]).abs().max(), (inv_c["latent_pair"][1] - inv_u[1]).abs().max())),
        "generation_final_pair_max_abs": float(max(
            (gen_c["latent_pair"][0] - gen_u[0]).abs().max(), (gen_c["latent_pair"][1] - gen_u[1]).abs().max())),
        "custom_roundtrip_inversion_pair_recovered_error_note": "custom gen->inv not run here; D0 stage A covers it",
        "state_scale_max_abs": float(max(inv_pair[0].abs().max(), inv_pair[1].abs().max())),
    }
    # Pair-level cross check: does the custom generation of the upstream-inverted pair match too?
    for r in rows:
        r.update({"job_id": job, "code_hash": meta["code_hash"], "S": S, "s": STRENGTH, "p": P, "G": G})
        append_csv(outdir / "events.csv", r)
    save_json(summary, outdir / "summary.json")
    save_state(outdir / "pairs.pt", {
        "inv_custom": [p.cpu() for p in inv_c["latent_pair"]], "inv_upstream": [p.cpu() for p in inv_u],
        "gen_custom": [p.cpu() for p in gen_c["latent_pair"]], "gen_upstream": [p.cpu() for p in gen_u],
    })
    print(json.dumps({k: v for k, v in summary.items() if k != "meta"}, indent=1), flush=True)
    print("REGRESSION_DONE")


if __name__ == "__main__":
    main()
