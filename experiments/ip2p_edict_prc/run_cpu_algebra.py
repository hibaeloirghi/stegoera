"""CPU check that the coupled loop inverts a state-dependent denoiser.

The denoiser reads the partner latent, which is what EDICT relies on. No UNet
and no image are loaded. Comments describe the lines. They do not change them.
"""
from __future__ import annotations  # annotations stay strings

import json  # write the numeric report
import sys  # exit status and import path
from pathlib import Path  # locate the repo and the results directory

import torch  # the fake latents and the exact arithmetic

ROOT = Path(__file__).resolve().parents[2]  # repository root: experiments/ip2p_edict_prc/this_file -> parents[2]
sys.path.insert(0, str(ROOT))  # allow `import experiments` when the job starts in the repo

from experiments.ip2p_edict_prc.sampler import EdictSchedule, coupled_loop, forward_step, reverse_step  # noqa: E402  # the coupling under test
from experiments.paths import RESULTS  # noqa: E402  # results directory, defaulting to the repo results/

OUT = RESULTS / "ip2p_edict_prc" / "cpu_algebra"  # this check's own folder; it does not touch the image runs


def main():
    OUT.mkdir(parents=True, exist_ok=True)  # create the output folder if the job has not already
    schedule = EdictSchedule(50)  # the published EDICT grid, 980 down to 0
    generator = torch.Generator().manual_seed(0)  # fixed CPU randomness so the check is repeatable
    sample = torch.randn(1, 4, 8, 8, generator=generator, dtype=torch.float64)  # one latent the affine step will move
    eps = torch.randn(1, 4, 8, 8, generator=generator, dtype=torch.float64)  # one frozen noise prediction
    timestep = schedule.timesteps[0]  # the first generative time, 980 at 50 steps
    nxt = forward_step(schedule, eps, timestep, sample)  # Eq. (14) step from the EDICT code
    back = reverse_step(schedule, eps, timestep, nxt)  # Eq. (15) with the same eps; should return sample
    step_error = float((back - sample).abs().max().item())  # largest coordinate error of that one-step inverse

    def denoise(partner, _timestep):
        return (0.05 * partner).to(dtype=torch.float64)  # a denoiser that depends on the partner and ignores time

    start = [  # two independent draws, replaced on the next line by one shared draw
        torch.randn(1, 4, 16, 16, generator=generator, dtype=torch.float64),  # temporary x
        torch.randn(1, 4, 16, 16, generator=generator, dtype=torch.float64),  # temporary y
    ]
    start[1] = start[0].clone()  # both streams start equal, which is the fresh-noise initialization in edict_functions.py lines 888-891
    final, _trace = coupled_loop(denoise, schedule, start, reverse=False, mix_weight=0.93, leapfrog=True)  # full generation at the paper's default p
    recovered, _back = coupled_loop(denoise, schedule, final, reverse=True, mix_weight=0.93, leapfrog=True)  # matched inverse from the final pair only
    pair_error = max(float((recovered[i] - start[i]).abs().max().item()) for i in (0, 1))  # worst error on either stream
    report = {  # the numbers written to disk
        "steps": 50,  # S
        "mix_weight": 0.93,  # p
        "leapfrog": True,  # the paper's alternating update order
        "fixed_eps_roundtrip_max_abs": step_error,  # one affine step, eps held fixed
        "coupled_roundtrip_max_abs": pair_error,  # all 50 coupled steps
        "denoiser": "0.05 * partner latent",  # what denoise returned
    }
    (OUT / "algebra.json").write_text(json.dumps(report, indent=2) + "\n")  # keep the measured errors
    ok = step_error < 1e-8 and pair_error < 1e-8  # float64 roundoff over 50 steps is far below this
    (OUT / "STATUS.txt").write_text("PASS\n" if ok else f"FAIL\n{report}\n")  # one-line status the job log can show
    print(json.dumps(report), flush=True)  # same numbers on stdout
    print("PASS" if ok else "FAIL", flush=True)  # final word of the log
    return 0 if ok else 2  # nonzero exits the SLURM script because it uses set -e


if __name__ == "__main__":
    sys.exit(main())  # run the check when this file is the program
