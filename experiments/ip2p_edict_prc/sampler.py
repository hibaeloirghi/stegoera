"""EDICT coupling around one denoiser.

The affine updates are the EDICT forward_step and reverse_step
(EDICT/edict_functions.py). The mixing layer uses the same in-place order as
that file, including the fact that the second mix reads the already updated
first stream. Noise is predicted from the partner stream, so each affine
update is linear in the stream it changes.

Paper: Wallace, Gokul, Ermon, "EDICT: Exact Diffusion Inversion via Coupled
Transformations", CVPR 2023. Local copy:
EDICT_Exact_Diffusion_Inversion_via_Coupled_Transformations.pdf.
Original code: EDICT/edict_functions.py.

This module does not load a UNet. The caller supplies denoise(partner, t).
Comments describe the lines. They do not change them.
"""
from __future__ import annotations  # keep annotations as strings so this file imports on older Python

import torch  # tensor math for the latent pair and the alpha schedule


class EdictSchedule:
    """EDICT's 50-step DDIM grid: 980, 960, ..., 0. Not Diffusers leading spacing.

    Original grid: EDICT/my_diffusers/schedulers/scheduling_ddim.py set_timesteps,
    lines 161-163, offset 0. EDICT/edict_functions.py lines 905-910 build the
    same beta schedule and then call set_timesteps.
    """

    def __init__(self, steps: int, device="cpu", dtype=torch.float64):
        from diffusers import DDIMScheduler  # only the alpha table is used; scheduler.step is never called

        if steps <= 0 or 1000 % steps != 0:  # EDICT's stride is 1000 // steps, so steps must divide 1000
            raise ValueError(f"steps must divide 1000, got {steps}")  # refuse a grid the original set_timesteps would not build
        base = DDIMScheduler(  # same constructor arguments as edict_functions.py lines 905-909
            beta_start=0.00085,  # Stable Diffusion beta start, also the InstructPix2Pix checkpoint
            beta_end=0.012,  # Stable Diffusion beta end
            beta_schedule="scaled_linear",  # square-root linear betas, as in EDICT and the checkpoint
            num_train_timesteps=1000,  # training horizon T
            clip_sample=False,  # EDICT turns clipping off so the affine step stays linear
            set_alpha_to_one=False,  # final_alpha_cumprod stays alphas_cumprod[0], matching EDICT
            prediction_type="epsilon",  # the UNet predicts noise, which is what forward_step expects
        )
        self.num_train_timesteps = 1000  # T in the prev-time formula timestep - T / S
        self.num_inference_steps = int(steps)  # S, the number of coupled steps
        self.alphas_cumprod = base.alphas_cumprod.to(device=device, dtype=dtype)  # alpha_bar[0..999], moved to the latent dtype
        self.final_alpha_cumprod = base.final_alpha_cumprod.to(device=device, dtype=dtype)  # alpha_bar used when the previous time is negative
        grid = list(range(0, 1000, 1000 // steps))[::-1]  # EDICT set_timesteps: arange(0, 1000, stride) then reverse, so 980 ... 0 at 50 steps
        self.timesteps = torch.tensor(grid, dtype=dtype, device=device)  # one tensor so the loop can index it


def _coeff(schedule, t, device):
    """alpha_bar and 1-alpha_bar. A negative time uses final_alpha_cumprod.

    This is get_alpha_and_beta in EDICT/edict_functions.py lines 673-691.
    Integer times hit the table directly. A float time linearly interpolates,
    which is what that function does when the tensor is not an integer dtype.
    """
    if not torch.is_tensor(t):  # a plain Python number, used for the previous timestep
        value = float(t)  # make it a Python float so the sign test below is exact
    else:  # a timestep taken from schedule.timesteps
        value = float(t.detach().to(dtype=torch.float64).cpu().item())  # one host float, detached from any graph
    alphas = schedule.alphas_cumprod  # local name for the alpha_bar table
    if value < 0:  # the step before t=0; EDICT lines 679-680
        alpha = schedule.final_alpha_cumprod  # alpha_bar_{-1}
    else:  # t is inside 0..999
        low = int(torch.tensor(value).floor().item())  # floor index, EDICT line 683
        high = int(torch.tensor(value).ceil().item())  # ceil index, EDICT line 684
        rem = value - low  # fractional part, EDICT line 685
        alpha = alphas[low] * rem + alphas[high] * (1 - rem)  # EDICT lines 687-689; an integer time returns alphas[t]
    alpha = alpha.to(device=device, dtype=torch.float64)  # put the scalar on the same device as the latent
    return alpha, 1 - alpha  # (alpha_bar_t, beta_bar_t) with beta_bar = 1 - alpha_bar


def forward_step(schedule, model_output, timestep, sample):
    """One generative DDIM step. Exact inverse is reverse_step for a fixed model_output.

    EDICT/edict_functions.py forward_step, lines 698-727.
    The local paper copy calls this the generative step, Eq. (14) in the code comments.
    Algebra, with q = sqrt(alpha_t / alpha_prev):
        x_prev = x / q - sqrt(beta_t) * eps / q + sqrt(1 - alpha_prev) * eps
    """
    sample = sample.to(dtype=torch.float64)  # EDICT runs the affine arithmetic in high precision
    eps = model_output.to(device=sample.device, dtype=torch.float64)  # the noise prediction, frozen for this update
    step = schedule.num_train_timesteps / schedule.num_inference_steps  # T/S, which is 20 when S is 50; EDICT line 714
    prev = float(timestep.detach().cpu().item()) - step if torch.is_tensor(timestep) else float(timestep) - step  # previous discrete time, EDICT line 714
    alpha_t, beta_t = _coeff(schedule, timestep, sample.device)  # alpha_bar_t and beta_bar_t, EDICT line 719
    alpha_prev, _ = _coeff(schedule, prev, sample.device)  # alpha_bar at the previous time, EDICT line 720
    alpha_quotient = torch.sqrt(alpha_t / alpha_prev)  # q in the comment above; EDICT line 723
    first = sample / alpha_quotient  # (1/q) * x_t ; EDICT line 724
    second = torch.sqrt(beta_t) * eps / alpha_quotient  # (1/q) * sqrt(beta_t) * eps ; EDICT line 725
    third = torch.sqrt(1 - alpha_prev) * eps  # sqrt(1 - alpha_prev) * eps ; EDICT line 726
    return first - second + third  # x_{t-1} ; EDICT line 727


def reverse_step(schedule, model_output, timestep, sample):
    """Algebraic inverse of forward_step for the same model_output.

    EDICT/edict_functions.py reverse_step, lines 733-764.
    The local paper copy calls this the inversion step, Eq. (15) in the code comments.
    It is exact only when model_output is the same tensor the forward step used.
    EDICT obtains that by reading epsilon from the partner stream, not from sample.
    """
    sample = sample.to(dtype=torch.float64)  # the cleaner latent that the forward step produced
    eps = model_output.to(device=sample.device, dtype=torch.float64)  # must be the epsilon from the partner, not a new guess
    step = schedule.num_train_timesteps / schedule.num_inference_steps  # same T/S as the forward step
    prev = float(timestep.detach().cpu().item()) - step if torch.is_tensor(timestep) else float(timestep) - step  # same previous time as the forward step
    alpha_t, beta_t = _coeff(schedule, timestep, sample.device)  # EDICT lines 756-757
    alpha_prev, _ = _coeff(schedule, prev, sample.device)  # EDICT line 757
    alpha_quotient = torch.sqrt(alpha_t / alpha_prev)  # same q as forward_step
    first = alpha_quotient * sample  # q * x_{t-1} ; EDICT line 761
    second = torch.sqrt(beta_t) * eps  # sqrt(beta_t) * eps ; EDICT line 762
    third = alpha_quotient * torch.sqrt(1 - alpha_prev) * eps  # q * sqrt(1 - alpha_prev) * eps ; EDICT line 763
    return first + second - third  # recovers x_t ; EDICT line 764


def mix(pair, p):
    """Forward mixing. The second line reads the updated first stream.

    EDICT/edict_functions.py lines 1048-1050.
    p is the paper's mixing coefficient. The published default is 0.93.
    x <- p x + (1-p) y
    y <- (1-p) x_new + p y
    The second assignment must use the new x. Using the old x is a different layer.
    """
    out = [pair[0].clone(), pair[1].clone()]  # copy both streams before either write, as the original does
    out[0] = (p * out[0] + (1 - p) * out[1]).clone()  # new x; EDICT line 1049
    out[1] = ((1 - p) * out[0] + p * out[1]).clone()  # new y from the new x; EDICT line 1050
    return out  # the contracted pair


def unmix(pair, p):
    """Inverse mixing, y then x, matching edict_functions.

    EDICT/edict_functions.py lines 956-957.
    Solving the forward layer for the old pair gives exactly these two lines,
    and only in this order: recover y from the mixed pair, then recover x
    using that recovered y.
    """
    out = [pair[0].clone(), pair[1].clone()]  # copy before writing, as the original does
    out[1] = (out[1] - (1 - p) * out[0]) / p  # recover y; EDICT line 956
    out[0] = (out[0] - (1 - p) * out[1]) / p  # recover x using the recovered y; EDICT line 957
    return out  # the pair before this mixing layer


def coupled_loop(denoise, schedule, latent_pair, *, reverse, mix_weight=0.93, leapfrog=True):
    """One direction of the coupled sampler.

    Body of coupled_stablediffusion in EDICT/edict_functions.py lines 943-1055,
    with the UNet call replaced by denoise(partner, timestep).
    reverse=False is generation (paper Eq. 14, then the mixing layer).
    reverse=True is the matched inverse (undo the mixing layer, then Eq. 15).
    Leapfrog alternates which stream is updated first. The inverse index rule
    is the original lines 969-975, not a rewritten order.
    """
    timesteps = [schedule.timesteps[i] for i in range(schedule.timesteps.shape[0])]  # the EDICT grid, one entry per step
    if reverse:  # inversion walks from the clean end back to the noise
        timesteps = list(reversed(timesteps))  # EDICT line 941 flips the slice when reverse is true
    pair = [latent_pair[0].to(dtype=torch.float64).clone(), latent_pair[1].to(dtype=torch.float64).clone()]  # x is [0], y is [1]; EDICT clones the start into both streams
    trace = [tuple(item.detach().clone() for item in pair)]  # record the pair before any step; the inverse does not read this
    for i, timestep in enumerate(timesteps):  # i is the position in this direction; EDICT line 943
        if reverse:  # a forward step mixed after the affine updates, so invert the mix first
            pair = unmix(pair, mix_weight)  # EDICT lines 955-958, before the reverse affine steps
        for latent_i in range(2):  # two affine updates, EDICT line 965; the loop variable is reassigned below and the next iteration still comes from range
            if reverse:  # choose which stream to update so the forward order is undone
                if leapfrog:  # EDICT lines 970-975
                    orig_i = len(timesteps) - (i + 1)  # the forward-step index this inverse step corresponds to
                    offset = (orig_i + 1) % 2  # the forward leapfrog offset, shifted by one
                    li = (latent_i + offset) % 2  # stream actually updated on this half-step
                else:  # non-leapfrog inverse does 1 then 0; EDICT lines 976-978
                    li = (latent_i + 1) % 2  # swap the range order
            elif leapfrog:  # forward leapfrog; EDICT lines 980-982, offset = i % 2
                li = (latent_i + (i % 2)) % 2  # even steps update 0 then 1; odd steps update 1 then 0
            else:  # forward without leapfrog updates 0 then 1
                li = latent_i  # the range index is the stream index
            lj = (li + 1) % 2  # the partner stream; EDICT line 984. Epsilon is computed here, not at li
            eps = denoise(pair[lj], timestep)  # noise prediction at the partner. This is the substitution for the UNet call at EDICT lines 990-1023
            step = reverse_step if reverse else forward_step  # EDICT line 1026
            pair[li] = step(schedule, eps, timestep, pair[li]).to(dtype=torch.float64)  # update only stream li; EDICT lines 1027-1033
        if not reverse:  # generation mixes after both affine updates
            pair = mix(pair, mix_weight)  # EDICT lines 1048-1051
        trace.append(tuple(item.detach().clone() for item in pair))  # one snapshot after this timestep; not an input to the inverse
    return pair, trace  # final pair, and the diagnostic snapshots
