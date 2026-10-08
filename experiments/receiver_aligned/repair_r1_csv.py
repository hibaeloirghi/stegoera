#!/usr/bin/env python3
"""Re-align rows of results/receiver_aligned/r1_smoke.csv written by job 7788181.

Cause: io_utils.append_csv writes the header of the FIRST row; the three later rows (true-channel row and two
controls) have different key sets, so DictWriter put their values under the wrong header (extra values were
silently spread or truncated by position). The values are positional in each row's own key order, which is
deterministic (dict insertion order in run_r1_smoke_v2.py). This script rebuilds each row from that order,
asserts the value count matches, and writes a NEW file; the original is left untouched.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
src = ROOT / "results/receiver_aligned/r1_smoke.csv"
dst = ROOT / "results/receiver_aligned/r1_smoke_7788181_true_rows_aligned.csv"

BASE = ["experiment_id", "job_id", "code_hash", "config_hash", "photo_id", "edit_prompt", "inversion_condition",
        "receiver_condition", "site", "active_step", "scheduler_timestep", "carrier_definition", "dimension", "pair_mode",
        "beta", "message_length", "key_id", "F", "t", "sigma", "sigma_squared", "tau", "iteration", "channel_stage"]
FINAL = ["surrogate_margin", "true_margin", "statistic", "threshold", "detect", "decode_non_none", "exact_message_match",
         "overall_success", "sign_agreement", "true_margin_sigma_squared_0.5", "detect_sigma_squared_0.5",
         "exact_message_match_sigma_squared_0.5", "terminal_pair_MSE", "terminal_x_recovery_MSE", "site_carrier_error_oracle",
         "site_carrier_error_vs_baseline_site", "LPIPS_vs_edit", "pixel_MSE_vs_edit", "LPIPS_vs_original", "edit_alignment",
         "time", "peak_memory", "status", "optimization_wall_s", "LPIPS_note"]
WRONG = ["key_id", "true_margin", "statistic", "threshold", "detect", "decode_non_none", "exact_message_match", "status",
         "LPIPS_vs_edit", "note"]
BASEL = ["true_margin", "statistic", "threshold", "detect", "decode_non_none", "exact_message_match", "sign_agreement",
         "LPIPS_vs_edit", "pixel_MSE_vs_edit", "site_carrier_error_oracle", "status", "note"]


def order(update):
    d = dict.fromkeys(BASE)
    d.update(dict.fromkeys(update))
    return list(d)


with open(src, newline="") as f:
    raw = list(csv.reader(f))
rows = raw[1:]
want = [("true_png_decoder_inversion_fp64", FINAL), ("control_wrong_key_true_png", WRONG),
        ("control_unwatermarked_baseline_true_png", BASEL)]
out = []
for stage, upd in want:
    cand = [r for r in rows if stage in r]
    assert len(cand) == 1, (stage, len(cand))
    keys = order(upd)
    vals = cand[0]
    # DictWriter wrote exactly one value per key of the row's own dict; csv.reader gives them back in order
    assert len(vals) == len(keys), (stage, len(vals), len(keys))
    out.append(dict(zip(keys, vals)))
fields = list(dict.fromkeys(k for r in out for k in r))
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out)
for r in out:
    print({k: r.get(k) for k in ("channel_stage", "true_margin", "detect", "decode_non_none", "exact_message_match",
                                  "overall_success", "sign_agreement", "LPIPS_vs_edit", "pixel_MSE_vs_edit",
                                  "LPIPS_vs_original", "edit_alignment", "terminal_pair_MSE", "terminal_x_recovery_MSE",
                                  "site_carrier_error_oracle", "status", "time")})
