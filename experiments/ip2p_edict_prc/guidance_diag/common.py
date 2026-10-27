"""Shared trace rows, hashes, and CSV publication for the guidance diagnostic.

Path D is scored against the forward snapshots only after the inverse has
returned. The inverse function in the runners does not receive those snapshots.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "ip2p_edict_prc" / "guidance_diag"
PARTS = OUT / "parts"

# Predeclared in GUIDANCE_DIAG_PLAN.md. Not a threshold to be moved after plots.
DEPARTURE_MSE = 1e-4

TRACE_FIELDS = [
    "pipeline", "edit", "guidance_config", "path", "step", "t",
    "mse_xy", "maxabs_xy", "norm_ratio",
    "state_err_x", "state_err_y", "state_corr_x", "state_corr_y",
]
RECEIVER_FIELDS = [
    "experiment_id", "job_id", "code_hash", "config_hash",
    "image_hash", "conditioning_hash", "key_id",
    "receiver_level", "guidance_text", "guidance_image", "guidance_kind",
    "recovered_stream",
    "initial_noise_MSE", "initial_noise_correlation", "initial_noise_role",
    "sign_agreement",
    "sigma", "sigma_squared",
    "detect", "statistic", "threshold", "margin",
    "decode_non_none", "exact_message_match",
    "terminal_xy_MSE", "terminal_latent_MSE", "terminal_latent_corr",
    "runtime", "peak_memory", "status",
    "wrong_key_detect", "wrong_key_decode_non_none",
]
QUALITY_FIELDS = [
    "experiment_id", "edit", "guidance_text", "guidance_image", "role",
    "lpips_vs_source", "lpips_error", "pixel_mse_vs_source",
    "directional_clip", "clip_instruction", "clip_source_caption",
    "source_caption", "instruction", "image_path",
]
DEPARTURE_FIELDS = [
    "pipeline", "edit", "guidance_config",
    "terminal_xy_mse", "terminal_xy_maxabs",
    "first_x_departure_step", "first_y_departure_step",
    "path_d_step0_state_err_x", "path_d_step0_state_err_y",
    "path_i_final_state_err_x", "path_i_final_state_err_y",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def git_rev() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def code_hash(paths: list[Path]) -> str:
    blob = b"".join(Path(path).read_bytes() for path in paths)
    return sha256_bytes(blob)


def require_hash(path: Path, expected: str, label: str) -> None:
    got = sha256_file(path)
    if got != expected:
        raise SystemExit(f"{label} hash mismatch for {path}: {got}")


def flatten(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(dtype=torch.float64).flatten().cpu()


def pair_gap(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    difference = flatten(x) - flatten(y)
    y_norm = float(flatten(y).norm().item())
    ratio = float(flatten(x).norm().item() / y_norm) if y_norm > 0 else float("inf")
    return (
        float(difference.pow(2).mean().item()),
        float(difference.abs().max().item()),
        ratio,
    )


def vector_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    difference = flatten(a) - flatten(b)
    return float(difference.pow(2).mean().item())


def vector_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    left = flatten(a)
    right = flatten(b)
    if torch.allclose(left, left[:1].expand_as(left)) or torch.allclose(right, right[:1].expand_as(right)):
        return float("nan")
    return float(torch.corrcoef(torch.stack([left, right]))[0, 1].item())


def snaps_from_trace(trace) -> list[list[torch.Tensor]]:
    return [
        [item.detach().to(dtype=torch.float64).cpu().clone() for item in pair]
        for pair in trace
    ]


def trace_rows(
    pipeline: str,
    edit: str,
    guidance_config: str,
    path: str,
    snaps: list[list[torch.Tensor]],
    step_times: list,
    forward_snaps: list[list[torch.Tensor]],
) -> list[dict]:
    """One row per snapshot. Inverse step k is aligned to forward snapshot 50-k."""
    if path == "F":
        reference = snaps
    else:
        reference = forward_snaps
    if len(snaps) != len(reference):
        raise RuntimeError(f"{path} has {len(snaps)} snapshots, forward has {len(reference)}")
    last = len(reference) - 1
    rows = []
    for step, pair in enumerate(snaps):
        ref = reference[step] if path == "F" else reference[last - step]
        mse_xy, maxabs_xy, norm_ratio = pair_gap(pair[0], pair[1])
        when = step_times[step]
        rows.append({
            "pipeline": pipeline,
            "edit": edit,
            "guidance_config": guidance_config,
            "path": path,
            "step": step,
            "t": "" if when is None else int(when),
            "mse_xy": mse_xy,
            "maxabs_xy": maxabs_xy,
            "norm_ratio": norm_ratio,
            "state_err_x": vector_mse(pair[0], ref[0]),
            "state_err_y": vector_mse(pair[1], ref[1]),
            "state_corr_x": vector_corr(pair[0], ref[0]),
            "state_corr_y": vector_corr(pair[1], ref[1]),
        })
    return rows


def departure_row(pipeline: str, edit: str, guidance_config: str, rows: list[dict]) -> dict:
    def select(path: str) -> list[dict]:
        return sorted((row for row in rows if row["path"] == path), key=lambda row: int(row["step"]))

    forward = select("F")
    duplicated = select("D")
    inverse = select("I")
    terminal = forward[-1]
    step0 = duplicated[0]
    final_i = inverse[-1]

    def first(path_rows: list[dict], field: str, minimum_step: int):
        for row in path_rows:
            if int(row["step"]) >= minimum_step and float(row[field]) > DEPARTURE_MSE:
                return int(row["step"])
        return ""

    return {
        "pipeline": pipeline,
        "edit": edit,
        "guidance_config": guidance_config,
        "terminal_xy_mse": terminal["mse_xy"],
        "terminal_xy_maxabs": terminal["maxabs_xy"],
        "first_x_departure_step": first(duplicated, "state_err_x", 1),
        "first_y_departure_step": first(duplicated, "state_err_y", 0),
        "path_d_step0_state_err_x": step0["state_err_x"],
        "path_d_step0_state_err_y": step0["state_err_y"],
        "path_i_final_state_err_x": final_i["state_err_x"],
        "path_i_final_state_err_y": final_i["state_err_y"],
    }


def write_new_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path = Path(path)
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _publish_csv(pattern: str, fields: list[str], destination: Path) -> None:
    rows: list[dict] = []
    for path in sorted(PARTS.glob(pattern)):
        rows.extend(_read_csv(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    os.replace(temporary, destination)


def publish() -> None:
    """Rebuild the combined deliverables from per-job part files."""
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / ".publish.lock"
    lock = lock_path.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    _publish_csv("trace_*.csv", TRACE_FIELDS, OUT / "trace_per_step.csv")
    _publish_csv("receiver_*.csv", RECEIVER_FIELDS, OUT / "receiver_results.csv")
    _publish_csv("quality_*.csv", QUALITY_FIELDS, OUT / "edit_quality.csv")
    _publish_csv("departure_*.csv", DEPARTURE_FIELDS, OUT / "departure_summary.csv")
    manifests = []
    for path in sorted(PARTS.glob("manifest_*.json")):
        manifests.append(json.loads(path.read_text()))
    temporary = OUT / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifests, indent=2, default=str) + "\n")
    os.replace(temporary, OUT / "manifest.json")
    status_lines = []
    for path in sorted(PARTS.glob("status_*.txt")):
        status_lines.append(path.read_text().rstrip())
    (OUT / "STATUS.txt").write_text("\n\n".join(status_lines) + "\n")
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()


def overlay_plot(rows: list[dict], title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, style in (("F", "-"), ("I", "--"), ("D", ":")):
        chosen = [row for row in rows if row["path"] == name]
        chosen.sort(key=lambda row: int(row["step"]))
        steps = [int(row["step"]) for row in chosen]
        axes[0].plot(steps, [float(row["mse_xy"]) for row in chosen], linestyle=style, label=name)
        if name != "F":
            axes[1].plot(steps, [float(row["state_err_x"]) for row in chosen], linestyle=style, label=f"{name} x")
            axes[1].plot(steps, [float(row["state_err_y"]) for row in chosen], linestyle=style, label=f"{name} y")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("MSE(x, y)")
    axes[0].legend()
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("state MSE vs forward")
    axes[1].set_yscale("symlog", linthresh=1e-8)
    axes[1].axhline(DEPARTURE_MSE, color="black", linewidth=0.6)
    axes[1].legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _self_check() -> None:
    generator = torch.Generator().manual_seed(0)
    forward = []
    current = [torch.randn(4, generator=generator), torch.randn(4, generator=generator)]
    forward.append([item.clone() for item in current])
    for _ in range(50):
        current = [item + 0.1 for item in current]
        forward.append([item.clone() for item in current])
    times = [None] + list(range(50))
    reversed_forward = list(reversed(forward))
    rows = trace_rows("test", "edit", "g", "I", reversed_forward, times, forward)
    if any(float(row["state_err_x"]) > 1e-12 for row in rows):
        raise SystemExit("inverse alignment self-check failed")
    duplicated = [[forward[-1][0].clone(), forward[-1][0].clone()]]
    cursor = [item.clone() for item in duplicated[0]]
    for _ in range(50):
        cursor = [item + 0.2 for item in cursor]
        duplicated.append([item.clone() for item in cursor])
    d_rows = trace_rows("test", "edit", "g", "D", duplicated, times, forward)
    if float(d_rows[0]["state_err_x"]) > 1e-12:
        raise SystemExit("duplicated x should match the forward final x at step 0")
    if float(d_rows[0]["state_err_y"]) <= DEPARTURE_MSE:
        raise SystemExit("step-0 y error should be the terminal gap")
    forward_rows = trace_rows("test", "edit", "g", "F", forward, times, forward)
    summary = departure_row("test", "edit", "g", forward_rows + rows + d_rows)
    if summary["first_y_departure_step"] != 0:
        raise SystemExit(f"y departure should be step 0, got {summary['first_y_departure_step']}")
    if int(summary["first_x_departure_step"]) < 1:
        raise SystemExit("x departure must be after the initial duplicate")
    print("self-check ok", float(d_rows[0]["state_err_y"]), "x_depart", summary["first_x_departure_step"])


if __name__ == "__main__":
    _self_check()
