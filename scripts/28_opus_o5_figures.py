#!/usr/bin/env python3
"""Common-scale VAL frame-1000 panels for a finished Opus O5 directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.evaluation.opus_o5_plan import (  # noqa: E402
    CHECKPOINTS,
    REQUIRED_RUN_IDS,
    shared_field_scales,
)


def load_panels(out_root: Path, checkpoint: str) -> list[dict]:
    panels = []
    for run_id in REQUIRED_RUN_IDS:
        path = out_root / run_id / "fields" / f"val_frame1000_{checkpoint}_cov05_rep0.npz"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        payload = np.load(path, allow_pickle=True)
        panels.append(
            {
                "run_id": run_id,
                "sim_id": [str(item) for item in payload["sim_id"].tolist()],
                "pred": np.asarray(payload["pred_T_K"], dtype=np.float32),
                "true": np.asarray(payload["true_T_K"], dtype=np.float32),
            }
        )
    return panels


def render(panels: list[dict], dest: Path, checkpoint: str) -> None:
    n_cond = len(panels[0]["sim_id"])
    trues = []
    errors = []
    for panel in panels:
        for i in range(n_cond):
            trues.append(panel["true"][i])
            errors.append(panel["pred"][i] - panel["true"][i])
    limits = shared_field_scales(trues, errors)
    fig, axes = plt.subplots(
        len(panels),
        n_cond * 2,
        figsize=(2.2 * n_cond * 2 + 1.5, 2.1 * len(panels) + 1.6),
        squeeze=False,
        layout="constrained",
    )
    im_pred = im_err = None
    for r, panel in enumerate(panels):
        for c in range(n_cond):
            pred_ax = axes[r, c]
            err_ax = axes[r, n_cond + c]
            im_pred = pred_ax.imshow(
                panel["pred"][c],
                vmin=limits["vmin"],
                vmax=limits["vmax"],
                cmap="inferno",
            )
            err = panel["pred"][c] - panel["true"][c]
            im_err = err_ax.imshow(err, vmin=-limits["emax"], vmax=limits["emax"], cmap="coolwarm")
            if r == 0:
                pred_ax.set_title(f"{panel['sim_id'][c]}\nprediction", fontsize=7)
                err_ax.set_title(f"{panel['sim_id'][c]}\nsigned error", fontsize=7)
            if c == 0:
                pred_ax.set_ylabel(panel["run_id"])
            pred_ax.set_xticks([])
            pred_ax.set_yticks([])
            err_ax.set_xticks([])
            err_ax.set_yticks([])
    fig.suptitle(
        f"VAL frame 1000, {checkpoint}, 5% replica 0. "
        f"Shared T [{limits['vmin']:.0f}, {limits['vmax']:.0f}] K, "
        f"shared error ±{limits['emax']:.0f} K. One frame, not the VAL set.",
        fontsize=9,
    )
    fig.colorbar(im_pred, ax=axes[:, :n_cond], shrink=0.5, label="T (K)", location="bottom")
    fig.colorbar(im_err, ax=axes[:, n_cond:], shrink=0.5, label="signed error (K)", location="bottom")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best", choices=CHECKPOINTS)
    args = parser.parse_args()
    if args.dest.exists():
        raise SystemExit(f"refusing to overwrite {args.dest}")
    render(load_panels(args.out_root, args.checkpoint), args.dest, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
