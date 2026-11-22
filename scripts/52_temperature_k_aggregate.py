#!/usr/bin/env python3
"""Tables and figures for the K sweep. No estimator is selected."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.evaluation.temperature_k import (  # noqa: E402
    per_frame_csv_paths,
    summarize_sweep,
)

_KEYS = [
    "run_id", "method", "p_conditioning", "model_seed", "checkpoint", "role",
    "mask_replica_id", "sim_id", "frame_index", "layout_id",
]


def _read(root: Path) -> pd.DataFrame:
    paths = per_frame_csv_paths(root)
    if not paths:
        raise SystemExit(f"no per-frame CSV under {root}")
    parts = [pd.read_csv(path) for path in paths]
    frame = pd.concat(parts, ignore_index=True)
    if not frame["full_rmse_K"].map(lambda value: value == value).all():
        raise SystemExit("a full-field RMSE is NaN")
    return frame


def _same(frame: pd.DataFrame, treatment: str, other: str, column: str, k: int) -> float:
    keys = _KEYS
    left = frame[(frame["K"] == k) & (frame["treatment"] == treatment)][keys + [column]]
    right = frame[(frame["K"] == k) & (frame["treatment"] == other)][keys + [column]]
    if left.empty or right.empty:
        raise SystemExit(f"missing rows at K={k}: {treatment}={len(left)} {other}={len(right)}")
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise SystemExit(f"duplicate identity for {treatment} or {other} at K={k}")
    merged = left.merge(right, on=keys, suffixes=("_a", "_b"))
    if len(merged) != len(left) or len(merged) != len(right):
        raise SystemExit(
            f"{treatment} and {other} do not share rows at K={k}: {len(left)} vs {len(right)} joined {len(merged)}"
        )
    return float((merged[f"{column}_a"] - merged[f"{column}_b"]).abs().max())


def _checks(frame: pd.DataFrame) -> dict[str, float]:
    primary = frame[frame["layout_kind"] == "primary"]
    bridge = frame[frame["layout_kind"] == "t9_bridge"]
    if primary.empty or bridge.empty:
        raise SystemExit("primary rows or the T9 bridge are missing")
    if not (primary["common_n"] == 16352).all():
        raise SystemExit("a primary row does not exclude the 32 layout sites")
    if not (bridge["common_n"] == 0).all():
        raise SystemExit("the T9 bridge was given a common-domain count")
    if set(frame["role"]) != {"seen_train_eval", "val_eval"}:
        raise SystemExit(f"unexpected roles {sorted(frame['role'].unique())}")
    neural = frame[frame["family"].isin(["sparse_unet", "adapted_glu"])]
    runs = set(neural["run_id"])
    if len(runs) != 12:
        raise SystemExit(f"{len(runs)} neural runs, expected 12")
    if set(neural["checkpoint"]) != {"best", "last"}:
        raise SystemExit("neural checkpoints are not best and last")
    if set(neural["mask_replica_id"]) != {0, 1, 2}:
        raise SystemExit("neural replicas are not 0, 1 and 2")
    if set(int(k) for k in primary["K"].unique()) != {0, 1, 4, 9, 16, 32}:
        raise SystemExit(f"unexpected K {sorted(primary['K'].unique())}")
    scalar = primary[primary["treatment"].isin(["free", "ao", "mean", "median"])]
    report = {
        "k0_ao_full": _same(scalar, "ao", "free", "full_rmse_K", 0),
        "k0_mean_full": _same(scalar, "mean", "free", "full_rmse_K", 0),
        "k0_median_full": _same(scalar, "median", "free", "full_rmse_K", 0),
        "k0_ao_centered": _same(scalar, "ao", "free", "full_centered_rmse_K", 0),
        "k0_mean_centered": _same(scalar, "mean", "free", "full_centered_rmse_K", 0),
        "k0_median_centered": _same(scalar, "median", "free", "full_centered_rmse_K", 0),
        "k1_mean_median": _same(primary, "mean", "median", "full_rmse_K", 1),
    }
    if any(value != 0.0 for value in report.values()):
        raise SystemExit(f"identity check failed: {report}")
    return report


def _figure(by_seed: pd.DataFrame, dest: Path) -> None:
    primary = by_seed[(by_seed["layout_kind"] == "primary") & (by_seed["checkpoint"].isin(["best", "fitted"]))]
    primary = primary[primary["role"] == "val_eval"]
    treatments = ["free", "ao", "mean", "median"]
    families = list(primary["method"].drop_duplicates())
    if not families:
        return
    fig, axes = plt.subplots(len(families), 1, figsize=(8, 2.4 * len(families)), sharex=True, squeeze=False)
    for ax, method in zip(axes[:, 0], families):
        part = primary[primary["method"] == method]
        for (p_on, seed, treatment), grp in part.groupby(["p_conditioning", "model_seed", "treatment"]):
            if treatment not in treatments:
                continue
            curve = grp.groupby("K")["common_rmse_K"].mean().sort_index()
            ax.plot(curve.index, curve.values, marker="o", label=f"p={int(p_on)} s={seed} {treatment}")
        ax.set_ylabel("common RMSE (K)")
        ax.set_title(str(method))
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("K")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--forward-sha", required=True)
    args = parser.parse_args()
    neural = _read(args.neural_root)
    baseline = _read(args.baseline_root)
    frame = pd.concat([neural, baseline], ignore_index=True)
    report = _checks(frame)
    tables = summarize_sweep(frame)
    args.dest.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.dest / f"{name}.csv", index=False)
    _figure(tables["by_seed"], args.dest / "figures" / "val_common_rmse_vs_K.png")
    worse = tables["versus_free"]
    n_worse = int(worse["worse_full"].fillna(False).sum()) if "worse_full" in worse else 0
    (args.dest / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.dest / "INDEX.md").write_text(
        "\n".join([
            "# Temperature K sweep",
            "",
            f"Forward and correction SHA `{args.forward_sha}`.",
            f"Aggregation SHA `{args.git_sha}`.",
            f"Rows: {len(frame)}. Neural CSV files live under `<run>/per_frame`; baseline CSV files live under `per_frame`.",
            f"Seeds, replicas, K and treatments are in `by_seed.csv`.",
            f"Paired changes versus free, including worsenings: {n_worse} cells with a higher full-field RMSE (`versus_free.csv`).",
            "K=0 AO, mean and median match free on full RMSE and centered RMSE. K=1 mean matches median. See `checks.json`.",
            "Historical T9 rows have `layout_kind=t9_bridge` and are not in the primary layout curves.",
            "No estimator is selected.",
            "",
        ])
    )
    print(args.dest / "INDEX.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
