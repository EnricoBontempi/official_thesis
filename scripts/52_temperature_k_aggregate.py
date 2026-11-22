#!/usr/bin/env python3
"""Tables and figures for the K sweep. No estimator is selected."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.evaluation.temperature_k import summarize_sweep  # noqa: E402


def _read(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("per_frame/**/*.csv"))
    if not paths:
        raise SystemExit(f"no per-frame CSV under {root}")
    parts = [pd.read_csv(path) for path in paths]
    frame = pd.concat(parts, ignore_index=True)
    if not frame["full_rmse_K"].map(lambda v: v == v).all():
        raise SystemExit("a full-field RMSE is NaN")
    return frame


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
    args = parser.parse_args()
    neural = _read(args.neural_root)
    baseline = _read(args.baseline_root)
    frame = pd.concat([neural, baseline], ignore_index=True)
    tables = summarize_sweep(frame)
    args.dest.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.dest / f"{name}.csv", index=False)
    _figure(tables["by_seed"], args.dest / "figures" / "val_common_rmse_vs_K.png")
    worse = tables["versus_free"]
    n_worse = int(worse["worse_full"].fillna(False).sum()) if "worse_full" in worse else 0
    (args.dest / "INDEX.md").write_text(
        "\n".join([
            "# Temperature K sweep",
            "",
            f"Git SHA `{args.git_sha}`.",
            f"Rows: {len(frame)}. Seeds, replicas, K and treatments are in `by_seed.csv`.",
            f"Paired changes versus free, including worsenings: {n_worse} cells with a higher full-field RMSE (`versus_free.csv`).",
            "Historical T9 rows have `layout_kind=t9_bridge` and are not in the primary layout curves.",
            "No estimator is selected.",
            "",
        ])
    )
    print(args.dest / "INDEX.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
