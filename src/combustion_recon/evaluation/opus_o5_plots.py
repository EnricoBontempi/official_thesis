"""Common-scale figures for the Opus O5 package. Descriptive only, no ranking."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combustion_recon.evaluation.opus_o5_plan import CHECKPOINTS, ROLE_LABEL

FAMILY_COLOR = {
    ("B19", 0): "#0072B2",
    ("A38", 0): "#D55E00",
    ("A38", 1): "#CC79A7",
    ("C149", 0): "#009E73",
}
CKPT_MARKER = {"best": "o", "best_seen_diagnostic": "s", "last": "^"}
ROLE_SHORT = {"train_fit": "reference760", "seen_train_eval": "seen220", "val_eval": "VAL123"}


def _style(row) -> dict:
    return {
        "color": FAMILY_COLOR[(row.pool_id, int(row.model_seed))],
        "linestyle": "-" if bool(row.p_conditioning) else "--",
    }


def _label(row) -> str:
    return f"{row.pool_id} s{int(row.model_seed)} p_{'on' if bool(row.p_conditioning) else 'off'}"


def _runs(runs: pd.DataFrame):
    return list(runs.itertuples())


def _save(fig, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def loop_curves(loop: pd.DataFrame, glob_rep: pd.DataFrame, runs: pd.DataFrame, dest: Path, x: str = "update") -> None:
    """Training-loop curves (not O5) with the O5 5% values at the three checkpoints."""
    cols = (
        ("train_rep0_rmse_K", "train_fit", "reference760, replica 0"),
        ("seen_rep0_rmse_K", "seen_train_eval", "seen220, replica 0"),
        ("val_mean3_rmse_K", "val_eval", "VAL123, mean of 3 masks"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="col", sharex=True, squeeze=False)
    o5 = glob_rep[np.isclose(glob_rep["coverage"], 0.05)]
    for row in _runs(runs):
        r = 0 if bool(row.p_conditioning) else 1
        curve = loop[loop.run_id == row.run_id].sort_values("epoch")
        for c, (col, role, _) in enumerate(cols):
            ax = axes[r, c]
            ax.plot(curve[x], curve[col], lw=1.2, label=_label(row), **_style(row))
            reps = (0,) if role != "val_eval" else (0, 1, 2)
            for ckpt in CHECKPOINTS:
                part = o5[(o5.run_id == row.run_id) & (o5.checkpoint_kind == ckpt) & (o5.role == role) & (o5.replica.isin(reps))]
                if part.empty:
                    continue
                epoch = int(part["epoch"].iloc[0])
                xv = curve.loc[curve.epoch == epoch, x]
                if xv.empty:
                    continue
                ax.scatter(xv.iloc[0], part["rmse_K"].mean(), marker=CKPT_MARKER[ckpt], s=46, color=FAMILY_COLOR[(row.pool_id, int(row.model_seed))], edgecolor="black", linewidth=0.7, zorder=3)
    for c, (_, _, title) in enumerate(cols):
        axes[0, c].set_title(title)
        axes[1, c].set_xlabel("optimizer updates" if x == "update" else "passes over the pool (presentations / pool size)")
    axes[0, 0].set_ylabel("p on: RMSE (K)")
    axes[1, 0].set_ylabel("p off: RMSE (K)")
    if x != "update":
        for ax in axes.ravel():
            ax.set_xscale("log")
    handles, labels = axes[0, 2].get_legend_handles_labels()
    marker_handles = [plt.Line2D([], [], marker=m, linestyle="", color="white", markeredgecolor="black", label=f"O5 5%: {k}") for k, m in CKPT_MARKER.items()]
    fig.legend(handles + marker_handles, labels + [h.get_label() for h in marker_handles], loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Training-loop curves (training_loop_not_o5), markers = O5 re-evaluation at 5%. Common y per column.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    _save(fig, dest)


def rmse_points_5pct(glob_rep: pd.DataFrame, runs: pd.DataFrame, dest: Path) -> None:
    sel = glob_rep[np.isclose(glob_rep["coverage"], 0.05)]
    roles = ("seen_train_eval", "val_eval")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row", squeeze=False)
    order = [r.run_id for r in _runs(runs)]
    for i, role in enumerate(roles):
        for j, ckpt in enumerate(CHECKPOINTS):
            ax = axes[i, j]
            for k, row in enumerate(_runs(runs)):
                part = sel[(sel.run_id == row.run_id) & (sel.checkpoint_kind == ckpt) & (sel.role == role)]
                color = FAMILY_COLOR[(row.pool_id, int(row.model_seed))]
                ax.scatter([k] * len(part), part["rmse_K"], s=18, color=color, alpha=0.55)
                ax.scatter([k], [part["rmse_K"].mean()], s=60, marker="D" if bool(row.p_conditioning) else "X", color=color, edgecolor="black", zorder=3)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([_label(r) for r in _runs(runs)], rotation=60, ha="right", fontsize=7)
            ax.set_title(f"{ROLE_SHORT[role]}, {ckpt}")
            if j == 0:
                ax.set_ylabel("global RMSE at 5% (K)")
    fig.suptitle("O5 at 5%: mean over conditions; dots = the three mask replicas, diamond/cross = replica mean (p on/off). Common y per row.")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, dest)


def coverage_curves(glob: pd.DataFrame, runs: pd.DataFrame, dest: Path) -> None:
    roles = ("seen_train_eval", "val_eval")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row", sharex=True, squeeze=False)
    for i, role in enumerate(roles):
        for j, ckpt in enumerate(CHECKPOINTS):
            ax = axes[i, j]
            for row in _runs(runs):
                part = glob[(glob.run_id == row.run_id) & (glob.checkpoint_kind == ckpt) & (glob.role == role)].sort_values("coverage")
                ax.errorbar(part["coverage"] * 100, part["rmse_K"], yerr=[part["rmse_K"] - part["rmse_rep_min_K"], part["rmse_rep_max_K"] - part["rmse_K"]], marker="o", ms=4, lw=1.1, capsize=2, label=_label(row), **_style(row))
            ax.set_xscale("log")
            ax.set_xticks([1, 5, 20])
            ax.set_xticklabels(["1%", "5%", "20%"])
            ax.set_title(f"{ROLE_SHORT[role]}, {ckpt}")
            if j == 0:
                ax.set_ylabel("global RMSE (K)")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Coverage 1/5/20% of a 5%-trained model (generalization from TRAIN density, not a sensor-count optimum). Bars = replica min/max.")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, dest)


def val_by_condition(per_condition: pd.DataFrame, runs: pd.DataFrame, dest: Path) -> None:
    sel = per_condition[(np.isclose(per_condition["coverage"], 0.05)) & (per_condition.role == "val_eval")]
    sims = sorted(sel["sim_id"].unique())
    fig, axes = plt.subplots(3, len(sims), figsize=(5 * len(sims), 11), sharey=True, squeeze=False)
    for j, ckpt in enumerate(CHECKPOINTS):
        for i, sim in enumerate(sims):
            ax = axes[j, i]
            for k, row in enumerate(_runs(runs)):
                part = sel[(sel.run_id == row.run_id) & (sel.checkpoint_kind == ckpt) & (sel.sim_id == sim)]
                color = FAMILY_COLOR[(row.pool_id, int(row.model_seed))]
                ax.errorbar([k], part["rmse_K"], yerr=[part["rmse_K"] - part["rmse_rep_min_K"], part["rmse_rep_max_K"] - part["rmse_K"]], fmt="D" if bool(row.p_conditioning) else "X", color=color, capsize=2, ms=6)
            ax.set_xticks(range(len(runs)))
            ax.set_xticklabels([_label(r) for r in _runs(runs)], rotation=60, ha="right", fontsize=7)
            ax.set_title(f"{sim}, {ckpt}")
            if i == 0:
                ax.set_ylabel("RMSE at 5% (K)")
    fig.suptitle("VAL conditions at 5%, replica mean with min/max bars. One common y axis.")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, dest)


def regional(glob: pd.DataFrame, runs: pd.DataFrame, ckpt: str, dest: Path) -> None:
    metrics = (
        ("high_rmse_K", "hot-region RMSE (K)"),
        ("peak_rmse_K", "peak RMSE (K)"),
        ("jet_rmse_K", "jet RMSE (K)"),
        ("non_high_rmse_K", "outside hot region RMSE (K)"),
        ("high_iou", "hot-region IoU"),
        ("high_rel_area_error", "hot-area relative error"),
    )
    sel = glob[np.isclose(glob["coverage"], 0.05) & (glob.checkpoint_kind == ckpt)]
    roles = ("seen_train_eval", "val_eval")
    fig, axes = plt.subplots(2, len(metrics), figsize=(3.4 * len(metrics), 8), sharey="col", squeeze=False)
    for i, role in enumerate(roles):
        for j, (col, title) in enumerate(metrics):
            ax = axes[i, j]
            for k, row in enumerate(_runs(runs)):
                part = sel[(sel.run_id == row.run_id) & (sel.role == role)]
                ax.scatter([k], part[col], s=50, marker="D" if bool(row.p_conditioning) else "X", color=FAMILY_COLOR[(row.pool_id, int(row.model_seed))], edgecolor="black")
            ax.set_xticks(range(len(runs)))
            ax.set_xticklabels([_label(r) for r in _runs(runs)], rotation=70, ha="right", fontsize=6)
            ax.set_title(f"{ROLE_SHORT[role]}: {title}", fontsize=8)
    fig.suptitle(f"Regional metrics at 5%, {ckpt}, replica mean. Common y per column; definitions as in the historical evaluator.")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, dest)


def deltas(summary: pd.DataFrame, dest: Path, metric: str = "rmse_K") -> None:
    sel = summary[(summary.metric == metric) & np.isclose(summary.coverage, 0.05) & (summary.family != "selection")]
    labels = list(dict.fromkeys(sel["label"]))
    roles = ("seen_train_eval", "val_eval")
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(labels) + 2), sharex=True, squeeze=False)
    for ax, role in zip(axes[0], roles):
        for yi, label in enumerate(labels):
            for ckpt in CHECKPOINTS:
                part = sel[(sel.label == label) & (sel.role == role) & (sel.checkpoint_a == ckpt)]
                if part.empty:
                    continue
                off = (CHECKPOINTS.index(ckpt) - 1) * 0.22
                ax.plot([part["min_delta"].iloc[0], part["max_delta"].iloc[0]], [yi + off] * 2, color="#999999", lw=1)
                ax.scatter(part["mean_delta"], [yi + off], marker=CKPT_MARKER[ckpt], s=40, color="black", zorder=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if role == roles[0] else [], fontsize=8)
        ax.set_title(f"{ROLE_SHORT[role]}: delta RMSE at 5% (K), a minus b")
        ax.invert_yaxis()
    handles = [plt.Line2D([], [], marker=m, linestyle="", color="black", label=k) for k, m in CKPT_MARKER.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Paired per-condition deltas (marker = mean over conditions, bar = min..max over conditions). Common x axis. Descriptive only.")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(fig, dest)


def make_all(dest: Path, tables: dict, loop: pd.DataFrame, runs: pd.DataFrame, cmp: dict) -> list[Path]:
    glob, glob_rep, per_condition = tables["global_by_setting"], tables["global_by_replica"], tables["per_condition"]
    out = [
        dest / "loop_curves_vs_update.png",
        dest / "loop_curves_vs_pool_passages.png",
        dest / "o5_rmse_points_5pct.png",
        dest / "o5_coverage_curves.png",
        dest / "o5_val_by_condition_5pct.png",
        dest / "o5_delta_rmse_5pct.png",
    ]
    loop_curves(loop, glob_rep, runs, out[0], "update")
    loop_curves(loop, glob_rep, runs, out[1], "pool_passages")
    rmse_points_5pct(glob_rep, runs, out[2])
    coverage_curves(glob, runs, out[3])
    val_by_condition(per_condition, runs, out[4])
    deltas(cmp["compare_summary"], out[5])
    for ckpt in CHECKPOINTS:
        path = dest / f"o5_regional_5pct_{ckpt}.png"
        regional(glob, runs, ckpt, path)
        out.append(path)
    return out
