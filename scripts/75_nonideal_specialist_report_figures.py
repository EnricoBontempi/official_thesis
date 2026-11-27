#!/usr/bin/env python3
"""Report figures for docs/nonideal/05_specialist_results.md, drawn only from the published aggregated tables.

Sign convention everywhere: gain = RMSE_original - RMSE_specialist (K), positive = specialist better (blue).
Clean cost = RMSE_specialist - RMSE_original on R0 (K), positive = specialist worse. Dots are the three training seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

BLUE, ORANGE, INK, INK2, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
REG = ["N1", "N2", "G1", "G1-eq", "N1G1"]
CELLS = ["R0", "N1", "N2", "G1", "G1-eq", "N1G1"]
FAM = {"sparse_unet": ("U-Net", BLUE), "adapted_glu": ("GLU", ORANGE)}
ROLE = {"seen_train_eval": "seen (frame 1600-2000 delle 20 condizioni TRAIN)", "val_eval": "VAL (3 condizioni mai viste in training)"}
plt.rcParams.update({"font.size": 10, "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
DIV = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)


def dots(ax, d, col, ck, role, ylabel=None):
    for i, (fam, (nm, c)) in enumerate(FAM.items()):
        off = -0.17 + 0.34 * i
        for j, reg in enumerate(REG):
            g = d[(d.family == fam) & (d.checkpoint_kind == ck) & (d.role == role) & (d.train_regime == reg)].sort_values("model_seed")[col].to_numpy()
            ax.scatter(np.full(3, j + off) + np.array([-0.05, 0, 0.05]), g, s=26, color=c, alpha=0.85, edgecolor=SURF, lw=0.8, zorder=3)
            ax.hlines(g.mean(), j + off - 0.11, j + off + 0.11, color=c, lw=3, zorder=4)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(REG); style(ax)
    ax.set_title(ROLE[role], loc="left", fontsize=10, color=INK2)
    if ylabel: ax.set_ylabel(ylabel)


def legend(fig, y=0.995):
    for k, (nm, c) in enumerate(FAM.values()):
        fig.text(0.07 + 0.12 * k, y, "● " + nm, color=c, fontsize=11, va="top", weight="bold")
    fig.text(0.32, y, "punti = 3 seed di training; tratto spesso = media", color=INK2, fontsize=9, va="top")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reading", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.reading / "own_scenario_by_run.csv")
    t = pd.read_csv(a.reading / "transfer_matrix_gain.csv")
    b = pd.read_csv(a.reading / "baseline_deterioration.csv")

    # A: how much the original models lose under each degradation (Block II, p_on)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, role in zip(axs, ["seen_train_eval", "val_eval"]):
        for i, (fam, (nm, c)) in enumerate(FAM.items()):
            v = [b[(b.family == fam) & (b.role == role) & (b.cell == k)].baseline_deterioration_K.iloc[0] for k in CELLS[1:]]
            ax.bar(np.arange(5) + (-0.2 + 0.4 * i), v, 0.36, color=c, label=nm)
            for x, y in zip(np.arange(5) + (-0.2 + 0.4 * i), v): ax.text(x, y + 0.12, f"{y:.1f}", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(range(5)); ax.set_xticklabels(CELLS[1:]); style(ax); ax.set_title(ROLE[role], loc="left", fontsize=10, color=INK2)
    axs[0].set_ylabel("peggioramento dell'originale\nrispetto al proprio R0 (K)"); axs[0].legend(frameon=False)
    fig.suptitle("Blocco II - quanto perdono i modelli originali p_on (media sui 3 seed)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(a.out / "figA_original_models_degradation.png", dpi=150); plt.close(fig)

    # B: gain on the training regime (best)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, role in zip(axs, ["seen_train_eval", "val_eval"]):
        dots(ax, d, "gain_vs_baseline_K", "best", role)
        ax.set_xlabel("regime di training = scenario valutato")
    axs[0].set_ylabel("guadagno = RMSE originale - RMSE specialista (K)\n↑ specialista migliore | ↓ peggiore")
    legend(fig); fig.suptitle("", y=1); fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(a.out / "figB_own_scenario_gain_best.png", dpi=150); plt.close(fig)

    # C: clean cost (best)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, role in zip(axs, ["seen_train_eval", "val_eval"]):
        dots(ax, d, "clean_cost_K", "best", role); ax.set_xlabel("regime di training dello specialista")
    axs[0].set_ylabel("costo sul pulito R0 = RMSE specialista - RMSE originale (K)\n↑ specialista PEGGIORE | ↓ migliore")
    legend(fig); fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(a.out / "figC_clean_cost_best.png", dpi=150); plt.close(fig)

    # D: transfer matrices, VAL best, gain sign
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, (fam, (nm, c)) in zip(axs, FAM.items()):
        M = np.array([[t[(t.family == fam) & (t.checkpoint_kind == "best") & (t.role == "val_eval") & (t.train_regime == r) & (t.cell == k)]["mean"].iloc[0] for k in CELLS] for r in REG])
        im = ax.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0, -5.5, 5.5), aspect="auto")
        for i in range(5):
            for j in range(6): ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=9, color=INK)
        ax.set_xticks(range(6)); ax.set_xticklabels(CELLS); ax.set_yticks(range(5)); ax.set_yticklabels(REG)
        ax.set_xlabel("scenario valutato (R0 = pulito)"); ax.set_title(f"{nm}, VAL, best", loc="left", fontsize=10, color=INK2)
        for s in ax.spines.values(): s.set_visible(False)
    axs[0].set_ylabel("regime di training")
    cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.02); cb.set_label("guadagno (K): blu = specialista meglio, rosso = peggio")
    fig.savefig(a.out / "figD_transfer_gain_val_best.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # E: best vs last, own scenario gain, VAL and seen
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, role in zip(axs, ["seen_train_eval", "val_eval"]):
        for i, (fam, (nm, c)) in enumerate(FAM.items()):
            for k, (ck, mk) in enumerate([("best", "o"), ("last", "s")]):
                mean = [d[(d.family == fam) & (d.checkpoint_kind == ck) & (d.role == role) & (d.train_regime == r)].gain_vs_baseline_K.mean() for r in REG]
                x = np.arange(5) + (-0.2 + 0.4 * i) + (-0.04 + 0.08 * k)
                ax.scatter(x, mean, marker=mk, s=45, color=c, facecolor=c if ck == "best" else SURF, edgecolor=c, lw=1.6, zorder=3)
        ax.axhline(0, color=INK2, lw=1); ax.set_xticks(range(5)); ax.set_xticklabels(REG); style(ax)
        ax.set_title(ROLE[role], loc="left", fontsize=10, color=INK2); ax.set_xlabel("regime di training = scenario valutato")
    axs[0].set_ylabel("guadagno medio sui 3 seed (K)\n↑ specialista migliore")
    for k, (nm, c) in enumerate(FAM.values()): fig.text(0.07 + 0.12 * k, 0.995, "● " + nm, color=c, fontsize=11, va="top", weight="bold")
    fig.text(0.32, 0.995, "pieno = best (selezionato su VAL pulita); vuoto = last", color=INK2, fontsize=9, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(a.out / "figE_best_vs_last_own_gain.png", dpi=150); plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
