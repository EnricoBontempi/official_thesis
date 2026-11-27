#!/usr/bin/env python3
"""Discussion figures for the finished Block II robustness package.

Reads only the published aggregate tables. Does not reload per-frame rows,
does not run inference, and does not rewrite the original package.

Outputs go to results/nonideal/block2/reading/.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/nonideal/block2/package/tables"
DEST = ROOT / "results/nonideal/block2/reading"
FIG = DEST / "figures"
TAB = DEST / "tables"

SERIES = [
    ("sparse_unet", "p_on", "U-Net, con φ e NH3", "#0072B2"),
    ("sparse_unet", "p_off", "U-Net, senza φ e NH3", "#009E73"),
    ("adapted_glu", "p_on", "GLU, con φ e NH3", "#D55E00"),
    ("adapted_glu", "p_off", "GLU, senza φ e NH3", "#CC79A7"),
]
OFFSET = { (f, v): o for (f, v, _, _), o in zip(SERIES, (-0.27, -0.09, 0.09, 0.27)) }
MARK = {0: "o", 1: "s", 2: "^"}
SEED_NAME = {0: "seed 0", 1: "seed 1", 2: "seed 2"}
ROLE = {"seen_train_eval": "seen220", "val_eval": "VAL123"}
CELLS_DEG = ["N1", "N2", "G1-eq", "G1", "N1G1"]
CELL_LABEL = {
    "N1": "N1\n0,5 m/s",
    "N2": "N2\n1,5 m/s",
    "G1-eq": "G1-eq\nsiti a caso",
    "G1": "G1\ndischi",
    "N1G1": "N1G1\n0,5 m/s + dischi",
}


def _series_handles() -> list:
    handles = [
        plt.Line2D([], [], color=c, marker="o", ls="", ms=8, label=lab)
        for _, _, lab, c in SERIES
    ]
    handles += [
        plt.Line2D([], [], color="0.25", marker=m, ls="", ms=8, label=SEED_NAME[s])
        for s, m in MARK.items()
    ]
    return handles


def _attach_legend(fig, caption: str, extra: list | None = None) -> None:
    handles = _series_handles() + (extra or [])
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055),
        ncol=4, frameon=False, fontsize=9,
    )
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8.5, color="0.15")


def _cond_label(name: str) -> str:
    nh3, phi = name.replace(".h5", "").split("NH3_")
    return f"{nh3}% NH3, φ={phi.replace('.', ',')}"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def _points(ax, df: pd.DataFrame, cats: list[str], y: str, xcol: str = "cell") -> None:
    for fam, var, _, color in SERIES:
        sub = df[(df.family == fam) & (df.variant == var)]
        off = OFFSET[(fam, var)]
        for seed, marker in MARK.items():
            s = sub[sub.model_seed == seed]
            if s.empty:
                continue
            xs = [cats.index(c) + off for c in s[xcol]]
            ax.scatter(xs, s[y], s=36, c=color, marker=marker, zorder=3, linewidths=0)


def _finish_cats(ax, cats: list[str], labels: list[str] | None = None) -> None:
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(labels or cats)
    ax.set_xlim(-0.55, len(cats) - 0.45)
    ax.axhline(0, color="0.45", lw=0.8, zorder=0)


def load() -> dict[str, pd.DataFrame]:
    names = [
        "absolute_per_model", "delta_vs_R0_per_model", "absolute_seed_summary",
        "delta_seed_summary", "structure_contrasts", "delta_vs_R0_by_condition",
        "regional_g1", "sites_and_area", "pon_minus_poff",
    ]
    return {n: pd.read_csv(SRC / f"{n}.csv") for n in names}


def verify(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Recompute the report's headline RMSE table from the aggregate CSVs."""
    expected = {
        ("seen_train_eval", "sparse_unet", "p_on"):  [42.32, 42.62, 44.91, 47.74, 43.81, 48.14],
        ("seen_train_eval", "sparse_unet", "p_off"): [44.77, 45.09, 47.92, 50.98, 46.88, 51.45],
        ("seen_train_eval", "adapted_glu", "p_on"):   [50.07, 50.33, 53.36, 57.60, 52.35, 57.85],
        ("seen_train_eval", "adapted_glu", "p_off"):  [53.34, 53.65, 58.08, 61.47, 55.84, 61.74],
        ("val_eval", "sparse_unet", "p_on"):          [39.73, 39.89, 41.06, 43.84, 41.74, 44.09],
        ("val_eval", "sparse_unet", "p_off"):         [40.59, 41.12, 43.97, 46.20, 44.19, 46.63],
        ("val_eval", "adapted_glu", "p_on"):          [42.73, 42.91, 45.67, 47.07, 44.02, 47.30],
        ("val_eval", "adapted_glu", "p_off"):         [44.49, 44.59, 48.63, 50.04, 46.00, 50.38],
    }
    cells = ["R0", "N1", "N2", "G1", "G1-eq", "N1G1"]
    abs_s = tables["absolute_seed_summary"]
    rows = []
    for (role, fam, var), exp in expected.items():
        for cell, e in zip(cells, exp):
            hit = abs_s[(abs_s.role == role) & (abs_s.family == fam) & (abs_s.variant == var) & (abs_s.cell == cell)]
            got = float(hit["rmse_K__mean"].iloc[0])
            rows.append({
                "check": "rmse_mean_vs_report", "role": role, "family": fam, "variant": var,
                "cell": cell, "report_K": e, "recomputed_K": round(got, 2),
                "abs_diff_K": abs(got - e), "match_0.02K": abs(got - e) <= 0.02,
            })
    out = pd.DataFrame(rows)
    if not bool(out["match_0.02K"].all()):
        raise SystemExit("RMSE means do not match the published report within 0.02 K")
    # 460 family x p x condition x cell x role, averaged over seeds and replicas
    dc = tables["delta_vs_R0_by_condition"]
    g = dc.groupby(["family", "variant", "role", "cell", "condition"])["d_rmse_K"].mean()
    if len(g) != 460:
        raise SystemExit(f"expected 460 condition contrasts, found {len(g)}")
    n_improve = int((g < 0).sum())
    if n_improve != 9:
        raise SystemExit(f"expected 9 improving contrasts at family x p level, found {n_improve}")
    return out


def fig_g1(dp: pd.DataFrame) -> None:
    wide = dp[dp.cell.isin(["G1", "G1-eq"])].pivot_table(
        index=["run_id", "family", "variant", "model_seed", "role"],
        columns="cell", values="d_rmse_K",
    ).reset_index()
    wide["ratio"] = wide["G1-eq"] / wide["G1"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
    for col, role in enumerate(("seen_train_eval", "val_eval")):
        sub = wide[wide.role == role]
        ax_d, ax_r = axes[0, col], axes[1, col]
        for i, (fam, var, _, color) in enumerate(SERIES):
            s = sub[(sub.family == fam) & (sub.variant == var)]
            for _, row in s.iterrows():
                y = i
                ax_d.plot([row["G1-eq"], row["G1"]], [y, y], color=color, lw=1.15, zorder=2)
                ax_d.scatter([row["G1-eq"]], [y], s=42, facecolors="white", edgecolors=color,
                             marker=MARK[int(row.model_seed)], linewidths=1.4, zorder=3)
                ax_d.scatter([row["G1"]], [y], s=42, c=color, marker=MARK[int(row.model_seed)],
                             linewidths=0, zorder=3)
                ax_r.scatter([row["ratio"]], [y], s=42, c=color, marker=MARK[int(row.model_seed)], zorder=3)
        for ax in (ax_d, ax_r):
            ax.set_yticks(range(len(SERIES)))
            ax.set_yticklabels([lab for _, _, lab, _ in SERIES])
            ax.set_ylim(len(SERIES) - 0.45, -0.55)
            ax.grid(axis="y", visible=False)
        ax_d.axvline(0, color="0.45", lw=0.8)
        ax_d.set_xlabel("Aumento della RMSE rispetto al proprio R0 (K)")
        ax_r.axvline(1, color="0.45", lw=0.8, ls="--")
        ax_r.set_xlim(0, 1.05)
        ax_r.set_xlabel("Rapporto descrittivo  Δ(G1-eq) / Δ(G1)")
        ax_d.set_title(ROLE[role])
        ax_r.set_title("Stesso ruolo: quanto del danno G1 torna con i siti a caso")
    fig.suptitle(
        "Lacune contigue e perdita di siti, a pari numero di osservazioni rimosse",
        fontsize=13,
    )
    fig.text(
        0.5, -0.08,
        "Ogni segno è un seed di training (0 cerchio, 1 quadrato, 2 triangolo), già mediato sulle tre repliche di maschera.\n"
        "Il pieno è G1 (dischi contigui); il vuoto è G1-eq (stesso numero di siti, posizioni casuali). "
        "Il rapporto descrive questi due scenari: non è una quota causale della sola contiguità.",
        ha="center", va="top", fontsize=9, color="0.2",
    )
    fig.savefig(FIG / "fig_g1_vs_g1eq.png", dpi=160)
    plt.close(fig)
    wide.to_csv(TAB / "g1_vs_g1eq_by_seed.csv", index=False)


def fig_bias(dp: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.28, hspace=0.48, wspace=0.22)
    specs = [
        ("d_centered_rmse_K", "Variazione della centered-RMSE (K)", (-1, 10.5)),
        ("d_bias_K", "Variazione del bias (K)\npredizione − verità, media sul campo", (-13.5, 2.5)),
    ]
    for col, role in enumerate(("seen_train_eval", "val_eval")):
        sub = dp[(dp.role == role) & (dp.cell.isin(CELLS_DEG))]
        for row, (ycol, ylab, ylim) in enumerate(specs):
            ax = axes[row, col]
            _points(ax, sub, CELLS_DEG, ycol)
            _finish_cats(ax, CELLS_DEG, [CELL_LABEL[c] for c in CELLS_DEG])
            ax.set_ylim(*ylim)
            if col == 0:
                ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(ROLE[role])
    fig.suptitle("Bias e centered-RMSE: variazione rispetto al proprio caso pulito", fontsize=13)
    _attach_legend(
        fig,
        "Sul singolo frame, RMSE² = bias² + centered-RMSE². Questi punti sono medie su frame, condizioni e tre repliche di maschera:\n"
        "non si sommano e non sono una percentuale della RMSE. Bias negativo: la predizione è più fredda della temperatura LES.\n"
        "Ogni segno è un seed di training. Non è un intervallo di confidenza.",
    )
    fig.savefig(FIG / "fig_bias_centered.png", dpi=160)
    plt.close(fig)


def fig_regional(dp: pd.DataFrame) -> None:
    metrics = [
        ("d_jet_rmse_K", "Getti freddi\nT vera < 1800 K", (-2, 30)),
        ("d_high_rmse_K", "Zona calda\nT vera ≥ 2100 K", (-1, 8)),
        ("d_peak_rmse_K", "Picco\nT vera ≥ 2200 K", (-1, 12)),
    ]
    cats = ["N1", "N2", "G1-eq", "G1"]
    labels = [CELL_LABEL[c] for c in cats]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.26, hspace=0.42, wspace=0.28)
    for col, (ycol, title, ylim) in enumerate(metrics):
        for row, role in enumerate(("seen_train_eval", "val_eval")):
            ax = axes[row, col]
            sub = dp[(dp.role == role) & (dp.cell.isin(cats))]
            _points(ax, sub, cats, ycol)
            _finish_cats(ax, cats, labels)
            ax.set_ylim(*ylim)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{ROLE[role]}\naumento della RMSE regionale (K)")
    fig.suptitle("Dove cresce l'errore rispetto al proprio caso pulito", fontsize=13)
    _attach_legend(
        fig,
        "Le soglie sono sulla temperatura vera, quindi la stessa regione esiste anche nel caso pulito: la differenza è un deterioramento.\n"
        "Ogni segno è un seed, mediato sulle tre repliche di maschera. Non è un intervallo di confidenza: i frame sono correlati.",
    )
    fig.savefig(FIG / "fig_regional_jets_hot.png", dpi=160)
    plt.close(fig)


def _val_pair(ax, val: pd.DataFrame, ycol: str, conds: list[str]) -> None:
    for fam, var, _, color in SERIES:
        off = OFFSET[(fam, var)] * 0.35
        sub = val[(val.family == fam) & (val.variant == var)]
        for seed, marker in MARK.items():
            s = sub[sub.model_seed == seed]
            for cell, face, shift in (("G1", color, -0.15), ("G1-eq", "white", 0.15)):
                ss = s[s.cell == cell]
                xs = [conds.index(c) + off + shift for c in ss["condition"]]
                ax.scatter(
                    xs, ss[ycol], s=36, c=face, edgecolors=color, marker=marker,
                    linewidths=1.25, zorder=3,
                )
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([_cond_label(c) for c in conds])
    ax.set_xlim(-0.55, len(conds) - 0.45)
    ax.axhline(0, color="0.45", lw=0.8, zorder=0)


def fig_conditions(dc: pd.DataFrame) -> None:
    # Replica -> seed, then keep seeds visible.
    m = dc.groupby(
        ["family", "variant", "model_seed", "role", "cell", "condition"]
    )[["d_rmse_K", "d_bias_K", "d_centered_rmse_K"]].mean().reset_index()
    seen = m[(m.role == "seen_train_eval") & (m.cell == "G1")].copy()
    order = [c.replace(".h5", "") for c in seen.groupby("condition")["d_rmse_K"].mean().sort_values().index]
    seen["condition"] = seen["condition"].str.replace(".h5", "", regex=False)

    val = m[(m.role == "val_eval") & (m.cell.isin(["G1", "G1-eq"]))].copy()
    val["condition"] = val["condition"].str.replace(".h5", "", regex=False)
    val_order = ["0NH3_1.05", "20NH3_1.1", "40NH3_0.9"]

    fig = plt.figure(figsize=(13.4, 10.6))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.05, 1], left=0.18, right=0.98, top=0.90, bottom=0.20,
        hspace=0.36, wspace=0.28,
    )
    ax_s = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 1])

    for fam, var, _, color in SERIES:
        sub = seen[(seen.family == fam) & (seen.variant == var)]
        off = OFFSET[(fam, var)] * 0.55
        for seed, marker in MARK.items():
            s = sub[sub.model_seed == seed]
            ys = [order.index(c) + off for c in s["condition"]]
            ax_s.scatter(s["d_rmse_K"], ys, s=28, c=color, marker=marker, zorder=3, linewidths=0)
    ax_s.axvline(0, color="0.45", lw=0.8)
    ax_s.set_yticks(range(len(order)))
    ax_s.set_yticklabels([_cond_label(c) for c in order])
    ax_s.set_ylim(-0.6, len(order) - 0.4)
    ax_s.set_xlabel("Aumento della RMSE con le lacune contigue (K)")
    ax_s.set_title("seen220 · 20 condizioni già nel training")
    ax_s.grid(axis="y", visible=False)

    _val_pair(ax_v, val, "d_rmse_K", val_order)
    ax_v.set_ylabel("Aumento della RMSE (K)")
    ax_v.set_title("VAL123 · RMSE\npieno = dischi, vuoto = stessi siti a caso")
    ax_v.set_ylim(-1.5, 14)

    _val_pair(ax_b, val, "d_bias_K", val_order)
    ax_b.set_ylabel("Variazione del bias (K)")
    ax_b.set_title("VAL123 · livello\npieno = dischi, vuoto = stessi siti a caso")
    ax_b.set_ylim(-28, 5)

    fig.suptitle("Quali condizioni si allontanano dalla media", fontsize=13)
    _attach_legend(
        fig,
        "seen220 è il dominio di valutazione sulle condizioni di training, non un insieme di condizioni nuove.\n"
        "VAL123 ha tre condizioni ed è già stata usata per scegliere i checkpoint. "
        "I tre segni di uno stesso colore sono i tre seed, non un intervallo di confidenza.",
    )
    fig.savefig(FIG / "fig_conditions.png", dpi=160)
    plt.close(fig)
    seen.to_csv(TAB / "seen_g1_by_condition_seed.csv", index=False)
    val.to_csv(TAB / "val_g1_g1eq_by_condition_seed.csv", index=False)


def export_points(dp: pd.DataFrame) -> None:
    keep = [
        "run_id", "family", "variant", "model_seed", "role", "cell",
        "d_rmse_K", "d_bias_K", "d_centered_rmse_K",
        "d_jet_rmse_K", "d_high_rmse_K", "d_peak_rmse_K", "d_high_iou",
    ]
    dp[keep].to_csv(TAB / "delta_by_seed_plotted.csv", index=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    _style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    tables = load()
    checks = verify(tables)
    checks.to_csv(TAB / "rmse_report_check.csv", index=False)
    dp = tables["delta_vs_R0_per_model"]
    fig_g1(dp)
    fig_bias(dp)
    fig_regional(dp)
    fig_conditions(tables["delta_vs_R0_by_condition"])
    export_points(dp)
    rows = []
    for p in sorted(SRC.glob("*.csv")):
        rows.append({"kind": "source_table", "path": str(p.relative_to(ROOT)), "sha256": sha256(p)})
    for p in sorted(FIG.glob("*.png")):
        rows.append({"kind": "figure", "path": str(p.relative_to(ROOT)), "sha256": sha256(p)})
    pd.DataFrame(rows).to_csv(TAB / "file_sha256.csv", index=False)
    print(f"checks {len(checks)} all matched; figures in {FIG}")


if __name__ == "__main__":
    main()
