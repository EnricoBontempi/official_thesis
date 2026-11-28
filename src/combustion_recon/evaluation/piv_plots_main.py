"""Curated confirmatory main figures for Mission 02 (Virtual PIV).

Generates ``figures/main/figM1`` .. ``figM8`` directly from aggregated tables and
selected field archives.  No neural inference is performed here.
Every figure carries clear Italian/bilingual scientific labels, explicit sign
conventions, individual seed visibility, and hollow markers for ``budget_exhausted``
runs (never declared converged).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from combustion_recon.evaluation.piv_metrics import DISTANCE_BINS

FAM_LABEL = {"sparse_unet": "U-Net", "adapted_glu": "GLU", "dense_unet": "U-Net densa"}
FAM_COLOR = {"sparse_unet": "#1b6ca8", "adapted_glu": "#c0392b", "dense_unet": "#2e8b57"}
ROLE_LABEL = {"seen_train_eval": "seen220 (TRAIN futuro)", "val_eval": "VAL123 (sviluppo)"}
CELL_ORDER = ("IDEAL-A", "PIV-A", "IDEAL-B", "PIV-B", "BRIDGE-5pct-IDEAL")
EVAL_ORDER_MAIN = (
    "IDEAL-full",
    "IDEAL-A",
    "FILT-A",
    "PIV-A",
    "IDEAL-B-in",
    "FILT-B-in",
    "PIV-B-in",
    "IDEAL-B-down",
    "PIV-B-down",
    "IDEAL-B-off",
    "PIV-B-off",
)


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 150,
        }
    )


def _ordered(present: Sequence[str], ref: Sequence[str]) -> list[str]:
    p = list(dict.fromkeys(present))
    head = [x for x in ref if x in p]
    tail = sorted(x for x in p if x not in head)
    return head + tail


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_m1_campaign_overview(per_run: pd.DataFrame, dest: Path) -> Path | None:
    """figM1: Own-scenario RMSE (best) across T1-T7 cells, both families, all seeds."""
    if per_run.empty:
        return None
    sub = per_run[
        (per_run["checkpoint_kind"] == "best") & (per_run["eval_config"] == per_run["train_config_id"])
    ].copy()
    if sub.empty:
        return None
    roles = _ordered(sub["role"].unique(), ("val_eval", "seen_train_eval"))
    cells = _ordered(sub["train_cell"].unique(), CELL_ORDER)
    fams = _ordered(sub["family"].unique(), ("sparse_unet", "adapted_glu"))

    fig, axes = plt.subplots(1, len(roles), figsize=(5.6 * len(roles), 4.4), sharey=True, squeeze=False)
    for ax, role in zip(axes[0], roles):
        sr = sub[sub["role"] == role]
        for fi, fam in enumerate(fams):
            col = FAM_COLOR.get(fam, "#444")
            off = -0.16 + 0.32 * fi if len(fams) > 1 else 0.0
            for ci, cell in enumerate(cells):
                g = sr[(sr["family"] == fam) & (sr["train_cell"] == cell)]
                if g.empty:
                    continue
                x = ci + off
                bud = g["budget_exhausted"].astype(bool).to_numpy()
                vals = g["rmse_K"].to_numpy(dtype=float)
                if (~bud).any():
                    ax.scatter(
                        np.full(int((~bud).sum()), x),
                        vals[~bud],
                        color=col,
                        s=34,
                        zorder=4,
                        label=f"{FAM_LABEL.get(fam, fam)} (plateau)" if ci == 0 else None,
                    )
                if bud.any():
                    ax.scatter(
                        np.full(int(bud.sum()), x),
                        vals[bud],
                        facecolors="white",
                        edgecolors=col,
                        linewidths=1.4,
                        s=42,
                        zorder=5,
                        label=f"{FAM_LABEL.get(fam, fam)} (budget_exhausted)" if ci == 0 else None,
                    )
                mu = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
                ax.errorbar(x, mu, yerr=sd if sd > 0 else None, fmt="_", color=col, markersize=10, elinewidth=1.4, capsize=3, zorder=3)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(cells, rotation=18, ha="right")
        ax.set_title(ROLE_LABEL.get(role, role), loc="left")
        ax.set_xlabel("Cella di training (= configurazione valutata)")
    axes[0][0].set_ylabel("RMSE macro su T (K) — best\n(punti = seed; barre = media ± std fra seed)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=min(4, len(uniq)), frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("M1 — Campagna principale (confermativa): accuratezza diagonale per cella e seed", y=1.08, fontsize=11, fontweight="bold")
    return _save(fig, dest / "figM1_campaign_overview_best.png")


def _plot_m2_confirmatory_contrasts(contrasts: pd.DataFrame, dest: Path) -> Path | None:
    """figM2: Confirmatory paired contrasts vs control inter-seed std band."""
    if contrasts.empty:
        return None
    sub = contrasts[contrasts["checkpoint_kind"] == "best"].copy()
    if sub.empty:
        return None
    role = "val_eval" if "val_eval" in set(sub["role"]) else str(sub["role"].iloc[0])
    sub = sub[sub["role"] == role]
    cids = list(dict.fromkeys(sub["contrast_id"]))
    fams = _ordered(sub["family"].unique(), ("sparse_unet", "adapted_glu"))

    fig, ax = plt.subplots(figsize=(9.5, max(4.2, 0.52 * len(cids) + 1.4)))
    y_pos = np.arange(len(cids))
    ax.axvline(0.0, color="#222", lw=1.0, ls="--", zorder=1)

    for fi, fam in enumerate(fams):
        col = FAM_COLOR.get(fam, "#444")
        off = (-0.16 + 0.32 * fi) if len(fams) > 1 else 0.0
        for yi, cid in enumerate(cids):
            hit = sub[(sub["contrast_id"] == cid) & (sub["family"] == fam)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            y = yi + off
            mu = float(r["delta_rmse_mean_K"])
            lo = float(r["delta_rmse_min_K"])
            hi = float(r["delta_rmse_max_K"])
            ctrl_sd = float(r["control_B_seed_std_K"]) if pd.notna(r["control_B_seed_std_K"]) else float("nan")
            if math.isfinite(ctrl_sd) and ctrl_sd > 0:
                ax.barh(y, 2 * ctrl_sd, left=-ctrl_sd, height=0.22, color=col, alpha=0.14, zorder=1)
            ax.plot([lo, hi], [y, y], color=col, lw=1.8, zorder=3)
            mk = "s" if bool(r["exceeds_control_seed_std"]) else "o"
            ax.scatter(
                [mu],
                [y],
                color=col,
                marker=mk,
                s=44,
                zorder=4,
                label=FAM_LABEL.get(fam, fam) if yi == 0 else None,
            )
            ax.text(hi + 0.15, y, f"{mu:+.2f} K", va="center", ha="left", fontsize=7.5, color=col)

    labels = [
        f"{cid.split('_', 1)[0]}: {sub[sub['contrast_id'] == cid]['label'].iloc[0]}"
        for cid in cids
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ΔRMSE appaiato per seed = A − B (K)  |  banda chiara = ±1 std fra seed del controllo B")
    ax.set_title(
        f"M2 — Contrasti confermativi ({ROLE_LABEL.get(role, role)}, best; quadrato = |Δ| > std controllo)",
        loc="left",
        fontweight="bold",
    )
    ax.legend(loc="lower right", frameon=True)
    return _save(fig, dest / "figM2_confirmatory_contrasts.png")


def _plot_m3_transfer_matrix(cell_summary: pd.DataFrame, dest: Path, *, checkpoint_kind: str = "best") -> Path | None:
    """figM3: Transfer matrix (train_cell x eval_config) for U-Net & GLU, VAL & seen."""
    if cell_summary.empty:
        return None
    sub = cell_summary[
        (cell_summary["checkpoint_kind"] == checkpoint_kind)
        & (~cell_summary["eval_config"].astype(str).str.startswith("SENS-"))
    ].copy()
    if sub.empty:
        return None

    fams = _ordered(sub["family"].unique(), ("sparse_unet", "adapted_glu"))
    roles = _ordered(sub["role"].unique(), ("val_eval", "seen_train_eval"))
    tcells = _ordered(sub["train_cell"].unique(), CELL_ORDER)
    ecfgs = _ordered(sub["eval_config"].unique(), EVAL_ORDER_MAIN)

    vmin = float(np.nanpercentile(sub["rmse_K__mean"], 2))
    vmax = float(np.nanpercentile(sub["rmse_K__mean"], 98))
    if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmax <= vmin:
        vmin, vmax = 0.0, max(1.0, float(np.nanmax(sub["rmse_K__mean"])))

    fig, axes = plt.subplots(
        len(fams),
        len(roles),
        figsize=(max(6.2, 0.78 * len(ecfgs)) * len(roles), max(3.0, 0.65 * len(tcells)) * len(fams) + 0.8),
        squeeze=False,
        constrained_layout=True,
    )
    im = None
    for i, fam in enumerate(fams):
        for j, role in enumerate(roles):
            ax = axes[i, j]
            M = np.full((len(tcells), len(ecfgs)), np.nan)
            S = np.full((len(tcells), len(ecfgs)), np.nan)
            B = np.zeros((len(tcells), len(ecfgs)), dtype=int)
            sr = sub[(sub["family"] == fam) & (sub["role"] == role)]
            for r_i, tc in enumerate(tcells):
                for c_j, ec in enumerate(ecfgs):
                    hit = sr[(sr["train_cell"] == tc) & (sr["eval_config"] == ec)]
                    if not hit.empty:
                        M[r_i, c_j] = float(hit["rmse_K__mean"].iloc[0])
                        S[r_i, c_j] = float(hit["rmse_K__std"].iloc[0]) if pd.notna(hit["rmse_K__std"].iloc[0]) else np.nan
                        B[r_i, c_j] = int(hit["n_budget_exhausted"].iloc[0])
            im = ax.imshow(M, cmap="viridis_r", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(ecfgs)))
            ax.set_xticklabels(ecfgs, rotation=32, ha="right", fontsize=7.5)
            ax.set_yticks(range(len(tcells)))
            ax.set_yticklabels(tcells, fontsize=8)
            ax.set_title(f"{FAM_LABEL.get(fam, fam)} — {ROLE_LABEL.get(role, role)} ({checkpoint_kind})", loc="left")
            if j == 0:
                ax.set_ylabel("Cella di training")
            if i == len(fams) - 1:
                ax.set_xlabel("Configurazione di valutazione")
            ax.grid(False)
            mid = 0.5 * (vmin + vmax)
            for r_i in range(len(tcells)):
                for c_j in range(len(ecfgs)):
                    val = M[r_i, c_j]
                    if math.isfinite(val):
                        sd = S[r_i, c_j]
                        badge = "*" if B[r_i, c_j] > 0 else ""
                        txt = f"{val:.1f}{badge}\n±{sd:.1f}" if math.isfinite(sd) else f"{val:.1f}{badge}"
                        color = "white" if val < mid else "#111"
                        ax.text(c_j, r_i, txt, ha="center", va="center", fontsize=6.8, color=color)

    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02)
        cb.set_label("RMSE macro medio fra seed (K)  |  * = include run budget_exhausted")
    fig.suptitle(f"M3 — Matrice di trasferimento ({checkpoint_kind}): media ± std fra seed", fontsize=11, fontweight="bold")
    return _save(fig, dest / f"figM3_transfer_matrix_{checkpoint_kind}.png")


def _plot_m4_b_positions_and_distance(b_dist: pd.DataFrame, dest: Path) -> Path | None:
    """figM4: B-in/down/off in-field vs out-of-field AND RMSE vs distance from valid vector."""
    if b_dist.empty:
        return None
    sub = b_dist[b_dist["checkpoint_kind"] == "best"].copy()
    role = "val_eval" if "val_eval" in set(sub["role"]) else str(sub["role"].iloc[0])
    sub = sub[sub["role"] == role]
    b_cfgs = [c for c in ("PIV-B-in", "PIV-B-down", "PIV-B-off", "IDEAL-B-in", "IDEAL-B-down", "IDEAL-B-off") if c in set(sub["eval_config"])]
    dist_cols = [f"dist_{b[0]}_rmse_K" for b in DISTANCE_BINS if f"dist_{b[0]}_rmse_K" in sub.columns]
    if not b_cfgs and not dist_cols:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), constrained_layout=True)

    # Left panel: in-field vs out-of-field on B positions for PIV-B / IDEAL-B models
    ax0 = axes[0]
    has_left = False
    if b_cfgs and {"in_field_rmse_K", "out_field_rmse_K"}.issubset(sub.columns):
        # Match PIV-B models on PIV-B-* and IDEAL-B models on IDEAL-B-* (fallback to all)
        filt = sub[
            ((sub["train_cell"] == "PIV-B") & sub["eval_config"].isin(["PIV-B-in", "PIV-B-down", "PIV-B-off"]))
            | ((sub["train_cell"] == "IDEAL-B") & sub["eval_config"].isin(["IDEAL-B-in", "IDEAL-B-down", "IDEAL-B-off"]))
        ]
        if filt.empty:
            filt = sub[sub["eval_config"].isin(b_cfgs)]
        show_cfgs = _ordered(filt["eval_config"].unique(), b_cfgs)
        fams = _ordered(filt["family"].unique(), ("sparse_unet", "adapted_glu"))
        x = np.arange(len(show_cfgs))
        width = 0.35 / max(1, len(fams))
        for fi, fam in enumerate(fams):
            col = FAM_COLOR.get(fam, "#444")
            sf = filt[filt["family"] == fam]
            in_vals = [float(sf[sf["eval_config"] == c]["in_field_rmse_K"].mean()) if (sf["eval_config"] == c).any() else np.nan for c in show_cfgs]
            out_vals = [float(sf[sf["eval_config"] == c]["out_field_rmse_K"].mean()) if (sf["eval_config"] == c).any() else np.nan for c in show_cfgs]
            dx = (fi - 0.5 * (len(fams) - 1)) * (2.2 * width)
            ax0.bar(x + dx - 0.5 * width, in_vals, width=width, color=col, alpha=0.88, label=f"{FAM_LABEL.get(fam, fam)} dentro campo")
            ax0.bar(x + dx + 0.5 * width, out_vals, width=width, color=col, alpha=0.35, edgecolor=col, hatch="//", label=f"{FAM_LABEL.get(fam, fam)} fuori campo")
            has_left = True
        ax0.set_xticks(x)
        ax0.set_xticklabels(show_cfgs, rotation=24, ha="right")
        ax0.set_ylabel("RMSE macro (K)")
        ax0.set_title(f"Posizioni B: dentro vs fuori dal campo ({ROLE_LABEL.get(role, role)})", loc="left")
        ax0.legend(fontsize=7.5)
    if not has_left:
        ax0.text(0.5, 0.5, "Configurazioni B non presenti nel manifest", ha="center", va="center", transform=ax0.transAxes)
        ax0.set_axis_off()

    # Right panel: RMSE vs distance bin (frozen R1 §8 classes)
    ax1 = axes[1]
    has_right = False
    if dist_cols:
        bin_labels = [b[3] for b in DISTANCE_BINS if f"dist_{b[0]}_rmse_K" in sub.columns]
        curves_spec = [
            ("PIV-A", "PIV-A", "#1b6ca8", "-"),
            ("PIV-B", "PIV-B-in", "#c0392b", "-"),
            ("PIV-B", "PIV-B-down", "#e67e22", "--"),
            ("PIV-B", "PIV-B-off", "#8e44ad", ":"),
        ]
        # Prefer U-Net if present, else first family
        fam0 = "sparse_unet" if "sparse_unet" in set(sub["family"]) else str(sub["family"].iloc[0])
        sf = sub[sub["family"] == fam0]
        for tcell, ecfg, col, ls in curves_spec:
            hit = sf[(sf["train_cell"] == tcell) & (sf["eval_config"] == ecfg)]
            if hit.empty:
                continue
            ys = [float(hit[c].mean()) for c in dist_cols]
            if any(math.isfinite(y) for y in ys):
                ax1.plot(range(len(dist_cols)), ys, marker="o", lw=1.8, ls=ls, color=col, label=f"{tcell} → {ecfg}")
                has_right = True
        ax1.set_xticks(range(len(bin_labels)))
        ax1.set_xticklabels(bin_labels, rotation=20, ha="right")
        ax1.set_xlabel("Distanza euclidea dal vettore valido più vicino (classi R1 §8)")
        ax1.set_ylabel("RMSE (K)")
        ax1.set_title(f"Errore vs distanza dalle osservazioni ({FAM_LABEL.get(fam0, fam0)}, best)", loc="left")
        if has_right:
            ax1.legend(fontsize=8)
    if not has_right:
        ax1.text(0.5, 0.5, "Colonne distanza non presenti", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_axis_off()

    fig.suptitle("M4 — Copertura, posizioni del campo B e accuratezza in funzione della distanza", fontsize=11, fontweight="bold")
    return _save(fig, dest / "figM4_b_positions_and_distance.png")


def _plot_m5_sensitivities(sens: pd.DataFrame, cell_summary: pd.DataFrame, dest: Path) -> Path | None:
    """figM5: Forest plot of sensitivity deltas vs main PIV-A / PIV-B-in + seed-std band."""
    if sens.empty:
        return None
    sub = sens[sens["checkpoint_kind"] == "best"].copy()
    if sub.empty:
        return None
    role = "val_eval" if "val_eval" in set(sub["role"]) else str(sub["role"].iloc[0])
    # Focus on own PIV-A models on A sensitivities and PIV-B models on B-in sensitivities
    own = sub[
        ((sub["train_cell"] == "PIV-A") & (sub["ref_main_config"] == "PIV-A"))
        | ((sub["train_cell"] == "PIV-B") & (sub["ref_main_config"] == "PIV-B-in"))
    ]
    if own.empty:
        own = sub
    own = own[own["role"] == role]
    if own.empty:
        return None

    ecfgs = sorted(own["eval_config"].unique())
    fams = _ordered(own["family"].unique(), ("sparse_unet", "adapted_glu"))

    fig, ax = plt.subplots(figsize=(8.8, max(4.0, 0.42 * len(ecfgs) + 1.4)))
    ax.axvline(0.0, color="#222", lw=1.0, ls="--")

    # Shade U-Net T2 (PIV-A) seed-std band if available (exact R4 criterion reference!)
    if not cell_summary.empty:
        ref = cell_summary[
            (cell_summary["family"] == "sparse_unet")
            & (cell_summary["train_cell"] == "PIV-A")
            & (cell_summary["eval_config"] == "PIV-A")
            & (cell_summary["checkpoint_kind"] == "best")
            & (cell_summary["role"] == role)
        ]
        if not ref.empty and pd.notna(ref["rmse_K__std"].iloc[0]):
            sd = float(ref["rmse_K__std"].iloc[0])
            ax.axvspan(-sd, sd, color="#1b6ca8", alpha=0.10, label=f"±1 std fra seed T2 PIV-A (±{sd:.2f} K)")

    for fi, fam in enumerate(fams):
        col = FAM_COLOR.get(fam, "#444")
        off = (-0.15 + 0.30 * fi) if len(fams) > 1 else 0.0
        for yi, ec in enumerate(ecfgs):
            hit = own[(own["eval_config"] == ec) & (own["family"] == fam)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            y = yi + off
            ax.plot([r["delta_rmse_min_K"], r["delta_rmse_max_K"]], [y, y], color=col, lw=1.7)
            ax.scatter([r["delta_rmse_mean_K"]], [y], color=col, s=36, label=FAM_LABEL.get(fam, fam) if yi == 0 else None)

    ax.set_yticks(range(len(ecfgs)))
    ax.set_yticklabels(ecfgs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ΔRMSE rispetto alla configurazione principale (SENS − PIV principale, K)  |  > 0 peggiore")
    ax.set_title(f"M5 — Analisi di sensibilità ({ROLE_LABEL.get(role, role)}, best, modelli PIV)", loc="left", fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=8)
    return _save(fig, dest / "figM5_sensitivities_forest.png")


def _plot_m6_thermal_and_isotherms(per_run: pd.DataFrame, dest: Path) -> Path | None:
    """figM6: Thermal regions (hot RMSE/IoU, jets RMSE, peak/p95 error) & isotherm distances."""
    if per_run.empty:
        return None
    sub = per_run[
        (per_run["checkpoint_kind"] == "best") & (per_run["eval_config"] == per_run["train_config_id"])
    ].copy()
    if sub.empty:
        return None
    role = "val_eval" if "val_eval" in set(sub["role"]) else str(sub["role"].iloc[0])
    sub = sub[sub["role"] == role]
    cells = _ordered(sub["train_cell"].unique(), CELL_ORDER)
    fams = _ordered(sub["family"].unique(), ("sparse_unet", "adapted_glu"))

    panels = [
        ("high_rmse_K", "Zona calda (T ≥ 2100 K) RMSE (K)"),
        ("high_iou", "Zona calda (T ≥ 2100 K) IoU"),
        ("jet_rmse_K", "Getti freddi (T < 1800 K) RMSE (K)"),
        ("peak_abs_error_K", "Errore assoluto su picco T (K)"),
        ("iso1800_sym_dist_mm", "Distanza isoterma 1800 K (mm)"),
        ("iso2100_sym_dist_mm", "Distanza isoterma 2100 K (mm)"),
    ]
    panels = [(c, lbl) for c, lbl in panels if c in sub.columns and sub[c].notna().any()]
    if not panels:
        return None

    ncols = 3
    nrows = int(math.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), constrained_layout=True, squeeze=False)
    for idx, (col_name, ylabel) in enumerate(panels):
        ax = axes[idx // ncols, idx % ncols]
        for fi, fam in enumerate(fams):
            col = FAM_COLOR.get(fam, "#444")
            off = (-0.15 + 0.30 * fi) if len(fams) > 1 else 0.0
            for ci, cell in enumerate(cells):
                g = sub[(sub["family"] == fam) & (sub["train_cell"] == cell)][col_name].dropna().to_numpy(dtype=float)
                if g.size == 0:
                    continue
                ax.scatter(np.full(g.size, ci + off), g, color=col, s=26, alpha=0.85, label=FAM_LABEL.get(fam, fam) if (ci == 0 and idx == 0) else None)
                ax.plot([ci + off - 0.1, ci + off + 0.1], [np.mean(g), np.mean(g)], color=col, lw=2.0)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(cells, rotation=22, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
    for k in range(len(panels), nrows * ncols):
        axes[k // ncols, k % ncols].set_axis_off()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=len(uniq), frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"M6 — Metriche fisiche, regioni termiche e isoterme ({ROLE_LABEL.get(role, role)}, best)", y=1.07, fontsize=11, fontweight="bold")
    return _save(fig, dest / "figM6_thermal_regions_and_isotherms.png")


def _load_seed0_fields_for_frame(
    audit_table: pd.DataFrame,
    *,
    role: str,
    condition: str,
    frame_index: int,
) -> list[dict[str, Any]]:
    """Load saved selected fields for the SAME (role, condition, frame) across cells/families."""
    records: list[dict[str, Any]] = []
    if audit_table.empty:
        return records
    best_rows = audit_table[(audit_table["checkpoint_kind"] == "best") & (audit_table["eval_valid"])].copy()
    # Pick lowest seed per (family, train_cell) for the concise main panel
    for (fam, tcell), g in best_rows.groupby(["family", "train_cell"], sort=True):
        r = g.sort_values("model_seed").iloc[0]
        src = Path(str(r["eval_source_dir"]))
        mp = src / "fields_manifest.json"
        if not mp.is_file():
            continue
        try:
            man = json.loads(mp.read_text(encoding="utf-8"))
            npz_p = src / man["file"]
            if not npz_p.is_file():
                continue
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        tcfg = str(r["train_config_id"])
        key = f"{tcfg}|{role}|{condition}|{int(frame_index)}"
        with np.load(npz_p) as z:
            if f"{key}|pred_T_K" not in z:
                continue
            records.append(
                {
                    "family": str(fam),
                    "train_cell": str(tcell),
                    "eval_config": tcfg,
                    "run_id": str(r["run_id"]),
                    "model_seed": int(r["model_seed"]),
                    "pred_T_K": z[f"{key}|pred_T_K"].copy(),
                    "true_T_K": z[f"{key}|true_T_K"].copy(),
                    "error_signed_K": z[f"{key}|error_signed_K"].copy(),
                    "field_mask": z[f"{key}|field_mask"].copy() if f"{key}|field_mask" in z else None,
                }
            )
    return records


def _plot_m7_representative_fields(
    audit_table: pd.DataFrame,
    rep_frames: Mapping[str, Sequence[Mapping[str, Any]]],
    dest: Path,
) -> Path | None:
    """figM7: Same a-priori representative frame across cells with COMMON T & error scales."""
    candidates: list[tuple[str, str, int]] = []
    for role in ("val_eval", "seen_train_eval"):
        for it in rep_frames.get(role, []):
            candidates.append((role, str(it["condition"]), int(it["frame_index"])))
    chosen = None
    recs: list[dict[str, Any]] = []
    for cand in candidates:
        recs = _load_seed0_fields_for_frame(audit_table, role=cand[0], condition=cand[1], frame_index=cand[2])
        if recs:
            chosen = cand
            break
    if not recs or chosen is None:
        return None

    role, cond, fidx = chosen
    recs = sorted(recs, key=lambda d: (0 if d["family"] == "sparse_unet" else 1, CELL_ORDER.index(d["train_cell"]) if d["train_cell"] in CELL_ORDER else 99))
    true_T = recs[0]["true_T_K"]

    # Common color scales across all panels
    all_T = [true_T] + [r["pred_T_K"] for r in recs]
    tmin = float(min(np.nanmin(a) for a in all_T))
    tmax = float(max(np.nanmax(a) for a in all_T))
    all_e = [np.abs(r["error_signed_K"]) for r in recs]
    e_lim = float(max(50.0, min(400.0, np.nanpercentile(np.concatenate([x.ravel() for x in all_e]), 99))))

    ncols = len(recs) + 1
    fig, axes = plt.subplots(2, ncols, figsize=(2.85 * ncols, 5.4), constrained_layout=True)
    im_t = axes[0, 0].imshow(true_T, cmap="inferno", vmin=tmin, vmax=tmax, origin="upper")
    axes[0, 0].set_title(f"LES vera (T)\n{cond.replace('.h5', '')} f{fidx}", fontsize=8.5)
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, f"Ruolo: {role}\nReplica: 0\nScale comuni:\nT ∈ [{tmin:.0f}, {tmax:.0f}] K\nErr ∈ [±{e_lim:.0f}] K", ha="center", va="center", fontsize=8)

    im_e = None
    for idx, rec in enumerate(recs, start=1):
        rmse = float(np.sqrt(np.mean(rec["error_signed_K"].astype(np.float64) ** 2)))
        ax_t = axes[0, idx]
        ax_e = axes[1, idx]
        ax_t.imshow(rec["pred_T_K"], cmap="inferno", vmin=tmin, vmax=tmax, origin="upper")
        ax_t.set_title(f"{FAM_LABEL.get(rec['family'], rec['family'])} {rec['train_cell']}\n(s{rec['model_seed']}) RMSE={rmse:.1f} K", fontsize=8)
        ax_t.set_xticks([]); ax_t.set_yticks([])

        im_e = ax_e.imshow(
            rec["error_signed_K"],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(0.0, vmin=-e_lim, vmax=e_lim),
            origin="upper",
        )
        if rec["field_mask"] is not None and rec["field_mask"].size and not rec["field_mask"].all():
            ax_e.contour(rec["field_mask"].astype(float), levels=[0.5], colors=["#111"], linewidths=1.1, linestyles="--")
            ax_t.contour(rec["field_mask"].astype(float), levels=[0.5], colors=["#00ffaa"], linewidths=0.9, linestyles="--")
        ax_e.set_title("Errore (pred − vera)", fontsize=7.8)
        ax_e.set_xticks([]); ax_e.set_yticks([])

    cb_t = fig.colorbar(im_t, ax=axes[0, :], shrink=0.85, pad=0.01)
    cb_t.set_label("Temperatura T (K)")
    if im_e is not None:
        cb_e = fig.colorbar(im_e, ax=axes[1, 1:], shrink=0.85, pad=0.01)
        cb_e.set_label("Errore firmato pred − vera (K)")
    fig.suptitle(f"M7 — Confronto visivo su frame rappresentativo a priori (scale comuni) — {cond.replace('.h5', '')} f{fidx}", fontsize=10.5, fontweight="bold")
    return _save(fig, dest / "figM7_representative_fields_common_scale.png")


def _plot_m8_best_vs_last_and_bridge(per_run: pd.DataFrame, glu_bridge: pd.DataFrame, dest: Path) -> Path | None:
    """figM8: best vs last stability + T7 GLU batching bridge."""
    if per_run.empty:
        return None
    diag = per_run[per_run["eval_config"] == per_run["train_config_id"]].copy()
    if diag.empty:
        return None
    role = "val_eval" if "val_eval" in set(diag["role"]) else str(diag["role"].iloc[0])
    diag = diag[diag["role"] == role]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)

    # Left: best vs last
    ax0 = axes[0]
    cells = _ordered(diag["train_cell"].unique(), CELL_ORDER)
    fams = _ordered(diag["family"].unique(), ("sparse_unet", "adapted_glu"))
    for fi, fam in enumerate(fams):
        col = FAM_COLOR.get(fam, "#444")
        for ck, mk, fill in (("best", "o", True), ("last", "s", False)):
            dx = (-0.18 + 0.36 * fi if len(fams) > 1 else 0.0) + (-0.05 if ck == "best" else 0.05)
            for ci, cell in enumerate(cells):
                g = diag[(diag["family"] == fam) & (diag["train_cell"] == cell) & (diag["checkpoint_kind"] == ck)]
                if g.empty:
                    continue
                ax0.scatter(
                    [ci + dx],
                    [float(g["rmse_K"].mean())],
                    marker=mk,
                    s=46,
                    facecolors=col if fill else "white",
                    edgecolors=col,
                    linewidths=1.5,
                    label=f"{FAM_LABEL.get(fam, fam)} ({ck})" if ci == 0 else None,
                )
    ax0.set_xticks(range(len(cells)))
    ax0.set_xticklabels(cells, rotation=20, ha="right")
    ax0.set_ylabel("RMSE macro medio fra seed (K)")
    ax0.set_title(f"Stabilità best vs last ({ROLE_LABEL.get(role, role)})", loc="left")
    handles, labels = ax0.get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        ax0.legend(uniq.values(), uniq.keys(), fontsize=7.8)

    # Right: GLU T7 batching bridge
    ax1 = axes[1]
    if not glu_bridge.empty:
        gb = glu_bridge[glu_bridge["checkpoint_kind"] == "best"]
        if not gb.empty:
            x = np.arange(len(gb))
            ax1.bar(x - 0.16, gb["t7_rmse_mean_K"], width=0.30, color="#c0392b", alpha=0.85, label="T7 GLU (microbatch 1, accum 8)")
            if "m00_glu_orig_rmse_mean_K" in gb.columns and gb["m00_glu_orig_rmse_mean_K"].notna().any():
                ax1.bar(x + 0.16, gb["m00_glu_orig_rmse_mean_K"], width=0.30, color="#7f8c8d", alpha=0.8, label="Missione 00 GLU orig (batch 8)")
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"{r}\n({c})" for r, c in zip(gb["role"], gb["eval_config"])], fontsize=8)
            ax1.set_ylabel("RMSE macro (K)")
            ax1.set_title("Ponte di batching GLU (T7 vs Missione 00)", loc="left")
            ax1.legend(fontsize=8)
        else:
            ax1.text(0.5, 0.5, "Ponte T7 (best) non presente", ha="center", va="center", transform=ax1.transAxes)
            ax1.set_axis_off()
    else:
        ax1.text(0.5, 0.5, "Gruppo ponte T7 non presente nel manifest", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_axis_off()

    fig.suptitle("M8 — Checkpoint best vs last e controllo di batching GLU (T7)", fontsize=11, fontweight="bold")
    return _save(fig, dest / "figM8_best_vs_last_and_glu_bridge.png")


def generate_main_figures(
    agg: Mapping[str, Any],
    audit: Mapping[str, Any],
    manifest: Mapping[str, Any],
    dest_main: Path,
) -> list[Path]:
    """Generate all curated confirmatory main figures into ``dest_main``."""
    _apply_style()
    dest_main = Path(dest_main)
    dest_main.mkdir(parents=True, exist_ok=True)
    conf = agg["confirmatory"]

    produced: list[Path] = []
    for p in (
        _plot_m1_campaign_overview(conf["per_run_scenario"], dest_main),
        _plot_m2_confirmatory_contrasts(conf["confirmatory_contrasts"], dest_main),
        _plot_m3_transfer_matrix(conf["cell_seed_summary"], dest_main, checkpoint_kind="best"),
        _plot_m4_b_positions_and_distance(conf["b_positions_and_distance"], dest_main),
        _plot_m5_sensitivities(conf["sensitivity_summary"], conf["cell_seed_summary"], dest_main),
        _plot_m6_thermal_and_isotherms(conf["per_run_scenario"], dest_main),
        _plot_m7_representative_fields(audit["table"], agg["exploratory"]["selected_cases"]["a_priori_representative"], dest_main),
        _plot_m8_best_vs_last_and_bridge(conf["per_run_scenario"], conf["glu_bridge_t7"], dest_main),
    ):
        if p is not None:
            produced.append(p)
    return produced


__all__ = ["generate_main_figures"]
