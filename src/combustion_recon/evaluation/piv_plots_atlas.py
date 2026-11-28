"""Comprehensive exploratory diagnostic atlas for Mission 02 (Virtual PIV).

Generates structured, clearly labelled diagnostic figures under ``figures/atlas/``
organized into seven thematic sections (never blindly enumerating meaningless pairs):

* ``01_training_curves/`` — training loss, validation RMSE trajectories, LR & best epoch
* ``02_distributions_and_paired/`` — frame error ECDFs/boxes, paired frame scatters
* ``03_by_condition_and_region/`` — per-condition VAL & seen vs v_ref/sigma_u, bias vs cRMSE
* ``04_spatial_maps_and_seed_var/`` — spatial mean-error maps, spatial RMSE maps,
  and inter-seed variability maps (both cross-run spatial dispersion and selected-frame
  pixelwise seed std)
* ``05_transfer_and_coverage/`` — last/isotherm/IoU transfer matrices, coverage-distance
  profiles, SENS-REJ node-level comparison (flag vs sub)
* ``06_case_studies/`` — common-scale visual cards for a-priori representative frames
  plus explicit-rule best and worst VAL frames
* ``07_compute_and_status/`` — run status matrix (plateau / budget_exhausted / incomplete /
  failed / missing), GPU-hours & preemptions, end-to-end eval wall time
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

from combustion_recon.evaluation.piv_metrics import DISTANCE_BINS, compute_interseed_field_stats
from combustion_recon.evaluation.piv_plots_main import (
    CELL_ORDER,
    EVAL_ORDER_MAIN,
    FAM_COLOR,
    FAM_LABEL,
    ROLE_LABEL,
    _apply_style,
    _ordered,
    _save,
)


# --------------------------------------------------------------------------- 01: Training curves

def _atlas_training_curves(agg: Mapping[str, Any], audit: Mapping[str, Any], out: Path) -> list[Path]:
    produced: list[Path] = []
    expl = agg["exploratory"]
    tr_df: pd.DataFrame = expl.get("train_epoch_curves", pd.DataFrame())
    va_df: pd.DataFrame = expl.get("val_eval_curves", pd.DataFrame())
    audit_runs = audit["table"].drop_duplicates(subset=["run_id"]) if not audit["table"].empty else pd.DataFrame()
    bud_map = dict(zip(audit_runs["run_id"], audit_runs["budget_exhausted"])) if not audit_runs.empty else {}

    if not tr_df.empty:
        cells = _ordered(tr_df["train_cell"].unique(), CELL_ORDER)
        fams = _ordered(tr_df["family"].unique(), ("sparse_unet", "adapted_glu"))
        fig, axes = plt.subplots(len(fams), len(cells), figsize=(3.4 * len(cells), 2.8 * len(fams) + 0.5), squeeze=False, sharex=False, sharey="row", constrained_layout=True)
        for i, fam in enumerate(fams):
            for j, cell in enumerate(cells):
                ax = axes[i, j]
                g = tr_df[(tr_df["family"] == fam) & (tr_df["train_cell"] == cell)]
                for rid, gr in g.groupby("run_id"):
                    ls = "--" if bud_map.get(rid, False) else "-"
                    ax.plot(gr["epoch"], gr["train_loss_mean"], lw=1.1, ls=ls, alpha=0.85, label=f"s{int(gr['model_seed'].iloc[0])}")
                ax.set_title(f"{FAM_LABEL.get(fam, fam)} — {cell}", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel("Loss training (MSE norm.)")
                if i == len(fams) - 1:
                    ax.set_xlabel("Epoca")
                if not g.empty:
                    ax.legend(fontsize=6.5, ncol=2, frameon=False)
        fig.suptitle("Atlante 01a — Curve di training per cella e seed (tratteggiato = budget_exhausted)", fontsize=10.5, fontweight="bold")
        produced.append(_save(fig, out / "atlas_curves_train_loss.png"))

    if not va_df.empty:
        cells = _ordered(va_df["train_cell"].unique(), CELL_ORDER)
        fams = _ordered(va_df["family"].unique(), ("sparse_unet", "adapted_glu"))
        fig, axes = plt.subplots(len(fams), len(cells), figsize=(3.4 * len(cells), 2.8 * len(fams) + 0.5), squeeze=False, sharey="row", constrained_layout=True)
        for i, fam in enumerate(fams):
            for j, cell in enumerate(cells):
                ax = axes[i, j]
                g = va_df[(va_df["family"] == fam) & (va_df["train_cell"] == cell)]
                for rid, gr in g.groupby("run_id"):
                    ls = "--" if bud_map.get(rid, False) else "-"
                    ax.plot(gr["epoch"], gr["val_rmse_K"], lw=1.2, ls=ls, alpha=0.88, label=f"s{int(gr['model_seed'].iloc[0])}")
                    v = pd.to_numeric(gr["val_rmse_K"], errors="coerce")
                    if v.notna().any():
                        bi = v.idxmin()
                        ax.scatter([gr.loc[bi, "epoch"]], [v.loc[bi]], s=22, zorder=5)
                ax.set_title(f"{FAM_LABEL.get(fam, fam)} — {cell}", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel("VAL macro RMSE (K)")
                if i == len(fams) - 1:
                    ax.set_xlabel("Epoca")
                if not g.empty:
                    ax.legend(fontsize=6.5, ncol=2, frameon=False)
        fig.suptitle("Atlante 01b — Curve di validazione (stessa osservazione; punto = epoca best)", fontsize=10.5, fontweight="bold")
        produced.append(_save(fig, out / "atlas_curves_val_rmse_by_group.png"))

    csum: pd.DataFrame = expl.get("training_curves_summary", pd.DataFrame())
    if not csum.empty and csum["best_val_epoch"].notna().any():
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for fam in _ordered(csum["family"].unique(), ("sparse_unet", "adapted_glu")):
            g = csum[csum["family"] == fam]
            bud = g["run_id"].map(lambda r: bool(bud_map.get(r, False))).to_numpy()
            col = FAM_COLOR.get(fam, "#444")
            if (~bud).any():
                ax.scatter(g["last_eval_epoch"][~bud], g["best_val_epoch"][~bud], color=col, s=36, label=f"{FAM_LABEL.get(fam, fam)} (plateau)")
            if bud.any():
                ax.scatter(g["last_eval_epoch"][bud], g["best_val_epoch"][bud], facecolors="white", edgecolors=col, lw=1.4, s=44, label=f"{FAM_LABEL.get(fam, fam)} (budget_exhausted)")
        lim = max(10.0, float(csum["last_eval_epoch"].max()) * 1.05)
        ax.plot([0, lim], [0, lim], color="#999", ls=":", lw=1.0, label="best = last")
        ax.set_xlabel("Ultima epoca valutata")
        ax.set_ylabel("Epoca di best VAL RMSE")
        ax.set_title("Atlante 01c — Epoca best vs durata del training", loc="left", fontweight="bold")
        ax.legend(fontsize=8)
        produced.append(_save(fig, out / "atlas_curves_lr_and_best_epoch.png"))

    return produced


# --------------------------------------------------------------------------- 02: Distributions & paired

def _atlas_distributions_and_paired(agg: Mapping[str, Any], out: Path) -> list[Path]:
    produced: list[Path] = []
    pf: pd.DataFrame = agg["per_frame"]
    paired: pd.DataFrame = agg["exploratory"].get("paired_frame_deltas", pd.DataFrame())

    if not pf.empty:
        sub = pf[(pf["checkpoint_kind"] == "best") & (pf["eval_config"] == pf["train_config_id"])].copy()
        if not sub.empty:
            role = "val_eval" if "val_eval" in set(sub["role"]) else str(sub["role"].iloc[0])
            sub = sub[sub["role"] == role]
            fmean = sub.groupby(["family", "train_cell", "condition", "frame_index"], as_index=False)["rmse_K"].mean()
            fams = _ordered(fmean["family"].unique(), ("sparse_unet", "adapted_glu"))
            cells = _ordered(fmean["train_cell"].unique(), CELL_ORDER)

            fig, axes = plt.subplots(1, len(fams), figsize=(5.4 * len(fams), 4.0), sharey=True, squeeze=False)
            for ax, fam in zip(axes[0], fams):
                sf = fmean[fmean["family"] == fam]
                data = [sf[sf["train_cell"] == c]["rmse_K"].to_numpy(dtype=float) for c in cells if (sf["train_cell"] == c).any()]
                lbls = [c for c in cells if (sf["train_cell"] == c).any()]
                if data:
                    bp = ax.boxplot(data, tick_labels=lbls, patch_artist=True, showfliers=True)
                    for box in bp["boxes"]:
                        box.set_facecolor(FAM_COLOR.get(fam, "#444"))
                        box.set_alpha(0.35)
                ax.set_title(f"{FAM_LABEL.get(fam, fam)} — {ROLE_LABEL.get(role, role)}", loc="left")
                ax.tick_params(axis="x", rotation=20)
                ax.set_xlabel("Cella (valutata sulla propria configurazione)")
            axes[0][0].set_ylabel("RMSE per frame (media fra seed/repliche, K)")
            fig.suptitle("Atlante 02a — Distribuzione degli errori per frame", fontsize=10.5, fontweight="bold")
            produced.append(_save(fig, out / "atlas_frame_error_distributions.png"))

    if not paired.empty:
        role = "val_eval" if "val_eval" in set(paired["role"]) else str(paired["role"].iloc[0])
        pr = paired[paired["role"] == role]

        # PIV-A vs IDEAL-A paired scatter
        piv_pairs = [k for k in sorted(pr["pair_kind"].unique()) if "PIV-A_vs_IDEAL-A" in k]
        if piv_pairs:
            fig, axes = plt.subplots(1, len(piv_pairs), figsize=(5.2 * len(piv_pairs), 4.2), squeeze=False)
            for ax, pk in zip(axes[0], piv_pairs):
                g = pr[pr["pair_kind"] == pk]
                lo = float(min(g["rmse_left_K"].min(), g["rmse_right_K"].min())) * 0.95
                hi = float(max(g["rmse_left_K"].max(), g["rmse_right_K"].max())) * 1.05
                ax.scatter(g["rmse_right_K"], g["rmse_left_K"], s=18, alpha=0.7, color="#1b6ca8")
                ax.plot([lo, hi], [lo, hi], color="#222", ls="--", lw=1.0)
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
                ax.set_xlabel("IDEAL-A RMSE (K)")
                ax.set_ylabel("PIV-A RMSE (K)")
                ax.set_title(f"{pk}\nmediana Δ = {g['delta_K'].median():+.2f} K", fontsize=9)
            fig.suptitle(f"Atlante 02b — Confronto appaiato frame-per-frame: PIV-A vs IDEAL-A ({ROLE_LABEL.get(role, role)})", fontsize=10.5, fontweight="bold")
            produced.append(_save(fig, out / "atlas_paired_frame_scatter_piv_vs_ideal.png"))

        # U-Net vs GLU paired scatter
        fam_pairs = [k for k in sorted(pr["pair_kind"].unique()) if k.startswith("U-Net_vs_GLU:")]
        if fam_pairs:
            fig, axes = plt.subplots(1, len(fam_pairs), figsize=(4.2 * len(fam_pairs), 3.8), squeeze=False)
            for ax, pk in zip(axes[0], fam_pairs):
                g = pr[pr["pair_kind"] == pk]
                lo = float(min(g["rmse_left_K"].min(), g["rmse_right_K"].min())) * 0.95
                hi = float(max(g["rmse_left_K"].max(), g["rmse_right_K"].max())) * 1.05
                ax.scatter(g["rmse_right_K"], g["rmse_left_K"], s=16, alpha=0.7, color="#c0392b")
                ax.plot([lo, hi], [lo, hi], color="#222", ls="--", lw=1.0)
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
                ax.set_xlabel("GLU RMSE (K)")
                ax.set_ylabel("U-Net RMSE (K)")
                ax.set_title(f"{pk.split(':', 1)[1]}\nmediana (U-Net − GLU) = {g['delta_K'].median():+.2f} K", fontsize=8.8)
            fig.suptitle(f"Atlante 02c — Confronto appaiato frame-per-frame: U-Net vs GLU ({ROLE_LABEL.get(role, role)})", fontsize=10.5, fontweight="bold")
            produced.append(_save(fig, out / "atlas_paired_frame_scatter_unet_vs_glu.png"))

    return produced


# --------------------------------------------------------------------------- 03: By condition & region

def _atlas_condition_and_region(agg: Mapping[str, Any], out: Path) -> list[Path]:
    produced: list[Path] = []
    cv: pd.DataFrame = agg["exploratory"].get("condition_vs_vref", pd.DataFrame())
    pr: pd.DataFrame = agg["confirmatory"]["per_run_scenario"]

    if not cv.empty:
        sub = cv[cv["checkpoint_kind"] == "best"].copy()
        # VAL condition breakdown (own config)
        # Map train_cell to own eval_config
        own_map = {"IDEAL-A": "IDEAL-A", "PIV-A": "PIV-A", "IDEAL-B": "IDEAL-B-in", "PIV-B": "PIV-B-in", "BRIDGE-5pct-IDEAL": "E0"}
        sub_own = sub[sub["eval_config"] == sub["train_cell"].map(own_map)]
        val_sub = sub_own[sub_own["role"] == "val_eval"]
        if not val_sub.empty:
            conds = sorted(val_sub["condition"].unique())
            cells = _ordered(val_sub["train_cell"].unique(), CELL_ORDER)
            fams = _ordered(val_sub["family"].unique(), ("sparse_unet", "adapted_glu"))
            fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.8), sharey=True, squeeze=False)
            for ax, cond in zip(axes[0], conds):
                gc = val_sub[val_sub["condition"] == cond]
                for fi, fam in enumerate(fams):
                    col = FAM_COLOR.get(fam, "#444")
                    off = (-0.16 + 0.32 * fi) if len(fams) > 1 else 0.0
                    ys = [float(gc[(gc["family"] == fam) & (gc["train_cell"] == c)]["rmse_mean_K"].mean()) if ((gc["family"] == fam) & (gc["train_cell"] == c)).any() else np.nan for c in cells]
                    es = [float(gc[(gc["family"] == fam) & (gc["train_cell"] == c)]["rmse_std_K"].mean()) if ((gc["family"] == fam) & (gc["train_cell"] == c)).any() else 0.0 for c in cells]
                    ax.bar(np.arange(len(cells)) + off, ys, width=0.30, color=col, alpha=0.82, yerr=[e if math.isfinite(e) else 0.0 for e in es], capsize=2, label=FAM_LABEL.get(fam, fam))
                vr = gc["v_ref_m_s"].dropna().iloc[0] if gc["v_ref_m_s"].notna().any() else float("nan")
                vt = f" (v_ref={vr:.1f} m/s)" if math.isfinite(vr) else ""
                ax.set_title(f"{cond.replace('.h5', '')}{vt}", fontsize=9)
                ax.set_xticks(range(len(cells)))
                ax.set_xticklabels(cells, rotation=22, ha="right", fontsize=7.5)
            axes[0][0].set_ylabel("RMSE per condizione (K, best)")
            axes[0][0].legend(fontsize=8)
            fig.suptitle("Atlante 03a — Accuratezza per singola condizione VAL123", fontsize=10.5, fontweight="bold")
            produced.append(_save(fig, out / "atlas_rmse_by_condition_val.png"))

        # seen220 vs v_ref / sigma_u
        seen_sub = sub_own[(sub_own["role"] == "seen_train_eval") & sub_own["v_ref_m_s"].notna()]
        if not seen_sub.empty:
            fig, ax = plt.subplots(figsize=(7.6, 4.2))
            for (fam, cell), g in seen_sub.groupby(["family", "train_cell"]):
                if cell not in ("IDEAL-A", "PIV-A", "PIV-B"):
                    continue
                mk = "o" if cell == "PIV-A" else ("s" if cell == "IDEAL-A" else "^")
                ax.scatter(g["v_ref_m_s"], g["rmse_mean_K"], color=FAM_COLOR.get(fam, "#444"), marker=mk, s=30, alpha=0.8, label=f"{FAM_LABEL.get(fam, fam)} {cell}")
            ax.set_xlabel("Velocità di riferimento condizione v_ref (m/s)  [σ_u ∝ v_ref]")
            ax.set_ylabel("RMSE per condizione su seen220 (K)")
            ax.set_title("Atlante 03b — Accuratezza per condizione seen220 in funzione di v_ref", loc="left", fontweight="bold")
            ax.legend(fontsize=7.8, ncol=2)
            produced.append(_save(fig, out / "atlas_rmse_by_condition_seen_vs_vref.png"))

    # Bias vs Centered RMSE decomposition
    if not pr.empty and {"abs_bias_K", "centered_rmse_K"}.issubset(pr.columns):
        sub = pr[(pr["checkpoint_kind"] == "best") & (pr["eval_config"] == pr["train_config_id"])]
        role = "val_eval" if "val_eval" in set(sub["role"]) else str(sub["role"].iloc[0])
        sub = sub[sub["role"] == role]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(6.8, 4.2))
            for (fam, cell), g in sub.groupby(["family", "train_cell"]):
                mk = {"IDEAL-A": "o", "PIV-A": "s", "IDEAL-B": "^", "PIV-B": "D"}.get(cell, "x")
                ax.scatter(g["centered_rmse_K"], g["abs_bias_K"], color=FAM_COLOR.get(fam, "#444"), marker=mk, s=38, alpha=0.85, label=f"{FAM_LABEL.get(fam, fam)} {cell}")
            ax.set_xlabel("RMSE centrata (pattern error, K)")
            ax.set_ylabel("|Bias| medio firmato (K)")
            ax.set_title(f"Atlante 03c — Decomposizione RMSE² = Bias² + RMSE_centrata² ({ROLE_LABEL.get(role, role)})", loc="left", fontweight="bold")
            ax.legend(fontsize=7.5, ncol=2)
            produced.append(_save(fig, out / "atlas_region_and_bias_breakdown.png"))

    return produced


# --------------------------------------------------------------------------- 04: Spatial maps & seed variability

def _load_spatial_maps(audit_table: pd.DataFrame, *, role: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Load compact spatial error accumulators per (family, train_cell) on their own eval config."""
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if audit_table.empty:
        return out
    best_rows = audit_table[(audit_table["checkpoint_kind"] == "best") & (audit_table["eval_valid"])]
    for _, r in best_rows.iterrows():
        tcfg = str(r["train_config_id"])
        sp = Path(str(r["eval_source_dir"])) / "spatial" / tcfg / f"{role}_rep0.npz"
        if not sp.is_file():
            continue
        with np.load(sp) as z:
            if "macro|mean_error_K" not in z or "macro|rmse_K" not in z:
                continue
            out.setdefault((str(r["family"]), str(r["train_cell"])), []).append(
                {
                    "run_id": str(r["run_id"]),
                    "model_seed": int(r["model_seed"]),
                    "mean_error_K": z["macro|mean_error_K"].copy(),
                    "rmse_K": z["macro|rmse_K"].copy(),
                }
            )
    return out


def _atlas_spatial_and_seed_var(audit: Mapping[str, Any], out: Path) -> list[Path]:
    produced: list[Path] = []
    role = "val_eval"
    smaps = _load_spatial_maps(audit["table"], role=role)
    if not smaps:
        role = "seen_train_eval"
        smaps = _load_spatial_maps(audit["table"], role=role)
    if not smaps:
        return produced

    keys = sorted(smaps.keys(), key=lambda k: (0 if k[0] == "sparse_unet" else 1, CELL_ORDER.index(k[1]) if k[1] in CELL_ORDER else 99))
    mean_err_by_key = {k: np.mean([d["mean_error_K"] for d in smaps[k]], axis=0) for k in keys}
    mean_rmse_by_key = {k: np.mean([d["rmse_K"] for d in smaps[k]], axis=0) for k in keys}
    seed_std_rmse_by_key = {
        k: (np.std([d["rmse_K"] for d in smaps[k]], axis=0, ddof=1) if len(smaps[k]) >= 2 else np.zeros((128, 128)))
        for k in keys
    }

    ncols = min(4, len(keys))
    nrows = int(math.ceil(len(keys) / ncols))

    # 04a: Spatial mean signed error maps (common diverging scale)
    elim = float(max(20.0, np.nanpercentile(np.concatenate([np.abs(v).ravel() for v in mean_err_by_key.values()]), 98)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.0 * nrows + 0.4), squeeze=False, constrained_layout=True)
    im = None
    for idx, k in enumerate(keys):
        ax = axes[idx // ncols, idx % ncols]
        im = ax.imshow(mean_err_by_key[k], cmap="RdBu_r", norm=TwoSlopeNorm(0.0, vmin=-elim, vmax=elim), origin="upper")
        ax.set_title(f"{FAM_LABEL.get(k[0], k[0])} {k[1]} (n={len(smaps[k])})", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(keys), nrows * ncols):
        axes[j // ncols, j % ncols].set_axis_off()
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.82, label="Errore medio firmato pred − vera (K)")
    fig.suptitle(f"Atlante 04a — Mappe dell'errore medio spaziale ({ROLE_LABEL.get(role, role)}, best, scala comune)", fontsize=10.5, fontweight="bold")
    produced.append(_save(fig, out / "atlas_spatial_mean_error_maps.png"))

    # 04b: Spatial RMSE maps (common sequential scale)
    rmax = float(max(30.0, np.nanpercentile(np.concatenate([v.ravel() for v in mean_rmse_by_key.values()]), 98)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.0 * nrows + 0.4), squeeze=False, constrained_layout=True)
    im = None
    for idx, k in enumerate(keys):
        ax = axes[idx // ncols, idx % ncols]
        im = ax.imshow(mean_rmse_by_key[k], cmap="magma", vmin=0.0, vmax=rmax, origin="upper")
        ax.set_title(f"{FAM_LABEL.get(k[0], k[0])} {k[1]} (n={len(smaps[k])})", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(keys), nrows * ncols):
        axes[j // ncols, j % ncols].set_axis_off()
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.82, label="RMSE spaziale locale (K)")
    fig.suptitle(f"Atlante 04b — Mappe della RMSE spaziale ({ROLE_LABEL.get(role, role)}, best, scala comune)", fontsize=10.5, fontweight="bold")
    produced.append(_save(fig, out / "atlas_spatial_rmse_maps.png"))

    # 04c: Inter-seed spatial variability maps
    multi_keys = [k for k in keys if len(smaps[k]) >= 2]
    if multi_keys:
        nc2 = min(4, len(multi_keys))
        nr2 = int(math.ceil(len(multi_keys) / nc2))
        smax = float(max(5.0, np.nanpercentile(np.concatenate([seed_std_rmse_by_key[k].ravel() for k in multi_keys]), 98)))
        fig, axes = plt.subplots(nr2, nc2, figsize=(3.1 * nc2, 3.0 * nr2 + 0.4), squeeze=False, constrained_layout=True)
        im = None
        for idx, k in enumerate(multi_keys):
            ax = axes[idx // nc2, idx % nc2]
            im = ax.imshow(seed_std_rmse_by_key[k], cmap="plasma", vmin=0.0, vmax=smax, origin="upper")
            ax.set_title(f"{FAM_LABEL.get(k[0], k[0])} {k[1]} (std su {len(smaps[k])} seed)", fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
        for j in range(len(multi_keys), nr2 * nc2):
            axes[j // nc2, j % nc2].set_axis_off()
        if im is not None:
            fig.colorbar(im, ax=axes, shrink=0.82, label="Deviazione standard fra seed della RMSE spaziale (K)")
        fig.suptitle(f"Atlante 04c — Variabilità spaziale fra seed ({ROLE_LABEL.get(role, role)}, best, scala comune)", fontsize=10.5, fontweight="bold")
        produced.append(_save(fig, out / "atlas_spatial_interseed_variability.png"))

    return produced


# --------------------------------------------------------------------------- 05: Transfer & coverage

def _atlas_transfer_and_coverage(agg: Mapping[str, Any], out: Path) -> list[Path]:
    produced: list[Path] = []
    conf = agg["confirmatory"]
    cs: pd.DataFrame = conf["cell_seed_summary"]
    b_dist: pd.DataFrame = conf["b_positions_and_distance"]
    rej_df: pd.DataFrame = conf["rej_node_metrics"]

    # Transfer matrices for last checkpoint, isotherm 2100 K, and hot IoU
    for fname, ck, col, cmap, title, cbar in (
        ("atlas_transfer_matrix_last.png", "last", "rmse_K__mean", "viridis_r", "Matrice di trasferimento — checkpoint last (RMSE, K)", "RMSE macro (K)"),
        ("atlas_transfer_matrix_isotherm2100.png", "best", "iso2100_sym_dist_mm__mean", "cividis_r", "Matrice di trasferimento — distanza isoterma 2100 K (mm, best)", "Distanza media simmetrica (mm)"),
        ("atlas_transfer_matrix_hot_iou.png", "best", "high_iou__mean", "viridis", "Matrice di trasferimento — IoU zona calda ≥ 2100 K (best)", "IoU zona calda"),
    ):
        if cs.empty or col not in cs.columns:
            continue
        sub = cs[(cs["checkpoint_kind"] == ck) & (~cs["eval_config"].astype(str).str.startswith("SENS-")) & cs[col].notna()]
        if sub.empty:
            continue
        role = "val_eval" if "val_eval" in set(sub["role"]) else str(sub["role"].iloc[0])
        sub = sub[sub["role"] == role]
        fams = _ordered(sub["family"].unique(), ("sparse_unet", "adapted_glu"))
        tcells = _ordered(sub["train_cell"].unique(), CELL_ORDER)
        ecfgs = _ordered(sub["eval_config"].unique(), EVAL_ORDER_MAIN)
        vmin, vmax = float(sub[col].min()), float(sub[col].max())
        if vmax <= vmin:
            vmax = vmin + 1e-3
        fig, axes = plt.subplots(1, len(fams), figsize=(max(5.5, 0.72 * len(ecfgs)) * len(fams), max(2.8, 0.65 * len(tcells)) + 0.8), squeeze=False, constrained_layout=True)
        im = None
        for j, fam in enumerate(fams):
            ax = axes[0, j]
            M = np.full((len(tcells), len(ecfgs)), np.nan)
            sf = sub[sub["family"] == fam]
            for ri, tc in enumerate(tcells):
                for cj, ec in enumerate(ecfgs):
                    hit = sf[(sf["train_cell"] == tc) & (sf["eval_config"] == ec)]
                    if not hit.empty:
                        M[ri, cj] = float(hit[col].iloc[0])
            im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(ecfgs))); ax.set_xticklabels(ecfgs, rotation=32, ha="right", fontsize=7.5)
            ax.set_yticks(range(len(tcells))); ax.set_yticklabels(tcells, fontsize=8)
            ax.set_title(f"{FAM_LABEL.get(fam, fam)} — {ROLE_LABEL.get(role, role)}", loc="left")
            ax.grid(False)
            for ri in range(len(tcells)):
                for cj in range(len(ecfgs)):
                    if math.isfinite(M[ri, cj]):
                        ax.text(cj, ri, f"{M[ri, cj]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[ri, cj] < 0.5 * (vmin + vmax) else "#111")
        if im is not None:
            fig.colorbar(im, ax=axes, shrink=0.82, label=cbar)
        fig.suptitle(f"Atlante 05 — {title}", fontsize=10.5, fontweight="bold")
        produced.append(_save(fig, out / fname))

    # Coverage vs accuracy scatter
    if not b_dist.empty and "coverage_observed_frac" in b_dist.columns:
        sub = b_dist[b_dist["checkpoint_kind"] == "best"]
        role = "val_eval" if "val_eval" in set(sub["role"]) else str(sub["role"].iloc[0])
        sub = sub[sub["role"] == role]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for fam in _ordered(sub["family"].unique(), ("sparse_unet", "adapted_glu")):
                g = sub[sub["family"] == fam]
                ax.scatter(100.0 * g["coverage_observed_frac"], g["rmse_K"], color=FAM_COLOR.get(fam, "#444"), s=36, alpha=0.8, label=FAM_LABEL.get(fam, fam))
                for _, r in g.iterrows():
                    if r["eval_config"] in ("PIV-A", "PIV-B-in", "PIV-B-down", "PIV-B-off", "IDEAL-A", "IDEAL-B-in"):
                        ax.annotate(f"{r['train_cell']}→{r['eval_config']}", (100.0 * r["coverage_observed_frac"], r["rmse_K"]), fontsize=6.5, alpha=0.8)
            ax.set_xlabel("Copertura osservata (% nodi della slice)")
            ax.set_ylabel("RMSE macro (K)")
            ax.set_title(f"Atlante 05d — Copertura osservata vs accuratezza ({ROLE_LABEL.get(role, role)}, best)", loc="left", fontweight="bold")
            ax.legend(fontsize=8)
            produced.append(_save(fig, out / "atlas_coverage_distance_profiles.png"))

    # SENS-REJ node-level comparison (flag vs sub, iso vs clu)
    if not rej_df.empty:
        sub = rej_df[rej_df["checkpoint_kind"] == "best"]
        role = "val_eval" if "val_eval" in set(sub["role"]) else str(sub["role"].iloc[0])
        sub = sub[sub["role"] == role]
        if not sub.empty:
            piv_a = sub[sub["train_cell"] == "PIV-A"]
            if not piv_a.empty:
                sub = piv_a
            ecfgs = sorted(sub["eval_config"].unique())
            fig, ax = plt.subplots(figsize=(8.2, 4.0))
            x = np.arange(len(ecfgs))
            w = 0.25
            for col_i, (cname, lbl, c) in enumerate(
                (
                    ("nonrej_valid_rmse_K", "Nodi validi non rigettati", "#2e8b57"),
                    ("rej_nodes_rmse_K", "Nodi rigettati (flag o sub)", "#c0392b"),
                    ("rmse_K", "Campo completo", "#1b6ca8"),
                )
            ):
                if cname not in sub.columns:
                    continue
                ys = [float(sub[sub["eval_config"] == e][cname].mean()) for e in ecfgs]
                ax.bar(x + (col_i - 1) * w, ys, width=w, color=c, alpha=0.85, label=lbl)
            ax.set_xticks(x)
            ax.set_xticklabels(ecfgs, rotation=18, ha="right", fontsize=8)
            ax.set_ylabel("RMSE (K)")
            ax.set_title(f"Atlante 05e — SENS-REJ (non calibrata): errore sui nodi rigettati/sostituiti ({ROLE_LABEL.get(role, role)})", loc="left", fontweight="bold")
            ax.legend(fontsize=8)
            produced.append(_save(fig, out / "atlas_rej_flag_vs_sub_nodes.png"))

    return produced


# --------------------------------------------------------------------------- 06: Case studies (rep + best/worst)

def _render_case_card(
    audit_table: pd.DataFrame,
    *,
    role: str,
    condition: str,
    frame_index: int,
    title_prefix: str,
    out_path: Path,
) -> Path | None:
    """Common-scale visual card across cells for one (role, condition, frame_index), with inter-seed std if n>=2."""
    best_rows = audit_table[(audit_table["checkpoint_kind"] == "best") & (audit_table["eval_valid"])]
    grouped_fields: list[dict[str, Any]] = []
    for (fam, tcell), g in best_rows.groupby(["family", "train_cell"], sort=True):
        preds_by_seed: dict[int, np.ndarray] = {}
        true_T = None
        fmask = None
        tcfg = str(g["train_config_id"].iloc[0])
        key = f"{tcfg}|{role}|{condition}|{int(frame_index)}"
        for _, r in g.sort_values("model_seed").iterrows():
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
            with np.load(npz_p) as z:
                if f"{key}|pred_T_K" not in z:
                    continue
                preds_by_seed[int(r["model_seed"])] = z[f"{key}|pred_T_K"].copy()
                if true_T is None:
                    true_T = z[f"{key}|true_T_K"].copy()
                if fmask is None and f"{key}|field_mask" in z:
                    fmask = z[f"{key}|field_mask"].copy()
        if preds_by_seed and true_T is not None:
            stats = compute_interseed_field_stats(preds_by_seed, true_T)
            grouped_fields.append(
                {
                    "family": str(fam),
                    "train_cell": str(tcell),
                    "eval_config": tcfg,
                    "true_T_K": true_T,
                    "field_mask": fmask,
                    **stats,
                }
            )
    if not grouped_fields:
        return None

    grouped_fields.sort(key=lambda d: (0 if d["family"] == "sparse_unet" else 1, CELL_ORDER.index(d["train_cell"]) if d["train_cell"] in CELL_ORDER else 99))
    true_T = grouped_fields[0]["true_T_K"]
    tmin = float(min([np.nanmin(true_T)] + [np.nanmin(d["mean_pred_T_K"]) for d in grouped_fields]))
    tmax = float(max([np.nanmax(true_T)] + [np.nanmax(d["mean_pred_T_K"]) for d in grouped_fields]))
    elim = float(max(50.0, min(400.0, np.nanpercentile(np.concatenate([np.abs(d["error_of_mean_K"]).ravel() for d in grouped_fields]), 99))))
    has_std = any(d["n_seeds"] >= 2 for d in grouped_fields)
    smax = (
        float(max(5.0, np.nanpercentile(np.concatenate([d["seed_std_T_K"].ravel() for d in grouped_fields if d["n_seeds"] >= 2]), 98)))
        if has_std
        else 1.0
    )

    nrows = 3 if has_std else 2
    ncols = len(grouped_fields) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.75 * ncols, 2.55 * nrows + 0.5), constrained_layout=True)
    im_t = axes[0, 0].imshow(true_T, cmap="inferno", vmin=tmin, vmax=tmax, origin="upper")
    axes[0, 0].set_title(f"LES vera\n{condition.replace('.h5', '')} f{frame_index}", fontsize=8.2)
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    for rr in range(1, nrows):
        axes[rr, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, f"{title_prefix}\n{role}, rep 0\nScale comuni:\nT ∈ [{tmin:.0f},{tmax:.0f}] K\nErr ∈ [±{elim:.0f}] K", ha="center", va="center", fontsize=7.5)

    im_e = None
    im_s = None
    for idx, d in enumerate(grouped_fields, start=1):
        axes[0, idx].imshow(d["mean_pred_T_K"], cmap="inferno", vmin=tmin, vmax=tmax, origin="upper")
        axes[0, idx].set_title(
            f"{FAM_LABEL.get(d['family'], d['family'])} {d['train_cell']} (n={d['n_seeds']})\nμ(RMSE)={d['mean_of_seed_rmse_K']:.1f} | RMSE(μ)={d['rmse_of_mean_pred_K']:.1f}",
            fontsize=7.3,
        )
        axes[0, idx].set_xticks([]); axes[0, idx].set_yticks([])

        im_e = axes[1, idx].imshow(d["error_of_mean_K"], cmap="RdBu_r", norm=TwoSlopeNorm(0.0, vmin=-elim, vmax=elim), origin="upper")
        if d["field_mask"] is not None and d["field_mask"].size and not d["field_mask"].all():
            axes[1, idx].contour(d["field_mask"].astype(float), levels=[0.5], colors=["#111"], linewidths=1.0, linestyles="--")
        axes[1, idx].set_title("Errore della media (pred_μ − vera)", fontsize=7.2)
        axes[1, idx].set_xticks([]); axes[1, idx].set_yticks([])

        if has_std:
            if d["n_seeds"] >= 2:
                im_s = axes[2, idx].imshow(d["seed_std_T_K"], cmap="plasma", vmin=0.0, vmax=smax, origin="upper")
                axes[2, idx].set_title(f"Std fra seed (media={d['mean_spatial_seed_std_K']:.1f} K)", fontsize=7.2)
            else:
                axes[2, idx].text(0.5, 0.5, "n_seed = 1", ha="center", va="center", fontsize=7.5)
            axes[2, idx].set_xticks([]); axes[2, idx].set_yticks([])

    fig.colorbar(im_t, ax=axes[0, :], shrink=0.82, label="T (K)")
    if im_e is not None:
        fig.colorbar(im_e, ax=axes[1, 1:], shrink=0.82, label="Errore (K)")
    if im_s is not None:
        fig.colorbar(im_s, ax=axes[2, 1:], shrink=0.82, label="Std fra seed (K)")
    fig.suptitle(f"Atlante 06 — {title_prefix}: {condition.replace('.h5', '')} frame {frame_index} ({role}, scale comuni)", fontsize=10, fontweight="bold")
    return _save(fig, out_path)


def _atlas_case_studies(agg: Mapping[str, Any], audit: Mapping[str, Any], out: Path) -> list[Path]:
    produced: list[Path] = []
    sel = agg["exploratory"].get("selected_cases", {})

    # 1. A-priori fixed representative frames (up to 6 to stay clean & interpretable)
    rep = sel.get("a_priori_representative") or {}
    count = 0
    for role in ("val_eval", "seen_train_eval"):
        for item in rep.get(role, []):
            if count >= 6:
                break
            cond = str(item["condition"])
            fidx = int(item["frame_index"])
            safe_c = cond.replace(".h5", "")
            p = _render_case_card(
                audit["table"],
                role=role,
                condition=cond,
                frame_index=fidx,
                title_prefix="Rappresentativo a priori",
                out_path=out / f"atlas_rep_{role}_{safe_c}_f{fidx}.png",
            )
            if p is not None:
                produced.append(p)
                count += 1

    # 2. Best & worst cases selected by explicit rule
    bw = (sel.get("best_worst_val") or {}).get("selections") or []
    for kind in ("best", "worst"):
        # Deduplicate (condition, frame_index) taking top rank 1
        hits = [x for x in bw if x["selection_kind"] == kind and int(x["rank"]) == 1]
        seen_cf: set[tuple[str, int]] = set()
        for h in hits:
            cf = (str(h["condition"]), int(h["frame_index"]))
            if cf in seen_cf:
                continue
            seen_cf.add(cf)
            safe_c = cf[0].replace(".h5", "")
            p = _render_case_card(
                audit["table"],
                role=str(h["role"]),
                condition=cf[0],
                frame_index=cf[1],
                title_prefix=f"Caso {kind.upper()} (regola esplicita; {h['train_cell']} RMSE={h['mean_rmse_K']:.1f} K)",
                out_path=out / f"atlas_{kind}_case_{safe_c}_f{cf[1]}.png",
            )
            if p is not None:
                produced.append(p)
                break
    return produced


# --------------------------------------------------------------------------- 07: Compute & run status

def _atlas_compute_and_status(audit: Mapping[str, Any], accounting: Mapping[str, Any], out: Path) -> list[Path]:
    produced: list[Path] = []
    adf: pd.DataFrame = audit["table"]

    # 07a: Visual run-status matrix separating plateau / budget_exhausted / incomplete / failed / missing
    if not adf.empty:
        runs = adf.drop_duplicates(subset=["run_id"]).sort_values(["group", "family", "train_cell", "model_seed"])
        status_code = {
            "plateau": 0,
            "frozen_external": 0,
            "complete": 0,
            "same_as_best": 0,
            "budget_exhausted": 1,
            "incomplete": 2,
            "failed": 3,
            "missing": 4,
        }
        cmap = ListedColormap(["#2ecc71", "#f39c12", "#3498db", "#e74c3c", "#bdc3c7"])
        cols = ["training", "eval_best", "eval_last"]
        M = np.full((len(runs), len(cols)), 4, dtype=int)
        txt_m: list[list[str]] = []
        for ri, (_, r) in enumerate(runs.iterrows()):
            rid = r["run_id"]
            ts = str(r["training_status"])
            M[ri, 0] = status_code.get(ts, 4)
            row_txt = [ts]
            for ci, ck in enumerate(("best", "last"), start=1):
                hit = adf[(adf["run_id"] == rid) & (adf["checkpoint_kind"] == ck)]
                es = str(hit["eval_status"].iloc[0]) if not hit.empty else "missing"
                M[ri, ci] = status_code.get(es, 4)
                row_txt.append(es)
            txt_m.append(row_txt)

        fig, ax = plt.subplots(figsize=(6.8, max(3.2, 0.28 * len(runs) + 1.2)))
        ax.imshow(M, cmap=cmap, vmin=0, vmax=4, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(["Training", "Eval (best)", "Eval (last)"])
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels([f"{r['run_id']} ({r['group']})" for _, r in runs.iterrows()], fontsize=7.5)
        ax.grid(False)
        for ri in range(len(runs)):
            for ci in range(len(cols)):
                ax.text(ci, ri, txt_m[ri][ci], ha="center", va="center", fontsize=7.2, color="#111", fontweight="bold")
        ax.set_title("Atlante 07a — Stato dei run (plateau / budget_exhausted / incomplete / failed / missing)", loc="left", fontweight="bold")
        produced.append(_save(fig, out / "atlas_run_status_matrix.png"))

    # 07b: GPU-hours & preemptions from sacct/ledgers
    by_run: pd.DataFrame = accounting.get("by_run", pd.DataFrame())
    if not by_run.empty and (by_run["sacct_gpu_h"].notna().any() or by_run["train_summary_elapsed_s"].notna().any()):
        use_sacct = by_run["sacct_gpu_h"].notna().any()
        vals = by_run["sacct_gpu_h"] if use_sacct else (by_run["train_summary_elapsed_s"] / 3600.0)
        ylabel = "GPU-ore misurate (sacct -D)" if use_sacct else "Ore wall-time da summary.json"
        fig, ax = plt.subplots(figsize=(max(7.2, 0.35 * len(by_run) + 2.0), 4.2))
        colors = [FAM_COLOR.get(f, "#555") for f in by_run["family"]]
        bars = ax.bar(range(len(by_run)), vals.fillna(0.0), color=colors, alpha=0.85)
        for idx, (_, r) in enumerate(by_run.iterrows()):
            if bool(r["budget_exhausted"]):
                bars[idx].set_hatch("//")
                bars[idx].set_edgecolor("#111")
            pre = r.get("sacct_preemptions") if use_sacct else r.get("ledger_signals")
            if pd.notna(pre) and int(pre) > 0:
                ax.text(idx, float(vals.iloc[idx]) + 0.05, f"P:{int(pre)}", ha="center", va="bottom", fontsize=6.5, color="#c0392b")
        ax.set_xticks(range(len(by_run)))
        ax.set_xticklabels(by_run["run_id"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title("Atlante 07b — Costi di calcolo per run (tratteggio = budget_exhausted; P:n = preemption)", loc="left", fontweight="bold")
        produced.append(_save(fig, out / "atlas_compute_gpu_hours_and_preemptions.png"))

    # 07c: Eval wall-time by config
    ec: pd.DataFrame = accounting.get("eval_costs_by_config", pd.DataFrame())
    if not ec.empty:
        fig, ax = plt.subplots(figsize=(max(7.0, 0.4 * ec["eval_config"].nunique() + 2.0), 4.0))
        cfgs = sorted(ec["eval_config"].unique())
        fams = _ordered(ec["family"].unique(), ("sparse_unet", "adapted_glu"))
        x = np.arange(len(cfgs))
        w = 0.36
        for fi, fam in enumerate(fams):
            sf = ec[ec["family"] == fam]
            ys = [float(sf[sf["eval_config"] == c]["mean_wall_s"].mean()) if (sf["eval_config"] == c).any() else 0.0 for c in cfgs]
            ax.bar(x + (fi - 0.5 * (len(fams) - 1)) * w, ys, width=w, color=FAM_COLOR.get(fam, "#555"), alpha=0.85, label=FAM_LABEL.get(fam, fam))
        ax.set_xticks(x)
        ax.set_xticklabels(cfgs, rotation=32, ha="right", fontsize=7.5)
        ax.set_ylabel("Wall-time medio per unità (s, end-to-end)")
        ax.set_title("Atlante 07c — Tempi di valutazione end-to-end (osservazione + I/O + forward + metriche)", loc="left", fontweight="bold")
        ax.legend(fontsize=8)
        produced.append(_save(fig, out / "atlas_eval_walltime_by_config.png"))

    return produced


def generate_diagnostic_atlas(
    agg: Mapping[str, Any],
    audit: Mapping[str, Any],
    accounting: Mapping[str, Any],
    dest_atlas: Path,
) -> dict[str, list[Path]]:
    """Generate the complete structured diagnostic atlas under ``dest_atlas``."""
    _apply_style()
    dest_atlas = Path(dest_atlas)
    sections = {
        "01_training_curves": _atlas_training_curves(agg, audit, dest_atlas / "01_training_curves"),
        "02_distributions_and_paired": _atlas_distributions_and_paired(agg, dest_atlas / "02_distributions_and_paired"),
        "03_by_condition_and_region": _atlas_condition_and_region(agg, dest_atlas / "03_by_condition_and_region"),
        "04_spatial_maps_and_seed_var": _atlas_spatial_and_seed_var(audit, dest_atlas / "04_spatial_maps_and_seed_var"),
        "05_transfer_and_coverage": _atlas_transfer_and_coverage(agg, dest_atlas / "05_transfer_and_coverage"),
        "06_case_studies": _atlas_case_studies(agg, audit, dest_atlas / "06_case_studies"),
        "07_compute_and_status": _atlas_compute_and_status(audit, accounting, dest_atlas / "07_compute_and_status"),
    }
    return sections


__all__ = ["generate_diagnostic_atlas"]
