#!/usr/bin/env python3
"""Astra A3 technical aggregation. Reads the frozen Astra per-frame outputs; NO inference.

Macro definition (uniform everywhere): per-frame metric -> mean over frames of one condition
(sim_id; NaN skipped, valid-frame count reported) -> mean over conditions (sim_id) = one replica
value. Replicas (mask replicas 0/1/2, NOT independent seeds) are then aggregated explicitly:
``replica='mean'`` = arithmetic mean of the three replica values, with ``rep_min``/``rep_max``.
seen_train_eval (20 conditions) and val_eval (3 conditions) are never pooled.
No significance tests. No selection or re-weighting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = {
    "s5_sparse_unet_pon_seed0": ("U-Net", True),
    "s5_sparse_unet_poff_seed0": ("U-Net", False),
    "s5_adapted_glu_pon_seed0": ("GLU", True),
    "s5_adapted_glu_poff_seed0": ("GLU", False),
}
ROLES = ("seen_train_eval", "val_eval")
BASE_METHOD = {  # baseline method id -> (family, p_conditioning)
    "GappyPOD_p_off": ("GappyPOD", False), "GappyPOD_p_on": ("GappyPOD", True),
    "ParamMean_p_on": ("ParamMean", True), "GlobalMean": ("GlobalMean", False),
    "T_only_const": ("T-only", False),
}
KEY = ["role", "model", "treatment", "replica_id", "sim_id", "frame_index"]
HEADLINE = ["rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K", "hot_rmse_K", "peak_rmse_K",
            "jet_rmse_K", "nonhot_rmse_K", "hot_iou", "hot_area_pred", "hot_area_true"]
COUNTS = ["hot_n", "peak_n", "jet_n", "nonhot_n", "hot_n_true", "hot_n_pred", "hot_n_inter", "hot_n_union"]
OFFSETS = ["delta_oracle_K", "delta_ao9_K", "delta_ao9_minus_oracle_K", "anchor_mean_resid_K"]
DOMAINS = ("all", "off")
METRIC_COLS = [f"{d}_{m}" for d in DOMAINS for m in HEADLINE]
COUNT_COLS = [f"{d}_{m}" for d in DOMAINS for m in COUNTS]
# regional metrics whose valid support can be empty (NaN legitimate, never zero-filled)
REGION_COLS = [f"{d}_{m}" for d in DOMAINS for m in ("hot_rmse_K", "peak_rmse_K", "jet_rmse_K",
                                                      "nonhot_rmse_K", "hot_iou")]

INFO = [  # information available to each (model, treatment); labels, not scores
    ("neural U-Net/GLU", "free", "velocity at 5% random sites (persisted mask) [+ p only if p_on]", "no T observation"),
    ("neural U-Net/GLU", "oracle", "same as free + FULL true T field (via one scalar offset)", "diagnostic, not deployable"),
    ("neural U-Net/GLU", "ao9", "same as free + nine ideal T (3x3 grid rows/cols 32,64,95; no noise; tau=1)", "ideal simulation values, not thermocouples"),
    ("GappyPOD p_off/p_on", "free/oracle/ao9", "velocity at the same 5% sites [+ p if p_on]; frozen fitted POD, no refit", "oracle/AO9 as for neural"),
    ("ParamMean (p_on)", "free/oracle/ao9", "p only (NH3, phi); NO velocity; coefficients frozen from Gappy artifact", "receives p: not matched to the p_off prohibition"),
    ("GlobalMean", "free/oracle/ao9", "nothing (train_fit mean T field)", "reference floor"),
    ("T-only constant", "t_only_const", "only the nine ideal T: field = mean of the nine T", "no velocity, no p, no other pixels"),
]


def load(astra: Path) -> pd.DataFrame:
    frames = []
    for run, (fam, p_on) in RUNS.items():
        for role in ROLES:
            d = pd.read_csv(astra / run / f"per_frame_{role}.csv.gz")
            d["model"] = run
            d["family"] = fam
            d["p_on"] = p_on
            frames.append(d)
    for role in ROLES:
        d = pd.read_csv(astra / "baselines" / f"per_frame_{role}.csv.gz")
        d["model"] = d["method"]
        d["family"] = d["method"].map(lambda m: BASE_METHOD[m][0])
        d["p_on"] = d["method"].map(lambda m: BASE_METHOD[m][1])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    return df


def per_condition(df: pd.DataFrame, cols: list[str], count_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(["role", "model", "family", "p_on", "treatment", "replica_id", "sim_id"], sort=True)
    out = g[cols + count_cols + OFFSETS + ["mse_recovery_vs_oracle"]].mean()  # skipna
    out["n_frames"] = g.size()
    for c in REGION_COLS + ["mse_recovery_vs_oracle"]:
        out[f"{c}__n_valid_frames"] = g[c].count()
    return out.reset_index()


def to_macro(pc: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """condition table -> replica macro (mean over sim_id) -> replica-mean with min/max."""
    keys = ["role", "model", "family", "p_on", "treatment"]
    rep = pc.groupby(keys + ["replica_id"])[value_cols].mean().reset_index()  # NaN skipped; counts below
    ncond = pc.groupby(keys + ["replica_id"]).size().rename("n_conditions").reset_index()
    rep = rep.merge(ncond, on=keys + ["replica_id"])
    long = rep.melt(id_vars=keys + ["replica_id", "n_conditions"], value_vars=value_cols,
                    var_name="metric", value_name="value")
    long["replica"] = long["replica_id"].astype(str)
    agg = long.groupby(keys + ["metric"], sort=True).agg(
        value=("value", "mean"), rep_min=("value", "min"), rep_max=("value", "max"),
        n_conditions=("n_conditions", "first")).reset_index()
    agg["replica"] = "mean"
    return pd.concat([long.drop(columns="replica_id"), agg], ignore_index=True)[
        keys + ["metric", "replica", "value", "rep_min", "rep_max", "n_conditions"]]


def pairs() -> list[tuple[str, str, str, str, str, str]]:
    """(comparison, model_A, treat_A, model_B, treat_B, note) with diff = A - B."""
    out = []
    for m in list(RUNS) + ["GappyPOD_p_off", "GappyPOD_p_on", "ParamMean_p_on", "GlobalMean"]:
        out.append(("ao9_minus_free", m, "ao9", m, "free"))
        out.append(("oracle_minus_free", m, "oracle", m, "free"))
        out.append(("ao9_minus_oracle", m, "ao9", m, "oracle"))
    for t in ("free", "oracle", "ao9"):
        out.append(("pon_minus_poff", "s5_sparse_unet_pon_seed0", t, "s5_sparse_unet_poff_seed0", t))
        out.append(("pon_minus_poff", "s5_adapted_glu_pon_seed0", t, "s5_adapted_glu_poff_seed0", t))
        out.append(("pon_minus_poff", "GappyPOD_p_on", t, "GappyPOD_p_off", t))
        out.append(("unet_minus_glu", "s5_sparse_unet_pon_seed0", t, "s5_adapted_glu_pon_seed0", t))
        out.append(("unet_minus_glu", "s5_sparse_unet_poff_seed0", t, "s5_adapted_glu_poff_seed0", t))
    return out


def paired_frame_diffs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    idx = df.set_index(["role", "model", "treatment", "replica_id", "sim_id", "frame_index"])[cols].sort_index()
    for comp, ma, ta, mb, tb in pairs():
        for role in ROLES:
            a = idx.loc[(role, ma, ta)]
            b = idx.loc[(role, mb, tb)]
            if not a.index.equals(b.index):
                raise SystemExit(f"unpaired frame keys {comp} {ma}/{mb} {role}")
            d = (a - b).reset_index()
            d["role"], d["comparison"], d["model_A"], d["treat_A"], d["model_B"], d["treat_B"] = role, comp, ma, ta, mb, tb
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--astra-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df = load(Path(a.astra_root))

    # ---- completeness
    comp = df.groupby(["role", "model", "treatment", "replica_id"]).agg(
        n_rows=("frame_index", "size"), n_conditions=("sim_id", "nunique")).reset_index()
    nonfinite = df.select_dtypes("number").replace([np.inf, -np.inf], np.nan).isna().sum()
    nancols = nonfinite[nonfinite > 0].rename("n_nan_or_inf_total").reset_index().rename(columns={"index": "column"})
    comp.to_csv(out / "tables" / "completeness_rows.csv", index=False)
    nancols.to_csv(out / "tables" / "completeness_nan_columns.csv", index=False)

    # ---- per condition (replica level) and macro
    pc = per_condition(df, METRIC_COLS, COUNT_COLS)
    pc.to_csv(out / "tables" / "per_condition_metrics.csv", index=False, float_format="%.10g")
    macro = to_macro(pc, METRIC_COLS)
    macro.to_csv(out / "tables" / "macro_metrics_long.csv", index=False, float_format="%.10g")
    head = macro[(macro.replica == "mean") & macro.metric.isin(
        [f"{d}_{m}" for d in DOMAINS for m in ("rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K")])]
    wide = head.pivot_table(index=["role", "family", "p_on", "model", "treatment"], columns="metric",
                            values="value", sort=False).reset_index()
    wide.to_csv(out / "tables" / "macro_headline_wide.csv", index=False, float_format="%.6f")
    regional = macro[(macro.replica == "mean") & macro.metric.isin(REGION_COLS + [f"{d}_hot_area_pred" for d in DOMAINS]
                                                                  + [f"{d}_hot_area_true" for d in DOMAINS])]
    regional.to_csv(out / "tables" / "macro_regional_long.csv", index=False, float_format="%.10g")
    counts = to_macro(pc, COUNT_COLS)
    counts = counts[counts.replica == "mean"]
    counts.to_csv(out / "tables" / "macro_valid_counts_long.csv", index=False, float_format="%.10g")
    nvalid = [c for c in pc.columns if c.endswith("__n_valid_frames")]
    nv = pc.groupby(["role", "model", "treatment", "replica_id"])[nvalid].sum().reset_index()
    nv.to_csv(out / "tables" / "region_valid_frame_counts.csv", index=False)

    # ---- offsets (free rows carry the per-frame offsets; identical across treatments of a frame)
    off_src = df[df.treatment == "free"].copy()
    off_src["abs_delta_oracle_K"] = off_src["delta_oracle_K"].abs()
    off_src["abs_delta_ao9_K"] = off_src["delta_ao9_K"].abs()
    off_src["abs_ao9_minus_oracle_K"] = off_src["delta_ao9_minus_oracle_K"].abs()
    ao = df[df.treatment == "ao9"][KEY + ["mse_recovery_vs_oracle"]]
    off_src = off_src.drop(columns="mse_recovery_vs_oracle").merge(
        ao.drop(columns="treatment"), on=[k for k in KEY if k != "treatment"], how="left")
    ocols = OFFSETS + ["abs_delta_oracle_K", "abs_delta_ao9_K", "abs_ao9_minus_oracle_K", "mse_recovery_vs_oracle"]
    g = off_src.groupby(["role", "model", "family", "p_on", "treatment", "replica_id", "sim_id"])
    opc = g[ocols].mean().reset_index()
    opc["mse_recovery_vs_oracle__n_valid_frames"] = g["mse_recovery_vs_oracle"].count().values
    opc["n_frames"] = g.size().values
    opc = opc[opc.model.isin(list(RUNS) + ["GappyPOD_p_off", "GappyPOD_p_on", "ParamMean_p_on", "GlobalMean"])]
    opc.to_csv(out / "tables" / "offsets_per_condition.csv", index=False, float_format="%.10g")
    omacro = to_macro(opc.drop(columns=["treatment"]).assign(treatment="free"), ocols)
    omacro.drop(columns="treatment").to_csv(out / "tables" / "offsets_macro_long.csv", index=False, float_format="%.10g")

    # ---- paired differences (frame-level pairing, then the same macro)
    pf = paired_frame_diffs(df, METRIC_COLS)
    gk = ["role", "comparison", "model_A", "treat_A", "model_B", "treat_B", "replica_id", "sim_id"]
    pcd = pf.groupby(gk)[METRIC_COLS].mean().reset_index()
    pcd["n_frames"] = pf.groupby(gk).size().values
    pcd.to_csv(out / "tables" / "paired_diff_per_condition.csv", index=False, float_format="%.10g")
    repd = pcd.groupby(gk[:-1])[METRIC_COLS].mean().reset_index()
    ncd = pcd.groupby(gk[:-1]).size().rename("n_conditions").reset_index()
    repd = repd.merge(ncd, on=gk[:-1])
    lg = repd.melt(id_vars=gk[:-1] + ["n_conditions"], value_vars=METRIC_COLS, var_name="metric", value_name="value")
    k2 = ["role", "comparison", "model_A", "treat_A", "model_B", "treat_B", "metric"]
    ag = lg.groupby(k2, sort=False).agg(value=("value", "mean"), rep_min=("value", "min"), rep_max=("value", "max"),
                                         n_conditions=("n_conditions", "first")).reset_index()
    ag["replica"] = "mean"
    lg["replica"] = lg["replica_id"].astype(str)
    pm = pd.concat([lg.drop(columns="replica_id"), ag], ignore_index=True)
    # descriptive sign count over (condition, replica) pairs, mean row only
    sc = pcd.melt(id_vars=gk + ["n_frames"], value_vars=METRIC_COLS, var_name="metric", value_name="v")
    sc = sc.groupby(k2).agg(n_cond_rep_cells=("v", "count"), n_cells_negative=("v", lambda s: int((s < 0).sum())),
                            n_cells_positive=("v", lambda s: int((s > 0).sum()))).reset_index()
    pm = pm.merge(sc, on=k2, how="left")
    pm.loc[pm.replica != "mean", ["n_cond_rep_cells", "n_cells_negative", "n_cells_positive"]] = np.nan
    pm.to_csv(out / "tables" / "paired_diff_macro_long.csv", index=False, float_format="%.10g")

    pd.DataFrame(INFO, columns=["model_group", "treatment", "information_available", "label"]).to_csv(
        out / "tables" / "information_labels.csv", index=False)
    meta = {"tables": sorted(p.name for p in (out / "tables").glob("*.csv")),
            "macro_definition": "frame -> condition(sim_id, NaN skipped) -> mean over conditions = replica value; replica 'mean' = mean of 3 replica values",
            "n_rows_per_frame_table": int(len(df))}
    (out / "tables" / "aggregation_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
