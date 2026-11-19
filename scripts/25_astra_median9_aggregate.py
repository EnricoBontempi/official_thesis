#!/usr/bin/env python3
"""Technical aggregation of the validated median9 full set. No inference.

Uses the Astra A3 macro: frame -> condition (sim_id) -> mean over conditions = one
replica, then the mean of the three mask replicas with rep_min/rep_max.
seen_train_eval and val_eval stay separate. T-only stays in its own tables.

Paired values are treatment A minus treatment B. For an error metric, negative
means A is closer to the truth than B and positive means A is worse. Both
counts are kept. Nothing in these tables selects an estimator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("seen_train_eval", "val_eval")
EXPECTED_ROWS = {
    "s5_sparse_unet_pon_seed0": {"seen_train_eval": 3300, "val_eval": 1845},
    "s5_sparse_unet_poff_seed0": {"seen_train_eval": 3300, "val_eval": 1845},
    "s5_adapted_glu_pon_seed0": {"seen_train_eval": 3300, "val_eval": 1845},
    "s5_adapted_glu_poff_seed0": {"seen_train_eval": 3300, "val_eval": 1845},
    "baselines": {"seen_train_eval": 13860, "val_eval": 7749},
}
LEVEL = ("free", "oracle", "ao9", "mean9", "median9")
CONTRASTS = (
    ("median9_minus_ao9", "median9", "ao9"),
    ("median9_minus_mean9", "median9", "mean9"),
    ("mean9_minus_ao9", "mean9", "ao9"),
    ("median9_minus_oracle", "median9", "oracle"),
    ("mean9_minus_oracle", "mean9", "oracle"),
    ("ao9_minus_oracle", "ao9", "oracle"),
    ("median9_minus_free", "median9", "free"),
    ("mean9_minus_free", "mean9", "free"),
    ("ao9_minus_free", "ao9", "free"),
    ("oracle_minus_free", "oracle", "free"),
)
EXTRA_OFFSETS = [
    "delta_mean9_K", "delta_median9_K", "delta_mean9_minus_oracle_K",
    "delta_median9_minus_oracle_K", "delta_applied_minus_oracle_K",
]
SIGN_METRICS = ("all_rmse_K", "off_rmse_K", "all_mae_K", "off_mae_K", "all_abs_bias_K", "off_abs_bias_K")


def _load_a3():
    path = ROOT / "scripts" / "20_astra_a3_aggregate.py"
    spec = importlib.util.spec_from_file_location("astra_a3_aggregate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def treatment_pairs(models: list[str]) -> list[tuple[str, str, str, str, str]]:
    """Within-model contrasts only. T-only is not a member."""
    out = []
    for model in models:
        if model == "T_only_const":
            continue
        for name, left, right in CONTRASTS:
            out.append((name, model, left, model, right))
    return out


def _signs(series: pd.Series) -> pd.Series:
    return pd.Series({
        "n_frames": int(series.size),
        "n_negative": int((series < 0).sum()),
        "n_positive": int((series > 0).sum()),
        "n_zero": int((series == 0).sum()),
    })


def frame_sign_counts(paired: pd.DataFrame) -> pd.DataFrame:
    """A - B at each frame and replica. Positive error means A is worse than B."""
    keys = ["role", "comparison", "model_A", "treat_A", "treat_B", "replica_id"]
    parts = []
    for metric in SIGN_METRICS:
        rec = paired.groupby(keys, sort=True)[metric].apply(_signs).unstack().reset_index()
        rec["metric"] = metric
        parts.append(rec)
    out = pd.concat(parts, ignore_index=True)
    total = out.groupby(
        ["role", "comparison", "model_A", "treat_A", "treat_B", "metric"], sort=True
    )[["n_frames", "n_negative", "n_positive", "n_zero"]].sum().reset_index()
    total["replica_id"] = "sum"
    return pd.concat([out, total], ignore_index=True)


def _require_validated(root: Path) -> list[dict]:
    records = []
    for name in EXPECTED_ROWS:
        gate = json.loads((root / name / "GATE.json").read_text())
        complete = json.loads((root / name / "COMPLETE.json").read_text())
        replay = json.loads((root / name / "replay_primary.json").read_text())
        if not (gate.get("replay_executed") and gate.get("scientific_validated") and complete.get("scientific_validated")):
            raise SystemExit(f"{name}: refusing to aggregate without a passed replay gate")
        for role, n in EXPECTED_ROWS[name].items():
            path = root / name / f"per_frame_{role}.csv.gz"
            got = sum(1 for _ in pd.read_csv(path, usecols=["treatment"]).itertuples())
            # usecols still reads every row; len via iterator above is the count
            if got != n:
                raise SystemExit(f"{name} {role}: {got} rows, expected {n}")
        maxima = []
        for block in replay.get("replays", {}).values():
            for cell in block.get("cells") or []:
                for key in ("delta_macro_rmse_K", "max_abs_frame_rmse_K", "max_abs_delta_applied_K"):
                    if cell.get(key) is not None:
                        maxima.append(abs(float(cell[key])))
        records.append({
            "name": name,
            "scientific_validated": True,
            "max_abs_reported_gate_K": max(maxima) if maxima else None,
            "primary_files_sha256": replay.get("primary_files_sha256"),
            "git_sha": json.loads((root / name / "provenance.json").read_text()).get("astra_git_sha"),
        })
    return records


def _anchor_tables(resid: pd.DataFrame, s20) -> tuple[pd.DataFrame, pd.DataFrame]:
    resid = resid.copy()
    resid["abs_resid_K"] = resid["resid_K"].abs()
    resid["abs_ao9_contribution_K"] = resid["ao9_contribution_K"].abs()
    resid["site_tag"] = np.where(resid["anchor_col"] == 95, "col95", "other")
    keys = ["role", "model", "family", "p_on", "replica_id", "sim_id", "anchor_id", "anchor_row", "anchor_col", "site_tag"]
    cols = ["resid_K", "ao9_contribution_K", "abs_resid_K", "abs_ao9_contribution_K"]
    per = resid.groupby(keys, sort=True)[cols].mean().reset_index()
    per["n_frames"] = resid.groupby(keys, sort=True).size().values
    # share of absolute AO9 contribution from the two col-95 anchors, per frame, then the same macro
    frame_keys = ["role", "model", "family", "p_on", "replica_id", "sim_id", "frame_index"]
    tot = resid.groupby(frame_keys)["abs_ao9_contribution_K"].sum().rename("abs_sum")
    col95 = resid[resid.site_tag == "col95"].groupby(frame_keys)["abs_ao9_contribution_K"].sum().rename("abs_col95")
    share = pd.concat([tot, col95], axis=1).reset_index()
    share["share_abs_ao9_from_col95"] = share["abs_col95"] / share["abs_sum"].where(share["abs_sum"] > 0)
    share_pc = share.groupby(
        ["role", "model", "family", "p_on", "replica_id", "sim_id"], sort=True
    )["share_abs_ao9_from_col95"].mean().reset_index()
    share_pc["treatment"] = "ao9_contribution_share"
    share_macro = s20.to_macro(share_pc, ["share_abs_ao9_from_col95"])
    return per, share_macro


def aggregate(root: Path, out: Path) -> dict:
    s20 = _load_a3()
    gate_records = _require_validated(root)
    df = s20.load(root)
    level = df[df.treatment.isin(LEVEL)].copy()
    t_only = df[df.treatment == "t_only_const"].copy()
    if set(level.treatment) != set(LEVEL):
        raise SystemExit(f"missing treatments {set(LEVEL) - set(level.treatment)}")
    if t_only.empty or t_only["delta_median9_K"].notna().any():
        raise SystemExit("T-only is missing or received median9")

    s20.OFFSETS = list(dict.fromkeys(list(s20.OFFSETS) + EXTRA_OFFSETS))
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=False)
    pc = s20.per_condition(level, s20.METRIC_COLS, s20.COUNT_COLS)
    pc.to_csv(tables / "per_condition_metrics.csv", index=False, float_format="%.10g")
    macro = s20.to_macro(pc, s20.METRIC_COLS)
    macro.to_csv(tables / "macro_metrics_long.csv", index=False, float_format="%.10g")
    head_metrics = [f"{d}_{m}" for d in s20.DOMAINS for m in ("rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K")]
    head = macro[(macro.replica == "mean") & macro.metric.isin(head_metrics)]
    wide = head.pivot_table(
        index=["role", "family", "p_on", "model", "treatment"], columns="metric", values="value", sort=False
    ).reset_index()
    wide.to_csv(tables / "macro_headline_wide.csv", index=False, float_format="%.6f")
    regional_metrics = s20.REGION_COLS + [f"{d}_hot_area_pred" for d in s20.DOMAINS] + [f"{d}_hot_area_true" for d in s20.DOMAINS]
    macro[(macro.replica == "mean") & macro.metric.isin(regional_metrics)].to_csv(
        tables / "macro_regional_long.csv", index=False, float_format="%.10g"
    )
    s20.to_macro(pc, s20.COUNT_COLS).query("replica == 'mean'").to_csv(
        tables / "macro_valid_counts_long.csv", index=False, float_format="%.10g"
    )

    tpc = s20.per_condition(t_only, s20.METRIC_COLS, s20.COUNT_COLS)
    tpc.to_csv(tables / "t_only_per_condition.csv", index=False, float_format="%.10g")
    s20.to_macro(tpc, s20.METRIC_COLS).to_csv(tables / "t_only_macro_long.csv", index=False, float_format="%.10g")

    off = level.copy()
    off["abs_offset_error_K"] = off["delta_applied_minus_oracle_K"].abs()
    ocols = ["delta_applied_K", "delta_applied_minus_oracle_K", "abs_offset_error_K", "delta_oracle_K",
             "delta_ao9_K", "delta_mean9_K", "delta_median9_K"]
    og = off.groupby(["role", "model", "family", "p_on", "treatment", "replica_id", "sim_id"], sort=True)
    opc = og[ocols].mean().reset_index()
    opc["n_frames"] = og.size().values
    opc.to_csv(tables / "offset_error_per_condition.csv", index=False, float_format="%.10g")
    s20.to_macro(opc, ocols).to_csv(tables / "offset_error_macro_long.csv", index=False, float_format="%.10g")

    models = [m for m in list(s20.RUNS) + list(s20.BASE_METHOD) if m != "T_only_const"]
    s20.pairs = lambda: treatment_pairs(models)
    paired = s20.paired_frame_diffs(level, s20.METRIC_COLS)
    gk = ["role", "comparison", "model_A", "treat_A", "model_B", "treat_B", "replica_id", "sim_id"]
    pcd = paired.groupby(gk, sort=True)[s20.METRIC_COLS].mean().reset_index()
    pcd["n_frames"] = paired.groupby(gk, sort=True).size().values
    pcd.to_csv(tables / "paired_diff_per_condition.csv", index=False, float_format="%.10g")
    repd = pcd.groupby(gk[:-1], sort=True)[s20.METRIC_COLS].mean().reset_index()
    ncd = pcd.groupby(gk[:-1], sort=True).size().rename("n_conditions").reset_index()
    repd = repd.merge(ncd, on=gk[:-1])
    lg = repd.melt(id_vars=gk[:-1] + ["n_conditions"], value_vars=s20.METRIC_COLS, var_name="metric", value_name="value")
    k2 = ["role", "comparison", "model_A", "treat_A", "model_B", "treat_B", "metric"]
    ag = lg.groupby(k2, sort=False).agg(
        value=("value", "mean"), rep_min=("value", "min"), rep_max=("value", "max"),
        n_conditions=("n_conditions", "first"),
    ).reset_index()
    ag["replica"] = "mean"
    lg["replica"] = lg["replica_id"].astype(str)
    pm = pd.concat([lg.drop(columns="replica_id"), ag], ignore_index=True)
    sc = pcd.melt(id_vars=gk + ["n_frames"], value_vars=list(s20.METRIC_COLS), var_name="metric", value_name="v")
    sc = sc.groupby(k2, sort=False).agg(
        n_cond_rep_cells=("v", "count"),
        n_cells_negative=("v", lambda s: int((s < 0).sum())),
        n_cells_positive=("v", lambda s: int((s > 0).sum())),
    ).reset_index()
    pm = pm.merge(sc, on=k2, how="left")
    pm.loc[pm.replica != "mean", ["n_cond_rep_cells", "n_cells_negative", "n_cells_positive"]] = np.nan
    pm.to_csv(tables / "paired_diff_macro_long.csv", index=False, float_format="%.10g")
    frame_sign_counts(paired).to_csv(tables / "paired_frame_signs.csv", index=False)

    centered = pm[(pm.replica == "mean") & (pm.metric == "all_centered_rmse_K") & pm.comparison.str.endswith("_minus_free")]
    max_centered = float(centered["value"].abs().max()) if len(centered) else float("nan")
    if not np.isfinite(max_centered) or max_centered > 1e-6:
        raise SystemExit(f"centered RMSE moved under a pure translation: {max_centered}")

    resid_parts = []
    for run, (fam, p_on) in s20.RUNS.items():
        for role in ROLES:
            part = pd.read_csv(root / run / f"anchor_residuals_{role}.csv.gz")
            part["model"] = run
            part["family"] = fam
            part["p_on"] = p_on
            resid_parts.append(part)
    for role in ROLES:
        part = pd.read_csv(root / "baselines" / f"anchor_residuals_{role}.csv.gz")
        part["model"] = part["method"]
        part["family"] = part["method"].map(lambda m: s20.BASE_METHOD[m][0])
        part["p_on"] = part["method"].map(lambda m: s20.BASE_METHOD[m][1])
        resid_parts.append(part)
    resid = pd.concat(resid_parts, ignore_index=True)
    if set(resid.model) & {"T_only_const"}:
        raise SystemExit("T-only residuals are not part of this offset")
    per_anchor, share_macro = _anchor_tables(resid, s20)
    per_anchor.to_csv(tables / "anchor_residual_per_condition.csv", index=False, float_format="%.10g")
    share_macro.to_csv(tables / "anchor_col95_share_macro_long.csv", index=False, float_format="%.10g")

    labels = [
        ("same base prediction", "free", "velocity at the persisted 5% sites, plus p only when p_on", "no T observation"),
        ("same base prediction", "oracle", "free plus the full true T field through one scalar", "diagnostic, not deployable"),
        ("same base prediction", "ao9", "free plus nine ideal T, tau=1, so 0.9 times the mean residual", "shrink and mean, together"),
        ("same base prediction", "mean9", "free plus the unshrunk mean of the same nine residuals", "no tau; separates the shrink from AO9"),
        ("same base prediction", "median9", "free plus the ordinary median of the same nine residuals", "no tau; median versus AO9 changes estimator and shrink"),
        ("historical T-only", "t_only_const", "field equals the mean of the nine observed T", "mean9 and median9 were not applied"),
    ]
    pd.DataFrame(labels, columns=["group", "treatment", "information", "note"]).to_csv(
        tables / "information_labels.csv", index=False
    )
    meta = {
        "scope": "full_seen220_val123_replicas_0_1_2",
        "not_the_example_frame_package": True,
        "example_frame_package": "results/month2/astra_median9_second_pass/example_frame1000_rep0",
        "macro_definition": "frame -> condition(sim_id, NaN skipped) -> mean over conditions = replica; replica mean = mean of the three mask replicas",
        "paired_definition": "value = treatment_A - treatment_B. For RMSE, MAE and absolute bias, negative means A is closer to the truth and positive means A is worse. Both are kept. No estimator is selected.",
        "centered_translation_max_abs_K": max_centered,
        "gates": gate_records,
        "n_level_rows": int(len(level)),
        "n_t_only_rows": int(len(t_only)),
        "tables": sorted(p.name for p in tables.glob("*.csv")),
        "source_root": str(root),
    }
    (tables / "aggregation_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"tables": meta["tables"], "centered_max_abs_K": max_centered, "n_level_rows": meta["n_level_rows"]}, indent=2))
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    aggregate(a.root, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
