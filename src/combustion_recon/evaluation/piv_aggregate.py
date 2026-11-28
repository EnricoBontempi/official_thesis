"""Mission 02 (Virtual PIV) multi-level aggregation, confirmatory contrasts & diagnostics.

Implements the aggregation hierarchy fixed in Mission 00 and ``roadmap_tecnica_piv.md`` (§7.3):

    per_frame  -->  per_condition (mean over frames, NaN-aware)
               -->  macro_by_replica (uniform mean over conditions)
               -->  per_run_scenario (mean across noise replicas; NOT independent trainings)
               -->  cell_seed_summary (mean / std / min / max across independent training seeds)

Explicitly separates:
* **Confirmatory outputs** planned by the roadmap (main matrix T1–T7, paired contrasts
  against control seed-std, R4 decision-support indicators D-REJ / D-mix / D-kern-train,
  transfer matrices, sensitivity sweeps, B positions & distance bins, SENS-REJ node
  breakdown, T7 GLU batching bridge, descriptive hierarchical block bootstrap).
* **Exploratory diagnostics** (frame error distributions & quantiles, paired frame-level
  deltas, per-condition performance vs ``v_ref`` / ``sigma_u`` / NH3% / phi, training &
  validation curves parsed from ``curves.csv.gz`` and ``eval_history.json``).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from combustion_recon.evaluation.piv_manifest import (
    DEFAULT_REPRESENTATIVE_FRAMES,
    EVAL_STATUS_COMPLETE,
    EVAL_STATUS_SAME_AS_BEST,
    select_best_worst_frames,
)
from combustion_recon.evaluation.piv_metrics import DISTANCE_BINS

FRAME_KEY_COLS: list[str] = [
    "run_id",
    "checkpoint_kind",
    "eval_config",
    "role",
    "replica",
    "condition",
    "frame_index",
]

CORE_METRICS: tuple[str, ...] = (
    "rmse_K",
    "mae_K",
    "bias_K",
    "abs_bias_K",
    "centered_rmse_K",
    "rel_l2",
    "nrmse_train_sigma",
    "r2",
    "high_rmse_K",
    "high_mae_K",
    "high_iou",
    "high_dice",
    "jet_rmse_K",
    "jet_mae_K",
    "peak_rmse_K",
    "peak_abs_error_K",
    "p95_abs_error_K",
    "mean_level_abs_error_K",
    "iso1800_sym_dist_mm",
    "iso1800_p95_dist_mm",
    "iso2100_sym_dist_mm",
    "iso2100_p95_dist_mm",
    "in_field_rmse_K",
    "in_field_mae_K",
    "in_field_bias_K",
    "out_field_rmse_K",
    "out_field_mae_K",
    "out_field_bias_K",
    "obs_nodes_rmse_K",
    "unobs_nodes_rmse_K",
    "rej_nodes_rmse_K",
    "rej_nodes_bias_K",
    "nonrej_valid_rmse_K",
    "sub_nodes_rmse_K",
    "sub_nodes_bias_K",
    "coverage_observed_frac",
    "coverage_valid_frac",
    "coverage_field_frac",
    "dist_mean_mm",
    "dist_max_mm",
    "vel_hg_rmse_K",
    "vel_jet_rmse_K",
    "vel_rest_rmse_K",
)

DISTANCE_METRIC_COLS: tuple[str, ...] = tuple(
    f"dist_{b[0]}_{m}" for b in DISTANCE_BINS for m in ("rmse_K", "mae_K", "bias_K", "n")
)


class PivAggregateError(RuntimeError):
    """Raised on data integrity violations (duplicates, NaNs, hash mismatches)."""


def _numeric_metric_cols(df: pd.DataFrame) -> list[str]:
    wanted = list(CORE_METRICS) + list(DISTANCE_METRIC_COLS)
    return [c for c in dict.fromkeys(wanted) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def load_audited_per_frame_tables(
    audit_table: pd.DataFrame,
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Read per-frame CSVs for all valid audited (run, checkpoint) rows.

    Verifies uniqueness of frame keys, finiteness of primary global metrics,
    and identical observation hashes (``obs_hash``) across runs evaluated on the
    same ``(eval_config, role, replica, condition, frame_index)``.
    """
    parts: list[pd.DataFrame] = []
    warnings: list[str] = []

    for _, row in audit_table.iterrows():
        if not (bool(row["training_valid"]) and bool(row["eval_valid"])):
            continue
        src = Path(str(row["eval_source_dir"]))
        pf_dir = src / "per_frame"
        csvs = sorted(pf_dir.rglob("rep*.csv"))
        if not csvs:
            msg = f"{row['run_id']}/{row['checkpoint_kind']}: no per_frame CSVs under {pf_dir}"
            if strict:
                raise PivAggregateError(msg)
            warnings.append(msg)
            continue

        df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
        # Attach authoritative manifest/audit metadata in one shot (avoids fragmentation)
        meta_assign: dict[str, Any] = {
            "run_id": str(row["run_id"]),
            "group": str(row["group"]),
            "family": str(row["family"]),
            "variant": str(row["variant"]),
            "train_cell": str(row["train_cell"]),
            "train_config_id": str(row["train_config_id"]),
            "b_position": str(row["b_position"]),
            "model_seed": int(row["model_seed"]),
            "conditional": bool(row["conditional"]),
            "checkpoint_kind": str(row["checkpoint_kind"]),
            "training_status": str(row["training_status"]),
            "training_outcome": str(row["training_outcome"]),
            "budget_exhausted": bool(row["budget_exhausted"]),
        }
        if "condition" not in df.columns and "sim_id" in df.columns:
            meta_assign["condition"] = df["sim_id"].astype(str)
        if "bias_K" not in df.columns and "mean_signed_error_K" in df.columns:
            meta_assign["bias_K"] = df["mean_signed_error_K"]
        df = df.assign(**meta_assign)
        parts.append(df)

    if not parts:
        return pd.DataFrame(), warnings

    allf = pd.concat(parts, ignore_index=True)

    # 1. Duplicate key check
    if allf.duplicated(FRAME_KEY_COLS).any():
        n_dup = int(allf.duplicated(FRAME_KEY_COLS).sum())
        raise PivAggregateError(f"found {n_dup} duplicate per-frame keys across loaded tables")

    # 2. Primary global metrics must be finite
    for col in ("rmse_K", "mae_K", "bias_K"):
        if col in allf.columns and not np.isfinite(allf[col].to_numpy(dtype=float)).all():
            raise PivAggregateError(f"non-finite values detected in primary metric {col!r}")

    # 3. Cross-model observation comparability: identical obs_hash for same (eval_config, role, replica, condition, frame_index)
    if "obs_hash" in allf.columns and allf["obs_hash"].notna().any():
        non_empty = allf[allf["obs_hash"].notna()]
        nunq = non_empty.groupby(["eval_config", "role", "replica", "condition", "frame_index"])["obs_hash"].nunique()
        if int(nunq.max()) > 1:
            n_bad = int((nunq > 1).sum())
            msg = f"observation hash mismatch across runs on {n_bad} (eval_config, role, replica, condition, frame) keys"
            if strict:
                raise PivAggregateError(msg)
            warnings.append(msg)

    return allf, warnings


def aggregate_hierarchy(per_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Perform exact hierarchy: frame -> condition -> macro(replica) -> per_run -> cell_seed_summary."""
    if per_frame.empty:
        empty = pd.DataFrame()
        return {
            "per_condition": empty,
            "macro_by_replica": empty,
            "per_run_scenario": empty,
            "cell_seed_summary": empty,
        }

    mcols = _numeric_metric_cols(per_frame)
    meta_keys = [
        "run_id",
        "group",
        "family",
        "variant",
        "train_cell",
        "train_config_id",
        "b_position",
        "model_seed",
        "conditional",
        "checkpoint_kind",
        "training_status",
        "budget_exhausted",
        "eval_config",
        "role",
        "replica",
    ]

    # Step 1: frame -> condition (mean over frames, NaN-aware, with explicit valid counts)
    g_cond = per_frame.groupby(meta_keys + ["condition"], dropna=False)
    pc = g_cond[mcols].mean().reset_index()
    pc["n_frames"] = g_cond.size().to_numpy()
    for flag_col, out_col in (
        ("iso1800_valid", "n_iso1800_valid"),
        ("iso1800_true_empty", "n_iso1800_true_empty"),
        ("iso1800_pred_empty", "n_iso1800_pred_empty"),
        ("iso2100_valid", "n_iso2100_valid"),
        ("iso2100_true_empty", "n_iso2100_true_empty"),
        ("iso2100_pred_empty", "n_iso2100_pred_empty"),
        ("omega_high_empty", "n_omega_high_empty"),
    ):
        if flag_col in per_frame.columns:
            pc[out_col] = g_cond[flag_col].sum().astype(int).to_numpy()

    # Step 2: condition -> macro per replica (uniform mean across conditions)
    count_cols = [c for c in pc.columns if c.startswith("n_") and c != "n_frames"]
    g_mac = pc.groupby(meta_keys, dropna=False)
    macro = g_mac[mcols].mean().reset_index()
    macro["n_conditions"] = g_mac["condition"].nunique().to_numpy()
    macro["n_frames_total"] = g_mac["n_frames"].sum().to_numpy()
    for cc in count_cols:
        macro[cc] = g_mac[cc].sum().astype(int).to_numpy()

    # Step 3: mean across noise replicas per run (descriptive; replicas are NOT independent trainings)
    run_keys = [k for k in meta_keys if k != "replica"]
    g_run = macro.groupby(run_keys, dropna=False)
    per_run = g_run[mcols].mean().reset_index()
    per_run["n_replicas"] = g_run["replica"].nunique().to_numpy()
    per_run["n_conditions"] = g_run["n_conditions"].first().to_numpy()
    per_run["rmse_K_replica_std"] = g_run["rmse_K"].std(ddof=1).to_numpy()
    for cc in count_cols:
        per_run[cc] = g_run[cc].mean().to_numpy()

    # Step 4: summary across independent training seeds within each cell
    cell_keys = ["group", "family", "variant", "train_cell", "train_config_id", "b_position", "conditional", "checkpoint_kind", "eval_config", "role"]
    summary_metrics = [
        c
        for c in (
            "rmse_K",
            "mae_K",
            "bias_K",
            "abs_bias_K",
            "centered_rmse_K",
            "high_rmse_K",
            "high_iou",
            "jet_rmse_K",
            "peak_rmse_K",
            "peak_abs_error_K",
            "p95_abs_error_K",
            "iso1800_sym_dist_mm",
            "iso2100_sym_dist_mm",
            "in_field_rmse_K",
            "out_field_rmse_K",
            "rej_nodes_rmse_K",
            "sub_nodes_rmse_K",
        )
        if c in per_run.columns
    ]
    g_cell = per_run.groupby(cell_keys, dropna=False)
    parts = []
    for stat_name, fn in (("mean", "mean"), ("std", "std"), ("min", "min"), ("max", "max")):
        t = getattr(g_cell[summary_metrics], fn)()
        t.columns = [f"{c}__{stat_name}" for c in t.columns]
        parts.append(t)
    cell_summ = pd.concat(parts, axis=1).reset_index()
    cell_summ["n_seeds"] = g_cell["model_seed"].nunique().to_numpy()
    cell_summ["n_budget_exhausted"] = g_cell["budget_exhausted"].sum().astype(int).to_numpy()
    cell_summ["seed_list"] = g_cell["model_seed"].apply(lambda s: ",".join(str(x) for x in sorted(s.unique()))).to_numpy()

    return {
        "per_condition": pc,
        "macro_by_replica": macro,
        "per_run_scenario": per_run,
        "cell_seed_summary": cell_summ,
    }


def compute_confirmatory_contrasts(
    per_run: pd.DataFrame,
    cell_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Confirmatory paired contrasts from ``roadmap_tecnica_piv.md`` §4.2 & §7.3.

    Each contrast compares two ``(train_cell, eval_config)`` entries matched by
    ``(family, model_seed, checkpoint_kind, role)`` and reports:
    * per-seed paired mean, std, min, max of ``delta_rmse_K = target - reference``;
    * ``control_seed_std_K``: inter-seed std of the reference control cell;
    * ``exceeds_control_seed_std``: whether ``|mean_delta| > control_seed_std_K``
      (per roadmap §7.3 interpretability rule — never a p-value).
    """
    if per_run.empty:
        return pd.DataFrame()

    # Defined confirmatory contrasts: (id, label, category, (cell_A, eval_A), (cell_B, eval_B))
    # Sign convention: delta = A - B (positive => A has higher RMSE than B)
    contrast_defs: list[tuple[str, str, str, tuple[str, str], tuple[str, str]]] = [
        (
            "C1_pivA_info_loss",
            "Perdita d'informazione PIV-A (T2 su PIV-A − T1 su IDEAL-A)",
            "confirmatory_main",
            ("PIV-A", "PIV-A"),
            ("IDEAL-A", "IDEAL-A"),
        ),
        (
            "C2_pivA_coherent_train_gain",
            "Effetto training coerente PIV-A (T2 su PIV-A − T1 su PIV-A; <0 = T2 migliore)",
            "confirmatory_main",
            ("PIV-A", "PIV-A"),
            ("IDEAL-A", "PIV-A"),
        ),
        (
            "C3_pivA_filter_only_on_idealA",
            "Solo filtro su modello IDEAL-A (T1 su FILT-A − T1 su IDEAL-A)",
            "confirmatory_decomposition",
            ("IDEAL-A", "FILT-A"),
            ("IDEAL-A", "IDEAL-A"),
        ),
        (
            "C4_pivA_noise_on_idealA",
            "Solo errore correlato su modello IDEAL-A (T1 su PIV-A − T1 su FILT-A)",
            "confirmatory_decomposition",
            ("IDEAL-A", "PIV-A"),
            ("IDEAL-A", "FILT-A"),
        ),
        (
            "C5_coverage_loss_idealB_vs_idealA",
            "Perdita da sola copertura ridotta (T3 su IDEAL-B-in − T1 su IDEAL-A)",
            "confirmatory_coverage",
            ("IDEAL-B", "IDEAL-B-in"),
            ("IDEAL-A", "IDEAL-A"),
        ),
        (
            "C6_pivB_info_loss",
            "Perdita d'informazione PIV-B (T4 su PIV-B-in − T3 su IDEAL-B-in)",
            "confirmatory_main",
            ("PIV-B", "PIV-B-in"),
            ("IDEAL-B", "IDEAL-B-in"),
        ),
        (
            "C7_pivB_coherent_train_gain",
            "Effetto training coerente PIV-B (T4 su PIV-B-in − T3 su PIV-B-in; <0 = T4 migliore)",
            "confirmatory_main",
            ("PIV-B", "PIV-B-in"),
            ("IDEAL-B", "PIV-B-in"),
        ),
        (
            "C8_pivB_transfer_down",
            "Trasferimento posizione B: B-down − B-in (modelli PIV-B)",
            "confirmatory_b_position",
            ("PIV-B", "PIV-B-down"),
            ("PIV-B", "PIV-B-in"),
        ),
        (
            "C9_pivB_transfer_off",
            "Trasferimento posizione B: B-off − B-in (modelli PIV-B)",
            "confirmatory_b_position",
            ("PIV-B", "PIV-B-off"),
            ("PIV-B", "PIV-B-in"),
        ),
        (
            "C10_transfer_A_to_B",
            "Trasferimento configurazione A→B (modelli PIV-A su PIV-B-in − modelli PIV-B su PIV-B-in)",
            "confirmatory_transfer",
            ("PIV-A", "PIV-B-in"),
            ("PIV-B", "PIV-B-in"),
        ),
        (
            "C11_transfer_B_to_A",
            "Trasferimento configurazione B→A (modelli PIV-B su PIV-A − modelli PIV-A su PIV-A)",
            "confirmatory_transfer",
            ("PIV-B", "PIV-A"),
            ("PIV-A", "PIV-A"),
        ),
        (
            "C12_edge_band_E3a_minus_E3",
            "Effetto fascia di bordo A (modelli IDEAL-A su IDEAL-A − IDEAL-full)",
            "confirmatory_layout",
            ("IDEAL-A", "IDEAL-A"),
            ("IDEAL-A", "IDEAL-full"),
        ),
    ]

    ctrl_std_lookup = {}
    if not cell_summary.empty and "rmse_K__std" in cell_summary.columns:
        for _, r in cell_summary.iterrows():
            ctrl_std_lookup[(r["family"], r["train_cell"], r["eval_config"], r["checkpoint_kind"], r["role"])] = float(
                r["rmse_K__std"]
            )

    rows: list[dict[str, Any]] = []
    for cid, label, cat, (ca, ea), (cb, eb) in contrast_defs:
        a = per_run[(per_run["train_cell"] == ca) & (per_run["eval_config"] == ea)]
        b = per_run[(per_run["train_cell"] == cb) & (per_run["eval_config"] == eb)]
        if a.empty or b.empty:
            continue
        keep_cols = ["family", "variant", "model_seed", "checkpoint_kind", "role", "rmse_K", "budget_exhausted"]
        m = a[keep_cols].merge(
            b[keep_cols],
            on=["family", "variant", "model_seed", "checkpoint_kind", "role"],
            suffixes=("_a", "_b"),
            how="inner",
        )
        if m.empty:
            continue
        delta = m["rmse_K_a"] - m["rmse_K_b"]
        m = m.assign(
            delta_rmse_K=delta,
            rel_delta_rmse=delta / m["rmse_K_b"].replace(0.0, np.nan),
        )
        for (fam, var, ck, role), g in m.groupby(["family", "variant", "checkpoint_kind", "role"], sort=True):
            mean_d = float(g["delta_rmse_K"].mean())
            std_d = float(g["delta_rmse_K"].std(ddof=1)) if len(g) >= 2 else float("nan")
            ctrl_std = ctrl_std_lookup.get((fam, cb, eb, ck, role), float("nan"))
            exceeds = bool(math.isfinite(ctrl_std) and abs(mean_d) > ctrl_std)
            rows.append(
                {
                    "contrast_id": cid,
                    "label": label,
                    "category": cat,
                    "family": fam,
                    "variant": var,
                    "checkpoint_kind": ck,
                    "role": role,
                    "cell_A": ca,
                    "eval_A": ea,
                    "cell_B": cb,
                    "eval_B": eb,
                    "n_paired_seeds": int(len(g)),
                    "rmse_A_mean_K": float(g["rmse_K_a"].mean()),
                    "rmse_B_mean_K": float(g["rmse_K_b"].mean()),
                    "delta_rmse_mean_K": mean_d,
                    "delta_rmse_std_K": std_d,
                    "delta_rmse_min_K": float(g["delta_rmse_K"].min()),
                    "delta_rmse_max_K": float(g["delta_rmse_K"].max()),
                    "rel_delta_rmse_mean": float(g["rel_delta_rmse"].mean()),
                    "control_B_seed_std_K": ctrl_std,
                    "exceeds_control_seed_std": exceeds,
                    "n_budget_exhausted_involved": int(
                        (g["budget_exhausted_a"] | g["budget_exhausted_b"]).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def compute_r4_decision_indicators(
    per_run: pd.DataFrame,
    cell_summary: pd.DataFrame,
    *,
    role: str = "val_eval",
    checkpoint_kind: str = "best",
) -> dict[str, Any]:
    """Compute roadmap §5 & decisione_R1 §5 R4 decision-support indicators (descriptive only).

    Evaluates the three conditional training gates on VAL123 ``best``:
    * ``D-REJ`` (T8): loss of U-Net T2 (PIV-A) on ``SENS-REJ-*`` vs T2 seed std on ``PIV-A``.
    * ``D-mix`` (T9): transfer cost A->B and B->A vs target cell seed std.
    * ``D-kern-train`` (T10): loss of U-Net T2 (PIV-A) on ``SENS-TH-A`` vs T2 seed std on ``PIV-A``.

    Never authorizes training automatically; outputs structured evidence for R4 review.
    """
    out: dict[str, Any] = {
        "role": role,
        "checkpoint_kind": checkpoint_kind,
        "note": "Quantitative indicators for R4 review (roadmap §5, decisione_R1 §5); decision belongs to Enrico.",
        "D_REJ": {"status": "not_available", "comparisons": []},
        "D_mix": {"status": "not_available", "comparisons": []},
        "D_kern_train_T10": {"status": "not_available", "comparisons": []},
    }
    if per_run.empty or cell_summary.empty:
        return out

    cs = cell_summary[(cell_summary["role"] == role) & (cell_summary["checkpoint_kind"] == checkpoint_kind)]
    pr = per_run[(per_run["role"] == role) & (per_run["checkpoint_kind"] == checkpoint_kind)]

    def _cell_stat(fam: str, tcell: str, ecfg: str) -> tuple[float | None, float | None, int]:
        hit = cs[(cs["family"] == fam) & (cs["train_cell"] == tcell) & (cs["eval_config"] == ecfg)]
        if hit.empty:
            return None, None, 0
        r = hit.iloc[0]
        m = float(r["rmse_K__mean"])
        s = float(r["rmse_K__std"]) if pd.notna(r["rmse_K__std"]) else None
        return m, s, int(r["n_seeds"])

    # D-REJ & D-kern-train use U-Net T2 (PIV-A) on PIV-A as reference dispersion
    t2_mean, t2_std, t2_n = _cell_stat("sparse_unet", "PIV-A", "PIV-A")
    if t2_mean is not None:
        # D-REJ
        rej_cfgs = sorted(c for c in pr["eval_config"].unique() if str(c).startswith("SENS-REJ-"))
        rej_rows = []
        for rc in rej_cfgs:
            rm, rs, rn = _cell_stat("sparse_unet", "PIV-A", rc)
            if rm is not None:
                loss = rm - t2_mean
                trig = bool(t2_std is not None and loss > t2_std)
                rej_rows.append(
                    {
                        "eval_config": rc,
                        "rmse_mean_K": rm,
                        "ref_pivA_mean_K": t2_mean,
                        "loss_K": loss,
                        "ref_pivA_seed_std_K": t2_std,
                        "exceeds_t2_seed_std": trig,
                    }
                )
        if rej_rows:
            out["D_REJ"] = {
                "status": "evaluated",
                "t2_n_seeds": t2_n,
                "t2_seed_std_K": t2_std,
                "any_exceeds_t2_seed_std": any(x["exceeds_t2_seed_std"] for x in rej_rows),
                "comparisons": rej_rows,
            }

        # D-kern-train (T10): T2 on SENS-TH-A vs T2 seed std
        th_mean, th_std, _ = _cell_stat("sparse_unet", "PIV-A", "SENS-TH-A")
        if th_mean is not None:
            loss_th = th_mean - t2_mean
            out["D_kern_train_T10"] = {
                "status": "evaluated",
                "t2_n_seeds": t2_n,
                "t2_seed_std_K": t2_std,
                "comparisons": [
                    {
                        "eval_config": "SENS-TH-A",
                        "rmse_mean_K": th_mean,
                        "ref_pivA_mean_K": t2_mean,
                        "loss_K": loss_th,
                        "ref_pivA_seed_std_K": t2_std,
                        "exceeds_t2_seed_std": bool(t2_std is not None and loss_th > t2_std),
                    }
                ],
            }

    # D-mix (T9): A->B and B->A transfer cost vs target seed std
    mix_rows = []
    for fam in ("sparse_unet", "adapted_glu"):
        for direction, src_cell, tgt_cell, tgt_eval in (
            ("A_to_B", "PIV-A", "PIV-B", "PIV-B-in"),
            ("B_to_A", "PIV-B", "PIV-A", "PIV-A"),
        ):
            src_m, _, _ = _cell_stat(fam, src_cell, tgt_eval)
            tgt_m, tgt_s, tgt_n = _cell_stat(fam, tgt_cell, tgt_eval)
            if src_m is not None and tgt_m is not None:
                cost = src_m - tgt_m
                mix_rows.append(
                    {
                        "family": fam,
                        "direction": direction,
                        "transfer_rmse_K": src_m,
                        "target_own_rmse_K": tgt_m,
                        "transfer_cost_K": cost,
                        "target_seed_std_K": tgt_s,
                        "target_n_seeds": tgt_n,
                        "exceeds_target_seed_std": bool(tgt_s is not None and cost > tgt_s),
                    }
                )
    if mix_rows:
        out["D_mix"] = {
            "status": "evaluated",
            "any_exceeds_seed_std": any(x["exceeds_target_seed_std"] for x in mix_rows),
            "comparisons": mix_rows,
        }
    return out


def compute_sensitivity_table(per_run: pd.DataFrame) -> pd.DataFrame:
    """Paired sensitivity delta (SENS-* minus matching main PIV-A or PIV-B-in) per run & cell."""
    if per_run.empty:
        return pd.DataFrame()
    sens = per_run[per_run["eval_config"].astype(str).str.startswith("SENS-")].copy()
    if sens.empty:
        return pd.DataFrame()
    sens["ref_main_config"] = np.where(
        sens["eval_config"].astype(str).str.endswith("-B-in"), "PIV-B-in", "PIV-A"
    )
    main = per_run[per_run["eval_config"].isin(["PIV-A", "PIV-B-in"])][
        ["run_id", "checkpoint_kind", "role", "eval_config", "rmse_K", "mae_K", "bias_K", "high_iou", "iso2100_sym_dist_mm"]
    ].rename(
        columns={
            "eval_config": "ref_main_config",
            "rmse_K": "ref_rmse_K",
            "mae_K": "ref_mae_K",
            "bias_K": "ref_bias_K",
            "high_iou": "ref_high_iou",
            "iso2100_sym_dist_mm": "ref_iso2100_sym_dist_mm",
        }
    )
    m = sens.merge(main, on=["run_id", "checkpoint_kind", "role", "ref_main_config"], how="left")
    m["delta_rmse_K"] = m["rmse_K"] - m["ref_rmse_K"]
    m["rel_delta_rmse"] = m["delta_rmse_K"] / m["ref_rmse_K"].replace(0.0, np.nan)
    m["delta_mae_K"] = m["mae_K"] - m["ref_mae_K"]
    m["delta_bias_K"] = m["bias_K"] - m["ref_bias_K"]
    if "iso2100_sym_dist_mm" in m.columns and "ref_iso2100_sym_dist_mm" in m.columns:
        m["delta_iso2100_mm"] = m["iso2100_sym_dist_mm"] - m["ref_iso2100_sym_dist_mm"]

    grp_cols = ["family", "train_cell", "checkpoint_kind", "role", "ref_main_config", "eval_config"]
    g = m.groupby(grp_cols, dropna=False)
    out = g.agg(
        n_seeds=("model_seed", "nunique"),
        rmse_mean_K=("rmse_K", "mean"),
        rmse_std_K=("rmse_K", "std"),
        ref_rmse_mean_K=("ref_rmse_K", "mean"),
        delta_rmse_mean_K=("delta_rmse_K", "mean"),
        delta_rmse_std_K=("delta_rmse_K", "std"),
        delta_rmse_min_K=("delta_rmse_K", "min"),
        delta_rmse_max_K=("delta_rmse_K", "max"),
        rel_delta_rmse_mean=("rel_delta_rmse", "mean"),
        n_budget_exhausted=("budget_exhausted", "sum"),
    ).reset_index()
    return out


def compute_b_positions_and_distance_table(per_run: pd.DataFrame) -> pd.DataFrame:
    """Summarize in-field / out-of-field and distance-binned RMSE across configs."""
    if per_run.empty:
        return pd.DataFrame()
    dist_cols = [f"dist_{b[0]}_rmse_K" for b in DISTANCE_BINS if f"dist_{b[0]}_rmse_K" in per_run.columns]
    extra_cols = [c for c in ("in_field_rmse_K", "out_field_rmse_K", "coverage_observed_frac", "dist_mean_mm") if c in per_run.columns]
    cols = ["rmse_K"] + extra_cols + dist_cols
    grp = ["family", "train_cell", "checkpoint_kind", "role", "eval_config"]
    g = per_run.groupby(grp, dropna=False)
    means = g[cols].mean().reset_index()
    means["n_seeds"] = g["model_seed"].nunique().to_numpy()
    return means


def compute_rej_node_table(per_run: pd.DataFrame) -> pd.DataFrame:
    """Summarize SENS-REJ errors on rejected/substituted vs non-rejected valid vectors."""
    if per_run.empty or "rej_nodes_rmse_K" not in per_run.columns:
        return pd.DataFrame()
    sub = per_run[per_run["eval_config"].astype(str).str.startswith("SENS-REJ-")]
    if sub.empty:
        return pd.DataFrame()
    cols = [c for c in ("rmse_K", "rej_nodes_rmse_K", "rej_nodes_bias_K", "sub_nodes_rmse_K", "sub_nodes_bias_K", "nonrej_valid_rmse_K") if c in sub.columns]
    grp = ["family", "train_cell", "checkpoint_kind", "role", "eval_config"]
    g = sub.groupby(grp, dropna=False)
    out = g[cols].mean().reset_index()
    out["n_seeds"] = g["model_seed"].nunique().to_numpy()
    return out


def compute_glu_bridge_table(
    per_run: pd.DataFrame,
    *,
    mission00_baseline_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Compare T7 (GLU microbatch 1 / accum 8 on 5% ideal) vs Mission 00 original GLU (batch 8)."""
    if per_run.empty:
        return pd.DataFrame()
    t7 = per_run[(per_run["group"] == "T7") | (per_run["train_cell"] == "BRIDGE-5pct-IDEAL")].copy()
    if t7.empty:
        return pd.DataFrame()
    t7_s = (
        t7.groupby(["family", "train_cell", "checkpoint_kind", "role", "eval_config"], dropna=False)
        .agg(
            n_seeds=("model_seed", "nunique"),
            t7_rmse_mean_K=("rmse_K", "mean"),
            t7_rmse_std_K=("rmse_K", "std"),
            t7_rmse_min_K=("rmse_K", "min"),
            t7_rmse_max_K=("rmse_K", "max"),
            n_budget_exhausted=("budget_exhausted", "sum"),
        )
        .reset_index()
    )
    if mission00_baseline_csv and Path(mission00_baseline_csv).is_file():
        b = pd.read_csv(mission00_baseline_csv)
        if {"family", "variant", "cell", "role", "rmse_K"}.issubset(b.columns):
            bg = b[(b["family"] == "adapted_glu") & (b["variant"] == "p_on") & (b["cell"] == "R0")]
            if not bg.empty:
                b_s = (
                    bg.groupby("role", dropna=False)["rmse_K"]
                    .agg(m00_glu_orig_rmse_mean_K="mean", m00_glu_orig_rmse_std_K="std")
                    .reset_index()
                )
                t7_s = t7_s.merge(b_s, on="role", how="left")
                t7_s["t7_minus_m00_orig_K"] = t7_s["t7_rmse_mean_K"] - t7_s["m00_glu_orig_rmse_mean_K"]
    return t7_s


def hierarchical_block_bootstrap(
    per_frame: pd.DataFrame,
    *,
    role: str = "val_eval",
    checkpoint_kind: str = "best",
    n_boot: int = 300,
    block_size: int = 5,
    seed: int = 20261220,
) -> pd.DataFrame:
    """Descriptive hierarchical bootstrap over seeds and temporal frame blocks (roadmap §7.3).

    * Frames within each condition are ordered by ``frame_index`` and resampled in
      contiguous blocks of length ``block_size`` to respect temporal autocorrelation.
    * Seeds within each cell are resampled with replacement when ``n_seeds >= 2``.
    * Output intervals [p2.5, p97.5] are explicitly **descriptive** (no hypothesis tests).
    """
    if per_frame.empty:
        return pd.DataFrame()
    sub = per_frame[(per_frame["role"] == role) & (per_frame["checkpoint_kind"] == checkpoint_kind)]
    if sub.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    # Average across noise replicas per (run, eval_config, condition, frame_index) first
    # because noise replicas are NOT independent samples (roadmap §7.3).
    fmean = (
        sub.groupby(["family", "train_cell", "eval_config", "model_seed", "condition", "frame_index"], as_index=False)[
            "rmse_K"
        ]
        .mean()
        .sort_values(["family", "train_cell", "eval_config", "model_seed", "condition", "frame_index"])
    )

    for (fam, tcell, ecfg), g in fmean.groupby(["family", "train_cell", "eval_config"], sort=True):
        seeds = sorted(g["model_seed"].unique())
        conds = sorted(g["condition"].unique())
        # Build array: shape (n_seeds, n_conds, n_frames_c) via per-condition matrices
        cond_mats: list[np.ndarray] = []
        for c in conds:
            gc = g[g["condition"] == c]
            piv = gc.pivot(index="model_seed", columns="frame_index", values="rmse_K").reindex(seeds)
            cond_mats.append(piv.to_numpy(dtype=np.float64))

        point_per_seed = np.mean([np.nanmean(M, axis=1) for M in cond_mats], axis=0)
        point_est = float(np.mean(point_per_seed))

        boots = np.empty(int(n_boot), dtype=np.float64)
        n_s = len(seeds)
        for b in range(int(n_boot)):
            s_idx = rng.integers(0, n_s, size=n_s) if n_s >= 2 else np.array([0])
            cond_means = []
            for M in cond_mats:
                nf = M.shape[1]
                blk = max(1, min(int(block_size), nf))
                n_blocks = int(math.ceil(nf / blk))
                starts = rng.integers(0, max(1, nf - blk + 1), size=n_blocks)
                f_idx = np.concatenate([np.arange(st, min(st + blk, nf)) for st in starts])[:nf]
                subM = M[np.ix_(s_idx, f_idx)]
                cond_means.append(float(np.nanmean(subM)))
            boots[b] = float(np.mean(cond_means))

        rows.append(
            {
                "family": str(fam),
                "train_cell": str(tcell),
                "eval_config": str(ecfg),
                "role": role,
                "checkpoint_kind": checkpoint_kind,
                "n_seeds": n_s,
                "n_conditions": len(conds),
                "n_boot": int(n_boot),
                "block_size": int(block_size),
                "point_mean_rmse_K": point_est,
                "boot_p025_K": float(np.percentile(boots, 2.5)),
                "boot_p500_K": float(np.percentile(boots, 50.0)),
                "boot_p975_K": float(np.percentile(boots, 97.5)),
                "boot_sd_K": float(np.std(boots, ddof=1)),
                "interpretation_note": "descriptive hierarchical block bootstrap (no significance test)",
            }
        )
    return pd.DataFrame(rows)


def build_transfer_matrix(
    cell_summary: pd.DataFrame,
    *,
    checkpoint_kind: str = "best",
    metric: str = "rmse_K",
) -> pd.DataFrame:
    """Tidy transfer matrix (family, role, train_cell x eval_config) for reporting."""
    if cell_summary.empty or f"{metric}__mean" not in cell_summary.columns:
        return pd.DataFrame()
    sub = cell_summary[cell_summary["checkpoint_kind"] == checkpoint_kind].copy()
    if sub.empty:
        return pd.DataFrame()
    piv = sub.pivot_table(
        index=["family", "role", "train_cell"],
        columns="eval_config",
        values=f"{metric}__mean",
        aggfunc="first",
    ).reset_index()
    piv.columns.name = None
    return piv


def compute_exploratory_tables(
    per_frame: pd.DataFrame,
    per_condition: pd.DataFrame,
    *,
    spec_vref: Mapping[str, float] | None = None,
    sigma_px_main: float = 0.10,
    d_max_px: float = 6.0,
) -> dict[str, pd.DataFrame]:
    """Exploratory diagnostics: frame quantiles, paired frame deltas, condition vs v_ref."""
    if per_frame.empty:
        return {
            "frame_error_quantiles": pd.DataFrame(),
            "paired_frame_deltas": pd.DataFrame(),
            "condition_vs_vref": pd.DataFrame(),
        }

    # 1. Frame-level error quantiles (averaged across replicas per frame first)
    fmean = per_frame.groupby(
        ["family", "train_cell", "checkpoint_kind", "eval_config", "role", "model_seed", "condition", "frame_index"],
        as_index=False,
    )[["rmse_K", "mae_K", "bias_K"]].mean()
    q_rows = []
    for (fam, tcell, ck, ecfg, role), g in fmean.groupby(
        ["family", "train_cell", "checkpoint_kind", "eval_config", "role"], sort=True
    ):
        vals = g["rmse_K"].to_numpy(dtype=float)
        q_rows.append(
            {
                "family": fam,
                "train_cell": tcell,
                "checkpoint_kind": ck,
                "eval_config": ecfg,
                "role": role,
                "n_seed_frames": int(len(vals)),
                "mean_K": float(np.mean(vals)),
                "std_K": float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan"),
                "p10_K": float(np.percentile(vals, 10)),
                "p25_K": float(np.percentile(vals, 25)),
                "p50_K": float(np.percentile(vals, 50)),
                "p75_K": float(np.percentile(vals, 75)),
                "p90_K": float(np.percentile(vals, 90)),
                "p95_K": float(np.percentile(vals, 95)),
                "p99_K": float(np.percentile(vals, 99)),
                "iqr_K": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            }
        )
    frame_quantiles = pd.DataFrame(q_rows)

    # 2. Paired frame-by-frame deltas on VAL best (cross-seed mean per frame):
    #    (a) PIV-A own vs IDEAL-A own, (b) U-Net vs GLU on own config
    f_cell = (
        fmean[fmean["checkpoint_kind"] == "best"]
        .groupby(["family", "train_cell", "eval_config", "role", "condition", "frame_index"], as_index=False)["rmse_K"]
        .mean()
    )
    pair_parts = []
    for fam in sorted(f_cell["family"].unique()):
        a = f_cell[(f_cell["family"] == fam) & (f_cell["train_cell"] == "PIV-A") & (f_cell["eval_config"] == "PIV-A")]
        b = f_cell[(f_cell["family"] == fam) & (f_cell["train_cell"] == "IDEAL-A") & (f_cell["eval_config"] == "IDEAL-A")]
        if not a.empty and not b.empty:
            m = a.merge(b, on=["role", "condition", "frame_index"], suffixes=("_piv", "_ideal"))
            m["pair_kind"] = f"{fam}:PIV-A_vs_IDEAL-A"
            m["rmse_left_K"] = m["rmse_K_piv"]
            m["rmse_right_K"] = m["rmse_K_ideal"]
            m["delta_K"] = m["rmse_K_piv"] - m["rmse_K_ideal"]
            pair_parts.append(m[["pair_kind", "role", "condition", "frame_index", "rmse_left_K", "rmse_right_K", "delta_K"]])

    for tcell, ecfg in (("IDEAL-A", "IDEAL-A"), ("PIV-A", "PIV-A"), ("IDEAL-B", "IDEAL-B-in"), ("PIV-B", "PIV-B-in")):
        u = f_cell[(f_cell["family"] == "sparse_unet") & (f_cell["train_cell"] == tcell) & (f_cell["eval_config"] == ecfg)]
        gl = f_cell[(f_cell["family"] == "adapted_glu") & (f_cell["train_cell"] == tcell) & (f_cell["eval_config"] == ecfg)]
        if not u.empty and not gl.empty:
            m = u.merge(gl, on=["role", "condition", "frame_index"], suffixes=("_unet", "_glu"))
            m["pair_kind"] = f"U-Net_vs_GLU:{tcell}"
            m["rmse_left_K"] = m["rmse_K_unet"]
            m["rmse_right_K"] = m["rmse_K_glu"]
            m["delta_K"] = m["rmse_K_unet"] - m["rmse_K_glu"]
            pair_parts.append(m[["pair_kind", "role", "condition", "frame_index", "rmse_left_K", "rmse_right_K", "delta_K"]])
    paired_deltas = pd.concat(pair_parts, ignore_index=True) if pair_parts else pd.DataFrame()

    # 3. Per-condition RMSE enriched with NH3%, phi, v_ref and sigma_u
    pc_cell = (
        per_condition.groupby(
            ["family", "train_cell", "checkpoint_kind", "eval_config", "role", "condition"], dropna=False
        )
        .agg(
            rmse_mean_K=("rmse_K", "mean"),
            rmse_std_K=("rmse_K", "std"),
            bias_mean_K=("bias_K", "mean"),
            n_seeds=("model_seed", "nunique"),
        )
        .reset_index()
    )
    nh3_list, phi_list, vref_list, sig_list = [], [], [], []
    for c in pc_cell["condition"].astype(str):
        mm = re.match(r"^(\d+)NH3_([0-9]+(?:\.[0-9]+)?)", c)
        nh3_list.append(int(mm.group(1)) if mm else np.nan)
        phi_list.append(float(mm.group(2)) if mm else np.nan)
        vr = float(spec_vref[c]) if (spec_vref and c in spec_vref) else float("nan")
        vref_list.append(vr)
        sig_list.append(sigma_px_main * vr / d_max_px if math.isfinite(vr) else float("nan"))
    pc_cell["nh3_pct"] = nh3_list
    pc_cell["phi"] = phi_list
    pc_cell["v_ref_m_s"] = vref_list
    pc_cell["sigma_u_main_m_s"] = sig_list

    return {
        "frame_error_quantiles": frame_quantiles,
        "paired_frame_deltas": paired_deltas,
        "condition_vs_vref": pc_cell,
    }


def load_training_curves(manifest: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Read ``curves.csv.gz`` and ``eval_history.json`` from every run's ``training_dir``."""
    train_parts: list[pd.DataFrame] = []
    eval_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for run in manifest["runs"]:
        tdir_raw = run.get("training_dir")
        if not tdir_raw or not Path(tdir_raw).is_dir():
            continue
        tdir = Path(tdir_raw)
        rid = str(run["run_id"])
        meta = {
            "run_id": rid,
            "group": str(run["group"]),
            "family": str(run["family"]),
            "train_cell": str(run["train_cell"]),
            "model_seed": int(run["model_seed"]),
        }
        # Epoch-aggregated training loss & LR from curves.csv.gz (or curves.csv)
        cpath = tdir / "curves.csv.gz"
        if not cpath.is_file():
            cpath = tdir / "curves.csv"
        if cpath.is_file():
            cdf = pd.read_csv(cpath)
            if not cdf.empty and "epoch" in cdf.columns:
                ep = (
                    cdf.groupby("epoch", as_index=False)
                    .agg(
                        train_loss_mean=("loss", "mean"),
                        train_loss_last=("loss", "last"),
                        lr_last=("lr", "last"),
                        grad_norm_mean=("grad_norm_pre_clip", "mean") if "grad_norm_pre_clip" in cdf.columns else ("loss", "count"),
                        n_updates=("update", "count") if "update" in cdf.columns else ("loss", "count"),
                    )
                )
                for k, v in meta.items():
                    ep[k] = v
                train_parts.append(ep)

        # Validation history from eval_history.json
        epath = tdir / "eval_history.json"
        if epath.is_file():
            try:
                eh = json.loads(epath.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                eh = []
            if isinstance(eh, list) and eh:
                erows = []
                for rec in eh:
                    erows.append(
                        {
                            **meta,
                            "epoch": int(rec.get("epoch", -1)),
                            "lr": rec.get("lr"),
                            "val_rmse_K": rec.get("val_mean_condition_rmse_K"),
                            "seen_rmse_K": rec.get("seen_mean_condition_rmse_K"),
                            "train_fit_rmse_K": rec.get("train_mean_condition_rmse_K"),
                        }
                    )
                edf = pd.DataFrame(erows)
                eval_parts.append(edf)
                val_s = pd.to_numeric(edf["val_rmse_K"], errors="coerce")
                best_idx = int(val_s.idxmin()) if val_s.notna().any() else None
                summary_rows.append(
                    {
                        **meta,
                        "n_eval_points": len(edf),
                        "last_eval_epoch": int(edf["epoch"].max()),
                        "best_val_epoch": int(edf.loc[best_idx, "epoch"]) if best_idx is not None else None,
                        "best_val_rmse_K": float(val_s.loc[best_idx]) if best_idx is not None else float("nan"),
                        "last_val_rmse_K": float(val_s.iloc[-1]) if pd.notna(val_s.iloc[-1]) else float("nan"),
                        "last_minus_best_val_K": (
                            float(val_s.iloc[-1] - val_s.loc[best_idx])
                            if (best_idx is not None and pd.notna(val_s.iloc[-1]))
                            else float("nan")
                        ),
                    }
                )

    return {
        "train_epoch_curves": pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(),
        "val_eval_curves": pd.concat(eval_parts, ignore_index=True) if eval_parts else pd.DataFrame(),
        "training_curves_summary": pd.DataFrame(summary_rows),
    }


def run_full_aggregation(
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    strict: bool = True,
    spec_vref: Mapping[str, float] | None = None,
    n_boot: int = 300,
) -> dict[str, Any]:
    """Execute complete confirmatory + exploratory aggregation from audited runs."""
    per_frame, warnings = load_audited_per_frame_tables(audit["table"], strict=strict)
    hier = aggregate_hierarchy(per_frame)
    contrasts = compute_confirmatory_contrasts(hier["per_run_scenario"], hier["cell_seed_summary"])
    r4_ind = compute_r4_decision_indicators(hier["per_run_scenario"], hier["cell_seed_summary"])
    sens = compute_sensitivity_table(hier["per_run_scenario"])
    b_dist = compute_b_positions_and_distance_table(hier["per_run_scenario"])
    rej_nodes = compute_rej_node_table(hier["per_run_scenario"])
    t7_bridge = compute_glu_bridge_table(
        hier["per_run_scenario"],
        mission00_baseline_csv=manifest.get("mission00_baseline_csv"),
    )
    boot_val = hierarchical_block_bootstrap(per_frame, role="val_eval", checkpoint_kind="best", n_boot=n_boot)
    tm_best = build_transfer_matrix(hier["cell_seed_summary"], checkpoint_kind="best", metric="rmse_K")
    tm_last = build_transfer_matrix(hier["cell_seed_summary"], checkpoint_kind="last", metric="rmse_K")
    expl = compute_exploratory_tables(per_frame, hier["per_condition"], spec_vref=spec_vref)
    curves = load_training_curves(manifest)
    selected_cases = {
        "a_priori_representative": manifest.get("representative_frames") or DEFAULT_REPRESENTATIVE_FRAMES,
        "best_worst_val": select_best_worst_frames(per_frame, role="val_eval", checkpoint_kind="best", replica=0, top_k=2),
    }
    return {
        "per_frame": per_frame,
        "warnings": warnings,
        "confirmatory": {
            "per_condition": hier["per_condition"],
            "macro_by_replica": hier["macro_by_replica"],
            "per_run_scenario": hier["per_run_scenario"],
            "cell_seed_summary": hier["cell_seed_summary"],
            "confirmatory_contrasts": contrasts,
            "r4_decision_indicators": r4_ind,
            "transfer_matrix_best": tm_best,
            "transfer_matrix_last": tm_last,
            "sensitivity_summary": sens,
            "b_positions_and_distance": b_dist,
            "rej_node_metrics": rej_nodes,
            "glu_bridge_t7": t7_bridge,
            "hierarchical_bootstrap_val": boot_val,
        },
        "exploratory": {
            **expl,
            **curves,
            "selected_cases": selected_cases,
        },
    }


__all__ = [
    "CORE_METRICS",
    "DISTANCE_METRIC_COLS",
    "FRAME_KEY_COLS",
    "PivAggregateError",
    "aggregate_hierarchy",
    "build_transfer_matrix",
    "compute_b_positions_and_distance_table",
    "compute_confirmatory_contrasts",
    "compute_exploratory_tables",
    "compute_glu_bridge_table",
    "compute_r4_decision_indicators",
    "compute_rej_node_table",
    "compute_sensitivity_table",
    "hierarchical_block_bootstrap",
    "load_audited_per_frame_tables",
    "load_training_curves",
    "run_full_aggregation",
]
