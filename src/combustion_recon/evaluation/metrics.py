"""Evaluation metrics for temperature reconstruction (proposal §5).

Always report **per operating condition**; never only a pooled average.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_T_HIGH = 2100.0
DEFAULT_T_PEAK = 2200.0
DEFAULT_T_JET = 1800.0


def _as_flat(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1)


def field_errors(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    train_sigma_T: float,
) -> dict[str, float]:
    """Family 1 — full-field reconstruction error."""
    p = _as_flat(pred)
    t = _as_flat(true)
    err = p - t
    sse = float(np.dot(err, err))
    sst = float(np.dot(t - t.mean(), t - t.mean()))
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    rel_l2 = float(np.sqrt(sse) / (np.linalg.norm(t) + 1e-12))
    nrmse = rmse / (float(train_sigma_T) + 1e-12)
    r2 = float(1.0 - sse / (sst + 1e-12))
    return {
        "rmse_K": rmse,
        "mae_K": mae,
        "rel_l2": rel_l2,
        "nrmse_train_sigma": nrmse,
        "r2": r2,
    }


def spatial_bias(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    """Family 2 — spatial / signed bias."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    bias_map = p - t
    # If time dimension present (T,H,W), average over time
    if bias_map.ndim == 3:
        bias_map_2d = bias_map.mean(axis=0)
        mean_signed = float(bias_map.mean())
    elif bias_map.ndim == 2:
        bias_map_2d = bias_map
        mean_signed = float(bias_map.mean())
    else:
        bias_map_2d = bias_map.reshape(-1)
        mean_signed = float(bias_map.mean())
    return {
        "mean_signed_error_K": mean_signed,
        "spatial_bias_map": bias_map_2d,
    }


def region_mask(true_T: np.ndarray, threshold: float, mode: str = "ge") -> np.ndarray:
    t = np.asarray(true_T)
    if mode == "ge":
        return t >= threshold
    if mode == "lt":
        return t < threshold
    raise ValueError(mode)


def _conditional_rmse_mae(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    p = _as_flat(pred)
    t = _as_flat(true)
    m = _as_flat(mask).astype(bool)
    n = int(m.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    err = p[m] - t[m]
    return float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err))), n


def high_temperature_region_errors(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    t_high: float = DEFAULT_T_HIGH,
    t_peak: float = DEFAULT_T_PEAK,
    t_jet: float = DEFAULT_T_JET,
) -> dict[str, float]:
    """Family 3 — errors inside Ω_high / Ω_peak / Ω_jet plus mask overlap."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    omega_high = region_mask(t, t_high, "ge")
    omega_hat = region_mask(p, t_high, "ge")
    omega_peak = region_mask(t, t_peak, "ge")
    omega_jet = region_mask(t, t_jet, "lt")

    rmse_h, mae_h, n_h = _conditional_rmse_mae(p, t, omega_high)
    rmse_peak, mae_peak, n_peak = _conditional_rmse_mae(p, t, omega_peak)
    rmse_jet, mae_jet, n_jet = _conditional_rmse_mae(p, t, omega_jet)

    inter = np.logical_and(omega_high, omega_hat)
    union = np.logical_or(omega_high, omega_hat)
    inter_n = float(inter.sum())
    union_n = float(union.sum())
    pred_n = float(omega_hat.sum())
    true_n = float(omega_high.sum())
    iou = inter_n / (union_n + 1e-12)
    dice = 2.0 * inter_n / (pred_n + true_n + 1e-12)
    area_true = true_n / (omega_high.size + 1e-12)
    area_pred = pred_n / (omega_hat.size + 1e-12)
    rel_area_err = abs(area_pred - area_true) / (area_true + 1e-12)

    return {
        "high_rmse_K": rmse_h,
        "high_mae_K": mae_h,
        "high_n_voxels": float(n_h),
        "high_iou": float(iou),
        "high_dice": float(dice),
        "high_rel_area_error": float(rel_area_err),
        "peak_rmse_K": rmse_peak,
        "peak_mae_K": mae_peak,
        "jet_rmse_K": rmse_jet,
        "jet_mae_K": mae_jet,
        "frac_omega_high_true": float(area_true),
        "frac_omega_high_pred": float(area_pred),
    }


def temperature_level_errors(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Family 4 — temperature level / peak errors."""
    p = _as_flat(pred)
    t = _as_flat(true)
    return {
        "mean_level_abs_error_K": float(abs(p.mean() - t.mean())),
        "p95_abs_error_K": float(abs(np.percentile(p, 95) - np.percentile(t, 95))),
        "peak_abs_error_K": float(abs(p.max() - t.max())),
        "true_mean_K": float(t.mean()),
        "pred_mean_K": float(p.mean()),
        "true_p95_K": float(np.percentile(t, 95)),
        "pred_p95_K": float(np.percentile(p, 95)),
        "true_peak_K": float(t.max()),
        "pred_peak_K": float(p.max()),
    }


def evaluate_sample(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    train_sigma_T: float,
    sim_id: str,
    split: str,
    t_high: float = DEFAULT_T_HIGH,
    t_peak: float = DEFAULT_T_PEAK,
    t_jet: float = DEFAULT_T_JET,
) -> dict[str, Any]:
    """Compute all four metric families for one sample / condition snapshot."""
    out: dict[str, Any] = {"sim_id": sim_id, "split": split}
    out.update(field_errors(pred, true, train_sigma_T=train_sigma_T))
    bias = spatial_bias(pred, true)
    out["mean_signed_error_K"] = bias["mean_signed_error_K"]
    # keep map out of flat tables by default
    out["_spatial_bias_map"] = bias["spatial_bias_map"]
    out.update(
        high_temperature_region_errors(
            pred, true, t_high=t_high, t_peak=t_peak, t_jet=t_jet
        )
    )
    out.update(temperature_level_errors(pred, true))
    return out


def evaluate_per_condition(
    results: Sequence[Mapping[str, Any]],
    *,
    drop_maps: bool = True,
) -> pd.DataFrame:
    """Aggregate a list of ``evaluate_sample`` dicts into a per-condition table.

    If multiple rows share the same ``(split, sim_id)``, numeric metrics are averaged.
    """
    rows = []
    for r in results:
        d = dict(r)
        if drop_maps:
            d.pop("_spatial_bias_map", None)
        rows.append(d)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    group_cols = ["split", "sim_id"]
    num_cols = [c for c in df.columns if c not in group_cols and np.issubdtype(df[c].dtype, np.number)]
    return df.groupby(group_cols, as_index=False)[num_cols].mean()


def summarize_by_split(per_condition_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± across conditions within each split (secondary summary only)."""
    if per_condition_df.empty:
        return per_condition_df
    num_cols = [
        c
        for c in per_condition_df.columns
        if c not in ("split", "sim_id") and np.issubdtype(per_condition_df[c].dtype, np.number)
    ]
    return per_condition_df.groupby("split", as_index=False)[num_cols].mean()
