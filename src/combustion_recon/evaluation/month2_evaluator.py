"""Shared Month-2 evaluation interface (Kelvin outputs; frame→condition→split)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.evaluation.metrics import (
    DEFAULT_T_HIGH,
    DEFAULT_T_JET,
    DEFAULT_T_PEAK,
    evaluate_per_condition,
    evaluate_sample,
    field_errors,
    spatial_bias,
    summarize_by_split,
    temperature_level_errors,
)


PRIMARY_SELECTION_METRIC = "rmse_K"
# Explicitly excluded from offset-invariant shape claims / selection:
EXCLUDED_SHAPE_METRICS = ("top25pct_iou", "top25pct_dice", "top25pct_threshold_K")


def denormalize_temperature(
    pred_norm: np.ndarray,
    normalizer: Month2FitOnlyNormalizer,
) -> np.ndarray:
    return normalizer.denormalize_field("Temperature", pred_norm)


def evaluate_frame_prediction(
    pred_T_K: np.ndarray,
    true_T_K: np.ndarray,
    *,
    train_sigma_T: float,
    sim_id: str,
    split: str,
    role: str,
    frame_index: int,
    mask_temperature: np.ndarray | None = None,
    t_high: float = DEFAULT_T_HIGH,
    t_peak: float = DEFAULT_T_PEAK,
    t_jet: float = DEFAULT_T_JET,
    include_top25pct_diagnostic: bool = False,
) -> dict[str, Any]:
    """Evaluate one frame in Kelvin.

    Assisted setups should pass ``mask_temperature`` so full-field, off-anchor,
    and at-anchor metrics are reported separately.
    """
    out = evaluate_sample(
        pred_T_K,
        true_T_K,
        train_sigma_T=train_sigma_T,
        sim_id=sim_id,
        split=split,
        t_high=t_high,
        t_peak=t_peak,
        t_jet=t_jet,
    )
    out["role"] = role
    out["frame_index"] = int(frame_index)

    # Empty-region bookkeeping already yields NaN RMSE with n=0 in metrics.py.
    out["omega_high_empty"] = bool(out.get("high_n_voxels", 0) == 0)

    if not include_top25pct_diagnostic:
        for k in EXCLUDED_SHAPE_METRICS:
            out.pop(k, None)
    else:
        out["top25pct_note"] = (
            "Diagnostic only; thresholds prediction with P75(true T); "
            "not offset-invariant; not used for selection."
        )

    if mask_temperature is not None:
        m = np.asarray(mask_temperature, dtype=bool)
        off = ~m
        # Full-field already in `out`.
        if m.any():
            fe_at = field_errors(pred_T_K[m], true_T_K[m], train_sigma_T=train_sigma_T)
            bias_at = spatial_bias(pred_T_K[m], true_T_K[m])
            out["anchor_rmse_K"] = fe_at["rmse_K"]
            out["anchor_mae_K"] = fe_at["mae_K"]
            out["anchor_bias_K"] = bias_at["mean_signed_error_K"]
            out["anchor_abs_bias_K"] = bias_at["abs_bias_K"]
            out["anchor_centered_rmse_K"] = bias_at["centered_rmse_K"]
            out["anchor_n"] = int(m.sum())
        else:
            out["anchor_rmse_K"] = float("nan")
            out["anchor_mae_K"] = float("nan")
            out["anchor_bias_K"] = float("nan")
            out["anchor_abs_bias_K"] = float("nan")
            out["anchor_centered_rmse_K"] = float("nan")
            out["anchor_n"] = 0

        if off.any():
            fe_off = field_errors(pred_T_K[off], true_T_K[off], train_sigma_T=train_sigma_T)
            bias_off = spatial_bias(pred_T_K[off], true_T_K[off])
            out["off_anchor_rmse_K"] = fe_off["rmse_K"]
            out["off_anchor_mae_K"] = fe_off["mae_K"]
            out["off_anchor_bias_K"] = bias_off["mean_signed_error_K"]
            out["off_anchor_abs_bias_K"] = bias_off["abs_bias_K"]
            out["off_anchor_centered_rmse_K"] = bias_off["centered_rmse_K"]
            out["off_anchor_n"] = int(off.sum())
        else:
            out["off_anchor_rmse_K"] = float("nan")
            out["off_anchor_n"] = 0

    # Level metrics already included via evaluate_sample → temperature_level_errors
    _ = temperature_level_errors  # imported for API clarity / reuse
    return out


def aggregate_frame_then_condition_then_split(
    frame_rows: Sequence[Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Aggregation order required by the Month-2 pilot: frame → condition → split."""
    per_condition = evaluate_per_condition(frame_rows, drop_maps=True)
    per_split = summarize_by_split(per_condition)
    # Selection helper: mean over conditions of full-field RMSE within a split
    selection = (
        per_condition.groupby("split", as_index=False)[PRIMARY_SELECTION_METRIC].mean()
        if not per_condition.empty and PRIMARY_SELECTION_METRIC in per_condition.columns
        else pd.DataFrame()
    )
    if not selection.empty:
        selection = selection.rename(columns={PRIMARY_SELECTION_METRIC: "mean_condition_rmse_K"})
    return {
        "per_frame": pd.DataFrame(
            [{k: v for k, v in r.items() if not str(k).startswith("_")} for r in frame_rows]
        ),
        "per_condition": per_condition,
        "per_split": per_split,
        "selection_mean_condition_rmse": selection,
    }
