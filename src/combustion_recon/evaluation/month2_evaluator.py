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

# Keys that identify an experimental configuration. When present on rows, mixed
# values are refused unless the caller aggregates one config at a time.
HOMOGENEITY_KEYS: tuple[str, ...] = (
    "role",
    "method",
    "p_conditioning",
    "K",
    "site_fraction",
    "model_seed",
    "mask_replica_id",
    "layout_id",
    "task_mode",
)


class MixedConfigurationError(ValueError):
    """Raised when aggregation would average across distinct experimental configs."""


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
    **config_tags: Any,
) -> dict[str, Any]:
    """Evaluate one frame in Kelvin.

    Assisted setups should pass ``mask_temperature`` so full-field, off-anchor,
    and at-anchor metrics are reported separately.

    Optional ``config_tags`` (method, p_conditioning, K, site_fraction,
    model_seed, mask_replica_id, ...) are stored on the row and used by
    :func:`aggregate_frame_then_condition_then_split` to refuse mixed configs.
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
    for k, v in config_tags.items():
        out[k] = v

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

    _ = temperature_level_errors
    return out


def _assert_homogeneous_config(
    frame_rows: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str] = HOMOGENEITY_KEYS,
) -> dict[str, Any]:
    """Refuse aggregation that would mix roles/methods/hyperparams.

    Only keys that appear on at least one row are checked. Missing keys on some
    rows while present on others also count as a mix.
    """
    if not frame_rows:
        return {}
    present = [k for k in keys if any(k in r for r in frame_rows)]
    fingerprint: dict[str, Any] = {}
    for k in present:
        values = []
        for r in frame_rows:
            if k not in r:
                raise MixedConfigurationError(
                    f"Aggregation refused: key {k!r} missing on some rows while present on others"
                )
            values.append(r[k])
        uniq = {json_safe(v) for v in values}
        if len(uniq) > 1:
            raise MixedConfigurationError(
                f"Aggregation refused: mixed values for {k!r}: {sorted(uniq)[:8]}"
            )
        fingerprint[k] = values[0]
    return fingerprint


def json_safe(v: Any) -> Any:
    if isinstance(v, (np.generic,)):
        return v.item()
    return v


def aggregate_frame_then_condition_then_split(
    frame_rows: Sequence[Mapping[str, Any]],
    *,
    require_homogeneous: bool = True,
    homogeneity_keys: Sequence[str] = HOMOGENEITY_KEYS,
) -> dict[str, pd.DataFrame]:
    """Aggregation order: frame → condition → split, for **one** configuration.

    Mean order
    ----------
    1. Per-frame metrics (input rows)
    2. Mean over frames within each ``(split, sim_id)`` → per_condition
       (delegates to Month-1 ``evaluate_per_condition``, unchanged)
    3. Mean over conditions within each ``split`` → per_split
    4. Selection helper: mean over VAL conditions of full-field RMSE only

    Guarantees
    ----------
    - ``train_fit`` and ``seen_train_eval`` are **not** fused (both have
      ``split=train`` but different ``role``); mixed roles raise
      :class:`MixedConfigurationError` when ``require_homogeneous`` is True.
    - Distinct method / p / K / coverage / seeds / mask replicas also refuse.
    - Call once per experimental configuration. Month-1 ``evaluate_per_condition``
      behavior is unchanged for callers that use it directly.
    """
    fingerprint = {}
    if require_homogeneous:
        fingerprint = _assert_homogeneous_config(frame_rows, keys=homogeneity_keys)

    per_condition = evaluate_per_condition(frame_rows, drop_maps=True)
    # Attach homogeneous config tags to per_condition for traceability
    for k, v in fingerprint.items():
        if k not in per_condition.columns:
            per_condition[k] = v

    per_split = summarize_by_split(per_condition)

    # Selection metric: VAL only (never train_fit/seen fused into selection)
    selection = pd.DataFrame()
    if not per_condition.empty and PRIMARY_SELECTION_METRIC in per_condition.columns:
        if "role" in fingerprint and fingerprint["role"] != "val_eval":
            selection = pd.DataFrame(
                {
                    "split": [],
                    "mean_condition_rmse_K": [],
                    "note": [],
                }
            )
            # Still compute a diagnostic table for non-VAL, clearly labeled
            diag = (
                per_condition.groupby("split", as_index=False)[PRIMARY_SELECTION_METRIC]
                .mean()
                .rename(columns={PRIMARY_SELECTION_METRIC: "mean_condition_rmse_K"})
            )
            diag["selection_eligible"] = False
            diag["note"] = (
                f"role={fingerprint['role']!r} is not val_eval; "
                "not used as Month-2 selection metric"
            )
            selection = diag
        else:
            # If role not tagged, fall back to split==val rows only when available
            pc = per_condition
            if "role" in pc.columns:
                pc = pc[pc["role"] == "val_eval"]
            elif "split" in pc.columns:
                pc = pc[pc["split"] == "val"]
            selection = (
                pc.groupby("split", as_index=False)[PRIMARY_SELECTION_METRIC].mean()
                if not pc.empty
                else pd.DataFrame()
            )
            if not selection.empty:
                selection = selection.rename(
                    columns={PRIMARY_SELECTION_METRIC: "mean_condition_rmse_K"}
                )
                selection["selection_eligible"] = True

    return {
        "per_frame": pd.DataFrame(
            [{k: v for k, v in r.items() if not str(k).startswith("_")} for r in frame_rows]
        ),
        "per_condition": per_condition,
        "per_split": per_split,
        "selection_mean_condition_rmse": selection,
        "config_fingerprint": fingerprint,
        "aggregation_order": [
            "frame",
            "condition(split,sim_id)",
            "split",
            "selection=mean_condition_rmse on val_eval only",
        ],
    }
