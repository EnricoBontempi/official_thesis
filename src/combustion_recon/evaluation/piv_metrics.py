"""Mission 02 (Virtual PIV) frame-level evaluation metrics and spatial accumulators.

Reuses Mission 00 / Month 2 evaluation primitives (:func:`evaluate_frame_prediction`,
:func:`field_errors`, :func:`spatial_bias`) and adds the PIV-specific physical metrics
frozen in ``decisione_R1.md`` (§8) and ``roadmap_tecnica_piv.md`` (§7.3):

1. Symmetric mean boundary distance (mm) between true and predicted isotherms at
   1800 K and 2100 K, using Euclidean distance transforms with the R1 physical grid
   steps (rows <-> x: 0.581 mm, cols <-> y: 0.758 mm). Empty masks are explicitly
   flagged and counted separately rather than silently imputed.
2. Error decomposition inside vs outside the observed field (crucial for B-in/down/off)
   and binned by Euclidean distance (mm) from the nearest observed valid vector,
   with frozen R1 classes: ``0``, ``(0, 2]``, ``(2, 5]``, ``(5, 10]``, ``(10, 20]``,
   ``> 20`` mm.
3. Node-level error decomposition on SENS-REJ rejected / substituted vs surviving
   valid vectors.
4. Optional velocity-defined region errors (HG, JET, REST) when velocity-region masks
   are supplied (kept distinct from temperature-defined regions).
5. Online spatial error accumulators (sum and sum-of-squares of ``pred - true`` on the
   128x128 grid) so full-dataset spatial bias and spatial RMSE maps are produced
   without retaining full predicted fields on disk.

All full-field metrics are computed while ``pred_T_K`` and ``true_T_K`` are in memory,
prior to discarding transient field arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from combustion_recon.evaluation.metrics import (
    DEFAULT_T_HIGH,
    DEFAULT_T_JET,
    DEFAULT_T_PEAK,
    field_errors,
    spatial_bias,
)
from combustion_recon.evaluation.month2_evaluator import evaluate_frame_prediction

# Default R1 physical steps (mm); callers should pass steps from the frozen spec.
DEFAULT_STEPS_MM: tuple[float, float] = (0.5810574316105278, 0.7581226848528722)

# Frozen R1 §8 distance-to-nearest-valid-vector classes (mm)
DISTANCE_BINS: tuple[tuple[str, float, float, str], ...] = (
    ("d0", 0.0, 0.0, "0 mm"),
    ("d0_2", 0.0, 2.0, "(0, 2] mm"),
    ("d2_5", 2.0, 5.0, "(2, 5] mm"),
    ("d5_10", 5.0, 10.0, "(5, 10] mm"),
    ("d10_20", 10.0, 20.0, "(10, 20] mm"),
    ("dgt20", 20.0, math.inf, "> 20 mm"),
)

ISOTHERM_THRESHOLDS_K: tuple[float, ...] = (1800.0, 2100.0)


def _subset_metrics(
    pred_T_K: np.ndarray,
    true_T_K: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> dict[str, Any]:
    """Compute RMSE, MAE, bias, centered RMSE and node count on a boolean subset."""
    m = np.asarray(mask, dtype=bool)
    n = int(m.sum())
    if n == 0:
        return {
            f"{prefix}_rmse_K": float("nan"),
            f"{prefix}_mae_K": float("nan"),
            f"{prefix}_bias_K": float("nan"),
            f"{prefix}_abs_bias_K": float("nan"),
            f"{prefix}_centered_rmse_K": float("nan"),
            f"{prefix}_n": 0,
        }
    p = np.asarray(pred_T_K, dtype=np.float64)[m]
    t = np.asarray(true_T_K, dtype=np.float64)[m]
    err = p - t
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    centered = float(np.sqrt(max(rmse**2 - bias**2, 0.0)))
    return {
        f"{prefix}_rmse_K": rmse,
        f"{prefix}_mae_K": mae,
        f"{prefix}_bias_K": bias,
        f"{prefix}_abs_bias_K": abs(bias),
        f"{prefix}_centered_rmse_K": centered,
        f"{prefix}_n": n,
    }


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    """4-connected interior boundary of a boolean mask.

    A True pixel is on the boundary iff it has at least one 4-neighbour that is False
    (or lies on the domain edge when the entire domain is True).
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    # Use border_value=0 so pixels on the domain boundary are treated as boundary
    # only when the mask is completely full; otherwise interior erosion with border_value=1
    # finds true interface pixels against ~m.
    if m.all():
        # Entire grid >= threshold: no internal isotherm crossing exists.
        return np.zeros_like(m, dtype=bool)
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    eroded = binary_erosion(m, structure=struct, border_value=1)
    return m & ~eroded


def isotherm_boundary_distance_mm(
    pred_T_K: np.ndarray,
    true_T_K: np.ndarray,
    *,
    threshold_K: float,
    steps_mm: tuple[float, float] = DEFAULT_STEPS_MM,
) -> dict[str, Any]:
    """Symmetric mean boundary distance (mm) between {T_true >= thr} and {T_pred >= thr}.

    Follows ``decisione_R1.md`` §8:
    - Computes borders of ``{T >= threshold_K}`` for true and predicted fields.
    - Uses Euclidean distance transform with physical steps ``(dr_mm, dc_mm)``.
    - Returns NaN and explicit boolean flags when either mask is empty or has no
      internal boundary (e.g. entire frame above threshold), so such frames are
      counted separately during aggregation.
    """
    tag = f"iso{int(round(threshold_K))}"
    p = np.asarray(pred_T_K, dtype=np.float64)
    t = np.asarray(true_T_K, dtype=np.float64)
    true_mask = t >= float(threshold_K)
    pred_mask = p >= float(threshold_K)

    true_empty = bool(not true_mask.any())
    pred_empty = bool(not pred_mask.any())
    true_full = bool(true_mask.all())
    pred_full = bool(pred_mask.all())

    b_true = _mask_boundary(true_mask)
    b_pred = _mask_boundary(pred_mask)
    n_b_true = int(b_true.sum())
    n_b_pred = int(b_pred.sum())

    if n_b_true == 0 or n_b_pred == 0:
        return {
            f"{tag}_sym_dist_mm": float("nan"),
            f"{tag}_true_to_pred_mm": float("nan"),
            f"{tag}_pred_to_true_mm": float("nan"),
            f"{tag}_p95_dist_mm": float("nan"),
            f"{tag}_valid": False,
            f"{tag}_true_empty": true_empty,
            f"{tag}_pred_empty": pred_empty,
            f"{tag}_true_full": true_full,
            f"{tag}_pred_full": pred_full,
            f"{tag}_n_border_true": n_b_true,
            f"{tag}_n_border_pred": n_b_pred,
        }

    sampling = (float(steps_mm[0]), float(steps_mm[1]))
    dt_to_pred = distance_transform_edt(~b_pred, sampling=sampling)
    dt_to_true = distance_transform_edt(~b_true, sampling=sampling)

    d_t2p = dt_to_pred[b_true]
    d_p2t = dt_to_true[b_pred]
    mean_t2p = float(np.mean(d_t2p))
    mean_p2t = float(np.mean(d_p2t))
    sym_mean = 0.5 * (mean_t2p + mean_p2t)
    p95 = float(np.percentile(np.concatenate([d_t2p, d_p2t]), 95))

    return {
        f"{tag}_sym_dist_mm": sym_mean,
        f"{tag}_true_to_pred_mm": mean_t2p,
        f"{tag}_pred_to_true_mm": mean_p2t,
        f"{tag}_p95_dist_mm": p95,
        f"{tag}_valid": True,
        f"{tag}_true_empty": False,
        f"{tag}_pred_empty": False,
        f"{tag}_true_full": False,
        f"{tag}_pred_full": False,
        f"{tag}_n_border_true": n_b_true,
        f"{tag}_n_border_pred": n_b_pred,
    }


def distance_to_valid_vector_mm(
    observed_mask: np.ndarray,
    *,
    steps_mm: tuple[float, float] = DEFAULT_STEPS_MM,
) -> np.ndarray:
    """Euclidean distance (mm) from every grid node to the nearest observed vector."""
    m = np.asarray(observed_mask, dtype=bool)
    if not m.any():
        return np.full(m.shape, np.inf, dtype=np.float64)
    return distance_transform_edt(~m, sampling=(float(steps_mm[0]), float(steps_mm[1])))


def distance_binned_errors(
    pred_T_K: np.ndarray,
    true_T_K: np.ndarray,
    observed_mask: np.ndarray,
    *,
    steps_mm: tuple[float, float] = DEFAULT_STEPS_MM,
    dist_map_mm: np.ndarray | None = None,
) -> dict[str, Any]:
    """Error metrics binned by distance (mm) from nearest observed valid vector (R1 §8)."""
    dmap = (
        np.asarray(dist_map_mm, dtype=np.float64)
        if dist_map_mm is not None
        else distance_to_valid_vector_mm(observed_mask, steps_mm=steps_mm)
    )
    out: dict[str, Any] = {
        "dist_max_mm": float(np.max(dmap)) if np.isfinite(dmap).any() else float("nan"),
        "dist_mean_mm": float(np.mean(dmap)) if np.isfinite(dmap).all() else float("nan"),
    }
    for name, lo, hi, _label in DISTANCE_BINS:
        if lo == 0.0 and hi == 0.0:
            bmask = dmap == 0.0
        elif math.isinf(hi):
            bmask = dmap > lo
        else:
            bmask = (dmap > lo) & (dmap <= hi)
        out.update(_subset_metrics(pred_T_K, true_T_K, bmask, f"dist_{name}"))
    return out


def field_rect_mask(rect_rows_cols: Sequence[int], hw: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Boolean mask True inside [row0..row1, col0..col1] inclusive."""
    r0, r1, c0, c1 = [int(v) for v in rect_rows_cols]
    m = np.zeros(hw, dtype=bool)
    m[r0 : r1 + 1, c0 : c1 + 1] = True
    return m


def evaluate_piv_frame(
    pred_T_K: np.ndarray,
    true_T_K: np.ndarray,
    *,
    train_sigma_T: float,
    sim_id: str,
    split: str,
    role: str,
    frame_index: int,
    observed_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
    field_mask: np.ndarray | None = None,
    rejected_mask: np.ndarray | None = None,
    substituted_mask: np.ndarray | None = None,
    velocity_region_masks: Mapping[str, np.ndarray] | None = None,
    steps_mm: tuple[float, float] = DEFAULT_STEPS_MM,
    dist_map_mm: np.ndarray | None = None,
    t_high: float = DEFAULT_T_HIGH,
    t_peak: float = DEFAULT_T_PEAK,
    t_jet: float = DEFAULT_T_JET,
    **config_tags: Any,
) -> dict[str, Any]:
    """Compute all Mission 02 frame metrics in Kelvin and millimetres.

    Reuses :func:`evaluate_frame_prediction` for the core thermal metrics and appends:
    - isotherm boundary distances at 1800 K and 2100 K;
    - inside-field vs outside-field errors (using ``field_mask``, falling back to
      ``valid_mask`` or ``observed_mask``);
    - observed-node vs unobserved-node errors;
    - distance-to-nearest-observed-vector binned errors (frozen R1 §8 classes);
    - SENS-REJ rejected / substituted vs non-rejected valid node errors;
    - optional velocity-defined region errors (HG, JET, REST).
    """
    pred = np.asarray(pred_T_K, dtype=np.float64)
    true = np.asarray(true_T_K, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"pred shape {pred.shape} != true shape {true.shape}")

    row = evaluate_frame_prediction(
        pred,
        true,
        train_sigma_T=train_sigma_T,
        sim_id=sim_id,
        split=split,
        role=role,
        frame_index=int(frame_index),
        mask_temperature=None,
        t_high=t_high,
        t_peak=t_peak,
        t_jet=t_jet,
        **config_tags,
    )
    row.pop("_spatial_bias_map", None)
    row["bias_K"] = float(row["mean_signed_error_K"])
    row["condition"] = str(sim_id)

    # Isotherm symmetric boundary distances (1800 K and 2100 K)
    for thr in ISOTHERM_THRESHOLDS_K:
        row.update(
            isotherm_boundary_distance_mm(
                pred, true, threshold_K=thr, steps_mm=steps_mm
            )
        )

    obs_m = np.asarray(observed_mask, dtype=bool)
    val_m = np.asarray(valid_mask, dtype=bool) if valid_mask is not None else obs_m
    fld_m = np.asarray(field_mask, dtype=bool) if field_mask is not None else val_m

    row["coverage_observed_frac"] = float(obs_m.mean())
    row["coverage_valid_frac"] = float(val_m.mean())
    row["coverage_field_frac"] = float(fld_m.mean())
    row["n_observed_nodes"] = int(obs_m.sum())
    row["n_valid_nodes"] = int(val_m.sum())
    row["n_field_nodes"] = int(fld_m.sum())

    # Inside vs outside observed field (e.g. B-in / B-down / B-off)
    row.update(_subset_metrics(pred, true, fld_m, "in_field"))
    row.update(_subset_metrics(pred, true, ~fld_m, "out_field"))

    # Observed vs unobserved nodes
    row.update(_subset_metrics(pred, true, obs_m, "obs_nodes"))
    row.update(_subset_metrics(pred, true, ~obs_m, "unobs_nodes"))

    # Distance-to-nearest-valid-vector bins (R1 §8)
    row.update(
        distance_binned_errors(
            pred, true, obs_m, steps_mm=steps_mm, dist_map_mm=dist_map_mm
        )
    )

    # SENS-REJ node-level decomposition (rejected / substituted vs non-rejected valid)
    has_rej = rejected_mask is not None and bool(np.asarray(rejected_mask, dtype=bool).any())
    row["has_rejections"] = has_rej
    if has_rej:
        rej_m = np.asarray(rejected_mask, dtype=bool)
        nonrej_m = val_m & ~rej_m
        row.update(_subset_metrics(pred, true, rej_m, "rej_nodes"))
        row.update(_subset_metrics(pred, true, nonrej_m, "nonrej_valid"))
        if substituted_mask is not None and bool(np.asarray(substituted_mask, dtype=bool).any()):
            sub_m = np.asarray(substituted_mask, dtype=bool)
            row.update(_subset_metrics(pred, true, sub_m, "sub_nodes"))
        else:
            row.update(_subset_metrics(pred, true, np.zeros_like(obs_m, dtype=bool), "sub_nodes"))
    else:
        empty = np.zeros_like(obs_m, dtype=bool)
        row.update(_subset_metrics(pred, true, empty, "rej_nodes"))
        row.update(_subset_metrics(pred, true, empty, "nonrej_valid"))
        row.update(_subset_metrics(pred, true, empty, "sub_nodes"))

    # Optional velocity-defined diagnostic regions (HG, JET, REST)
    if velocity_region_masks is not None:
        for reg_name in ("HG", "JET", "REST"):
            if reg_name in velocity_region_masks:
                row.update(
                    _subset_metrics(
                        pred, true, velocity_region_masks[reg_name], f"vel_{reg_name.lower()}"
                    )
                )

    return row


@dataclass
class SpatialErrorAccumulator:
    """Streaming per-condition spatial error statistics (pred - true) on a 2D grid.

    Stores sum of signed errors, sum of squared errors, and frame count per condition
    so spatial mean-error maps and spatial RMSE maps are computed with O(H*W) memory
    and compact disk footprint (tens of KB), without retaining all full predicted fields.
    """

    hw: tuple[int, int] = (128, 128)
    _sum_err: dict[str, np.ndarray] = field(default_factory=dict)
    _sum_sq_err: dict[str, np.ndarray] = field(default_factory=dict)
    _sum_abs_err: dict[str, np.ndarray] = field(default_factory=dict)
    _count: dict[str, int] = field(default_factory=dict)

    def update(self, condition: str, pred_T_K: np.ndarray, true_T_K: np.ndarray) -> None:
        err = np.asarray(pred_T_K, dtype=np.float64) - np.asarray(true_T_K, dtype=np.float64)
        if err.shape != self.hw:
            raise ValueError(f"expected shape {self.hw}, got {err.shape}")
        if condition not in self._count:
            self._sum_err[condition] = np.zeros(self.hw, dtype=np.float64)
            self._sum_sq_err[condition] = np.zeros(self.hw, dtype=np.float64)
            self._sum_abs_err[condition] = np.zeros(self.hw, dtype=np.float64)
            self._count[condition] = 0
        self._sum_err[condition] += err
        self._sum_sq_err[condition] += err * err
        self._sum_abs_err[condition] += np.abs(err)
        self._count[condition] += 1

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        """Serialize per-condition means + macro (uniform over conditions) summary."""
        out: dict[str, np.ndarray] = {}
        conds = sorted(self._count)
        out["conditions"] = np.asarray(conds, dtype="<U64")
        if not conds:
            return out
        mean_errs, rmse_maps, mae_maps = [], [], []
        for c in conds:
            n = self._count[c]
            me = (self._sum_err[c] / n).astype(np.float32)
            rm = np.sqrt(np.maximum(self._sum_sq_err[c] / n, 0.0)).astype(np.float32)
            ma = (self._sum_abs_err[c] / n).astype(np.float32)
            out[f"cond|{c}|mean_error_K"] = me
            out[f"cond|{c}|rmse_K"] = rm
            out[f"cond|{c}|mae_K"] = ma
            out[f"cond|{c}|n_frames"] = np.asarray(n, dtype=np.int64)
            mean_errs.append(me.astype(np.float64))
            rmse_maps.append(rm.astype(np.float64))
            mae_maps.append(ma.astype(np.float64))
        # Macro spatial maps: uniform average across conditions (matching scalar aggregation)
        out["macro|mean_error_K"] = np.mean(mean_errs, axis=0).astype(np.float32)
        out["macro|rmse_K"] = np.mean(rmse_maps, axis=0).astype(np.float32)
        out["macro|mae_K"] = np.mean(mae_maps, axis=0).astype(np.float32)
        out["macro|n_conditions"] = np.asarray(len(conds), dtype=np.int64)
        return out


def compute_interseed_field_stats(
    pred_fields_by_seed: Mapping[int, np.ndarray],
    true_T_K: np.ndarray,
) -> dict[str, Any]:
    """Pixelwise inter-seed variability on one frame.

    Explicitly separates:
    - ``mean_pred_T_K``: pointwise mean across seeds;
    - ``seed_std_T_K``: sample std across seeds (ddof=1 if n>=2, else NaN);
    - ``seed_pop_std_T_K``: population std across seeds (ddof=0, matching Mission 00);
    - ``error_of_mean_K``: ``mean_pred_T_K - true_T_K`` (its RMSE != mean of per-seed RMSEs);
    - ``mean_of_seed_rmse_K`` and ``rmse_of_mean_pred_K`` reported side-by-side.
    """
    seeds = sorted(pred_fields_by_seed)
    stack = np.stack([np.asarray(pred_fields_by_seed[s], dtype=np.float64) for s in seeds], axis=0)
    true = np.asarray(true_T_K, dtype=np.float64)
    mean_pred = np.mean(stack, axis=0)
    pop_std = np.std(stack, axis=0, ddof=0)
    sample_std = np.std(stack, axis=0, ddof=1) if len(seeds) >= 2 else np.full_like(mean_pred, np.nan)
    err_of_mean = mean_pred - true
    per_seed_rmses = [float(np.sqrt(np.mean((stack[i] - true) ** 2))) for i in range(len(seeds))]
    return {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "mean_pred_T_K": mean_pred.astype(np.float32),
        "seed_std_T_K": sample_std.astype(np.float32),
        "seed_pop_std_T_K": pop_std.astype(np.float32),
        "error_of_mean_K": err_of_mean.astype(np.float32),
        "rmse_of_mean_pred_K": float(np.sqrt(np.mean(err_of_mean**2))),
        "mean_of_seed_rmse_K": float(np.mean(per_seed_rmses)),
        "per_seed_rmse_K": {int(s): v for s, v in zip(seeds, per_seed_rmses)},
        "mean_spatial_seed_std_K": float(np.nanmean(sample_std)) if len(seeds) >= 2 else float("nan"),
        "max_spatial_seed_std_K": float(np.nanmax(sample_std)) if len(seeds) >= 2 else float("nan"),
    }


__all__ = [
    "DEFAULT_STEPS_MM",
    "DISTANCE_BINS",
    "ISOTHERM_THRESHOLDS_K",
    "SpatialErrorAccumulator",
    "compute_interseed_field_stats",
    "distance_binned_errors",
    "distance_to_valid_vector_mm",
    "evaluate_piv_frame",
    "field_rect_mask",
    "isotherm_boundary_distance_mm",
    "field_errors",
    "spatial_bias",
]
