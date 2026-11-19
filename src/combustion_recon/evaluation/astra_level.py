"""Astra: level (uniform offset) vs structure, oracle offset and nine-T anchor offset (AO9).

All quantities are in Kelvin after denormalisation. The anchor offset reuses the
consolidated :func:`combustion_recon.models.baselines.apply_anchor_offset`
(``b = sum_k (T_obs_k - T_hat_k) / (K + tau)``).

* ``oracle`` reads the full true field: diagnostic only, NOT deployable.
* ``ao9`` reads only nine observed T scalars and the prediction at those sites.
* Nine fixed sites are frozen a priori (3x3 grid, rows/cols ``[32, 64, 95]``).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from combustion_recon.evaluation.metrics import (
    DEFAULT_T_HIGH,
    DEFAULT_T_JET,
    DEFAULT_T_PEAK,
)
from combustion_recon.models.baselines import apply_anchor_offset

H = W = 128
T9_ROWS: tuple[int, ...] = (32, 64, 95)
T9_COLS: tuple[int, ...] = (32, 64, 95)
AO_TAU = 1.0
TREATMENTS: tuple[str, ...] = ("free", "oracle", "ao9")
# Second-pass diagnostics. They are not part of the primary A0–A3 treatments.
# mean9 is the unshrunk mean of the same nine residuals; median9 is their
# ordinary median. AO9 is mean9 * 9 / (9 + tau), so a median-versus-AO9
# contrast changes both the estimator and the shrink.
LEVEL_DIAGNOSTICS: tuple[str, ...] = ("mean9", "median9")
N_ANCHORS = 9


def t9_site_indices(height: int = H, width: int = W) -> np.ndarray:
    """Nine flat site indices (row-major, ``row * width + col``), fixed a priori."""
    sites = np.asarray(
        [r * width + c for r in T9_ROWS for c in T9_COLS], dtype=np.int64
    )
    if len(np.unique(sites)) != 9 or sites.min() < 0 or sites.max() >= height * width:
        raise ValueError("invalid T9 layout")
    return sites


def anchor_mask_from_sites(sites: np.ndarray, height: int = H, width: int = W) -> np.ndarray:
    m = np.zeros(height * width, dtype=bool)
    m[np.asarray(sites, dtype=np.int64)] = True
    return m.reshape(height, width)


def oracle_delta(pred_K: np.ndarray, true_K: np.ndarray) -> float:
    """``mean_grid(T_true - T_pred)`` (uses the whole true field: diagnostic)."""
    return float(np.mean(np.asarray(true_K, np.float64) - np.asarray(pred_K, np.float64)))


def ao_correct(
    pred_K: np.ndarray,
    anchor_values_K: np.ndarray,
    sites: np.ndarray,
    *,
    tau: float = AO_TAU,
) -> tuple[np.ndarray, float]:
    """Return ``(pred + delta, delta)`` from the K observed anchor scalars only.

    The observation grid handed to the legacy routine is NaN everywhere except at
    the anchors, so any read outside them would poison the result.
    """
    pred = np.asarray(pred_K, np.float64)
    sites = np.asarray(sites, dtype=np.int64).ravel()
    vals = np.asarray(anchor_values_K, np.float64).ravel()
    if vals.shape != sites.shape:
        raise ValueError("anchor values/sites shape mismatch")
    mask = anchor_mask_from_sites(sites, *pred.shape)
    t_obs = np.full(pred.shape, np.nan, dtype=np.float64)
    t_obs.ravel()[sites] = vals
    corrected, delta = apply_anchor_offset(pred, t_obs, mask, tau=tau)
    return corrected, float(delta)


def _err_block(pred: np.ndarray, true: np.ndarray, dom: np.ndarray, prefix: str) -> dict[str, float]:
    e = (pred - true)[dom]
    n = int(e.size)
    if n == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K")}
    bias = float(e.mean())
    rmse = float(np.sqrt(np.mean(e**2)))
    return {
        f"{prefix}_rmse_K": rmse,
        f"{prefix}_mae_K": float(np.mean(np.abs(e))),
        f"{prefix}_bias_K": bias,
        f"{prefix}_abs_bias_K": abs(bias),
        f"{prefix}_centered_rmse_K": float(np.sqrt(max(np.mean((e - bias) ** 2), 0.0))),
    }


def _region_block(
    pred: np.ndarray, true: np.ndarray, dom: np.ndarray, prefix: str,
    *, t_high: float, t_peak: float, t_jet: float,
) -> dict[str, float]:
    """Regional metrics inside ``dom``. Empty sets give NaN plus explicit counts."""
    hi_t = (true >= t_high) & dom
    hi_p = (pred >= t_high) & dom
    inter = int((hi_t & hi_p).sum())
    union = int((hi_t | hi_p).sum())
    n_dom = int(dom.sum())
    out: dict[str, float] = {
        f"{prefix}_hot_n_true": int(hi_t.sum()),
        f"{prefix}_hot_n_pred": int(hi_p.sum()),
        f"{prefix}_hot_n_inter": inter,
        f"{prefix}_hot_n_union": union,
        # both-empty: NaN (not 1.0); counts above disambiguate.
        f"{prefix}_hot_iou": (inter / union) if union > 0 else float("nan"),
        f"{prefix}_hot_area_true": int(hi_t.sum()) / max(n_dom, 1),
        f"{prefix}_hot_area_pred": int(hi_p.sum()) / max(n_dom, 1),
    }
    for name, m in (
        ("hot", hi_t),
        ("peak", (true >= t_peak) & dom),
        ("jet", (true < t_jet) & dom),
        ("nonhot", (~(true >= t_high)) & dom),
    ):
        k = int(m.sum())
        out[f"{prefix}_{name}_rmse_K"] = (
            float(np.sqrt(np.mean((pred - true)[m] ** 2))) if k else float("nan")
        )
        out[f"{prefix}_{name}_n"] = k
    return out


def frame_metrics(
    pred_K: np.ndarray,
    true_K: np.ndarray,
    anchor_mask: np.ndarray,
    *,
    t_high: float = DEFAULT_T_HIGH,
    t_peak: float = DEFAULT_T_PEAK,
    t_jet: float = DEFAULT_T_JET,
) -> dict[str, float]:
    """All-field (``all_``) and off-anchor (``off_``) metrics for one frame.

    Region masks use the true field (evaluation only) and are identical across
    treatments; the off-anchor domain excludes exactly the anchor sites.
    """
    pred = np.asarray(pred_K, np.float64)
    true = np.asarray(true_K, np.float64)
    am = np.asarray(anchor_mask, dtype=bool)
    all_dom = np.ones_like(am)
    off_dom = ~am
    out: dict[str, float] = {}
    for dom, pre in ((all_dom, "all"), (off_dom, "off")):
        out.update(_err_block(pred, true, dom, pre))
        out.update(
            _region_block(pred, true, dom, pre, t_high=t_high, t_peak=t_peak, t_jet=t_jet)
        )
    out["n_off_anchor"] = int(off_dom.sum())
    out["n_anchor"] = int(am.sum())
    return out


def treat_frame(
    pred_K: np.ndarray,
    true_K: np.ndarray,
    sites: np.ndarray,
    *,
    tau: float = AO_TAU,
) -> list[dict[str, Any]]:
    """Free / oracle / AO9 rows from ONE base prediction (single forward)."""
    pred = np.asarray(pred_K, np.float64)
    true = np.asarray(true_K, np.float64)
    sites = np.asarray(sites, dtype=np.int64)
    am = anchor_mask_from_sites(sites, *pred.shape)
    anchor_vals = true.ravel()[sites]
    d_or = oracle_delta(pred, true)
    ao_field, d_ao = ao_correct(pred, anchor_vals, sites, tau=tau)
    fields = {"free": (pred, 0.0), "oracle": (pred + d_or, d_or), "ao9": (ao_field, d_ao)}
    mse_free = float(np.mean((pred - true) ** 2))
    mse_or = float(np.mean((fields["oracle"][0] - true) ** 2))
    rows = []
    for name in TREATMENTS:
        f, d = fields[name]
        row: dict[str, Any] = {
            "treatment": name,
            "delta_applied_K": float(d),
            "delta_oracle_K": d_or,
            "delta_ao9_K": d_ao,
            "delta_ao9_minus_oracle_K": d_ao - d_or,
            "anchor_mean_resid_K": float(np.mean(anchor_vals - pred.ravel()[sites])),
        }
        row.update(frame_metrics(f, true, am))
        mse_t = float(np.mean((f - true) ** 2))
        den = mse_free - mse_or
        row["mse_recovery_vs_oracle"] = (
            (mse_free - mse_t) / den if den > 1e-6 else float("nan")
        )
        rows.append(row)
    return rows


def anchor_residuals_K(
    pred_K: np.ndarray,
    true_K: np.ndarray,
    sites: np.ndarray,
) -> np.ndarray:
    """Nine residuals ``T_obs - T_pred`` at ``sites``, in site order, kelvin."""
    pred = np.asarray(pred_K, np.float64).ravel()
    true = np.asarray(true_K, np.float64).ravel()
    sites = np.asarray(sites, dtype=np.int64).ravel()
    if sites.size != N_ANCHORS:
        raise ValueError(f"expected {N_ANCHORS} anchors, got {sites.size}")
    if np.unique(sites).size != N_ANCHORS:
        raise ValueError("anchor sites must be unique")
    return true[sites] - pred[sites]


def delta_mean9(residuals_K: np.ndarray) -> float:
    """Unregularised mean of the nine residuals. No tau, shrink, or clip."""
    resid = np.asarray(residuals_K, np.float64).ravel()
    if resid.size != N_ANCHORS:
        raise ValueError(f"mean9 requires {N_ANCHORS} residuals, got {resid.size}")
    return float(resid.sum() / N_ANCHORS)


def delta_median9(residuals_K: np.ndarray) -> float:
    """Ordinary median: the fifth value of the nine residuals sorted ascending."""
    resid = np.sort(np.asarray(residuals_K, np.float64).ravel())
    if resid.size != N_ANCHORS:
        raise ValueError(f"median9 requires {N_ANCHORS} residuals, got {resid.size}")
    return float(resid[N_ANCHORS // 2])


def ao9_contributions_K(residuals_K: np.ndarray, *, tau: float = AO_TAU) -> np.ndarray:
    """Per-anchor term of AO9. Their sum is ``delta_ao9`` for the same tau."""
    resid = np.asarray(residuals_K, np.float64).ravel()
    if resid.size != N_ANCHORS:
        raise ValueError(f"AO9 contributions require {N_ANCHORS} residuals")
    return resid / (N_ANCHORS + float(tau))


def saved_columns_have_anchor_residuals(columns: Any) -> bool:
    """True only when the nine residuals themselves are present.

    ``anchor_mean_resid_K`` is not enough: the median is not a function of the mean.
    """
    names = set(columns)
    return all(f"resid_K_{i}" in names for i in range(N_ANCHORS))


def treat_frame_with_level_diagnostics(
    pred_K: np.ndarray,
    true_K: np.ndarray,
    sites: np.ndarray,
    *,
    tau: float = AO_TAU,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Primary free/oracle/AO9 rows plus mean9 and median9 from the same prediction.

    The primary rows are exactly :func:`treat_frame`. mean9 and median9 add a
    uniform translation and do not replace AO9. T-only is not produced here.
    """
    pred = np.asarray(pred_K, np.float64)
    true = np.asarray(true_K, np.float64)
    sites = np.asarray(sites, dtype=np.int64)
    resid = anchor_residuals_K(pred, true, sites)
    d_mean = delta_mean9(resid)
    d_med = delta_median9(resid)
    d_ao = float(resid.sum() / (N_ANCHORS + float(tau)))
    rows = treat_frame(pred, true, sites, tau=tau)
    am = anchor_mask_from_sites(sites, *pred.shape)
    d_or = float(rows[0]["delta_oracle_K"])
    mse_free = float(np.mean((pred - true) ** 2))
    mse_or = float(np.mean((pred + d_or - true) ** 2))
    den = mse_free - mse_or
    for name, delta in (("mean9", d_mean), ("median9", d_med)):
        field = pred + delta
        row: dict[str, Any] = {
            "treatment": name,
            "delta_applied_K": float(delta),
            "delta_oracle_K": d_or,
            "delta_ao9_K": d_ao,
            "delta_mean9_K": d_mean,
            "delta_median9_K": d_med,
            "delta_ao9_minus_oracle_K": d_ao - d_or,
            "delta_mean9_minus_oracle_K": d_mean - d_or,
            "delta_median9_minus_oracle_K": d_med - d_or,
            "anchor_mean_resid_K": d_mean,
        }
        row.update(frame_metrics(field, true, am))
        mse_t = float(np.mean((field - true) ** 2))
        row["mse_recovery_vs_oracle"] = (
            (mse_free - mse_t) / den if den > 1e-6 else float("nan")
        )
        rows.append(row)
    detail = {
        "anchor_flat_index": sites.astype(np.int64).copy(),
        "anchor_row": (sites // pred.shape[1]).astype(np.int64),
        "anchor_col": (sites % pred.shape[1]).astype(np.int64),
        "resid_K": resid.copy(),
        "ao9_contribution_K": ao9_contributions_K(resid, tau=tau),
        "delta_mean9_K": d_mean,
        "delta_median9_K": d_med,
        "delta_ao9_K": d_ao,
        "delta_oracle_K": d_or,
    }
    return rows, detail


_REPLAY_FRAME_KEYS = ("method", "role", "replica_id", "sim_id", "frame_index", "treatment")


def prediction_tables(
    pred_K: np.ndarray,
    true_K: np.ndarray,
    sites: np.ndarray,
    identity: Mapping[str, Any],
    *,
    tau: float = AO_TAU,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Metric rows and one residual row per anchor, from one base prediction.

    Free, oracle and AO9 stay the primary rows. mean9 and median9 are extra
    translations of that same prediction. The identity is copied onto every row.
    """
    rows, detail = treat_frame_with_level_diagnostics(pred_K, true_K, sites, tau=tau)
    d_mean = float(detail["delta_mean9_K"])
    d_med = float(detail["delta_median9_K"])
    metric_rows: list[dict[str, Any]] = []
    for row in rows:
        applied = float(row["delta_applied_K"])
        oracle = float(row["delta_oracle_K"])
        metric_rows.append(
            {
                **dict(identity),
                **row,
                "delta_mean9_K": d_mean,
                "delta_median9_K": d_med,
                "delta_applied_minus_oracle_K": applied - oracle,
                "level_recipe": "same_base_prediction",
            }
        )
    residual_rows: list[dict[str, Any]] = []
    for i in range(N_ANCHORS):
        residual_rows.append(
            {
                **dict(identity),
                "anchor_id": i,
                "anchor_flat_index": int(detail["anchor_flat_index"][i]),
                "anchor_row": int(detail["anchor_row"][i]),
                "anchor_col": int(detail["anchor_col"][i]),
                "resid_K": float(detail["resid_K"][i]),
                "ao9_contribution_K": float(detail["ao9_contribution_K"][i]),
                "delta_mean9_K": d_mean,
                "delta_median9_K": d_med,
                "delta_ao9_K": float(detail["delta_ao9_K"]),
                "delta_oracle_K": float(detail["delta_oracle_K"]),
            }
        )
    return metric_rows, residual_rows


def t_only_constant_row(
    true_K: np.ndarray,
    sites: np.ndarray,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Historical constant field: the mean of the nine observed T values.

    mean9 and median9 are not applied. Offset columns stay NaN.
    """
    true = np.asarray(true_K, np.float64)
    sites = np.asarray(sites, dtype=np.int64)
    const = np.full_like(true, float(np.mean(true.ravel()[sites])))
    row: dict[str, Any] = {
        **dict(identity),
        "treatment": "t_only_const",
        "level_recipe": "historical_t_only_const",
        "delta_applied_K": float("nan"),
        "delta_oracle_K": float("nan"),
        "delta_ao9_K": float("nan"),
        "delta_mean9_K": float("nan"),
        "delta_median9_K": float("nan"),
        "delta_applied_minus_oracle_K": float("nan"),
        "anchor_mean_resid_K": float("nan"),
        "mse_recovery_vs_oracle": float("nan"),
    }
    row.update(frame_metrics(const, true, anchor_mask_from_sites(sites, *true.shape)))
    return row


def _macro_rmse(frame: pd.DataFrame, col: str = "all_rmse_K") -> float:
    """Mean over conditions of the per-condition mean. One weight per condition."""
    return float(frame.groupby("sim_id")[col].mean().mean())


def _replay_norm(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["method"] = out["method"].astype(str)
    out["role"] = out["role"].astype(str)
    out["replica_id"] = out["replica_id"].astype(int)
    out["sim_id"] = out["sim_id"].astype(str)
    out["frame_index"] = out["frame_index"].astype(int)
    out["treatment"] = out["treatment"].astype(str)
    return out


def primary_level_replay(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    atol_K: float = 0.05,
    treatments: tuple[str, ...] = TREATMENTS,
    roles: tuple[str, ...] = ("seen_train_eval", "val_eval"),
    replicas: tuple[int, ...] = (0, 1, 2),
    expected_frames: Mapping[str, int] | None = None,
    compare_delta_applied: bool = True,
) -> dict[str, Any]:
    """Compare free/oracle/AO9 with a non-empty primary table.

    The pass/fail cell is method × role × replica × treatment. Macro RMSE and
    the largest per-frame RMSE difference both have to sit within ``atol_K``.
    When ``compare_delta_applied`` is set, the applied offset is held to the
    same tolerance. An empty reference is not a replay: ``replay_executed``
    stays false and ``validated`` stays false.
    """
    empty_frames = pd.DataFrame(
        columns=[
            "method", "role", "replica_id", "sim_id", "frame_index", "treatment",
            "all_rmse_K_new", "all_rmse_K_ref", "delta_all_rmse_K",
            "delta_applied_K_new", "delta_applied_K_ref", "delta_delta_applied_K",
            "pair",
        ]
    )
    base: dict[str, Any] = {
        "replay_executed": False,
        "validated": False,
        "reason": None,
        "atol_K": float(atol_K),
        "treatments": list(treatments),
        "keys": list(_REPLAY_FRAME_KEYS),
        "compare_delta_applied": bool(compare_delta_applied),
        "n_cells_expected": 0,
        "n_cells_pass": 0,
        "max_abs_macro_rmse_K": None,
        "max_abs_frame_rmse_K": None,
        "max_abs_delta_applied_K": None,
        "cells": [],
        "frames": empty_frames,
    }
    if reference is None or len(reference) == 0:
        base["reason"] = "empty_reference"
        return base
    need = set(_REPLAY_FRAME_KEYS) | {"all_rmse_K"}
    if compare_delta_applied:
        need.add("delta_applied_K")
    missing_ref = sorted(need - set(reference.columns))
    missing_cand = sorted(need - set(candidate.columns))
    if missing_cand or missing_ref:
        base["reason"] = "missing_columns"
        base["missing_columns"] = {"candidate": missing_cand, "reference": missing_ref}
        return base

    cand = _replay_norm(candidate)
    ref = _replay_norm(reference)
    cand = cand[cand["treatment"].isin(treatments)]
    ref = ref[ref["treatment"].isin(treatments)]
    if len(ref) == 0 or len(cand) == 0:
        base["reason"] = "empty_reference"
        return base
    if cand.duplicated(list(_REPLAY_FRAME_KEYS)).any() or ref.duplicated(list(_REPLAY_FRAME_KEYS)).any():
        base["replay_executed"] = True
        base["reason"] = "duplicate_keys"
        return base

    methods = sorted(set(cand["method"]) | set(ref["method"]))
    if not methods or set(cand["method"]) != set(ref["method"]):
        base["replay_executed"] = True
        base["reason"] = "method_set_mismatch"
        base["methods_candidate"] = sorted(set(cand["method"]))
        base["methods_reference"] = sorted(set(ref["method"]))
        return base

    base["replay_executed"] = True
    cells: list[dict[str, Any]] = []
    frame_parts: list[pd.DataFrame] = []
    macro_abs: list[float] = []
    frame_abs: list[float] = []
    applied_abs: list[float] = []
    for method in methods:
        for role in roles:
            for replica in replicas:
                for treatment in treatments:
                    c = cand[
                        (cand["method"] == method)
                        & (cand["role"] == role)
                        & (cand["replica_id"] == int(replica))
                        & (cand["treatment"] == treatment)
                    ]
                    r = ref[
                        (ref["method"] == method)
                        & (ref["role"] == role)
                        & (ref["replica_id"] == int(replica))
                        & (ref["treatment"] == treatment)
                    ]
                    merged = c[["sim_id", "frame_index", "all_rmse_K"]].merge(
                        r[["sim_id", "frame_index", "all_rmse_K"]],
                        on=["sim_id", "frame_index"],
                        how="outer",
                        suffixes=("_new", "_ref"),
                        indicator=True,
                    )
                    if compare_delta_applied:
                        applied = c[["sim_id", "frame_index", "delta_applied_K"]].merge(
                            r[["sim_id", "frame_index", "delta_applied_K"]],
                            on=["sim_id", "frame_index"],
                            how="outer",
                            suffixes=("_new", "_ref"),
                        )
                        merged = merged.merge(applied, on=["sim_id", "frame_index"], how="left")
                    else:
                        merged["delta_applied_K_new"] = np.nan
                        merged["delta_applied_K_ref"] = np.nan
                    merged["delta_all_rmse_K"] = merged["all_rmse_K_new"] - merged["all_rmse_K_ref"]
                    merged["delta_delta_applied_K"] = (
                        merged["delta_applied_K_new"] - merged["delta_applied_K_ref"]
                    )
                    merged.insert(0, "treatment", treatment)
                    merged.insert(0, "replica_id", int(replica))
                    merged.insert(0, "role", role)
                    merged.insert(0, "method", method)
                    merged = merged.rename(columns={"_merge": "pair"})
                    frame_parts.append(merged)

                    paired = merged[merged["pair"] == "both"]
                    n_expected = None if expected_frames is None else int(expected_frames[role])
                    macro_new = _macro_rmse(c) if len(c) else None
                    macro_ref = _macro_rmse(r) if len(r) else None
                    macro_delta = None if macro_new is None or macro_ref is None else macro_new - macro_ref
                    max_frame = float(paired["delta_all_rmse_K"].abs().max()) if len(paired) else None
                    max_applied = (
                        float(paired["delta_delta_applied_K"].abs().max())
                        if compare_delta_applied and len(paired)
                        else None
                    )
                    ok = (
                        len(c) > 0
                        and len(r) > 0
                        and bool((merged["pair"] == "both").all())
                        and (n_expected is None or len(c) == n_expected)
                        and macro_delta is not None
                        and abs(macro_delta) <= atol_K
                        and max_frame is not None
                        and max_frame <= atol_K
                        and (
                            not compare_delta_applied
                            or (max_applied is not None and max_applied <= atol_K)
                        )
                    )
                    if macro_delta is not None:
                        macro_abs.append(abs(macro_delta))
                    if max_frame is not None:
                        frame_abs.append(max_frame)
                    if max_applied is not None:
                        applied_abs.append(max_applied)
                    cells.append(
                        {
                            "method": method,
                            "role": role,
                            "replica_id": int(replica),
                            "treatment": treatment,
                            "n_frames_new": int(len(c)),
                            "n_frames_ref": int(len(r)),
                            "n_frames_expected": n_expected,
                            "n_paired": int(len(paired)),
                            "macro_rmse_new_K": macro_new,
                            "macro_rmse_ref_K": macro_ref,
                            "delta_macro_rmse_K": macro_delta,
                            "max_abs_frame_rmse_K": max_frame,
                            "max_abs_delta_applied_K": max_applied,
                            "ok": bool(ok),
                        }
                    )
    frames = pd.concat(frame_parts, ignore_index=True) if frame_parts else empty_frames
    n_pass = int(sum(bool(c["ok"]) for c in cells))
    validated = bool(cells) and n_pass == len(cells)
    base.update(
        {
            "reason": None if validated else "out_of_tolerance_or_unpaired",
            "n_cells_expected": len(cells),
            "n_cells_pass": n_pass,
            "max_abs_macro_rmse_K": max(macro_abs) if macro_abs else None,
            "max_abs_frame_rmse_K": max(frame_abs) if frame_abs else None,
            "max_abs_delta_applied_K": max(applied_abs) if applied_abs else None,
            "cells": cells,
            "frames": frames,
            "validated": validated,
        }
    )
    return base
