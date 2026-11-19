"""Astra: level (uniform offset) vs structure, oracle offset and nine-T anchor offset (AO9).

All quantities are in Kelvin after denormalisation. The anchor offset reuses the
consolidated :func:`combustion_recon.models.baselines.apply_anchor_offset`
(``b = sum_k (T_obs_k - T_hat_k) / (K + tau)``).

* ``oracle`` reads the full true field: diagnostic only, NOT deployable.
* ``ao9`` reads only nine observed T scalars and the prediction at those sites.
* Nine fixed sites are frozen a priori (3x3 grid, rows/cols ``[32, 64, 95]``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

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
