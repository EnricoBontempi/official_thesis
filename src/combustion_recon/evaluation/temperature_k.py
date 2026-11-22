"""Scalar temperature corrections for the C149 K sweep.

The K=9 formulas are the tested median9 functions in
:mod:`combustion_recon.evaluation.astra_level` (ordinary median, unshrunk mean,
AO with tau=1). This module applies those same formulas at K in
``{0,1,4,9,16,32}``. It does not choose sites: the nested layouts come from
the frozen manifest, byte for byte. Historical T9 stays a separate bridge.

K=0: AO, mean and median return the free field exactly. The oracle offset is
the mean residual of the whole field at every K, including 0; it is diagnostic
only. Even K: the median is the mean of the two central residuals. No clipping
and no new estimator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from combustion_recon.data.masks import nested_temperature_anchor_permutation, stable_seed
from combustion_recon.evaluation.astra_level import (
    H,
    W,
    AO_TAU,
    anchor_mask_from_sites,
    anchor_residuals_K,
    ao_correct,
    delta_mean9,
    delta_median9,
    frame_metrics,
    oracle_delta,
    t9_site_indices,
    t_only_constant_row,
)

K_VALUES: tuple[int, ...] = (0, 1, 4, 9, 16, 32)
PRIMARY_TREATMENTS: tuple[str, ...] = ("free", "ao", "mean", "median", "oracle")
SHIFT_TREATMENTS: tuple[str, ...] = ("ao", "mean", "median", "oracle")
MANIFEST_RELATIVE = "configs/observations/month2_temperature_anchors_nested_v1.json"
PROTOCOL_RELATIVE = "configs/experiments/month2_temperature_study_protocol.yaml"
MANIFEST_FILE_SHA256 = "a9c997c96cb44489dd1fc0d4484304b44d8beddbf00b73c25baf4e1922a19e54"
MANIFEST_CONTENT_SHA256 = "83fd3b5f2a07e5febc787c384285e265cb4764884fcbc6fb90d22218932f6ae6"
HISTORICAL_BASELINE_MANIFEST = "results/meta/month2_baseline_run_manifest.json"
HISTORICAL_BASELINE_MANIFEST_SHA256 = "112af741d94e5fd4e23d077e05d5e867c8026801a207741dc8ee7091c5b4d088"
# Selected on the A38 K=9 search and frozen. The C149 artifact stores w_T=5.0
# from the temperature-free fit; that stored value is not the T-block weight.
TBLOCK_P_OFF = {"ridge_lambda": 0.001, "w_T": 0.5}
TBLOCK_P_ON = {"ridge_lambda": 0.01, "w_T": 0.5}
C149_BASELINE_ARTIFACT_SHA256 = "a18f1bf2f4f5a170a04ebd73fb5d438aaa3c3e390cf44e678377381eecfe3e5a"
GATE_ATOL_K = 0.05
SWEEP_ROLES: tuple[str, ...] = ("seen_train_eval", "val_eval")
SWEEP_CHECKPOINTS: tuple[str, ...] = ("best", "last")
N_FRAMES = {"seen_train_eval": 220, "val_eval": 123}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash used by ``scripts/50_t_write_anchor_manifest.py`` (excludes itself)."""
    body = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    payload = json.dumps(body, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _prefix_sha256(order: np.ndarray, k: int) -> str:
    return hashlib.sha256(np.ascontiguousarray(order[:k], dtype=np.int64).tobytes()).hexdigest()


def load_frozen_manifest(path: Path) -> dict[str, Any]:
    """Load the published manifest and refuse any other bytes."""
    path = Path(path)
    file_sha = sha256_file(path)
    if file_sha != MANIFEST_FILE_SHA256:
        raise ValueError(
            f"manifest file sha256 {file_sha} != frozen {MANIFEST_FILE_SHA256}"
        )
    manifest = json.loads(path.read_text())
    got = content_sha256(manifest)
    if got != MANIFEST_CONTENT_SHA256 or manifest.get("manifest_content_sha256") != MANIFEST_CONTENT_SHA256:
        raise ValueError(
            f"manifest content sha256 {got} != frozen {MANIFEST_CONTENT_SHA256}"
        )
    if list(manifest["K_values"]) != list(K_VALUES):
        raise ValueError(f"manifest K_values {manifest['K_values']} != {list(K_VALUES)}")
    layouts = manifest["layouts"]
    if len(layouts) != 3:
        raise ValueError(f"expected 3 layouts, found {len(layouts)}")
    if [int(item["replica_id"]) for item in layouts] != [0, 1, 2]:
        raise ValueError("layouts are not replica 0, 1, 2 in order")
    for layout in layouts:
        order = np.asarray(layout["site_order_flat"], dtype=np.int64)
        if order.shape != (32,) or np.unique(order).size != 32:
            raise ValueError(f"replica {layout['replica_id']} does not have 32 unique sites")
        if order.min() < 0 or order.max() >= H * W:
            raise ValueError(f"replica {layout['replica_id']} site outside the grid")
        for k in K_VALUES:
            if layout["prefix_sha256_by_K"][str(k)] != _prefix_sha256(order, k):
                raise ValueError(f"replica {layout['replica_id']} prefix hash mismatch at K={k}")
        if layout["common_domain_excluded_sites"] != layout["site_order_flat"]:
            raise ValueError(f"replica {layout['replica_id']} common-domain sites differ from the order")
        if int(layout["common_domain_n_pixels"]) != H * W - 32:
            raise ValueError("common domain is not 16352 pixels")
        for left, right in zip(K_VALUES, K_VALUES[1:]):
            if order[:left].tolist() != order[:right][:left].tolist():
                raise ValueError(f"replica {layout['replica_id']} is not nested")
        seed = stable_seed(manifest["namespace"], int(manifest["base_seed"]), int(layout["replica_id"]))
        if int(layout["seed"]) != int(seed):
            raise ValueError(f"replica {layout['replica_id']} seed != stable_seed")
        generated = nested_temperature_anchor_permutation(seed=seed, height=H, width=W, max_k=32)
        if not np.array_equal(generated, order):
            raise ValueError(
                f"replica {layout['replica_id']} does not match nested_temperature_anchor_permutation"
            )
    if len({tuple(item["site_order_flat"]) for item in layouts}) != 3:
        raise ValueError("the three layouts are not distinct")
    return manifest


def layout_by_replica(manifest: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["replica_id"]): item for item in manifest["layouts"]}


def sites_for_k(order: Sequence[int], k: int) -> np.ndarray:
    if int(k) not in K_VALUES:
        raise ValueError(f"K={k} is not in the frozen grid")
    sites = np.asarray(order, dtype=np.int64).ravel()[: int(k)]
    if sites.size != int(k) or np.unique(sites).size != int(k):
        raise ValueError(f"K={k} sites are not {k} unique indices")
    return sites


def ordinary_median(residuals: np.ndarray) -> float:
    """Ordinary median. Even length: mean of the two central sorted values."""
    ordered = np.sort(np.asarray(residuals, dtype=np.float64).ravel())
    n = int(ordered.size)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return float(ordered[n // 2])
    return float(0.5 * (ordered[n // 2 - 1] + ordered[n // 2]))


def correction_deltas(
    pred_K: np.ndarray,
    true_K: np.ndarray,
    sites: np.ndarray,
    *,
    tau: float = AO_TAU,
) -> dict[str, Any]:
    """Deltas from one base prediction. K=0 corrections are exactly zero.

    Residuals are ``T_obs - T_pred`` at the sites only. Mutating the true field
    off those sites does not change AO, mean or median. The oracle does read
    the whole true field.
    """
    pred = np.asarray(pred_K)
    true = np.asarray(true_K)
    sites = np.asarray(sites, dtype=np.int64).ravel()
    if sites.size and np.unique(sites).size != sites.size:
        raise ValueError("anchor sites must be unique")
    d_oracle = oracle_delta(pred, true)
    if sites.size == 0:
        return {
            "free": 0.0,
            "ao": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "oracle": d_oracle,
            "residuals": np.zeros((0,), dtype=np.float64),
            "k": 0,
        }
    residuals = np.asarray(true, dtype=np.float64).ravel()[sites] - np.asarray(pred, dtype=np.float64).ravel()[sites]
    k = int(residuals.size)
    if k == 9:
        legacy = anchor_residuals_K(pred, true, sites)
        if not np.array_equal(legacy, residuals):
            raise ValueError("K=9 residuals diverged from the tested median9 residual")
        mean = delta_mean9(legacy)
        median = delta_median9(legacy)
        _, ao = ao_correct(pred, np.asarray(true, dtype=np.float64).ravel()[sites], sites, tau=tau)
        ao = float(ao)
    else:
        mean = float(residuals.sum() / k)
        median = ordinary_median(residuals)
        ao = float(residuals.sum() / (k + float(tau)))
    return {
        "free": 0.0,
        "ao": ao,
        "mean": mean,
        "median": median,
        "oracle": d_oracle,
        "residuals": residuals,
        "k": k,
    }


def apply_correction(pred_K: np.ndarray, delta: float, *, identity: bool) -> np.ndarray:
    """Return the free field itself when ``identity`` is set, otherwise ``pred + delta``."""
    pred = np.asarray(pred_K)
    if identity:
        return pred.copy()
    return np.asarray(pred, dtype=np.float64) + float(delta)


def _anchor_block(pred: np.ndarray, true: np.ndarray, sites: np.ndarray) -> dict[str, float]:
    sites = np.asarray(sites, dtype=np.int64).ravel()
    n = int(sites.size)
    if n == 0:
        return {
            "anchor_rmse_K": float("nan"),
            "anchor_mae_K": float("nan"),
            "anchor_bias_K": float("nan"),
            "anchor_abs_bias_K": float("nan"),
            "anchor_centered_rmse_K": float("nan"),
            "anchor_n": 0,
            "anchor_empty": 1,
        }
    err = np.asarray(pred, dtype=np.float64).ravel()[sites] - np.asarray(true, dtype=np.float64).ravel()[sites]
    bias = float(err.mean())
    return {
        "anchor_rmse_K": float(np.sqrt(np.mean(err**2))),
        "anchor_mae_K": float(np.mean(np.abs(err))),
        "anchor_bias_K": bias,
        "anchor_abs_bias_K": abs(bias),
        "anchor_centered_rmse_K": float(np.sqrt(max(np.mean((err - bias) ** 2), 0.0))),
        "anchor_n": n,
        "anchor_empty": 0,
    }


def _take(metrics: Mapping[str, float], src: str, dest: str) -> dict[str, float]:
    keys = (
        "rmse_K",
        "mae_K",
        "bias_K",
        "abs_bias_K",
        "centered_rmse_K",
        "hot_rmse_K",
        "hot_n",
        "hot_n_true",
        "hot_iou",
        "peak_rmse_K",
        "peak_n",
        "jet_rmse_K",
        "jet_n",
        "nonhot_rmse_K",
        "nonhot_n",
    )
    return {f"{dest}_{key}": metrics[f"{src}_{key}"] for key in keys}


def domain_metrics(
    pred_K: np.ndarray,
    true_K: np.ndarray,
    sites: np.ndarray,
    common_sites: np.ndarray,
) -> dict[str, float]:
    """Full field, off-anchor(K), the 32-site common domain, and the anchor sites.

    A non-finite full-field RMSE is refused. An empty region stays NaN with its
    count. K=0 has an empty anchor set and an off-anchor domain equal to the
    full field; the common domain still excludes the 32 layout sites.
    """
    pred = np.asarray(pred_K)
    true = np.asarray(true_K)
    sites = np.asarray(sites, dtype=np.int64).ravel()
    common = np.asarray(common_sites, dtype=np.int64).ravel()
    if common.size != 32:
        raise ValueError("the common domain excludes the 32 sites of the layout")
    full = frame_metrics(pred, true, anchor_mask_from_sites(sites, *np.asarray(pred).shape))
    common_m = frame_metrics(pred, true, anchor_mask_from_sites(common, *np.asarray(pred).shape))
    if not np.isfinite(full["all_rmse_K"]) or not np.isfinite(common_m["off_rmse_K"]):
        raise ValueError("global RMSE is not finite")
    out = {}
    out.update(_take(full, "all", "full"))
    out.update(_take(full, "off", "off"))
    out.update(_take(common_m, "off", "common"))
    out["full_n"] = int(H * W)
    out["off_n"] = int(full["n_off_anchor"])
    out["common_n"] = int(common_m["n_off_anchor"])
    out["full_peak_empty"] = int(out["full_peak_n"] == 0)
    out["full_jet_empty"] = int(out["full_jet_n"] == 0)
    out["full_hot_empty"] = int(out["full_hot_n"] == 0)
    out.update(_anchor_block(pred, true, sites))
    return out


def rows_for_prediction(
    pred_K: np.ndarray,
    true_K: np.ndarray,
    order: Sequence[int],
    *,
    identity: Mapping[str, Any],
    layout_id: str,
    layout_kind: str,
    include_t_only: bool = False,
) -> list[dict[str, Any]]:
    """One base prediction, every K, the five treatments. T9 is not added here."""
    pred = np.asarray(pred_K)
    true = np.asarray(true_K)
    order_arr = np.asarray(order, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        sites = sites_for_k(order_arr, k)
        deltas = correction_deltas(pred, true, sites)
        identity_k0 = k == 0
        fields = {
            "free": apply_correction(pred, 0.0, identity=True),
            "ao": apply_correction(pred, deltas["ao"], identity=identity_k0),
            "mean": apply_correction(pred, deltas["mean"], identity=identity_k0),
            "median": apply_correction(pred, deltas["median"], identity=identity_k0),
            "oracle": apply_correction(pred, deltas["oracle"], identity=False),
        }
        if identity_k0:
            for name in ("ao", "mean", "median"):
                if not np.array_equal(fields[name], pred):
                    raise ValueError(f"K=0 {name} is not exactly the free field")
        for name in PRIMARY_TREATMENTS:
            row: dict[str, Any] = dict(identity)
            row.update(
                {
                    "layout_id": layout_id,
                    "layout_kind": layout_kind,
                    "K": int(k),
                    "treatment": name,
                    "delta_applied_K": float(deltas[name]),
                    "delta_oracle_K": float(deltas["oracle"]),
                    "delta_ao_K": float(deltas["ao"]),
                    "delta_mean_K": float(deltas["mean"]),
                    "delta_median_K": float(deltas["median"]),
                    "level_recipe": "scalar_translation" if name != "free" else "free",
                }
            )
            row.update(domain_metrics(fields[name], true, sites, order_arr))
            rows.append(row)
        free_centered = next(item["full_centered_rmse_K"] for item in rows if item["K"] == k and item["treatment"] == "free")
        for item in rows:
            if item["K"] == k and item["treatment"] in SHIFT_TREATMENTS:
                if abs(item["full_centered_rmse_K"] - free_centered) > 1e-8:
                    raise ValueError(f"centered RMSE moved under a translation at K={k} {item['treatment']}")
                if abs(item["common_centered_rmse_K"] - next(
                    row["common_centered_rmse_K"] for row in rows if row["K"] == k and row["treatment"] == "free"
                )) > 1e-8:
                    raise ValueError(f"common-domain centered RMSE moved at K={k} {item['treatment']}")
        if include_t_only and k > 0:
            t_row = t_only_constant_row(true, sites, {})
            row = dict(identity)
            row.update(
                {
                    "layout_id": layout_id,
                    "layout_kind": layout_kind,
                    "K": int(k),
                    "treatment": "t_only_const",
                    "delta_applied_K": float("nan"),
                    "delta_oracle_K": float(deltas["oracle"]),
                    "delta_ao_K": float("nan"),
                    "delta_mean_K": float("nan"),
                    "delta_median_K": float("nan"),
                    "level_recipe": "historical_t_only_const",
                }
            )
            const = np.full(np.asarray(true).shape, float(np.mean(np.asarray(true, dtype=np.float64).ravel()[sites])))
            row.update(domain_metrics(const, true, sites, order_arr))
            # The historical row is the same constant; keep its full RMSE as a check.
            if abs(row["full_rmse_K"] - t_row["all_rmse_K"]) > 1e-8:
                raise ValueError("t_only_const diverged from the tested historical row")
            rows.append(row)
    return rows


def rows_for_historical_t9(
    pred_K: np.ndarray,
    true_K: np.ndarray,
    *,
    identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Historical 3x3 bridge at K=9 only. Not one of the three primary layouts."""
    sites = t9_site_indices()
    pred = np.asarray(pred_K)
    true = np.asarray(true_K)
    deltas = correction_deltas(pred, true, sites)
    rows = []
    for name in PRIMARY_TREATMENTS:
        field = apply_correction(pred, float(deltas[name]), identity=(name == "free"))
        metrics = frame_metrics(field, true, anchor_mask_from_sites(sites))
        row: dict[str, Any] = dict(identity)
        row.update(
            {
                "layout_id": "historical_t9",
                "layout_kind": "t9_bridge",
                "K": 9,
                "treatment": name,
                "delta_applied_K": float(deltas[name]),
                "delta_oracle_K": float(deltas["oracle"]),
                "delta_ao_K": float(deltas["ao"]),
                "delta_mean_K": float(deltas["mean"]),
                "delta_median_K": float(deltas["median"]),
                "level_recipe": "historical_t9_bridge" if name != "free" else "free",
            }
        )
        row.update(_take(metrics, "all", "full"))
        row.update(_take(metrics, "off", "off"))
        row["full_n"] = int(H * W)
        row["off_n"] = int(metrics["n_off_anchor"])
        row["common_n"] = 0
        for key in ("rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K", "hot_rmse_K", "hot_n",
                    "hot_n_true", "hot_iou", "peak_rmse_K", "peak_n", "jet_rmse_K", "jet_n",
                    "nonhot_rmse_K", "nonhot_n"):
            row[f"common_{key}"] = float("nan")
        row["full_peak_empty"] = int(row["full_peak_n"] == 0)
        row["full_jet_empty"] = int(row["full_jet_n"] == 0)
        row["full_hot_empty"] = int(row["full_hot_n"] == 0)
        row.update(_anchor_block(field, true, sites))
        if not np.isfinite(row["full_rmse_K"]):
            raise ValueError("historical T9 full RMSE is not finite")
        rows.append(row)
    t_only = t_only_constant_row(true, sites, {})
    const = np.full(true.shape, float(np.mean(np.asarray(true, dtype=np.float64).ravel()[sites])))
    row = dict(identity)
    row.update(
        {
            "layout_id": "historical_t9",
            "layout_kind": "t9_bridge",
            "K": 9,
            "treatment": "t_only_const",
            "delta_applied_K": float("nan"),
            "delta_oracle_K": float("nan"),
            "delta_ao_K": float("nan"),
            "delta_mean_K": float("nan"),
            "delta_median_K": float("nan"),
            "level_recipe": "historical_t_only_const",
        }
    )
    metrics = frame_metrics(const, true, anchor_mask_from_sites(sites))
    row.update(_take(metrics, "all", "full"))
    row.update(_take(metrics, "off", "off"))
    row["full_n"] = int(H * W)
    row["off_n"] = int(metrics["n_off_anchor"])
    row["common_n"] = 0
    for key in ("rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K", "hot_rmse_K", "hot_n",
                "hot_n_true", "hot_iou", "peak_rmse_K", "peak_n", "jet_rmse_K", "jet_n",
                "nonhot_rmse_K", "nonhot_n"):
        row[f"common_{key}"] = float("nan")
    row["full_peak_empty"] = int(row["full_peak_n"] == 0)
    row["full_jet_empty"] = int(row["full_jet_n"] == 0)
    row["full_hot_empty"] = int(row["full_hot_n"] == 0)
    row.update(_anchor_block(const, true, sites))
    if abs(row["full_rmse_K"] - t_only["all_rmse_K"]) > 1e-8:
        raise ValueError("historical t_only diverged")
    rows.append(row)
    return rows


def field_row(
    pred_K: np.ndarray,
    true_K: np.ndarray,
    sites: np.ndarray,
    common_sites: np.ndarray,
    *,
    identity: Mapping[str, Any],
    layout_id: str,
    layout_kind: str,
    k: int,
    treatment: str,
    level_recipe: str,
    delta_applied: float,
) -> dict[str, Any]:
    """Metrics for one already-built field (used for T-block, which is not a translation)."""
    row: dict[str, Any] = dict(identity)
    row.update(
        {
            "layout_id": layout_id,
            "layout_kind": layout_kind,
            "K": int(k),
            "treatment": treatment,
            "delta_applied_K": float(delta_applied),
            "delta_oracle_K": float("nan"),
            "delta_ao_K": float("nan"),
            "delta_mean_K": float("nan"),
            "delta_median_K": float("nan"),
            "level_recipe": level_recipe,
        }
    )
    row.update(domain_metrics(pred_K, true_K, sites, common_sites))
    return row


def prediction_sha256(pred_K: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(pred_K))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_historical_tblock(manifest_path: Path) -> dict[str, dict[str, float]]:
    """Read the frozen A38 selection. Refuse a different file or a different pair."""
    path = Path(manifest_path)
    file_sha = sha256_file(path)
    if file_sha != HISTORICAL_BASELINE_MANIFEST_SHA256:
        raise ValueError(
            f"baseline manifest sha256 {file_sha} != {HISTORICAL_BASELINE_MANIFEST_SHA256}"
        )
    selected = json.loads(path.read_text())["selected"]
    got = {
        "p_off": {
            "ridge_lambda": float(selected["tblock_p_off"]["ridge_lambda"]),
            "w_T": float(selected["tblock_p_off"]["w_T"]),
        },
        "p_on": {
            "ridge_lambda": float(selected["tblock_p_on"]["ridge_lambda"]),
            "w_T": float(selected["tblock_p_on"]["w_T"]),
        },
    }
    if got["p_off"] != TBLOCK_P_OFF or got["p_on"] != TBLOCK_P_ON:
        raise ValueError(f"historical T-block pair changed: {got}")
    if float(selected["free_p_off_lambda"]) != TBLOCK_P_OFF["ridge_lambda"]:
        raise ValueError("free p_off lambda is not the frozen T-block lambda")
    if float(selected["free_p_on_lambda"]) != TBLOCK_P_ON["ridge_lambda"]:
        raise ValueError("free p_on lambda is not the frozen T-block lambda")
    return got


def _finite_macro(frame: pd.DataFrame, column: str) -> float:
    values = frame[column].to_numpy(dtype=float)
    if np.isnan(values).all():
        return float("nan")
    if not np.isfinite(values).all():
        raise ValueError(f"{column} mixes finite values with NaN; refusing a global NaN")
    return macro_rmse(frame, column)


def summarize_sweep(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-seed macros, the cross-seed range, and every paired change versus free.

    Seeds are not collapsed in ``by_seed``. A positive ``delta_vs_free_K`` is a
    worsening and is kept. Anchor NaNs are kept only when every anchor count is 0.
    """
    if frame.empty:
        raise ValueError("empty sweep frame")
    keys = [
        "method", "p_conditioning", "model_seed", "checkpoint", "role",
        "mask_replica_id", "layout_id", "layout_kind", "K", "treatment",
    ]
    missing = [key for key in keys + ["sim_id", "full_rmse_K", "off_rmse_K", "common_rmse_K", "anchor_rmse_K", "anchor_n"] if key not in frame.columns]
    if missing:
        raise ValueError(f"sweep frame missing {missing}")
    rows = []
    condition_rows = []
    for values, part in frame.groupby(keys, dropna=False):
        ident = dict(zip(keys, values))
        anchor_n = part["anchor_n"].to_numpy()
        if np.isnan(part["anchor_rmse_K"].to_numpy(dtype=float)).any() and not np.all(anchor_n == 0):
            raise ValueError(f"anchor RMSE is NaN while anchors exist: {ident}")
        rec = dict(ident)
        rec["n_frames"] = int(part[["sim_id", "frame_index"]].drop_duplicates().shape[0]) if "frame_index" in part.columns else int(len(part))
        rec["n_conditions"] = int(part["sim_id"].nunique())
        for column, name in (
            ("full_rmse_K", "full_rmse_K"),
            ("full_bias_K", "full_bias_K"),
            ("full_abs_bias_K", "full_abs_bias_K"),
            ("full_centered_rmse_K", "full_centered_rmse_K"),
            ("off_rmse_K", "off_rmse_K"),
            ("off_centered_rmse_K", "off_centered_rmse_K"),
            ("common_rmse_K", "common_rmse_K"),
            ("common_bias_K", "common_bias_K"),
            ("common_centered_rmse_K", "common_centered_rmse_K"),
            ("anchor_rmse_K", "anchor_rmse_K"),
        ):
            rec[name] = _finite_macro(part, column)
        for region in ("hot", "peak", "jet", "nonhot"):
            col = f"full_{region}_rmse_K"
            n_col = f"full_{region}_n"
            if col not in part.columns:
                continue
            empty = part[n_col].to_numpy() == 0
            rec[f"full_{region}_n_empty"] = int(empty.sum())
            rec[f"full_{region}_n_defined"] = int((~empty).sum())
            rec[col] = float(part.loc[~empty].groupby("sim_id")[col].mean().mean()) if (~empty).any() else float("nan")
        rows.append(rec)
        for sim_id, cond in part.groupby("sim_id"):
            item = dict(ident)
            item["sim_id"] = sim_id
            item["full_rmse_K"] = float(cond["full_rmse_K"].mean())
            item["common_rmse_K"] = float(cond["common_rmse_K"].mean()) if cond["common_rmse_K"].notna().all() else float("nan")
            item["off_rmse_K"] = float(cond["off_rmse_K"].mean())
            item["full_bias_K"] = float(cond["full_bias_K"].mean())
            condition_rows.append(item)
    by_seed = pd.DataFrame(rows)
    free = by_seed[by_seed["treatment"] == "free"][
        ["method", "p_conditioning", "model_seed", "checkpoint", "role", "mask_replica_id", "layout_id", "K",
         "full_rmse_K", "common_rmse_K", "off_rmse_K"]
    ].rename(columns={"full_rmse_K": "free_full_rmse_K", "common_rmse_K": "free_common_rmse_K", "off_rmse_K": "free_off_rmse_K"})
    paired = by_seed.merge(
        free,
        on=["method", "p_conditioning", "model_seed", "checkpoint", "role", "mask_replica_id", "layout_id", "K"],
        how="left",
    )
    paired["delta_full_vs_free_K"] = paired["full_rmse_K"] - paired["free_full_rmse_K"]
    paired["delta_common_vs_free_K"] = paired["common_rmse_K"] - paired["free_common_rmse_K"]
    paired["worse_full"] = paired["delta_full_vs_free_K"] > 0
    paired["worse_common"] = paired["delta_common_vs_free_K"] > 0
    worsen = paired[paired["treatment"] != "free"].copy()
    group = ["method", "p_conditioning", "checkpoint", "role", "mask_replica_id", "layout_kind", "K", "treatment"]
    across = (
        by_seed.groupby(group, dropna=False)
        .agg(
            n_seeds=("model_seed", "nunique"),
            full_rmse_mean_K=("full_rmse_K", "mean"),
            full_rmse_min_K=("full_rmse_K", "min"),
            full_rmse_max_K=("full_rmse_K", "max"),
            common_rmse_mean_K=("common_rmse_K", "mean"),
            common_rmse_min_K=("common_rmse_K", "min"),
            common_rmse_max_K=("common_rmse_K", "max"),
            off_rmse_mean_K=("off_rmse_K", "mean"),
        )
        .reset_index()
    )
    return {"by_seed": by_seed, "per_condition": pd.DataFrame(condition_rows), "versus_free": worsen, "across_seeds": across}


def macro_rmse(frame: pd.DataFrame, column: str = "full_rmse_K") -> float:
    """Mean over conditions of the per-condition mean. One weight per condition."""
    if frame.empty or column not in frame.columns:
        raise ValueError("macro RMSE requires a non-empty frame and the metric column")
    selected = frame.loc[:, column]
    if isinstance(selected, pd.DataFrame):
        raise ValueError(f"{column} is duplicated; refusing to average two columns of the same name")
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError(f"{column} has a non-finite value; refusing a global NaN")
    return float(selected.groupby(frame["sim_id"]).mean().mean())


def gate_free_against_reference(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    reference_path: str,
    reference_sha256: str,
    atol_K: float = GATE_ATOL_K,
    candidate_column: str = "s7_rmse_K",
) -> dict[str, Any]:
    """Compare free full-field RMSE with one non-empty S7 CSV.

    The pass/fail quantity is the macro RMSE. Every per-frame residual is
    returned. An empty reference, a missing key, or a hashless file is a failure.
    """
    if not reference_path or not reference_sha256:
        return {"ok": False, "replay_executed": False, "reason": "missing reference path or hash"}
    if reference is None or reference.empty:
        return {"ok": False, "replay_executed": False, "reason": "empty reference"}
    if candidate is None or candidate.empty:
        return {"ok": False, "replay_executed": False, "reason": "empty candidate"}
    needed = {"sim_id", "frame_index", "rmse_K"}
    if not needed <= set(reference.columns):
        return {"ok": False, "replay_executed": False, "reason": f"reference missing {sorted(needed - set(reference.columns))}"}
    if candidate_column not in candidate.columns:
        return {"ok": False, "replay_executed": False, "reason": f"candidate missing {candidate_column}"}
    ref = reference.copy()
    cand = candidate.copy()
    ref["sim_id"] = ref["sim_id"].astype(str)
    cand["sim_id"] = cand["sim_id"].astype(str)
    ref["frame_index"] = ref["frame_index"].astype(int)
    cand["frame_index"] = cand["frame_index"].astype(int)
    ref_keys = set(zip(ref["sim_id"], ref["frame_index"]))
    cand_keys = set(zip(cand["sim_id"], cand["frame_index"]))
    missing = sorted(ref_keys - cand_keys)
    extra = sorted(cand_keys - ref_keys)
    if ref["rmse_K"].isna().any() or not np.isfinite(ref["rmse_K"].to_numpy(dtype=float)).all():
        return {"ok": False, "replay_executed": False, "reason": "reference RMSE is empty or non-finite",
                "reference_path": reference_path, "reference_sha256": reference_sha256}
    merged = cand.merge(
        ref[["sim_id", "frame_index", "rmse_K"]],
        on=["sim_id", "frame_index"],
        how="inner",
        suffixes=("", "_ref"),
    )
    if merged.empty:
        return {"ok": False, "replay_executed": False, "reason": "no shared keys",
                "n_missing": len(missing), "n_extra": len(extra),
                "reference_path": reference_path, "reference_sha256": reference_sha256}
    delta = merged[candidate_column].to_numpy(dtype=float) - merged["rmse_K"].to_numpy(dtype=float)
    per_frame = merged[["sim_id", "frame_index"]].copy()
    per_frame["candidate_rmse_K"] = merged[candidate_column].to_numpy(dtype=float)
    per_frame["reference_rmse_K"] = merged["rmse_K"].to_numpy(dtype=float)
    per_frame["delta_rmse_K"] = delta
    # Do not rename onto full_rmse_K: the candidate already has that column, and a
    # duplicated name makes groupby.mean() return a Series.
    macro_c = macro_rmse(merged, candidate_column)
    macro_r = macro_rmse(merged, "rmse_K")
    macro_delta = float(macro_c - macro_r)
    ok = abs(macro_delta) <= float(atol_K) and not missing and not extra
    return {
        "ok": bool(ok),
        "replay_executed": True,
        "reason": "pass" if ok else "macro or key mismatch",
        "atol_K": float(atol_K),
        "macro_candidate_K": macro_c,
        "macro_reference_K": macro_r,
        "macro_delta_K": macro_delta,
        "max_abs_frame_delta_K": float(np.max(np.abs(delta))),
        "n_frames": int(len(merged)),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "missing_keys": missing[:20],
        "extra_keys": extra[:20],
        "reference_path": reference_path,
        "reference_sha256": reference_sha256,
        "per_frame": per_frame,
    }
