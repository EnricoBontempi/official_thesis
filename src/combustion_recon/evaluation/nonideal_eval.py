"""Block II evaluation of frozen models on degraded velocity observations.

Pure-Python/pandas/numpy helpers plus one evaluation loop that takes the model forward as a
callable, so everything except the real checkpoints can be tested on CPU.  No weights are
changed, no temperature is used to build any observation, and TEST roles are refused.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from combustion_recon.data.nonideal_observations import (
    DegradationSpec,
    DegradedObservation,
    VelocityScale,
    degrade_observation,
    file_sha256,
)
from combustion_recon.evaluation.month2_evaluator import (
    denormalize_temperature,
    evaluate_frame_prediction,
)
from combustion_recon.training.sparse_dataset import collate_sparse_batch

CELL_ORDER = ("R0", "N1", "G1", "N1G1", "G1-eq", "N2")
EVAL_ROLES = ("seen_train_eval", "val_eval")
REPLICAS = (0, 1, 2)
ROLE_N_FRAMES = {"seen_train_eval": 220, "val_eval": 123}
ROLE_N_CONDITIONS = {"seen_train_eval": 20, "val_eval": 3}
N_MODELS = 12
NOMINAL_PER_MODEL = len(CELL_ORDER) * len(REPLICAS) * sum(ROLE_N_FRAMES.values())  # 6174
NOMINAL_TOTAL = N_MODELS * NOMINAL_PER_MODEL  # 74088
GATE_TOLERANCE_K = 0.05
DIAG_CELL = "R0_bs8"
UNET_BATCH = 8
GLU_BATCH = 1

REGION_COLS = ("rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K", "n")
RENAME_ANCHOR = {
    "anchor_rmse_K": "g1in_rmse_K", "anchor_mae_K": "g1in_mae_K", "anchor_bias_K": "g1in_bias_K",
    "anchor_abs_bias_K": "g1in_abs_bias_K", "anchor_centered_rmse_K": "g1in_centered_rmse_K", "anchor_n": "g1in_n",
    "off_anchor_rmse_K": "g1out_rmse_K", "off_anchor_mae_K": "g1out_mae_K", "off_anchor_bias_K": "g1out_bias_K",
    "off_anchor_abs_bias_K": "g1out_abs_bias_K", "off_anchor_centered_rmse_K": "g1out_centered_rmse_K",
    "off_anchor_n": "g1out_n",
}


class NonidealEvalError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- batching

def batch_positions(n_items: int, family: str, *, diagnostic_s7_batching: bool = False) -> list[list[int]]:
    """Uniform policy per family. GLU: batch 1 (R0 diagnostic replay: batch 8). U-Net: batch 8."""
    if family == "sparse_unet":
        size = UNET_BATCH
    elif family == "adapted_glu":
        size = UNET_BATCH if diagnostic_s7_batching else GLU_BATCH
    else:
        raise NonidealEvalError(f"unknown family {family!r}")
    return [list(range(s, min(s + size, n_items))) for s in range(0, n_items, size)]


# ----------------------------------------------------------------------------- items

def sites_hash(sites: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(sites, dtype=np.int64)).tobytes()).hexdigest()


def degrade_item(ds, item: dict[str, Any], cell, spec: DegradationSpec, scale: VelocityScale,
                 replica_id: int) -> tuple[dict[str, Any], DegradedObservation]:
    """Replace the observation of an already built dataset item by the degraded packet.

    R0 returns the historical item object untouched (bit-identical input). Everything else rebuilds
    the U-Net grid and the GLU tokens from the surviving sites only. ``replica_id`` selects the
    realization (evaluation: mask replica; specialist training: mask epoch).
    """
    meta = item["metadata"]
    obs = item["observation"]
    deg = degrade_observation(
        obs, cell, spec, scale=scale, sim_id=str(meta["sim_id"]), frame_index=int(meta["frame_index"]),
        replica_id=int(replica_id), x_m=ds.x, y_m=ds.y,
    )
    if deg.obs is obs:
        return item, deg
    idx = int(meta["cache_index"])
    grid = deg.obs.to_unet_grid(
        x_scaled=ds.x_scaled, y_scaled=ds.y_scaled,
        nh3_norm=float(ds.p_norm[idx, 0]), phi_norm=float(ds.p_norm[idx, 1]),
        p_conditioning=bool(ds.p_conditioning),
    )
    new = dict(item)
    mi = dict(item["model_inputs"])
    mi["input_tensor"] = torch.from_numpy(np.array(grid, dtype=np.float32, copy=True))
    mi["mask_hash"] = deg.obs.mask_hash
    mi["n_sites"] = deg.obs.n_sites
    new["model_inputs"] = mi
    new["observation"] = deg.obs
    new["glu_tokens"] = deg.obs.to_glu_tokens()
    new["baseline_physical"] = deg.obs.to_baseline_physical()
    nm = dict(meta)
    nm["mask_hash"] = deg.obs.mask_hash
    new["metadata"] = nm
    return new, deg


def degraded_item(ds, i: int, cell, spec: DegradationSpec, scale: VelocityScale) -> tuple[dict[str, Any], DegradedObservation]:
    """Evaluation path: realization = mask replica of the dataset."""
    return degrade_item(ds, ds[i], cell, spec, scale, int(ds.replica_id))


def _check_glu_homogeneous(batch: Mapping[str, Any]) -> None:
    ns = {int(np.asarray(t["site_index"]).size) for t in batch["glu_tokens"]}
    if len(ns) != 1:
        raise NonidealEvalError(f"ragged GLU batch (N={sorted(ns)}); the wrapper needs homogeneous N")
    if 0 in ns:
        raise NonidealEvalError("GLU batch with zero tokens")


def _row_regional(row: dict[str, Any], region_defined: bool) -> None:
    for old, new in RENAME_ANCHOR.items():
        if old in row:
            row[new] = row.pop(old)
    row["region_defined"] = bool(region_defined)
    if not region_defined:
        for suffix in REGION_COLS:
            row[f"g1in_{suffix}"] = 0 if suffix == "n" else float("nan")
            row[f"g1out_{suffix}"] = 0 if suffix == "n" else float("nan")


def evaluate_cell(
    *,
    ds,
    role: str,
    replica_id: int,
    cell,
    spec: DegradationSpec,
    scale: VelocityScale,
    family: str,
    p_conditioning: bool,
    forward_fn: Callable[[dict[str, Any]], np.ndarray],
    normalizer,
    run_tags: Mapping[str, Any],
    diagnostic_s7_batching: bool = False,
    keep_fields_for: Mapping[tuple[str, int], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one (role, replica, cell) for one model. Returns rows, timing, kept fields."""
    if role not in EVAL_ROLES:
        raise NonidealEvalError(f"role {role!r} not allowed")
    ds.replica_id = int(replica_id)
    ds.site_fraction = 0.05
    ds.mask_mode = "evaluation"
    n = len(ds)
    sigma = float(normalizer.temperature_std)
    rows: list[dict[str, Any]] = []
    kept: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    t0 = time.perf_counter()
    for positions in batch_positions(n, family, diagnostic_s7_batching=diagnostic_s7_batching):
        items, degs = zip(*(degraded_item(ds, i, cell, spec, scale) for i in positions))
        batch = collate_sparse_batch(list(items))
        if family == "adapted_glu":
            batch["p_norm_tensor"] = torch.tensor(
                [[float(ds.p_norm[int(m["cache_index"]), 0]), float(ds.p_norm[int(m["cache_index"]), 1])]
                 for m in batch["metadata"]], dtype=torch.float32)
            _check_glu_homogeneous(batch)
        pred_n = forward_fn(batch)
        true_K = batch["target_T_K"].numpy()
        for j, (meta, deg) in enumerate(zip(batch["metadata"], degs)):
            pred_K = denormalize_temperature(pred_n[j, 0], normalizer)
            region = deg.region_mask
            row = evaluate_frame_prediction(
                pred_K, true_K[j, 0], train_sigma_T=sigma, sim_id=str(meta["sim_id"]), split=str(meta.get("split", role)),
                role=role, frame_index=int(meta["frame_index"]), mask_temperature=region,
                method=family, variant="p_on" if p_conditioning else "p_off", p_conditioning=bool(p_conditioning),
                K=0, site_fraction=0.05, mask_replica_id=int(replica_id), layout_id=str(ds.layout_id),
                task_mode="temperature_free", checkpoint_kind="best", run_id=str(run_tags["run_id"]),
            )
            row.pop("_spatial_bias_map", None)
            row["bias_K"] = row["mean_signed_error_K"]
            _row_regional(row, region is not None)
            m = deg.meta
            row.update({
                "cell": cell.cell_id if not diagnostic_s7_batching else DIAG_CELL,
                "replica": int(replica_id),
                "condition": str(meta["sim_id"]),
                "batching": f"{family}:bs{len(positions)}" if not diagnostic_s7_batching else f"{family}:s7_bs{UNET_BATCH}",
                "n_original": m["n_original"], "n_surviving": m["n_surviving"], "n_removed": m["n_removed"],
                "removed_site_fraction": m["removed_site_fraction"],
                "region_cells": m.get("region_cells"),
                "area_fraction_effective": m.get("area_fraction_effective", float("nan")),
                "area_mm2": m.get("area_mm2", float("nan")),
                "n_discs": m.get("n_discs"), "g1_attempt": m.get("attempt"),
                "noise_sigma_ms": m["noise_sigma_ms"] if m["noise_sigma_ms"] is not None else 0.0,
                "original_sites_hash": sites_hash(deg.original_sites),
                "surviving_sites_hash": sites_hash(deg.obs.sites),
                "original_mask_hash": m["original_mask_hash"], "surviving_mask_hash": m["surviving_mask_hash"],
            })
            row.update(run_tags)
            rows.append(row)
            key = (str(meta["sim_id"]), int(meta["frame_index"]))
            if keep_fields_for is not None and key in keep_fields_for and replica_id == 0:
                kept[key] = {
                    "pred_T_K": np.asarray(pred_K, dtype=np.float32),
                    "true_T_K": np.asarray(true_K[j, 0], dtype=np.float32),
                    "kept_sites": np.asarray(deg.obs.sites, dtype=np.int64),
                    "removed_sites": np.asarray(deg.removed_sites, dtype=np.int64),
                    "original_sites": np.asarray(deg.original_sites, dtype=np.int64),
                    "rmse_K_csv": np.asarray(row["rmse_K"], dtype=np.float64),
                    "region_mask": (np.zeros((0, 0), bool) if region is None else np.asarray(region, dtype=bool)),
                }
    return {"rows": rows, "wall_s": time.perf_counter() - t0, "n_frames": len(rows), "kept": kept}


# ----------------------------------------------------------------------------- aggregation primitives

def macro_rmse(frame_rows: pd.DataFrame, col: str = "rmse_K") -> float:
    """frame -> condition (mean over frames) -> uniform mean over conditions."""
    if frame_rows.empty:
        return float("nan")
    per_cond = frame_rows.groupby("condition")[col].mean()
    return float(per_cond.mean())


def gate_against_reference(
    r0_rows: pd.DataFrame, reference: pd.DataFrame, run_id: str, *, tolerance_k: float = GATE_TOLERANCE_K
) -> dict[str, Any]:
    """R0 macro vs the S7 reference table (global_by_replica.csv: best, coverage 0.05)."""
    ref = reference[
        (reference["run_id"] == run_id) & (reference["checkpoint_kind"] == "best")
        & (np.isclose(reference["coverage"].astype(float), 0.05)) & (reference["role"].isin(EVAL_ROLES))
    ]
    out = []
    for role in EVAL_ROLES:
        for rep in REPLICAS:
            sub = r0_rows[(r0_rows["role"] == role) & (r0_rows["replica"] == rep)]
            hit = ref[(ref["role"] == role) & (ref["replica"] == rep)]
            value = macro_rmse(sub) if len(sub) else None
            refv = float(hit["rmse_K"].iloc[0]) if len(hit) == 1 else None
            delta = None if (value is None or refv is None) else value - refv
            ok = bool(delta is not None and math.isfinite(delta) and abs(delta) <= tolerance_k
                      and len(sub) == ROLE_N_FRAMES[role])
            out.append({"run_id": run_id, "role": role, "replica": rep, "n_frames": int(len(sub)),
                        "r0_macro_rmse_K": value, "s7_reference_rmse_K": refv, "delta_K": delta,
                        "reference_present": refv is not None, "within_tolerance": ok})
    return {
        "tolerance_K": tolerance_k, "rows": out,
        "n_expected": len(EVAL_ROLES) * len(REPLICAS),
        "n_within_tolerance": int(sum(r["within_tolerance"] for r in out)),
        "max_abs_delta_K": max((abs(r["delta_K"]) for r in out if r["delta_K"] is not None), default=None),
        "all_pass": bool(len(out) == len(EVAL_ROLES) * len(REPLICAS) and all(r["within_tolerance"] for r in out)),
    }


def per_frame_discrepancy(r0_rows: pd.DataFrame, s7_frame_csv: pd.DataFrame, *, role: str, replica: int) -> pd.DataFrame:
    """Per-frame |RMSE difference| against the S7 per-frame CSV (diagnostic; joins on condition/frame)."""
    a = r0_rows[(r0_rows["role"] == role) & (r0_rows["replica"] == replica)][["condition", "frame_index", "rmse_K"]]
    b = s7_frame_csv.rename(columns={"sim_id": "condition"})[["condition", "frame_index", "rmse_K"]]
    j = a.merge(b, on=["condition", "frame_index"], suffixes=("_now", "_s7"), how="outer", indicator=True)
    j["delta_K"] = j["rmse_K_now"] - j["rmse_K_s7"]
    return j


KEY_COLS = ["run_id", "role", "condition", "frame_index", "replica", "cell"]


def check_run_frame_table(df: pd.DataFrame, *, family: str, expect_diag: bool) -> dict[str, Any]:
    """Completeness/uniqueness of one run's frame table. Raises on any violation."""
    main = df[df["cell"] != DIAG_CELL]
    diag = df[df["cell"] == DIAG_CELL]
    if main.duplicated(KEY_COLS).any() or diag.duplicated(KEY_COLS).any():
        raise NonidealEvalError("duplicate frame keys")
    if len(main) != NOMINAL_PER_MODEL:
        raise NonidealEvalError(f"{len(main)} nominal rows != {NOMINAL_PER_MODEL}")
    if set(main["cell"]) != set(CELL_ORDER):
        raise NonidealEvalError(f"cells {sorted(set(main['cell']))}")
    for role in EVAL_ROLES:
        sub = main[main["role"] == role]
        if len(sub) != len(CELL_ORDER) * len(REPLICAS) * ROLE_N_FRAMES[role]:
            raise NonidealEvalError(f"{role}: {len(sub)} rows")
        if sub["condition"].nunique() != ROLE_N_CONDITIONS[role]:
            raise NonidealEvalError(f"{role}: {sub['condition'].nunique()} conditions")
    # same frame set for every cell and replica: inputs comparable across cells
    frames = main.groupby(["cell", "replica", "role"])[["condition", "frame_index"]].apply(
        lambda g: hashlib.sha256("|".join(sorted(f"{c}:{f}" for c, f in zip(g["condition"], g["frame_index"]))).encode()).hexdigest()
    )
    if frames.groupby(level=["replica", "role"]).nunique().max() != 1:
        raise NonidealEvalError("frame sets differ between cells")
    # same original sites for every cell
    osh = main.groupby(["role", "condition", "frame_index", "replica"])["original_sites_hash"].nunique()
    if osh.max() != 1:
        raise NonidealEvalError("original sites differ between cells for the same frame")
    if not np.isfinite(main[["rmse_K", "mae_K", "bias_K"]].to_numpy(dtype=float)).all():
        raise NonidealEvalError("non-finite global metrics")
    n_diag = len(diag)
    if expect_diag != (n_diag > 0) or (expect_diag and n_diag != len(REPLICAS) * sum(ROLE_N_FRAMES.values())):
        raise NonidealEvalError(f"diagnostic replay rows {n_diag}")
    # structural invariants of the operators
    r0 = main[main["cell"] == "R0"]
    if (r0["n_removed"] != 0).any() or (r0["n_surviving"] != r0["n_original"]).any():
        raise NonidealEvalError("R0 removed sites")
    for c in ("N1", "N2"):
        s = main[main["cell"] == c]
        if (s["n_removed"] != 0).any():
            raise NonidealEvalError(f"{c} removed sites")
    g1 = main[main["cell"] == "G1"].set_index(["role", "condition", "frame_index", "replica"])
    for c in ("N1G1", "G1-eq"):
        o = main[main["cell"] == c].set_index(["role", "condition", "frame_index", "replica"])
        if not (o["n_removed"].reindex(g1.index) == g1["n_removed"]).all():
            raise NonidealEvalError(f"{c} removed count differs from G1")
    n1g1 = main[main["cell"] == "N1G1"].set_index(["role", "condition", "frame_index", "replica"])
    if not (n1g1["surviving_sites_hash"].reindex(g1.index) == g1["surviving_sites_hash"]).all():
        raise NonidealEvalError("N1G1 survivors differ from G1")
    if family == "adapted_glu" and (main["batching"] != "adapted_glu:bs1").any():
        raise NonidealEvalError("GLU nominal rows are not batch 1")
    return {"n_nominal": int(len(main)), "n_diagnostic": int(n_diag), "ok": True}


# ----------------------------------------------------------------------------- analysis tables

METRICS = ("rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K")


def metric_columns(df: pd.DataFrame) -> list[str]:
    cand = list(METRICS) + [c for c in df.columns if c.startswith(("jet_", "high_", "peak_", "g1in_", "g1out_"))]
    return [c for c in dict.fromkeys(cand) if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and (not c.endswith("_n") or c.startswith("g1"))]


def per_condition_table(frames: pd.DataFrame) -> pd.DataFrame:
    """frame -> condition: mean over frames, NaN-aware (regional NaN excluded, count reported)."""
    keys = ["run_id", "family", "variant", "model_seed", "cell", "role", "replica", "condition"]
    cols = metric_columns(frames)
    g = frames.groupby(keys, dropna=False)
    out = g[cols].mean().reset_index()
    out["n_frames"] = g.size().values
    return out


def macro_table(percond: pd.DataFrame) -> pd.DataFrame:
    """condition -> uniform mean over conditions, per replica."""
    keys = ["run_id", "family", "variant", "model_seed", "cell", "role", "replica"]
    cols = [c for c in percond.columns if c not in keys + ["condition", "n_frames"]]
    out = percond.groupby(keys, dropna=False)[cols].mean().reset_index()
    out["n_conditions"] = percond.groupby(keys, dropna=False)["condition"].nunique().values
    return out


def delta_vs_r0(table: pd.DataFrame, keys: Sequence[str], cols: Sequence[str]) -> pd.DataFrame:
    """scenario - R0 of the SAME model (same keys other than cell). Absolute and relative (rmse-like only)."""
    r0 = table[table["cell"] == "R0"].set_index(list(keys))[list(cols)]
    cur = table[table["cell"] != "R0"].copy()
    ref = r0.reindex(pd.MultiIndex.from_frame(cur[list(keys)]))
    if ref.isna().all(axis=1).any():
        raise NonidealEvalError("missing R0 reference for a scenario row")
    for c in cols:
        cur[f"d_{c}"] = cur[c].to_numpy() - ref[c].to_numpy()
        if c in ("rmse_K", "mae_K", "centered_rmse_K", "abs_bias_K"):
            cur[f"rel_{c}"] = cur[f"d_{c}"] / ref[c].to_numpy()
    return cur


def seed_summary(df: pd.DataFrame, value_cols: Sequence[str], group: Sequence[str]) -> pd.DataFrame:
    """mean/min/max over seeds of per-seed values; individual seeds stay in their own table."""
    g = df.groupby(list(group))[list(value_cols)]
    parts = []
    for name, fn in (("mean", "mean"), ("min", "min"), ("max", "max")):
        t = getattr(g, fn)().add_suffix(f"__{name}")
        parts.append(t)
    out = pd.concat(parts, axis=1).reset_index()
    out["n_seeds"] = df.groupby(list(group))["model_seed"].nunique().values
    return out


def pon_poff_contrast(per_model_delta: pd.DataFrame, value: str) -> pd.DataFrame:
    """For each family, seed index, cell, role: deterioration with p_on, with p_off and their difference.

    Seeds of p_on and p_off with the same index have different initializations; the pairing is
    descriptive only (as in the S7 package).
    """
    keys = ["family", "model_seed", "cell", "role"]
    d = per_model_delta.groupby(keys + ["variant"])[value].mean().unstack("variant").reset_index()
    d = d.rename(columns={"p_on": f"{value}_p_on", "p_off": f"{value}_p_off"})
    d[f"{value}_p_on_minus_p_off"] = d[f"{value}_p_on"] - d[f"{value}_p_off"]
    return d


def structure_contrasts(macro: pd.DataFrame, value: str = "rmse_K") -> pd.DataFrame:
    """G1 vs G1-eq, N1G1 vs singles and interaction, per model/role (mean over replicas)."""
    m = macro.groupby(["run_id", "family", "variant", "model_seed", "role", "cell"])[value].mean().unstack("cell")
    out = pd.DataFrame(index=m.index)
    out["G1_minus_G1eq"] = m["G1"] - m["G1-eq"]
    out["N1G1_minus_G1"] = m["N1G1"] - m["G1"]
    out["N1G1_minus_N1"] = m["N1G1"] - m["N1"]
    out["interaction_N1G1_N1_G1_R0"] = m["N1G1"] - m["N1"] - m["G1"] + m["R0"]
    out["N2_minus_N1"] = m["N2"] - m["N1"]
    return out.reset_index()
