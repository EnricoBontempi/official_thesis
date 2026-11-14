#!/usr/bin/env python3
"""Step 1.3 — Pilot trajectory deep audit (full 128×128, no spatial subsample).

Loads the three downloaded numerical Arrow shards under
data/raw/combustion/hf_dataset/numerical/, inspects schema / coordinates /
channel ranges / velocity magnitude consistency / data integrity / temporal
autocorrelation, writes manifests, and saves diagnostic figures.

Usage:
    python scripts/02_audit_pilot_trajectories.py
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import torch
import yaml
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
RAW_NUM = ROOT / "data" / "raw" / "combustion" / "hf_dataset" / "numerical"
META = ROOT / "data" / "manifests" / "hf_small_metadata"
MANIFESTS = ROOT / "data" / "manifests"
FIGDIR = ROOT / "results" / "figures" / "month1_audit"
CHANNELS_PATH = META / "combustion__channels.json"
REMAIN_PATH = META / "combustion__remain_params_numerical.json"
MEAN_STD_PATH = ROOT / "data" / "raw" / "combustion" / "mean_std.pt"
PILOT_MAP_PATH = MANIFESTS / "pilot_shard_map.json"

CHANNEL_NAMES = [
    "Absolute_Pressure",
    "Chemistry_Heat_Release_Rate",
    "Mole_Fraction_of_CH4",
    "Mole_Fraction_of_CO",
    "Mole_Fraction_of_CO2",
    "Mole_Fraction_of_H2O",
    "Mole_Fraction_of_NH2",
    "Mole_Fraction_of_NH3",
    "Mole_Fraction_of_OH",
    "Pressure",
    "Temperature",
    "Velocity[i]",
    "Velocity[j]",
    "Velocity[k]",
    "Velocity_Magnitude",
]

T_FRAME = 1000  # instantaneous snapshot index for fig02
AUTOCORR_MAX_LAG = 500


@dataclass
class PilotTrajectory:
    sim_id: str
    shard: str
    nh3_pct: int
    ch4_pct: int
    phi: float
    observed: np.ndarray  # (T,H,W) float32
    numerical: np.ndarray  # (T,H,W,15) float32
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    shape_t: int
    shape_h: int
    shape_w: int
    numerical_channels: int
    x_shape_meta: Any = None
    y_shape_meta: Any = None
    t_shape_meta: Any = None


def _decode_bytes(buf: bytes, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=dtype)
    if arr.size != int(np.prod(shape)):
        # try float64 for coordinate grids
        arr64 = np.frombuffer(buf, dtype=np.float64)
        if arr64.size == int(np.prod(shape)):
            return arr64.reshape(shape)
        raise ValueError(f"Cannot reshape buffer of {arr.size} float32 / {arr64.size} float64 into {shape}")
    return arr.reshape(shape)


def load_arrow_trajectory(path: Path) -> dict[str, Any]:
    """Load a single-row HF Arrow shard as a dict of decoded arrays.

    RealPDEBench shards are Arrow *IPC streams* (not IPC files). Prefer
    ``open_stream``; fall back to ``Dataset.from_file`` if needed.
    """
    table = None
    try:
        with pa.memory_map(str(path), "r") as mm:
            table = pa.ipc.open_stream(mm).read_all()
    except Exception as err_stream:
        try:
            from datasets import Dataset

            table = Dataset.from_file(str(path)).data.table
        except Exception as err_ds:
            raise RuntimeError(
                f"Failed to read {path} as Arrow stream ({err_stream}) "
                f"or datasets.Dataset ({err_ds})"
            ) from err_ds
    if table.num_rows != 1:
        warnings.warn(f"{path.name}: expected 1 row, got {table.num_rows}; using row 0")
    row = {col: table.column(col)[0].as_py() for col in table.column_names}

    shape_t = int(row["shape_t"])
    shape_h = int(row["shape_h"])
    shape_w = int(row["shape_w"])
    nchan = int(row.get("numerical_channels", 15))

    observed = _decode_bytes(row["observed"], (shape_t, shape_h, shape_w), np.float32)
    numerical = _decode_bytes(row["numerical"], (shape_t, shape_h, shape_w, nchan), np.float32)

    # Coordinate arrays: try documented (H,W), then (H,), then (H*W,)
    def decode_xy(key: str, shape_key: str | None = None) -> np.ndarray:
        buf = row[key]
        meta_shape = row.get(shape_key) if shape_key else None
        candidates = []
        if meta_shape is not None:
            candidates.append((int(meta_shape),))
        candidates.extend([(shape_h, shape_w), (shape_h,), (shape_w,), (shape_h * shape_w,)])
        last_err = None
        for shp in candidates:
            try:
                return _decode_bytes(buf, shp)
            except Exception as e:
                last_err = e
        raise ValueError(f"Failed to decode {key}: {last_err}")

    def decode_t() -> np.ndarray:
        buf = row["t"]
        meta = row.get("t_shape")
        for shp in ([ (int(meta),) ] if meta is not None else []) + [(shape_t,), (shape_t, 1)]:
            try:
                return _decode_bytes(buf, shp).reshape(-1)
            except Exception:
                continue
        # fallback: float64
        return np.frombuffer(buf, dtype=np.float64)

    x = decode_xy("x", "x_shape")
    y = decode_xy("y", "y_shape")
    t = decode_t()

    return {
        "sim_id": row["sim_id"],
        "keys": list(row.keys()),
        "observed": observed,
        "numerical": numerical,
        "x": x,
        "y": y,
        "t": t,
        "shape_t": shape_t,
        "shape_h": shape_h,
        "shape_w": shape_w,
        "numerical_channels": nchan,
        "x_shape_meta": row.get("x_shape"),
        "y_shape_meta": row.get("y_shape"),
        "t_shape_meta": row.get("t_shape"),
        "raw_key_types": {k: type(v).__name__ for k, v in row.items()},
    }


def channel_stats(arr: np.ndarray) -> dict[str, float]:
    flat = arr.reshape(-1).astype(np.float64)
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "q01": float(np.quantile(flat, 0.01)),
        "q50": float(np.quantile(flat, 0.50)),
        "q99": float(np.quantile(flat, 0.99)),
    }


def spatial_mean_series(field: np.ndarray) -> np.ndarray:
    """field: (T,H,W) -> mean over space -> (T,)"""
    return field.reshape(field.shape[0], -1).mean(axis=1)


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    x = series.astype(np.float64)
    x = x - x.mean()
    var = np.dot(x, x)
    if var <= 0:
        return np.zeros(max_lag + 1, dtype=np.float64)
    r = np.empty(max_lag + 1, dtype=np.float64)
    r[0] = 1.0
    for lag in range(1, max_lag + 1):
        r[lag] = np.dot(x[:-lag], x[lag:]) / var
    return r


def integral_timescale(r: np.ndarray, dt: float) -> float:
    """Integrate positive lobe of R(τ) until first zero crossing (or end)."""
    crossing = np.where(r[1:] <= 0)[0]
    end = int(crossing[0] + 1) if len(crossing) else len(r) - 1
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(r[: end + 1], dx=dt))


def integrity_scan(num: np.ndarray, obs: np.ndarray) -> dict[str, Any]:
    T = num[:, :, :, 10]
    species = num[:, :, :, 2:9]  # mole fractions CH4..OH
    report = {
        "nan_numerical": int(np.isnan(num).sum()),
        "inf_numerical": int(np.isinf(num).sum()),
        "nan_observed": int(np.isnan(obs).sum()),
        "inf_observed": int(np.isinf(obs).sum()),
        "negative_temperature_count": int((T < 0).sum()),
        "negative_mole_fraction_count": int((species < -1e-6).sum()),
        "mole_fraction_above_one_count": int((species > 1.0 + 1e-3).sum()),
        "temperature_min": float(T.min()),
        "temperature_max": float(T.max()),
    }
    # Duplicate frame check (hash a few bytes of consecutive frames)
    diffs = np.diff(num.reshape(num.shape[0], -1).astype(np.float64), axis=0)
    max_abs = np.max(np.abs(diffs), axis=1)
    report["identical_consecutive_frames"] = int(np.sum(max_abs == 0))
    report["near_identical_consecutive_frames_1e-12"] = int(np.sum(max_abs < 1e-12))
    return report


def infer_orientation(traj: PilotTrajectory) -> dict[str, Any]:
    """Heuristics for burner location and velocity component roles."""
    T = traj.numerical[:, :, :, 10]
    Vi = traj.numerical[:, :, :, 11]
    Vj = traj.numerical[:, :, :, 12]
    Vk = traj.numerical[:, :, :, 13]
    T_mean = T.mean(axis=0)
    Vi_mean = Vi.mean(axis=0)
    Vj_mean = Vj.mean(axis=0)
    Vk_mean = Vk.mean(axis=0)
    mag_mean = traj.numerical[:, :, :, 14].mean(axis=0)

    H, W = T_mean.shape
    # Edge mean temperatures / speeds to locate hot jet and inlet
    edges = {
        "bottom_rows": float(T_mean[: max(2, H // 16), :].mean()),
        "top_rows": float(T_mean[-max(2, H // 16) :, :].mean()),
        "left_cols": float(T_mean[:, : max(2, W // 16)].mean()),
        "right_cols": float(T_mean[:, -max(2, W // 16) :].mean()),
        "center": float(T_mean[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4].mean()),
    }
    speed = {
        "bottom": float(mag_mean[: max(2, H // 16), :].mean()),
        "top": float(mag_mean[-max(2, H // 16) :, :].mean()),
        "left": float(mag_mean[:, : max(2, W // 16)].mean()),
        "right": float(mag_mean[:, -max(2, W // 16) :].mean()),
        "center": float(mag_mean[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4].mean()),
    }

    # Component means and |mean| to identify dominant axial jet
    comp_stats = {}
    for name, field in [("i", Vi_mean), ("j", Vj_mean), ("k", Vk_mean)]:
        comp_stats[name] = {
            "mean": float(field.mean()),
            "abs_mean": float(np.abs(field).mean()),
            "std": float(field.std()),
            "max_abs": float(np.max(np.abs(field))),
            "bottom_mean": float(field[: max(2, H // 16), :].mean()),
            "top_mean": float(field[-max(2, H // 16) :, :].mean()),
            "center_mean": float(field[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4].mean()),
        }

    # Coordinate extents
    x, y = traj.x, traj.y
    if x.ndim == 1 and y.ndim == 1:
        xx, yy = np.meshgrid(x, y, indexing="xy")
        coord_note = "1D x,y broadcast to meshgrid(indexing='xy')"
    elif x.ndim == 2:
        xx, yy = x, y
        coord_note = "2D x,y grids"
    else:
        xx = x.reshape(H, W)
        yy = y.reshape(H, W)
        coord_note = "reshaped to (H,W)"

    dx = np.diff(xx, axis=1)
    dy = np.diff(yy, axis=0)

    return {
        "edge_mean_temperature": edges,
        "edge_mean_speed": speed,
        "component_stats": comp_stats,
        "coord_note": coord_note,
        "x_min": float(np.min(xx)),
        "x_max": float(np.max(xx)),
        "y_min": float(np.min(yy)),
        "y_max": float(np.max(yy)),
        "dx_median": float(np.median(dx)) if dx.size else None,
        "dy_median": float(np.median(dy)) if dy.size else None,
        "x_increases_along_axis1": bool(np.median(dx) > 0) if dx.size else None,
        "y_increases_along_axis0": bool(np.median(dy) > 0) if dy.size else None,
        "xx": xx,
        "yy": yy,
    }


def load_mean_std(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    out: dict[str, Any] = {"type": type(obj).__name__}
    if isinstance(obj, (list, tuple)):
        out["n_items"] = len(obj)
        labels = ["mean_inputs", "mean_targets", "std_inputs", "std_targets"]
        for i, item in enumerate(obj):
            key = labels[i] if i < len(labels) else f"item_{i}"
            if torch.is_tensor(item):
                out[key] = {
                    "shape": list(item.shape),
                    "dtype": str(item.dtype),
                    "values": item.detach().cpu().numpy().tolist(),
                }
            else:
                out[key] = repr(item)
    elif isinstance(obj, dict):
        out["keys"] = list(obj.keys())
        for k, v in obj.items():
            if torch.is_tensor(v):
                out[k] = {
                    "shape": list(v.shape),
                    "values": v.detach().cpu().numpy().tolist(),
                }
            else:
                out[k] = repr(v)
    else:
        out["repr"] = repr(obj)
    return out


def mesh_for_plot(traj: PilotTrajectory) -> tuple[np.ndarray, np.ndarray]:
    x, y = traj.x, traj.y
    H, W = traj.shape_h, traj.shape_w
    if x.ndim == 1 and y.ndim == 1:
        if x.size == W and y.size == H:
            xx, yy = np.meshgrid(x, y, indexing="xy")
        elif x.size == H and y.size == W:
            xx, yy = np.meshgrid(y, x, indexing="xy")
        else:
            xx, yy = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    elif x.ndim == 2:
        xx, yy = x, y
    else:
        xx = x.reshape(H, W)
        yy = y.reshape(H, W)
    return xx, yy


def style_axes(ax, xx, yy, title: str, cbar_label: str | None = None):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    remain = json.loads(REMAIN_PATH.read_text())
    channels_doc = json.loads(CHANNELS_PATH.read_text()) if CHANNELS_PATH.exists() else {}
    pilot_map = json.loads(PILOT_MAP_PATH.read_text()) if PILOT_MAP_PATH.exists() else {}

    # Discover pilot shards present on disk
    shard_files = sorted(RAW_NUM.glob("data-*-of-00030.arrow"))
    if len(shard_files) < 1:
        raise FileNotFoundError(f"No pilot arrow shards found in {RAW_NUM}")

    trajectories: list[PilotTrajectory] = []
    schema_reports: list[dict[str, Any]] = []
    integrity_reports: dict[str, Any] = {}
    stats_reports: dict[str, Any] = {}
    orientation_reports: dict[str, Any] = {}
    autocorr_reports: dict[str, Any] = {}
    vel_mag_checks: dict[str, Any] = {}

    print(f"Found {len(shard_files)} local numerical shards")
    for path in shard_files:
        print(f"\n=== Loading {path.name} ===")
        raw = load_arrow_trajectory(path)
        sim_id = raw["sim_id"]
        nh3, phi = remain.get(sim_id, [None, None])
        if nh3 is None:
            # parse from name
            import re

            m = re.match(r"(\d+)NH3_(\d+\.?\d*)\.h5", sim_id)
            nh3, phi = (int(m.group(1)), float(m.group(2))) if m else (-1, float("nan"))
        traj = PilotTrajectory(
            sim_id=sim_id,
            shard=path.name,
            nh3_pct=int(nh3),
            ch4_pct=100 - int(nh3),
            phi=float(phi),
            observed=raw["observed"],
            numerical=raw["numerical"],
            x=raw["x"],
            y=raw["y"],
            t=raw["t"],
            shape_t=raw["shape_t"],
            shape_h=raw["shape_h"],
            shape_w=raw["shape_w"],
            numerical_channels=raw["numerical_channels"],
            x_shape_meta=raw["x_shape_meta"],
            y_shape_meta=raw["y_shape_meta"],
            t_shape_meta=raw["t_shape_meta"],
        )
        trajectories.append(traj)

        # Schema
        dt = np.diff(traj.t)
        schema = {
            "sim_id": sim_id,
            "shard": path.name,
            "keys": raw["keys"],
            "raw_key_types": raw["raw_key_types"],
            "observed_shape": list(traj.observed.shape),
            "observed_dtype": str(traj.observed.dtype),
            "numerical_shape": list(traj.numerical.shape),
            "numerical_dtype": str(traj.numerical.dtype),
            "x_shape": list(traj.x.shape),
            "x_dtype": str(traj.x.dtype),
            "y_shape": list(traj.y.shape),
            "y_dtype": str(traj.y.dtype),
            "t_shape": list(traj.t.shape),
            "t_dtype": str(traj.t.dtype),
            "x_shape_meta": traj.x_shape_meta,
            "y_shape_meta": traj.y_shape_meta,
            "t_shape_meta": traj.t_shape_meta,
            "t_min": float(traj.t.min()),
            "t_max": float(traj.t.max()),
            "dt_median": float(np.median(dt)) if dt.size else None,
            "dt_unique_count": int(np.unique(np.round(dt, decimals=12)).size) if dt.size else None,
        }
        schema_reports.append(schema)
        print(json.dumps({k: schema[k] for k in schema if k not in ("keys", "raw_key_types")}, indent=2))

        # Stats per channel
        ch_stats = {"observed": channel_stats(traj.observed)}
        for c, name in enumerate(CHANNEL_NAMES):
            ch_stats[f"{c:02d}_{name}"] = channel_stats(traj.numerical[:, :, :, c])
        stats_reports[sim_id] = ch_stats

        # Velocity magnitude consistency
        vi, vj, vk = traj.numerical[..., 11], traj.numerical[..., 12], traj.numerical[..., 13]
        mag = traj.numerical[..., 14]
        mag_calc = np.sqrt(vi * vi + vj * vj + vk * vk)
        abs_err = np.abs(mag - mag_calc)
        vel_mag_checks[sim_id] = {
            "max_abs_err": float(abs_err.max()),
            "mean_abs_err": float(abs_err.mean()),
            "rel_err_mean": float((abs_err / (mag_calc + 1e-12)).mean()),
            "allclose_rtol_1e-5_atol_1e-6": bool(np.allclose(mag, mag_calc, rtol=1e-5, atol=1e-6)),
        }
        print("Velocity magnitude check:", vel_mag_checks[sim_id])

        # Integrity
        integrity_reports[sim_id] = integrity_scan(traj.numerical, traj.observed)
        print("Integrity:", integrity_reports[sim_id])

        # Orientation
        orient = infer_orientation(traj)
        # drop large arrays from JSON dump
        orient_json = {k: v for k, v in orient.items() if k not in ("xx", "yy")}
        orientation_reports[sim_id] = orient_json
        print("Orientation heuristics:", json.dumps(orient_json, indent=2))

        # Autocorrelation on spatially averaged series
        dt_med = schema["dt_median"] or 2.5e-4
        T_series = spatial_mean_series(traj.numerical[..., 10])
        Vi_series = spatial_mean_series(traj.numerical[..., 11])
        Vj_series = spatial_mean_series(traj.numerical[..., 12])
        Vk_series = spatial_mean_series(traj.numerical[..., 13])
        Mag_series = spatial_mean_series(traj.numerical[..., 14])
        rT = autocorrelation(T_series, AUTOCORR_MAX_LAG)
        rMag = autocorrelation(Mag_series, AUTOCORR_MAX_LAG)
        # also pointwise center-pixel autocorr as a local check
        cy, cx = traj.shape_h // 2, traj.shape_w // 2
        rT_center = autocorrelation(traj.numerical[:, cy, cx, 10], AUTOCORR_MAX_LAG)
        autocorr_reports[sim_id] = {
            "dt": dt_med,
            "tau_c_T_spatial_mean_s": integral_timescale(rT, dt_med),
            "tau_c_Mag_spatial_mean_s": integral_timescale(rMag, dt_med),
            "tau_c_T_center_s": integral_timescale(rT_center, dt_med),
            "lag_T_R_below_1e-1_frames": int(np.argmax(rT < 0.1)) if np.any(rT < 0.1) else None,
            "lag_T_R_below_1e-2_frames": int(np.argmax(rT < 0.01)) if np.any(rT < 0.01) else None,
            "rT": rT,
            "rMag": rMag,
            "rVi": autocorrelation(Vi_series, AUTOCORR_MAX_LAG),
            "rVj": autocorrelation(Vj_series, AUTOCORR_MAX_LAG),
            "rVk": autocorrelation(Vk_series, AUTOCORR_MAX_LAG),
            "rT_center": rT_center,
        }
        print(
            f"Autocorr tau_c T={autocorr_reports[sim_id]['tau_c_T_spatial_mean_s']:.6f}s, "
            f"Mag={autocorr_reports[sim_id]['tau_c_Mag_spatial_mean_s']:.6f}s"
        )

    # mean_std.pt
    mean_std_info = None
    if MEAN_STD_PATH.exists():
        mean_std_info = load_mean_std(MEAN_STD_PATH)
        print("\nmean_std.pt:", json.dumps(mean_std_info, indent=2)[:2000])
    else:
        print("WARNING: mean_std.pt not found at", MEAN_STD_PATH)

    # ---------- operating_conditions.yaml ----------
    sim_ids_sorted = sorted(remain.keys())
    hypothesized = {
        f"data-{i:05d}-of-00030.arrow": sid for i, sid in enumerate(sim_ids_sorted)
    }
    # Correct mapping from actually loaded pilots
    verified_map = {t.shard: t.sim_id for t in trajectories}
    conditions = []
    pilot_sim_ids = {t.sim_id for t in trajectories}
    for i, sid in enumerate(sim_ids_sorted):
        nh3, phi = remain[sid]
        shard = f"data-{i:05d}-of-00030.arrow"
        # If hypothesis failed for any pilot, mark shard as hypothesized only
        status = "not_downloaded"
        if shard in verified_map:
            status = "pilot_downloaded_verified"
            if verified_map[shard] != sid:
                shard_actual = next(t.shard for t in trajectories if t.sim_id == sid)
                shard = shard_actual
        conditions.append(
            {
                "sim_id": sid,
                "nh3_pct": int(nh3),
                "ch4_pct": 100 - int(nh3),
                "equivalence_ratio": float(phi),
                "hypothesized_shard_lexicographic": f"data-{i:05d}-of-00030.arrow",
                "shard_filename": verified_map and next(
                    (t.shard for t in trajectories if t.sim_id == sid),
                    f"data-{i:05d}-of-00030.arrow",
                ),
                "pilot_status": "pilot" if sid in pilot_sim_ids else "pending_full_download",
                "n_frames_expected": 2001,
                "grid_expected": [128, 128],
            }
        )

    # Fix shard_filename for non-pilots to hypothesized; for pilots use verified
    for c in conditions:
        if c["sim_id"] in pilot_sim_ids:
            c["shard_filename"] = next(t.shard for t in trajectories if t.sim_id == c["sim_id"])
            c["pilot_status"] = "pilot_downloaded"
        else:
            c["shard_filename"] = c["hypothesized_shard_lexicographic"]
            c["pilot_status"] = "pending_full_download"

    oc_doc = {
        "dataset": "RealPDEBench / combustion / numerical",
        "source_repo": "AI4Science-WestlakeU/RealPDEBench",
        "n_operating_conditions": len(conditions),
        "channel_names": CHANNEL_NAMES,
        "channels_doc_note": channels_doc.get("note"),
        "shard_mapping_note": (
            "Shard filenames for non-pilot conditions are hypothesized as "
            "lexicographic order of sim_id (sorted); pilot rows were verified by "
            "reading sim_id from the Arrow shard."
        ),
        "lexicographic_hypothesis_ok": all(
            hypothesized.get(t.shard) == t.sim_id for t in trajectories
        ),
        "pilot_sim_ids": sorted(pilot_sim_ids),
        "conditions": conditions,
        "pilot_temperature_summary": {
            t.sim_id: {
                "min": stats_reports[t.sim_id]["10_Temperature"]["min"],
                "mean": stats_reports[t.sim_id]["10_Temperature"]["mean"],
                "max": stats_reports[t.sim_id]["10_Temperature"]["max"],
            }
            for t in trajectories
        },
    }
    oc_path = MANIFESTS / "operating_conditions.yaml"
    with oc_path.open("w") as f:
        yaml.safe_dump(oc_doc, f, sort_keys=False)
    print(f"Wrote {oc_path}")

    # Audit JSON (arrays trimmed)
    audit_json = {
        "schema_reports": schema_reports,
        "stats_reports": stats_reports,
        "integrity_reports": integrity_reports,
        "velocity_magnitude_checks": vel_mag_checks,
        "orientation_reports": orientation_reports,
        "autocorr_summary": {
            sid: {k: v for k, v in rep.items() if not k.startswith("r")}
            for sid, rep in autocorr_reports.items()
        },
        "mean_std_pt": mean_std_info,
        "pilot_map_file": pilot_map,
    }
    audit_path = MANIFESTS / "pilot_audit_summary.json"
    audit_path.write_text(json.dumps(audit_json, indent=2) + "\n")
    print(f"Wrote {audit_path}")

    # ===================== FIGURES =====================
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )

    # fig01 — parameter space
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    all_nh3 = [c["nh3_pct"] for c in conditions]
    all_phi = [c["equivalence_ratio"] for c in conditions]
    ax.scatter(all_nh3, all_phi, s=36, c="#4c566a", alpha=0.85, label="All 30 conditions", zorder=2)
    for t in trajectories:
        ax.scatter(
            [t.nh3_pct],
            [t.phi],
            s=120,
            marker="*",
            zorder=3,
            label=f"Pilot: {t.sim_id}",
        )
        ax.annotate(t.sim_id.replace(".h5", ""), (t.nh3_pct, t.phi), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel(r"NH$_3$ fuel fraction (%)")
    ax.set_ylabel(r"Equivalence ratio $\phi$")
    ax.set_title("RealPDEBench Combustion — 30 operating conditions")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig01_parameter_space_30_conditions.png")
    plt.close(fig)

    # fig02 — instantaneous fields
    field_specs = [
        (10, "Temperature"),
        (11, "Velocity[i]"),
        (12, "Velocity[j]"),
        (13, "Velocity[k]"),
        (14, "Velocity_Magnitude"),
        (1, "Chemistry_Heat_Release_Rate"),
    ]
    n_rows = len(trajectories)
    n_cols = len(field_specs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.5 * n_rows), squeeze=False)
    for r, traj in enumerate(trajectories):
        xx, yy = mesh_for_plot(traj)
        frame = min(T_FRAME, traj.shape_t - 1)
        for c, (ch, name) in enumerate(field_specs):
            ax = axes[r][c]
            data = traj.numerical[frame, :, :, ch]
            if "Velocity[i]" in name or "Velocity[j]" in name or "Velocity[k]" in name:
                lim = np.nanpercentile(np.abs(data), 99)
                norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim) if lim > 0 else None
                im = ax.pcolormesh(xx, yy, data, shading="auto", cmap="RdBu_r", norm=norm)
            elif name == "Temperature":
                im = ax.pcolormesh(xx, yy, data, shading="auto", cmap="inferno")
            else:
                im = ax.pcolormesh(xx, yy, data, shading="auto", cmap="magma")
            style_axes(ax, xx, yy, f"{traj.sim_id}\n{name}" if r == 0 else name)
            if c == 0:
                ax.set_ylabel(f"{traj.sim_id}\ny")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Instantaneous fields at frame t_index={T_FRAME}", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig02_pilot_instantaneous_fields.png", bbox_inches="tight")
    plt.close(fig)

    # fig03 — mean and std
    metrics = [
        ("T_mean", 10, "mean", "Temperature mean"),
        ("T_std", 10, "std", "Temperature std"),
        ("Mag_mean", 14, "mean", "|U| mean"),
        ("Mag_std", 14, "std", "|U| std"),
        ("Vi_mean", 11, "mean", "Velocity[i] mean"),
        ("Vj_mean", 12, "mean", "Velocity[j] mean"),
    ]
    fig, axes = plt.subplots(n_rows, len(metrics), figsize=(2.5 * len(metrics), 2.4 * n_rows), squeeze=False)
    for r, traj in enumerate(trajectories):
        xx, yy = mesh_for_plot(traj)
        for c, (_, ch, how, title) in enumerate(metrics):
            ax = axes[r][c]
            field = traj.numerical[..., ch]
            data = field.mean(axis=0) if how == "mean" else field.std(axis=0)
            if "Velocity[i]" in title or "Velocity[j]" in title:
                lim = np.nanpercentile(np.abs(data), 99)
                norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim) if lim > 0 else None
                im = ax.pcolormesh(xx, yy, data, shading="auto", cmap="RdBu_r", norm=norm)
            else:
                cmap = "inferno" if "Temperature" in title else "viridis"
                im = ax.pcolormesh(xx, yy, data, shading="auto", cmap=cmap)
            ax.set_title(f"{traj.sim_id}\n{title}" if r == 0 else title, fontsize=8)
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Time-averaged mean and fluctuation intensity (all 2001 frames)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig03_pilot_mean_and_std_fields.png", bbox_inches="tight")
    plt.close(fig)

    # fig04 — histograms + autocorr
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    for traj in trajectories:
        T = traj.numerical[..., 10].reshape(-1)
        # subsample for histogram speed
        sample = T[:: max(1, T.size // 500_000)]
        ax.hist(sample, bins=80, density=True, histtype="step", linewidth=1.6, label=traj.sim_id)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("PDF")
    ax.set_title("(a) Temperature histograms (spatial–temporal)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for traj in trajectories:
        T = traj.numerical[..., 10]
        # marginal along presumed axial direction: mean over x, vs y index
        ax.plot(T.mean(axis=(0, 2)), label=f"{traj.sim_id} ⟨T⟩_x,t(y)")
    ax.set_xlabel("grid index along axis-0 (y rows)")
    ax.set_ylabel("Mean temperature")
    ax.set_title("(a2) Axial temperature profile (diagnostic)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for traj in trajectories:
        r = autocorr_reports[traj.sim_id]["rT"]
        lags = np.arange(len(r))
        ax.plot(lags, r, label=f"T {traj.sim_id}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"lag $\tau$ [frames]")
    ax.set_ylabel(r"$R(\tau)$")
    ax.set_title("(b) Temperature autocorr (spatial-mean series)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for traj in trajectories:
        r = autocorr_reports[traj.sim_id]["rMag"]
        lags = np.arange(len(r))
        ax.plot(lags, r, label=f"|U| {traj.sim_id}")
        rvi = autocorr_reports[traj.sim_id]["rVi"]
        ax.plot(lags, rvi, ls="--", alpha=0.7, label=f"Vi {traj.sim_id}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"lag $\tau$ [frames]")
    ax.set_ylabel(r"$R(\tau)$")
    ax.set_title("(b2) Velocity autocorr (spatial-mean)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig04_temperature_histograms_and_autocorr.png")
    plt.close(fig)

    print(f"\nFigures written to {FIGDIR}")
    print("Pilot sims:", [t.sim_id for t in trajectories])


if __name__ == "__main__":
    main()
