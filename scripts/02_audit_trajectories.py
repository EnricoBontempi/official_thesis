#!/usr/bin/env python3
"""Step 1.3b — Full 30-condition numerical combustion audit (low RAM).

Iterates one Arrow shard at a time under
``data/raw/combustion/hf_dataset/numerical/``, verifies ``sim_id`` mapping,
computes Temperature / Velocity statistics, updates manifests, and writes
``fig05_all30_temperature_velocity_trends.png``.

Usage:
    python scripts/02_audit_trajectories.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_NUM = ROOT / "data" / "raw" / "combustion" / "hf_dataset" / "numerical"
REMAIN_PATH = ROOT / "data" / "manifests" / "hf_small_metadata" / "combustion__remain_params_numerical.json"
CHANNELS_PATH = ROOT / "data" / "manifests" / "hf_small_metadata" / "combustion__channels.json"
MANIFESTS = ROOT / "data" / "manifests"
FIGDIR = ROOT / "results" / "figures" / "month1_audit"
TABLEDIR = ROOT / "results" / "tables"

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

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
T_THRESHOLDS = (600.0, 1200.0, 1800.0, 2000.0)


def load_arrow_row(path: Path) -> dict[str, Any]:
    with pa.memory_map(str(path), "r") as mm:
        table = pa.ipc.open_stream(mm).read_all()
    if table.num_rows != 1:
        raise ValueError(f"{path.name}: expected 1 row, got {table.num_rows}")
    return {col: table.column(col)[0].as_py() for col in table.column_names}


def decode_f32(buf: bytes, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.float32)
    if arr.size != int(np.prod(shape)):
        raise ValueError(f"float32 size {arr.size} != {shape}")
    return arr.reshape(shape)


def decode_f64(buf: bytes, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.float64)
    if arr.size != int(np.prod(shape)):
        raise ValueError(f"float64 size {arr.size} != {shape}")
    return arr.reshape(shape)


def field_stats(arr: np.ndarray) -> dict[str, float]:
    flat = arr.reshape(-1).astype(np.float64, copy=False)
    qs = np.percentile(flat, PERCENTILES)
    out = {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
    }
    for p, v in zip(PERCENTILES, qs):
        out[f"p{p:02d}"] = float(v)
    return out


def parse_sim_id(sim_id: str) -> tuple[int, float]:
    m = re.match(r"(\d+)NH3_(\d+\.?\d*)\.h5", sim_id)
    if not m:
        raise ValueError(f"Cannot parse sim_id={sim_id!r}")
    return int(m.group(1)), float(m.group(2))


def audit_shard(path: Path, remain: dict[str, list]) -> dict[str, Any]:
    row = load_arrow_row(path)
    sim_id = row["sim_id"]
    shape_t, shape_h, shape_w = int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"])
    nchan = int(row["numerical_channels"])
    numerical = decode_f32(row["numerical"], (shape_t, shape_h, shape_w, nchan))
    x = decode_f64(row["x"], (int(row["x_shape"]),))
    y = decode_f64(row["y"], (int(row["y_shape"]),))
    t = decode_f64(row["t"], (int(row["t_shape"]),))

    if sim_id in remain:
        nh3, phi = remain[sim_id]
        nh3, phi = int(nh3), float(phi)
    else:
        nh3, phi = parse_sim_id(sim_id)

    T = numerical[..., 10]
    vi = numerical[..., 11]
    vj = numerical[..., 12]
    vk = numerical[..., 13]
    mag_export = numerical[..., 14]
    mag = np.sqrt(vi * vi + vj * vj + vk * vk)

    dt = float(np.median(np.diff(t))) if t.size > 1 else float("nan")
    n_nan = int(np.isnan(numerical).sum())
    n_inf = int(np.isinf(numerical).sum())
    n_T_lt0 = int((T < 0).sum())
    n_T_lt200 = int((T < 200).sum())
    n_T_gt2600 = int((T > 2600).sum())
    n_tot = int(T.size)

    frac = {f"frac_T_lt_{int(th)}": float(np.mean(T < th)) for th in (600.0, 1200.0)}
    frac.update({f"frac_T_gt_{int(th)}": float(np.mean(T > th)) for th in (1800.0, 2000.0)})

    err = np.abs(mag_export - mag)
    record = {
        "shard_filename": path.name,
        "sim_id": sim_id,
        "nh3_pct": nh3,
        "ch4_pct": 100 - nh3,
        "equivalence_ratio": phi,
        "shape_t": shape_t,
        "shape_h": shape_h,
        "shape_w": shape_w,
        "numerical_channels": nchan,
        "t_start": float(t[0]),
        "t_end": float(t[-1]),
        "dt": dt,
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "nan_count": n_nan,
        "inf_count": n_inf,
        "n_T_lt_0": n_T_lt0,
        "n_T_lt_200": n_T_lt200,
        "n_T_gt_2600": n_T_gt2600,
        "n_voxels": n_tot,
        "frac_T_lt_0": n_T_lt0 / n_tot,
        "frac_T_lt_200": n_T_lt200 / n_tot,
        "frac_T_gt_2600": n_T_gt2600 / n_tot,
        **frac,
        "Temperature": field_stats(T),
        "Velocity_i": field_stats(vi),
        "Velocity_j": field_stats(vj),
        "Velocity_k": field_stats(vk),
        "Velocity_mag_recomputed": field_stats(mag),
        "Velocity_mag_exported": field_stats(mag_export),
        "vel_mag_export_vs_recomputed_mean_abs_err": float(err.mean()),
        "vel_mag_export_vs_recomputed_max_abs_err": float(err.max()),
        "vel_mag_export_vs_recomputed_corr": float(
            np.corrcoef(mag_export.ravel()[::200], mag.ravel()[::200])[0, 1]
        ),
    }
    # free large arrays before next shard
    del numerical, T, vi, vj, vk, mag, mag_export, err
    return record


def flatten_record(rec: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                flat[f"{k}_{sk}"] = sv
        else:
            flat[k] = v
    return flat


def write_fig05(records: list[dict[str, Any]], out_path: Path) -> None:
    nh3 = np.array([r["nh3_pct"] for r in records], dtype=float)
    phi = np.array([r["equivalence_ratio"] for r in records], dtype=float)
    T_mean = np.array([r["Temperature"]["mean"] for r in records])
    T_p95 = np.array([r["Temperature"]["p95"] for r in records])
    cold = np.array([r["frac_T_lt_1200"] for r in records])
    U_mean = np.array([r["Velocity_mag_recomputed"]["mean"] for r in records])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    metrics = [
        (axes[0, 0], T_mean, r"Mean $T$ [K]", "viridis"),
        (axes[0, 1], T_p95, r"$T$ 95th percentile [K]", "inferno"),
        (axes[1, 0], 100.0 * cold, r"Cold fraction $T<1200$ K [%]", "coolwarm"),
        (axes[1, 1], U_mean, r"Mean recomputed $|\mathbf{u}|$ [m/s]", "plasma"),
    ]
    for ax, vals, title, cmap in metrics:
        sc = ax.scatter(nh3, phi, c=vals, s=70, cmap=cmap, edgecolors="k", linewidths=0.4)
        for r, v in zip(records, vals):
            ax.annotate(
                f"{v:.0f}" if "fraction" not in title.lower() else f"{v:.1f}",
                (r["nh3_pct"], r["equivalence_ratio"]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=6,
                alpha=0.85,
            )
        ax.set_xlabel(r"NH$_3$ fuel fraction [%]")
        ax.set_ylabel(r"Equivalence ratio $\phi$")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "RealPDEBench combustion numerical — trends across all 30 operating conditions",
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABLEDIR.mkdir(parents=True, exist_ok=True)

    remain = json.loads(REMAIN_PATH.read_text())
    channels_doc = json.loads(CHANNELS_PATH.read_text()) if CHANNELS_PATH.exists() else {}
    shards = sorted(RAW_NUM.glob("data-*-of-00030.arrow"))
    if len(shards) != 30:
        raise FileNotFoundError(
            f"Expected 30 numerical shards in {RAW_NUM}, found {len(shards)}"
        )

    # Lexicographic expected map
    sim_ids_sorted = sorted(remain.keys())
    hypothesized = {f"data-{i:05d}-of-00030.arrow": sid for i, sid in enumerate(sim_ids_sorted)}

    records: list[dict[str, Any]] = []
    mapping_ok = True
    print(f"Auditing {len(shards)} shards one-at-a-time...", flush=True)
    for i, path in enumerate(shards, 1):
        print(f"[{i:02d}/30] {path.name}", flush=True)
        rec = audit_shard(path, remain)
        expected = hypothesized.get(path.name)
        rec["lexicographic_expected_sim_id"] = expected
        rec["lexicographic_match"] = expected == rec["sim_id"]
        if not rec["lexicographic_match"]:
            mapping_ok = False
            print(f"  *** MAP MISMATCH: got {rec['sim_id']} expected {expected}", flush=True)
        print(
            f"  {rec['sim_id']}  NH3={rec['nh3_pct']}%  phi={rec['equivalence_ratio']}  "
            f"T[mean]={rec['Temperature']['mean']:.1f}  "
            f"T[min/max]={rec['Temperature']['min']:.1f}/{rec['Temperature']['max']:.1f}  "
            f"negT={rec['n_T_lt_0']}  hot>{2600}={rec['n_T_gt_2600']}",
            flush=True,
        )
        records.append(rec)

    records_sorted = sorted(records, key=lambda r: (r["nh3_pct"], r["equivalence_ratio"], r["sim_id"]))

    # CSV
    flat_rows = [flatten_record(r) for r in records_sorted]
    csv_path = TABLEDIR / "month1_30_conditions_inventory.csv"
    fieldnames = list(flat_rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"Wrote {csv_path}", flush=True)

    # YAML operating conditions
    oc_doc = {
        "dataset": "RealPDEBench / combustion / numerical",
        "source_repo": "AI4Science-WestlakeU/RealPDEBench",
        "n_operating_conditions": len(records_sorted),
        "channel_names": CHANNEL_NAMES,
        "channels_doc_note": channels_doc.get("note"),
        "geometry": {
            "grid": [128, 128],
            "axis0": "y (m), increases with row index; physical range ~[0.063, 0.159] m",
            "axis1": "x (m), increases with column index; physical range ~[-0.037, +0.037] m",
            "symmetry_axis": "horizontal midline row≈64 (y≈0.111 m) — annular V-flame symmetry",
            "burner_inlet": (
                "Cold unburnt reactant jets enter from the RIGHT boundary "
                "(col≈127, x≈+0.037 m) and diverge in a V toward the LEFT "
                "(col≈0, x≈-0.037 m)"
            ),
            "Velocity[i]_ch11": "In-plane radial/transverse (expands ± away from row-64 centerline)",
            "Velocity[j]_ch12": "Axial along main jet (strongly + in V-branches, − in central IRZ)",
            "Velocity[k]_ch13": "Out-of-plane azimuthal swirl (opposite signs above/below row 64)",
            "Velocity_magnitude_note": (
                "Prefer recomputed |u|=sqrt(vi^2+vj^2+vk^2); exported ch14 is correlated "
                "but not identical and can be negative"
            ),
        },
        "lexicographic_shard_map_verified_all_30": mapping_ok,
        "conditions": [
            {
                "sim_id": r["sim_id"],
                "nh3_pct": r["nh3_pct"],
                "ch4_pct": r["ch4_pct"],
                "equivalence_ratio": r["equivalence_ratio"],
                "shard_filename": r["shard_filename"],
                "pilot_status": "downloaded_verified",
                "n_frames": r["shape_t"],
                "grid": [r["shape_h"], r["shape_w"]],
                "t_start": r["t_start"],
                "t_end": r["t_end"],
                "dt": r["dt"],
                "Temperature_min": r["Temperature"]["min"],
                "Temperature_mean": r["Temperature"]["mean"],
                "Temperature_max": r["Temperature"]["max"],
                "Temperature_p95": r["Temperature"]["p95"],
                "frac_T_lt_1200": r["frac_T_lt_1200"],
                "Velocity_mag_recomputed_mean": r["Velocity_mag_recomputed"]["mean"],
                "n_T_lt_0": r["n_T_lt_0"],
                "n_T_gt_2600": r["n_T_gt_2600"],
                "nan_count": r["nan_count"],
                "inf_count": r["inf_count"],
            }
            for r in records_sorted
        ],
    }
    oc_path = MANIFESTS / "operating_conditions.yaml"
    with oc_path.open("w") as f:
        yaml.safe_dump(oc_doc, f, sort_keys=False)
    print(f"Wrote {oc_path}", flush=True)

    # JSON summary for report generation
    summary_path = MANIFESTS / "full30_audit_summary.json"
    summary_path.write_text(json.dumps({"mapping_ok": mapping_ok, "records": records_sorted}, indent=2) + "\n")
    print(f"Wrote {summary_path}", flush=True)

    fig_path = FIGDIR / "fig05_all30_temperature_velocity_trends.png"
    write_fig05(records_sorted, fig_path)
    print(f"Wrote {fig_path}", flush=True)

    # Console rollup
    print("\n=== ROLLUP ===", flush=True)
    print(f"mapping_ok={mapping_ok}", flush=True)
    print(
        f"any NaN={any(r['nan_count'] for r in records)}  "
        f"any Inf={any(r['inf_count'] for r in records)}",
        flush=True,
    )
    print(
        f"conditions with T<0: {sum(1 for r in records if r['n_T_lt_0']>0)}  "
        f"with T>2600: {sum(1 for r in records if r['n_T_gt_2600']>0)}",
        flush=True,
    )
    by_nh3: dict[int, list] = {}
    for r in records_sorted:
        by_nh3.setdefault(r["nh3_pct"], []).append(r)
    for nh3, group in sorted(by_nh3.items()):
        means = [g["Temperature"]["mean"] for g in group]
        print(
            f"NH3={nh3:2d}%  n={len(group)}  "
            f"T_mean in [{min(means):.0f}, {max(means):.0f}] K  "
            f"avg cold%<1200K={100*np.mean([g['frac_T_lt_1200'] for g in group]):.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
