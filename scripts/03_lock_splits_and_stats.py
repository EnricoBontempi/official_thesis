#!/usr/bin/env python3
"""Step 1.4 — Lock splits plot, train-only norm stats, evaluation protocol.

Usage:
    python scripts/03_lock_splits_and_stats.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import yaml
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from combustion_recon.data.masks import stable_seed

ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = ROOT / "configs" / "splits" / "splits_interp_extrap.yaml"
RAW_NUM = ROOT / "data" / "raw" / "combustion" / "hf_dataset" / "numerical"
OC_PATH = ROOT / "data" / "manifests" / "operating_conditions.yaml"
FIGDIR = ROOT / "results" / "figures" / "month1_audit"
MANIFESTS = ROOT / "data" / "manifests"
PROCESSED = ROOT / "data" / "processed"

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
FIELDS = (
    "Temperature",
    "Velocity_i",
    "Velocity_j",
    "Velocity_k",
    "Velocity_mag_recomputed",
)


def load_splits() -> dict[str, list[str]]:
    with SPLITS_PATH.open() as f:
        cfg = yaml.safe_load(f)
    return {
        "train": list(cfg["train"]["sim_ids"]),
        "val": list(cfg["val"]["sim_ids"]),
        "test_interp": list(cfg["test_interp"]["sim_ids"]),
        "test_extrap": list(cfg["test_extrap"]["sim_ids"]),
    }


def parse_sim_id(sim_id: str) -> tuple[int, float]:
    m = re.match(r"(\d+)NH3_(\d+\.?\d*)\.h5", sim_id)
    if not m:
        raise ValueError(sim_id)
    return int(m.group(1)), float(m.group(2))


def sim_id_to_shard() -> dict[str, Path]:
    with OC_PATH.open() as f:
        oc = yaml.safe_load(f)
    return {c["sim_id"]: RAW_NUM / c["shard_filename"] for c in oc["conditions"]}


def load_numerical(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with pa.memory_map(str(path), "r") as mm:
        table = pa.ipc.open_stream(mm).read_all()
    row = {c: table.column(c)[0].as_py() for c in table.column_names}
    shape = (int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"]))
    nchan = int(row["numerical_channels"])
    numerical = np.frombuffer(row["numerical"], dtype=np.float32).reshape(*shape, nchan)
    x = np.frombuffer(row["x"], dtype=np.float64)
    y = np.frombuffer(row["y"], dtype=np.float64)
    return numerical, x, y


def running_stats() -> dict[str, Any]:
    """Online mean/var/min/max + reservoir-free histogram approx via Welford + quantiles via subsample."""
    return {
        "n": 0,
        "mean": 0.0,
        "M2": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        # store limited samples for quantiles (stratified per traj)
        "samples": [],
    }


def update_stats(st: dict[str, Any], arr: np.ndarray, max_samples_per_update: int = 200_000) -> None:
    flat = arr.reshape(-1).astype(np.float64, copy=False)
    chunk = 1_000_000
    for i in range(0, flat.size, chunk):
        x = flat[i : i + chunk]
        n_a = st["n"]
        n_b = int(x.size)
        if n_b == 0:
            continue
        mean_b = float(x.mean())
        M2_b = float(np.dot(x - mean_b, x - mean_b))
        if n_a == 0:
            st["mean"] = mean_b
            st["M2"] = M2_b
            st["n"] = n_b
        else:
            delta = mean_b - st["mean"]
            n = n_a + n_b
            st["mean"] = st["mean"] + delta * n_b / n
            st["M2"] = st["M2"] + M2_b + delta * delta * n_a * n_b / n
            st["n"] = n
        st["min"] = min(st["min"], float(x.min()))
        st["max"] = max(st["max"], float(x.max()))
    # subsample for quantiles
    rng = np.random.default_rng(stable_seed(int(flat.size), round(float(flat[0]), 4)))
    if flat.size > max_samples_per_update:
        samp = rng.choice(flat, size=max_samples_per_update, replace=False)
    else:
        samp = flat.copy()
    st["samples"].append(samp)


def finalize_stats(st: dict[str, Any]) -> dict[str, float]:
    n = st["n"]
    var = st["M2"] / n if n > 0 else float("nan")
    std = float(np.sqrt(var))
    samples = np.concatenate(st["samples"]) if st["samples"] else np.array([st["mean"]])
    out = {
        "n_voxels_total": int(n),
        "mean": float(st["mean"]),
        "std": std,
        "min": float(st["min"]),
        "max": float(st["max"]),
        "n_quantile_samples": int(samples.size),
    }
    for p in PERCENTILES:
        out[f"p{p:02d}"] = float(np.percentile(samples, p))
    return out


def plot_splits(splits: dict[str, list[str]], out_path: Path) -> None:
    styles = {
        "train": dict(c="#2e7d32", marker="o", s=55, label="train (20)"),
        "val": dict(c="#f9a825", marker="s", s=70, label="val (3)"),
        "test_interp": dict(c="#1565c0", marker="D", s=70, label="test_interp (3)"),
        "test_extrap": dict(c="#c62828", marker="^", s=80, label="test_extrap (4)"),
    }
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    train_pts = np.array([[parse_sim_id(s)[0], parse_sim_id(s)[1]] for s in splits["train"]], dtype=float)
    if len(train_pts) >= 3:
        hull = ConvexHull(train_pts)
        poly = Polygon(
            train_pts[hull.vertices],
            closed=True,
            facecolor="#2e7d32",
            edgecolor="#1b5e20",
            alpha=0.15,
            label="train convex hull",
        )
        ax.add_patch(poly)

    for split, sids in splits.items():
        pts = np.array([[parse_sim_id(s)[0], parse_sim_id(s)[1]] for s in sids], dtype=float)
        ax.scatter(pts[:, 0], pts[:, 1], **styles[split], zorder=3)
        for sid, (nh3, phi) in zip(sids, pts):
            ax.annotate(
                sid.replace(".h5", ""),
                (nh3, phi),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
                alpha=0.85,
            )

    ax.set_xlabel(r"NH$_3$ fuel fraction [%]")
    ax.set_ylabel(r"Equivalence ratio $\phi$")
    ax.set_title("Locked operating-condition splits (Step 1.4)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def high_t_coverage(
    splits: dict[str, list[str]],
    shard_map: dict[str, Path],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    cov: dict[str, Any] = {}
    for split, sids in splits.items():
        rows = []
        for sid in sids:
            numerical, _, _ = load_numerical(shard_map[sid])
            T = numerical[..., 10]
            n = T.size
            rows.append(
                {
                    "sim_id": sid,
                    "frac_T_ge_2100": float(np.mean(T >= thresholds["T_high"])),
                    "frac_T_ge_2200": float(np.mean(T >= thresholds["T_peak"])),
                    "frac_T_lt_1800": float(np.mean(T < thresholds["T_jet"])),
                    "T_mean": float(T.mean()),
                    "n_voxels": int(n),
                }
            )
            del numerical, T
        cov[split] = {
            "per_condition": rows,
            "mean_frac_T_ge_2100": float(np.mean([r["frac_T_ge_2100"] for r in rows])),
            "mean_frac_T_ge_2200": float(np.mean([r["frac_T_ge_2200"] for r in rows])),
            "mean_frac_T_lt_1800": float(np.mean([r["frac_T_lt_1800"] for r in rows])),
        }
    return cov


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    splits = load_splits()
    all_ids = [s for v in splits.values() for s in v]
    assert len(all_ids) == 30 and len(set(all_ids)) == 30

    print("Writing fig06...", flush=True)
    plot_splits(splits, FIGDIR / "fig06_locked_operating_condition_splits.png")

    shard_map = sim_id_to_shard()
    print("Computing train-only normalization statistics over 20 conditions...", flush=True)
    field_acc = {name: running_stats() for name in FIELDS}
    op_nh3 = []
    op_phi = []

    for i, sid in enumerate(splits["train"], 1):
        print(f"  [{i:02d}/20] {sid}", flush=True)
        numerical, _, _ = load_numerical(shard_map[sid])
        T = numerical[..., 10]
        vi = numerical[..., 11]
        vj = numerical[..., 12]
        vk = numerical[..., 13]
        mag = np.sqrt(vi * vi + vj * vj + vk * vk)
        update_stats(field_acc["Temperature"], T)
        update_stats(field_acc["Velocity_i"], vi)
        update_stats(field_acc["Velocity_j"], vj)
        update_stats(field_acc["Velocity_k"], vk)
        update_stats(field_acc["Velocity_mag_recomputed"], mag)
        nh3, phi = parse_sim_id(sid)
        op_nh3.append(nh3 / 100.0)
        op_phi.append(phi)
        del numerical, T, vi, vj, vk, mag

    fields_out = {k: finalize_stats(v) for k, v in field_acc.items()}
    op_nh3_a = np.asarray(op_nh3, dtype=np.float64)
    op_phi_a = np.asarray(op_phi, dtype=np.float64)
    op_out = {
        "nh3_fraction": {
            "n_conditions": int(op_nh3_a.size),
            "mean": float(op_nh3_a.mean()),
            "std": float(op_nh3_a.std(ddof=0)),
            "min": float(op_nh3_a.min()),
            "max": float(op_nh3_a.max()),
        },
        "equivalence_ratio": {
            "n_conditions": int(op_phi_a.size),
            "mean": float(op_phi_a.mean()),
            "std": float(op_phi_a.std(ddof=0)),
            "min": float(op_phi_a.min()),
            "max": float(op_phi_a.max()),
        },
    }

    stats_doc = {
        "meta": {
            "split": "train",
            "n_conditions": 20,
            "sim_ids": list(splits["train"]),
            "source": "scripts/03_lock_splits_and_stats.py",
            "note": "Statistics fitted exclusively on locked train operating conditions.",
        },
        "fields": fields_out,
        "operating_parameters": op_out,
    }

    for dest in (
        MANIFESTS / "train_norm_stats.json",
        PROCESSED / "train_norm_stats.json",
    ):
        dest.write_text(json.dumps(stats_doc, indent=2) + "\n")
        print(f"Wrote {dest}", flush=True)

    thresholds = {"T_high": 2100.0, "T_peak": 2200.0, "T_jet": 1800.0}
    print("Computing high-T coverage by split...", flush=True)
    coverage = high_t_coverage(splits, shard_map, thresholds)
    (MANIFESTS / "high_t_region_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n"
    )

    protocol = {
        "locked_date": "2026-11-14",
        "splits_file": "configs/splits/splits_interp_extrap.yaml",
        "train_norm_stats": "data/manifests/train_norm_stats.json",
        "geometry_note": (
            "Annular V-flame: cold jets enter from right boundary (col≈127); "
            "symmetry about row≈64; Velocity[j]=axial, Velocity[i]=transverse, "
            "Velocity[k]=swirl."
        ),
        "high_temperature_regions": {
            "Omega_high": {
                "definition": "T(x,y) >= T_high",
                "T_high_K": 2100.0,
                "role": "primary fixed high-temperature region for IoU / conditional RMSE",
            },
            "Omega_peak": {
                "definition": "T(x,y) >= T_peak",
                "T_peak_K": 2200.0,
                "role": "secondary peak thermal core diagnostic",
            },
            "Omega_jet": {
                "definition": "T(x,y) < T_jet",
                "T_jet_K": 1800.0,
                "role": "reaction/mixing layer & cold jet zone diagnostic",
            },
        },
        "train_temperature_summary": {
            "mean_K": fields_out["Temperature"]["mean"],
            "std_K": fields_out["Temperature"]["std"],
            "min_K": fields_out["Temperature"]["min"],
            "max_K": fields_out["Temperature"]["max"],
            "p50_K": fields_out["Temperature"]["p50"],
            "p95_K": fields_out["Temperature"]["p95"],
        },
        "high_t_coverage_by_split": {
            split: {
                "mean_frac_Omega_high": coverage[split]["mean_frac_T_ge_2100"],
                "mean_frac_Omega_peak": coverage[split]["mean_frac_T_ge_2200"],
                "mean_frac_Omega_jet": coverage[split]["mean_frac_T_lt_1800"],
            }
            for split in coverage
        },
        "reporting_rules": [
            "Always report metrics per operating condition (never only pooled).",
            "Temperature-free and temperature-assisted tasks reported separately.",
            "Observation-layout changes evaluated separately from operating-condition changes.",
            "Normalization statistics fitted on train split only.",
            "Missing observations never encoded as physical zeros.",
        ],
    }
    proto_path = ROOT / "configs" / "splits" / "evaluation_protocol.yaml"
    with proto_path.open("w") as f:
        yaml.safe_dump(protocol, f, sort_keys=False)
    print(f"Wrote {proto_path}", flush=True)

    print("\n=== Train Temperature stats ===", flush=True)
    print(json.dumps(fields_out["Temperature"], indent=2), flush=True)
    print("\n=== High-T coverage (mean frac) ===", flush=True)
    for split, v in protocol["high_t_coverage_by_split"].items():
        print(split, v, flush=True)


if __name__ == "__main__":
    main()
