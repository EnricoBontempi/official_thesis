#!/usr/bin/env python3
"""Month-2 non-neural baselines on the pilot protocol (replica-capable).

Usage:
  python scripts/06_run_month2_baselines.py --smoke --run-id smoke_local
  python scripts/06_run_month2_baselines.py --full --run-id consolidation_v1

Historical Step-2.3 tables under results/tables/ are never overwritten.
New runs write to results/baseline_consolidation/<run_id>/.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from combustion_recon.data.dataset import load_splits, parse_sim_id
from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.data.month2_protocol import (
    frames_for_role,
    load_frame_protocol,
    sha256_file,
)
from combustion_recon.data.shard_access import ShardStore
from combustion_recon.evaluation.month2_baseline_invariants import run_isolation_battery
from combustion_recon.evaluation.month2_evaluator import (
    aggregate_frame_then_condition_then_split,
    evaluate_frame_prediction,
)
from combustion_recon.models.baselines import (
    GappyPODReconstructor,
    GlobalTrainMeanPredictor,
    LocalVelocityTemperatureRegressor,
    ParamConditionedMeanPredictor,
)
from combustion_recon.models.month2_baseline_adapters import (
    observed_grids_from_model_inputs,
    predict_gappy,
    predict_global_mean,
    predict_param_mean,
    predict_param_mean_ao,
)

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "experiments" / "month2_baselines.yaml"
DEFAULT_RAW = ROOT / "data" / "raw" / "combustion" / "hf_dataset" / "numerical"
DEFAULT_STATS = ROOT / "data" / "manifests" / "train_norm_stats_month2_fit.json"
CONSOLIDATION_ROOT = ROOT / "results" / "baseline_consolidation"

# Comparison rule fixed before reading new scores (mission M4).
COMPARISON_RULE = {
    "metric_atol_K": 1.0e-6,
    "metric_rtol": 1.0e-8,
    "ids_must_match_exactly": True,
    "pairing_keys": [
        "method",
        "role",
        "p_conditioning",
        "K",
        "site_fraction",
        "variant",
        "mask_replica_id",
        "sim_id",
        "frame_index",
        "R",
        "ridge_lambda",
        "w_T",
        "observation_hash",
    ],
}


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_cfg() -> dict[str, Any]:
    with CFG_PATH.open() as f:
        return yaml.safe_load(f)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def _git_status_porcelain() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        )
    except Exception:
        return "UNKNOWN"


def resolve_contract(cfg: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    """Consume YAML contract or refuse incompatible values."""
    data = cfg["data"]
    methods = cfg["methods"]
    hs = cfg["hyperparameter_search"]
    ev = cfg["evaluation"]
    io = cfg["io"]

    site_fraction = float(data["site_fraction_primary"])
    if abs(site_fraction - 0.05) > 1e-12:
        raise ValueError(f"Unsupported site_fraction_primary={site_fraction} (contract requires 0.05)")
    if str(data.get("velocity")) != "planar_2C":
        raise ValueError("Contract requires data.velocity=planar_2C")
    if int(methods["GappyPOD"]["n_velocity_components"]) != 2:
        raise ValueError("Contract requires GappyPOD n_velocity_components=2")
    pod_R = int(methods["GappyPOD"]["R"])
    if pod_R != 32 and not smoke:
        raise ValueError(f"Full run requires GappyPOD.R=32, got {pod_R}")
    free_lams = [float(x) for x in hs["temperature_free"]["ridge_lambda"]]
    asst_lams = [float(x) for x in hs["temperature_assisted_primary"]["ridge_lambda"]]
    asst_wT = [float(x) for x in hs["temperature_assisted_primary"]["w_T"]]
    if len(free_lams) != int(hs["temperature_free"]["n_combinations"]):
        raise ValueError("temperature_free λ grid length != n_combinations")
    if len(asst_lams) * len(asst_wT) != int(hs["temperature_assisted_primary"]["n_combinations"]):
        raise ValueError("assisted λ×w_T grid size != n_combinations")
    K_values = [int(k) for k in ev["K_values"]]
    if K_values != [0, 9, 32] and not smoke:
        raise ValueError(f"Full run requires evaluation.K_values=[0,9,32], got {K_values}")
    ao_tau = float(ev["AO_tau"])
    if abs(ao_tau - 1.0) > 1e-12:
        raise ValueError(f"Contract requires AO_tau=1.0, got {ao_tau}")
    replicas = [int(r) for r in data["mask_replicas"]]
    layout_id = str(data["layout_id"])
    mask_seed = int(data["mask_seed"])
    stats_path = ROOT / str(data["normalizer"])
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    if smoke:
        return {
            "site_fraction": site_fraction,
            "pod_R": 4,
            "free_lams": [1e-2, 1e-1],
            "asst_lams": [1e-2],
            "asst_wT": [5.0],
            "K_values": [0, 9],
            "ao_tau": ao_tau,
            "replicas": [0],
            "layout_id": layout_id,
            "mask_seed": mask_seed,
            "stats_path": stats_path,
            "max_cached_frames": int(io["max_cached_frames"]),
            "run_hgb": False,
            "hgb_max_pixels": int(methods["LocalHGB_dense_diagnostic"]["max_pixels"]),
            "w_T_fixed_unused": float(hs["temperature_free"]["w_T_fixed_unused"]),
            "smoke_overrides": True,
        }
    return {
        "site_fraction": site_fraction,
        "pod_R": pod_R,
        "free_lams": free_lams,
        "asst_lams": asst_lams,
        "asst_wT": asst_wT,
        "K_values": K_values,
        "ao_tau": ao_tau,
        "replicas": replicas,
        "layout_id": layout_id,
        "mask_seed": mask_seed,
        "stats_path": stats_path,
        "max_cached_frames": int(io["max_cached_frames"]),
        "run_hgb": bool(methods["LocalHGB_dense_diagnostic"]["include"]),
        "hgb_max_pixels": int(methods["LocalHGB_dense_diagnostic"]["max_pixels"]),
        "w_T_fixed_unused": float(hs["temperature_free"]["w_T_fixed_unused"]),
        "smoke_overrides": False,
    }


def write_csv(df: pd.DataFrame, path: Path, *, gzip_enabled: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_enabled:
        out = path if str(path).endswith(".gz") else path.with_suffix(path.suffix + ".gz")
        with gzip.open(out, "wt", newline="") as f:
            df.to_csv(f, index=False)
        # round-trip check
        with gzip.open(out, "rt") as f:
            _ = pd.read_csv(f, nrows=2)
        return out
    df.to_csv(path, index=False)
    return path


def _compact_to_dict(
    fr,
    *,
    sid: str,
    role: str,
    split: str,
    nh3: float,
    phi: float,
    xx,
    yy,
) -> dict[str, Any]:
    from combustion_recon.data.temporal_quality import temporal_quality_from_mapping

    vi = fr.velocity_i.astype(np.float64)
    vj = fr.velocity_j.astype(np.float64)
    tq = temporal_quality_from_mapping(fr)
    return {
        "sim_id": sid,
        "frame": int(fr.frame_index),
        "role": role,
        "split": split,
        "nh3": nh3,
        "phi": float(phi),
        "T": fr.temperature.astype(np.float64),
        "vi": vi,
        "vj": vj,
        "vk": np.zeros_like(vi, dtype=np.float64),  # 2C: no vk
        "x": fr.x,
        "y": fr.y,
        "x2d": xx,
        "y2d": yy,
        "umag": np.sqrt(vi**2 + vj**2),
        # Temporal provenance for future per-frame tables (not model inputs).
        **tq,
    }


def materialize_role_frames(
    store: ShardStore,
    role: str,
    *,
    sim_ids: list[str] | None = None,
    max_per_cond: int | None = None,
) -> list[dict[str, Any]]:
    protocol = load_frame_protocol()
    splits = load_splits()
    split = {
        "train_fit": "train",
        "seen_train_eval": "train",
        "val_eval": "val",
    }[role]
    ids = list(sim_ids) if sim_ids is not None else list(splits[split])
    frames = frames_for_role(protocol, role)
    if max_per_cond is not None:
        frames = frames[: max_per_cond]
    out: list[dict[str, Any]] = []
    for sid in ids:
        store.prefetch(sid, frames)
        nh3_pct, phi = parse_sim_id(sid)
        xx = None
        yy = None
        for fi in frames:
            fr = store.get_frame(sid, fi)
            if xx is None:
                xx, yy = np.meshgrid(fr.x, fr.y, indexing="xy")
            out.append(
                _compact_to_dict(
                    fr,
                    sid=sid,
                    role=role,
                    split=split,
                    nh3=nh3_pct / 100.0,
                    phi=phi,
                    xx=xx,
                    yy=yy,
                )
            )
        # Drop this traj from cache after extracting (bound RAM across conditions)
        with store._lock:
            for k in [k for k in store._frame_cache if k[0] == sid]:
                store._frame_cache.pop(k, None)
        gc.collect()
    return out


def materialize_train_fit_and_seen(
    store: ShardStore,
    *,
    sim_ids: list[str],
    max_fit: int | None = None,
    max_seen: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One shard decode per TRAIN condition for both fit and seen roles."""
    protocol = load_frame_protocol()
    fit_idx = frames_for_role(protocol, "train_fit")
    seen_idx = frames_for_role(protocol, "seen_train_eval")
    if max_fit is not None:
        fit_idx = fit_idx[:max_fit]
    if max_seen is not None:
        seen_idx = seen_idx[:max_seen]
    need = sorted(set(fit_idx) | set(seen_idx))
    train_frames: list[dict[str, Any]] = []
    seen_frames: list[dict[str, Any]] = []
    fit_set, seen_set = set(fit_idx), set(seen_idx)
    for sid in sim_ids:
        store.prefetch(sid, need)
        nh3_pct, phi = parse_sim_id(sid)
        xx = yy = None
        for fi in need:
            fr = store.get_frame(sid, fi)
            if xx is None:
                xx, yy = np.meshgrid(fr.x, fr.y, indexing="xy")
            if fi in fit_set:
                train_frames.append(
                    _compact_to_dict(
                        fr,
                        sid=sid,
                        role="train_fit",
                        split="train",
                        nh3=nh3_pct / 100.0,
                        phi=phi,
                        xx=xx,
                        yy=yy,
                    )
                )
            if fi in seen_set:
                seen_frames.append(
                    _compact_to_dict(
                        fr,
                        sid=sid,
                        role="seen_train_eval",
                        split="train",
                        nh3=nh3_pct / 100.0,
                        phi=phi,
                        xx=xx,
                        yy=yy,
                    )
                )
        with store._lock:
            for k in [k for k in store._frame_cache if k[0] == sid]:
                store._frame_cache.pop(k, None)
        gc.collect()
    return train_frames, seen_frames


def make_sample(
    fr: dict[str, Any],
    *,
    K: int,
    p_conditioning: bool,
    replica_id: int,
    mask_seed: int,
    site_fraction: float,
    normalizer: Month2FitOnlyNormalizer,
    store: ShardStore | None = None,
) -> dict[str, Any]:
    """Build model_inputs from an already-materialized frame (no re-decode)."""
    from combustion_recon.data.month2_observations import build_month2_observation_pack

    T = fr["T"]
    vi = fr["vi"]
    vj = fr["vj"]
    vi_n = normalizer.normalize_field("Velocity_i", vi)
    vj_n = normalizer.normalize_field("Velocity_j", vj)
    t_n = normalizer.normalize_field("Temperature", T)
    K_eff = int(K)
    pack = build_month2_observation_pack(
        velocity_i=vi,
        velocity_j=vj,
        temperature=T,
        x_coords=fr["x"],
        y_coords=fr["y"],
        site_fraction=site_fraction,
        K=K_eff,
        layout_id="random_sparse_primary",
        mask_seed=mask_seed,
        sim_id=fr["sim_id"],
        frame_index=fr["frame"],
        replica_id=replica_id,
        mode="evaluation",
        include_dense_temperature_channels=(K_eff > 0),
        vi_norm=vi_n,
        vj_norm=vj_n,
        t_norm=t_n,
    )
    H, W = T.shape
    if pack.n_sites:
        ys = pack.velocity_site_index // W
        xs = pack.velocity_site_index % W
        vel_values_norm = np.stack([vi_n[ys, xs], vj_n[ys, xs]], axis=1)
    else:
        vel_values_norm = np.zeros((0, 2), dtype=np.float64)
    if K_eff > 0:
        ay = pack.anchor_site_index // W
        ax = pack.anchor_site_index % W
        anchor_values_norm = t_n[ay, ax]
        anchor_phys = pack.anchor_values
    else:
        anchor_values_norm = np.zeros((0,), dtype=np.float64)
        anchor_phys = pack.anchor_values

    model_inputs: dict[str, Any] = {
        "velocity_coords": pack.velocity_coords,
        "velocity_values": vel_values_norm,
        "velocity_values_physical": pack.velocity_values,
        "velocity_site_index": pack.velocity_site_index,
        "mask_velocity": pack.mask_velocity,
        "anchor_coords": pack.anchor_coords,
        "anchor_values": anchor_values_norm,
        "anchor_values_physical": anchor_phys,
        "anchor_site_index": pack.anchor_site_index,
        "mask_temperature": pack.mask_temperature,
        "dense_values": pack.dense_values,
        "K": K_eff,
        "site_fraction": float(site_fraction),
        "task_mode": "temperature_assisted" if K_eff > 0 else "temperature_free",
    }
    if p_conditioning:
        model_inputs["op_params"] = {
            "nh3_fraction": float(fr["nh3"]),
            "phi": float(fr["phi"]),
            "equivalence_ratio": float(fr["phi"]),
            "normalized": normalizer.normalize_op_params(fr["nh3"], phi=fr["phi"]),
        }
    metadata = {
        "sim_id": fr["sim_id"],
        "frame_index": fr["frame"],
        "role": fr["role"],
        "velocity_mask_hash": pack.velocity_mask_hash,
        "temperature_mask_hash": pack.temperature_mask_hash,
    }
    return {
        "model_inputs": model_inputs,
        "targets": {"temperature": T, "temperature_normalized": t_n},
        "metadata": metadata,
    }

def score_rows(rows: list[dict[str, Any]]) -> float:
    """Mean condition RMSE on VAL (or whatever is in rows), after homogeneity check."""
    if not rows:
        return float("nan")
    agg = aggregate_frame_then_condition_then_split(rows)
    pc = agg["per_condition"]
    if pc.empty or "rmse_K" not in pc.columns:
        return float("nan")
    return float(pc["rmse_K"].mean())


def evaluate_method_on_frames(
    *,
    method: str,
    frames: list[dict[str, Any]],
    predict_fn,
    normalizer: Month2FitOnlyNormalizer,
    cfg: dict[str, Any],
    K: int,
    p_conditioning: bool,
    replica_ids: list[int],
    site_fraction: float,
    store: ShardStore,
    extra_tags: dict[str, Any] | None = None,
    use_dense_for_hgb: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mask_seed = int(cfg["data"]["mask_seed"])
    tags = dict(extra_tags or {})
    for rep in replica_ids:
        for fr in frames:
            if use_dense_for_hgb:
                # Dense diagnostic: full velocity from materialized frame (not sparse inputs)
                sample = make_sample(
                    fr,
                    K=0,
                    p_conditioning=p_conditioning,
                    replica_id=rep,
                    mask_seed=mask_seed,
                    site_fraction=site_fraction,
                    normalizer=normalizer,
                    store=store,
                )
                pred = predict_fn(fr, sample)
                obs_hash = "dense_full_velocity"
            else:
                sample = make_sample(
                    fr,
                    K=K,
                    p_conditioning=p_conditioning,
                    replica_id=rep,
                    mask_seed=mask_seed,
                    site_fraction=site_fraction,
                    normalizer=normalizer,
                    store=store,
                )
                pred = predict_fn(sample["model_inputs"])
                obs_hash = sample["metadata"]["velocity_mask_hash"]
                if K > 0:
                    obs_hash = obs_hash + ":" + sample["metadata"]["temperature_mask_hash"]

            true = fr["T"]
            t_mask = (
                sample["model_inputs"]["mask_temperature"]
                if not use_dense_for_hgb
                else np.zeros_like(true, dtype=bool)
            )
            layout_id = tags.get("layout_id", cfg["data"]["layout_id"])
            temporal_tags = {
                k: fr[k]
                for k in (
                    "has_original_time_coordinate",
                    "timeline_regular",
                    "field_time_association_verified",
                    "time_validated",
                    "duplicate_of",
                    "time_coordinate_original",
                )
                if k in fr
            }
            row = evaluate_frame_prediction(
                pred,
                true,
                train_sigma_T=normalizer.temperature_std,
                sim_id=fr["sim_id"],
                split=fr["split"],
                role=fr["role"],
                frame_index=fr["frame"],
                mask_temperature=t_mask if K > 0 else None,
                method=method,
                p_conditioning=p_conditioning,
                K=K,
                site_fraction=site_fraction,
                model_seed=0,
                mask_replica_id=rep,
                observation_hash=obs_hash,
                layout_id=layout_id,
                task_mode="temperature_assisted" if K > 0 else "temperature_free",
                **temporal_tags,
                **{k: v for k, v in tags.items() if k != "layout_id"},
            )
            rows.append(row)
    return rows


def mean_over_replicas(per_frame_rows: list[dict[str, Any]]) -> float:
    """Aggregate: per replica → mean condition RMSE on VAL → mean over replicas."""
    if not per_frame_rows:
        return float("nan")
    df = pd.DataFrame(per_frame_rows)
    scores = []
    for rep, g in df.groupby("mask_replica_id"):
        rows = g.to_dict(orient="records")
        # one config at a time already
        scores.append(score_rows(rows))
    return float(np.mean(scores))


def run(
    smoke: bool = False,
    *,
    run_id: str,
    raw_dir: Path | None = None,
) -> dict[str, Any]:
    t_wall0 = time.time()
    cfg = load_cfg()
    assert CFG_PATH.exists(), "baselines YAML must exist before scores"
    contract = resolve_contract(cfg, smoke=smoke)
    protocol = load_frame_protocol()
    stats_path = contract["stats_path"]
    normalizer = Month2FitOnlyNormalizer(stats_path)
    raw_dir = Path(raw_dir) if raw_dir is not None else Path(
        os.environ.get("COMBUSTION_RAW_DIR", str(DEFAULT_RAW))
    )
    store = ShardStore(
        raw_dir=raw_dir, max_cached_frames=int(contract["max_cached_frames"])
    )

    out_dir = CONSOLIDATION_ROOT / run_id
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing run directory {out_dir}. Choose a new --run-id."
        )
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "logs").mkdir(exist_ok=True)

    # M4: provenance + comparison rule BEFORE fit / scores
    preflight = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "smoke": smoke,
        "git_head": _git_head(),
        "git_status_porcelain": _git_status_porcelain(),
        "cfg_sha256": sha256_file(CFG_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "adapters_sha256": sha256_file(
            ROOT / "src/combustion_recon/models/month2_baseline_adapters.py"
        ),
        "baselines_py_sha256": sha256_file(
            ROOT / "src/combustion_recon/models/baselines.py"
        ),
        "shard_access_sha256": sha256_file(
            ROOT / "src/combustion_recon/data/shard_access.py"
        ),
        "norm_stats_sha256": sha256_file(stats_path),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "executable": sys.executable,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "comparison_rule": COMPARISON_RULE,
        "contract": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in contract.items()
            if k != "smoke_overrides"
        },
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "historical_note": (
            "Historical 822ed52 run remains the dated record with incomplete provenance; "
            "this replica does not retroactively attest preregistration."
        ),
    }
    (out_dir / "preflight_manifest.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if preflight["git_status_porcelain"].strip() and not smoke:
        print(
            "WARNING: git working tree is dirty at preflight; full scientific "
            "replica should start from a clean freeze commit.",
            flush=True,
        )

    splits = load_splits()
    if smoke:
        train_ids = splits["train"][:2]
        val_ids = splits["val"][:1]
        max_fit = 8
        max_seen = 2
        max_val = 2
    else:
        train_ids = splits["train"]
        val_ids = splits["val"]
        max_fit = max_seen = max_val = None
    replicas = list(contract["replicas"])
    free_lams = list(contract["free_lams"])
    asst_lams = list(contract["asst_lams"])
    asst_wT = list(contract["asst_wT"])
    run_hgb = bool(contract["run_hgb"])
    pod_R = int(contract["pod_R"])
    ao_tau = float(contract["ao_tau"])
    layout_id = str(contract["layout_id"])
    K_values = list(contract["K_values"])
    site_fraction = float(contract["site_fraction"])
    w_T_fixed_unused = float(contract["w_T_fixed_unused"])
    hgb_max_pixels = int(contract["hgb_max_pixels"])

    print("Materializing TRAIN-fit + seen (one decode/condition)...", flush=True)
    train_frames, seen_frames = materialize_train_fit_and_seen(
        store, sim_ids=train_ids, max_fit=max_fit, max_seen=max_seen
    )
    print(
        f"  n_train_fit={len(train_frames)} n_seen={len(seen_frames)} "
        f"decodes={store.stats['shard_decodes']}",
        flush=True,
    )

    print("Materializing VAL...", flush=True)
    val_frames = materialize_role_frames(
        store, "val_eval", sim_ids=val_ids, max_per_cond=max_val
    )
    print(
        f"  n_val={len(val_frames)} "
        f"decodes_total={store.stats['shard_decodes']} "
        f"time_unvalidated={store.stats.get('time_unvalidated_frames', 0)} "
        f"rss={rss_mb():.0f}MB",
        flush=True,
    )

    # --- Fit zero-sensor baselines ---
    print("Fitting GlobalMean / ParamMean...", flush=True)
    t0 = time.time()
    # Equal weight per frame (and thus per condition × frames/cond)
    global_mean = GlobalTrainMeanPredictor.fit([f["T"] for f in train_frames])
    # Condition means: average T over frames of each condition
    by_cond: dict[str, list[np.ndarray]] = {}
    ops: dict[str, tuple[float, float]] = {}
    for f in train_frames:
        by_cond.setdefault(f["sim_id"], []).append(f["T"])
        ops[f["sim_id"]] = (f["nh3"], f["phi"])
    cond_means = [np.mean(np.stack(v, 0), 0) for _, v in sorted(by_cond.items())]
    cond_ops = [ops[sid] for sid, _ in sorted(by_cond.items())]
    param_mean = ParamConditionedMeanPredictor.fit(cond_means, cond_ops, ridge_alpha="auto")
    t_fit_means = time.time() - t0
    print(
        f"  ParamMean basis={param_mean.selected_basis} α={param_mean.selected_ridge_alpha} "
        f"LOCO={param_mean.loco_rmse_K:.2f}K  ({t_fit_means:.1f}s)",
        flush=True,
    )

    # --- Fit GappyPOD 2C ---
    print(f"Fitting GappyPOD 2C p_off / p_on (R={pod_R})...", flush=True)
    t0 = time.time()
    pod_off = GappyPODReconstructor.fit(
        train_frames,
        R=pod_R,
        ridge_lambda=1e-2,
        w_T=5.0,
        use_param_mean=False,
        n_velocity_components=2,
        seed=0,
    )
    pod_on = GappyPODReconstructor.fit(
        train_frames,
        R=pod_R,
        ridge_lambda=1e-2,
        w_T=5.0,
        use_param_mean=True,
        param_mean_predictor=param_mean,
        n_velocity_components=2,
        seed=0,
    )
    t_fit_pod = time.time() - t0
    print(
        f"  scales T={pod_off.scale_T:.4f} (POD internal) vs σ_T_norm={normalizer.temperature_std:.4f} "
        f"(Month-2 normalizer); both fit-only. fit_pod={t_fit_pod:.1f}s",
        flush=True,
    )

    search_rows: list[dict[str, Any]] = []

    # --- Hyperparameter search (contract grids; replica of historical search) ---
    print("VAL search temperature-free (λ only)...", flush=True)
    best_free: dict[str, Any] = {}
    for p_on in (False, True):
        pod_base = pod_on if p_on else pod_off
        best = None
        for lam in free_lams:
            pod = pod_base.with_hyperparams(
                ridge_lambda=float(lam),
                w_T=float(w_T_fixed_unused),
            )
            rows = evaluate_method_on_frames(
                method="GappyPOD_free",
                frames=val_frames,
                predict_fn=lambda mi, pod=pod, p_on=p_on: predict_gappy(
                    pod, mi, use_temperature_block=False, anchor_offset=False, p_conditioning=p_on
                ),
                normalizer=normalizer,
                cfg=cfg,
                K=0,
                p_conditioning=p_on,
                replica_ids=replicas,
                site_fraction=site_fraction,
                store=store,
                extra_tags={
                    "ridge_lambda": float(lam),
                    "w_T": float("nan"),
                    "R": pod_R,
                    "variant": "free",
                    "layout_id": layout_id,
                    "AO_tau": float("nan"),
                },
            )
            score = mean_over_replicas(rows)
            search_rows.append(
                {
                    "task": "temperature_free",
                    "p_conditioning": p_on,
                    "K": 0,
                    "ridge_lambda": float(lam),
                    "w_T": float("nan"),
                    "R": pod_R,
                    "score_mean_condition_rmse_K": score,
                    "n_val_frames": len(val_frames),
                    "n_replicas": len(replicas),
                }
            )
            print(f"  p_on={p_on} λ={lam:g} VAL={score:.2f}", flush=True)
            if best is None or score < best[0]:
                best = (score, float(lam), pod)
        assert best is not None
        best_free[p_on] = {"score": best[0], "ridge_lambda": best[1], "pod": best[2]}

    print("VAL search temperature-assisted T-block (K=9)...", flush=True)
    best_asst: dict[str, Any] = {}
    for p_on in (False, True):
        pod_base = pod_on if p_on else pod_off
        best = None
        for lam in asst_lams:
            for wT in asst_wT:
                pod = pod_base.with_hyperparams(ridge_lambda=float(lam), w_T=float(wT))
                rows = evaluate_method_on_frames(
                    method="GappyPOD_Tblock",
                    frames=val_frames,
                    predict_fn=lambda mi, pod=pod, p_on=p_on: predict_gappy(
                        pod,
                        mi,
                        use_temperature_block=True,
                        anchor_offset=False,
                        p_conditioning=p_on,
                    ),
                    normalizer=normalizer,
                    cfg=cfg,
                    K=9,
                    p_conditioning=p_on,
                    replica_ids=replicas,
                    site_fraction=site_fraction,
                    store=store,
                    extra_tags={
                        "ridge_lambda": float(lam),
                        "w_T": float(wT),
                        "R": pod_R,
                        "variant": "Tblock",
                        "layout_id": layout_id,
                        "AO_tau": float("nan"),
                    },
                )
                score = mean_over_replicas(rows)
                search_rows.append(
                    {
                        "task": "temperature_assisted_Tblock",
                        "p_conditioning": p_on,
                        "K": 9,
                        "ridge_lambda": float(lam),
                        "w_T": float(wT),
                        "R": pod_R,
                        "score_mean_condition_rmse_K": score,
                        "n_val_frames": len(val_frames),
                        "n_replicas": len(replicas),
                    }
                )
                print(f"  p_on={p_on} λ={lam:g} w_T={wT:g} VAL={score:.2f}", flush=True)
                if best is None or score < best[0]:
                    best = (score, float(lam), float(wT), pod)
        assert best is not None
        best_asst[p_on] = {
            "score": best[0],
            "ridge_lambda": best[1],
            "w_T": best[2],
            "pod": best[3],
        }

    # Selected free pods (for AO)
    selected = {
        "free_p_off": best_free[False],
        "free_p_on": best_free[True],
        "tblock_p_off": best_asst[False],
        "tblock_p_on": best_asst[True],
        "param_mean": {
            "basis": param_mean.selected_basis,
            "ridge_alpha": param_mean.selected_ridge_alpha,
            "loco_rmse_K": param_mean.loco_rmse_K,
        },
    }

    # --- LocalHGB dense diagnostic (optional) ---
    hgb_off = hgb_on = None
    if run_hgb:
        print("Fitting LocalHGB 2C dense diagnostic...", flush=True)
        t0 = time.time()
        hgb_off = LocalVelocityTemperatureRegressor.fit(
            train_frames,
            conditioned=False,
            include_vk=False,
            max_pixels=hgb_max_pixels,
            seed=0,
        )
        hgb_on = LocalVelocityTemperatureRegressor.fit(
            train_frames,
            conditioned=True,
            include_vk=False,
            max_pixels=hgb_max_pixels,
            seed=0,
        )
        print(f"  HGB fit {time.time()-t0:.1f}s", flush=True)

    # --- Full evaluation matrix ---
    print("Evaluating selected methods on train_fit / seen / VAL...", flush=True)
    all_rows: list[dict[str, Any]] = []
    roles_frames = {
        "train_fit": train_frames,
        "seen_train_eval": seen_frames,
        "val_eval": val_frames,
    }

    def add_eval(method, frames, predict_fn, K, p_on, tags, dense=False):
        nonlocal all_rows
        rows = evaluate_method_on_frames(
            method=method,
            frames=frames,
            predict_fn=predict_fn,
            normalizer=normalizer,
            cfg=cfg,
            K=K,
            p_conditioning=p_on,
            replica_ids=replicas,
            site_fraction=site_fraction,
            store=store,
            extra_tags=tags,
            use_dense_for_hgb=dense,
        )
        all_rows.extend(rows)

    for role, frames in roles_frames.items():
        # GlobalMean (p off)
        add_eval(
            "GlobalTrainMean",
            frames,
            lambda mi: predict_global_mean(global_mean),
            K=0,
            p_on=False,
            tags={"variant": "zero_sensor", "R": 0, "layout_id": layout_id, "AO_tau": float("nan")},
        )
        # ParamMean (p on)
        add_eval(
            "ParamConditionedMean",
            frames,
            lambda mi: predict_param_mean(param_mean, mi),
            K=0,
            p_on=True,
            tags={
                "variant": "zero_sensor",
                "R": 0,
                "ridge_alpha": param_mean.selected_ridge_alpha,
                "basis": param_mean.selected_basis,
                "layout_id": layout_id,
                "AO_tau": float("nan"),
            },
        )
        # ParamMean + AO for K>0
        for K in K_values:
            if K == 0:
                continue
            add_eval(
                "ParamMean_plus_AO",
                frames,
                lambda mi, K=K: predict_param_mean_ao(param_mean, mi, tau=ao_tau),
                K=K,
                p_on=True,
                tags={"variant": "AO", "AO_tau": ao_tau, "R": 0, "layout_id": layout_id},
            )

        for p_on, key_free, key_tb in (
            (False, "free_p_off", "tblock_p_off"),
            (True, "free_p_on", "tblock_p_on"),
        ):
            pod_free = selected[key_free]["pod"]
            pod_tb = selected[key_tb]["pod"]
            # Free K=0
            add_eval(
                "GappyPOD_free",
                frames,
                lambda mi, pod=pod_free, p_on=p_on: predict_gappy(
                    pod, mi, use_temperature_block=False, p_conditioning=p_on
                ),
                K=0,
                p_on=p_on,
                tags={
                    "variant": "free",
                    "R": pod_R,
                    "ridge_lambda": selected[key_free]["ridge_lambda"],
                    "w_T": float("nan"),
                    "layout_id": layout_id,
                    "AO_tau": float("nan"),
                },
            )
            for K in K_values:
                # AO on free selected
                if K > 0:
                    add_eval(
                        "GappyPOD_plus_AO",
                        frames,
                        lambda mi, pod=pod_free, p_on=p_on: predict_gappy(
                            pod,
                            mi,
                            use_temperature_block=False,
                            anchor_offset=True,
                            anchor_tau=ao_tau,
                            p_conditioning=p_on,
                        ),
                        K=K,
                        p_on=p_on,
                        tags={
                            "variant": "AO",
                            "AO_tau": ao_tau,
                            "R": pod_R,
                            "ridge_lambda": selected[key_free]["ridge_lambda"],
                            "w_T": float("nan"),
                            "free_source": "selected_free",
                            "layout_id": layout_id,
                        },
                    )
                # T-block with selected assisted hyperparams (K=0 must match free)
                add_eval(
                    "GappyPOD_Tblock",
                    frames,
                    lambda mi, pod=pod_tb, p_on=p_on, K=K: predict_gappy(
                        pod,
                        mi,
                        use_temperature_block=(K > 0),
                        anchor_offset=False,
                        p_conditioning=p_on,
                    ),
                    K=K,
                    p_on=p_on,
                    tags={
                        "variant": "Tblock",
                        "R": pod_R,
                        "ridge_lambda": selected[key_tb]["ridge_lambda"],
                        "w_T": selected[key_tb]["w_T"],
                        "layout_id": layout_id,
                        "AO_tau": float("nan"),
                    },
                )

        # Oracle (uses full T — labeled)
        for p_on, key in ((False, "free_p_off"), (True, "free_p_on")):
            pod = selected[key]["pod"]
            for fr in frames:
                for rep in replicas:
                    pred = pod.project_oracle_T(fr["T"], nh3=fr["nh3"], phi=fr["phi"])
                    temporal_tags = {
                        k: fr[k]
                        for k in (
                            "has_original_time_coordinate",
                            "timeline_regular",
                            "field_time_association_verified",
                            "time_validated",
                            "duplicate_of",
                            "time_coordinate_original",
                        )
                        if k in fr
                    }
                    row = evaluate_frame_prediction(
                        pred,
                        fr["T"],
                        train_sigma_T=normalizer.temperature_std,
                        sim_id=fr["sim_id"],
                        split=fr["split"],
                        role=fr["role"],
                        frame_index=fr["frame"],
                        method="OraclePOD",
                        p_conditioning=p_on,
                        K=-1,
                        site_fraction=1.0,
                        model_seed=0,
                        mask_replica_id=rep,
                        observation_hash="full_T_oracle",
                        variant="oracle_full_T",
                        R=pod_R,
                        deployable=False,
                        uses_full_T=True,
                        layout_id=layout_id,
                        AO_tau=float("nan"),
                        task_mode="oracle",
                        **temporal_tags,
                    )
                    all_rows.append(row)

        if run_hgb and hgb_off is not None and hgb_on is not None:
            for p_on, model in ((False, hgb_off), (True, hgb_on)):
                add_eval(
                    "LocalHGB_dense",
                    frames,
                    lambda fr, sample, model=model, p_on=p_on: model.predict_field(
                        fr["vi"],
                        fr["vj"],
                        fr["vk"],
                        fr["umag"],
                        fr["x2d"],
                        fr["y2d"],
                        nh3=fr["nh3"],
                        phi=fr["phi"],
                    ),
                    K=0,
                    p_on=p_on,
                    tags={
                        "variant": "dense_diagnostic",
                        "sparse_comparable": False,
                        "layout_id": layout_id,
                        "AO_tau": float("nan"),
                        "R": 0,
                    },
                    dense=True,
                )

    # --- Tables (never overwrite historical results/tables) ---
    df_frame = pd.DataFrame(all_rows)
    df_frame = df_frame[[c for c in df_frame.columns if not str(c).startswith("_")]]
    per_frame_path = write_csv(
        df_frame, out_dir / "month2_baseline_per_frame.csv", gzip_enabled=True
    )

    cond_parts = []
    group_cols = [
        c
        for c in [
            "method",
            "role",
            "p_conditioning",
            "K",
            "site_fraction",
            "variant",
            "ridge_lambda",
            "w_T",
            "R",
            "AO_tau",
            "layout_id",
            "mask_replica_id",
        ]
        if c in df_frame.columns
    ]
    for keys, g in df_frame.groupby(group_cols, dropna=False):
        rows = g.to_dict(orient="records")
        agg = aggregate_frame_then_condition_then_split(rows)
        pc = agg["per_condition"].copy()
        if isinstance(keys, tuple):
            for col, val in zip(group_cols, keys):
                pc[col] = val
        else:
            pc[group_cols[0]] = keys
        cond_parts.append(pc)
    if not cond_parts:
        raise RuntimeError("No per-condition aggregates produced")
    df_cond = pd.concat(cond_parts, ignore_index=True)
    write_csv(df_cond, out_dir / "month2_baseline_per_condition.csv")

    summary_rows = []
    metric_cols = [
        c
        for c in ["rmse_K", "abs_bias_K", "centered_rmse_K", "mae_K", "nrmse_train_sigma"]
        if c in df_cond.columns
    ]
    sum_group = [
        c
        for c in [
            "method",
            "role",
            "p_conditioning",
            "K",
            "site_fraction",
            "variant",
            "ridge_lambda",
            "w_T",
            "R",
            "AO_tau",
            "layout_id",
        ]
        if c in df_cond.columns
    ]
    for keys, g in df_cond.groupby(sum_group, dropna=False):
        rec = {}
        if isinstance(keys, tuple):
            for col, val in zip(sum_group, keys):
                rec[col] = val
        else:
            rec[sum_group[0]] = keys
        for m in metric_cols:
            rec[m] = float(g[m].mean())
        rec["n_condition_rows"] = int(len(g))
        summary_rows.append(rec)
    df_sum = pd.DataFrame(summary_rows)
    write_csv(df_sum, out_dir / "month2_baseline_summary.csv")

    df_search = pd.DataFrame(search_rows)
    write_csv(df_search, out_dir / "month2_baseline_search.csv")

    # Isolation battery on real VAL frames (fail-fast)
    print("Running isolation battery...", flush=True)
    checks = run_isolation_battery(
        frames=val_frames,
        make_sample=lambda fr, **kw: make_sample(
            fr,
            normalizer=normalizer,
            **kw,
        ),
        pods_free={
            False: selected["free_p_off"]["pod"],
            True: selected["free_p_on"]["pod"],
        },
        pods_tblock={
            False: selected["tblock_p_off"]["pod"],
            True: selected["tblock_p_on"]["pod"],
        },
        global_mean=global_mean,
        param_mean=param_mean,
        mask_seed=int(contract["mask_seed"]),
        site_fraction=site_fraction,
        max_frames=1 if smoke else 2,
    )
    checks.update(
        adapter_historical_parity(
            val_frames[:1],
            selected["free_p_off"]["pod"],
            normalizer,
            cfg,
        )
    )

    # Provenance for anomalous shard (verified when touched)
    anomalous_sha = None
    try:
        anomalous_sha = store.file_sha256("40NH3_1.2.h5")
    except Exception as e:
        anomalous_sha = f"unavailable:{e}"

    meta = {
        "experiment_id": cfg["experiment_id"],
        "run_id": run_id,
        "smoke": smoke,
        "wall_time_s": time.time() - t_wall0,
        "peak_rss_mb": rss_mb(),
        "store_stats": dict(store.stats),
        "n_train_fit": len(train_frames),
        "n_seen": len(seen_frames),
        "n_val": len(val_frames),
        "out_dir": str(out_dir),
        "per_frame_path": str(per_frame_path),
        "selected": {
            "free_p_off_lambda": selected["free_p_off"]["ridge_lambda"],
            "free_p_on_lambda": selected["free_p_on"]["ridge_lambda"],
            "tblock_p_off": {
                "ridge_lambda": selected["tblock_p_off"]["ridge_lambda"],
                "w_T": selected["tblock_p_off"]["w_T"],
            },
            "tblock_p_on": {
                "ridge_lambda": selected["tblock_p_on"]["ridge_lambda"],
                "w_T": selected["tblock_p_on"]["w_T"],
            },
            "param_mean": selected["param_mean"],
        },
        "fit_times_s": {"means": t_fit_means, "pod": t_fit_pod},
        "preflight": preflight,
        "hashes": {
            "baselines_yaml": sha256_file(CFG_PATH),
            "frame_protocol": sha256_file(ROOT / "configs/splits/month2_frame_protocol.yaml"),
            "norm_stats": sha256_file(stats_path),
            "splits": sha256_file(ROOT / "configs/splits/splits_interp_extrap.yaml"),
            "anomalous_shard_40NH3_1.2_sha256_verified": anomalous_sha,
        },
        "invariant_checks": checks,
        "comparison_rule": COMPARISON_RULE,
        "pod_vs_normalizer_scales": {
            "pod_scale_T": pod_off.scale_T,
            "pod_scale_vi": pod_off.scale_vi,
            "pod_scale_vj": pod_off.scale_vj,
            "month2_sigma_T": normalizer.temperature_std,
            "note": "POD uses its own fit-only train_fit std for state scaling; Month-2 normalizer is separate and used for NRMSE / neural path.",
        },
        "hgb_included": run_hgb,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")

    write_auto_report(cfg, df_sum, df_search, meta, df_cond, out_dir / "auto_report.md")
    print(
        json.dumps(
            {
                "status": "PASS",
                "run_id": run_id,
                "wall_s": meta["wall_time_s"],
                "rss_mb": meta["peak_rss_mb"],
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )
    return meta


def adapter_historical_parity(frames, pod, normalizer, cfg) -> dict[str, Any]:
    """Compare adapter path vs historical reconstruct API on identical observed grids."""
    if not frames:
        return {"parity_skipped": True}
    fr = frames[0]
    sample = make_sample(
        fr,
        K=9,
        p_conditioning=False,
        replica_id=0,
        mask_seed=int(cfg["data"]["mask_seed"]),
        site_fraction=float(cfg["data"]["site_fraction_primary"]),
        normalizer=normalizer,
    )
    mi = sample["model_inputs"]
    grids = observed_grids_from_model_inputs(mi)
    via_adapter = predict_gappy(
        pod, mi, use_temperature_block=True, p_conditioning=False
    )
    via_hist = pod.reconstruct(
        grids["vi"],
        grids["vj"],
        grids["vk"],
        vel_mask=grids["vel_mask"],
        T_obs=grids["T_obs"],
        T_mask=grids["T_mask"],
        nh3=0.0,
        phi=1.0,
        use_temperature_block=True,
        anchor_offset=False,
    )
    return {
        "adapter_vs_historical_maxabs": float(np.max(np.abs(via_adapter - via_hist))),
        "adapter_vs_historical_ok": bool(np.allclose(via_adapter, via_hist)),
    }



def write_auto_report(cfg, df_sum, df_search, meta, df_cond, path: Path) -> None:
    """Machine-generated report only (no editorial interpretation)."""
    def slice_sum(role):
        if df_sum.empty:
            return "(empty)"
        d = df_sum[df_sum["role"] == role]
        cols = [
            c
            for c in [
                "method",
                "p_conditioning",
                "K",
                "variant",
                "rmse_K",
                "abs_bias_K",
                "centered_rmse_K",
            ]
            if c in d.columns
        ]
        return d[cols].sort_values([c for c in ["method", "p_conditioning", "K"] if c in cols]).to_string(
            index=False
        )

    lines = [
        "# Month-2 baseline run — automatic report",
        "",
        f"- experiment_id: `{cfg['experiment_id']}`",
        f"- run_id: `{meta.get('run_id')}`",
        f"- smoke: {meta['smoke']}",
        f"- wall_time_s: {meta['wall_time_s']:.3f}",
        f"- peak_rss_mb: {meta['peak_rss_mb']:.3f}",
        f"- shard_decodes: {meta['store_stats'].get('shard_decodes')}",
        f"- git_head_preflight: `{meta.get('preflight', {}).get('git_head')}`",
        "",
        "## Selected hyperparameters",
        "```json",
        json.dumps(meta["selected"], indent=2),
        "```",
        "",
        "## Search table",
        "```",
        df_search.to_string(index=False) if not df_search.empty else "(empty)",
        "```",
        "",
        "## Summary seen_train_eval",
        "```",
        slice_sum("seen_train_eval"),
        "```",
        "",
        "## Summary val_eval",
        "```",
        slice_sum("val_eval"),
        "```",
        "",
        "## Invariant checks / costs",
        "```json",
        json.dumps(
            {
                "invariant_checks": meta["invariant_checks"],
                "fit_times_s": meta["fit_times_s"],
                "store_stats": meta["store_stats"],
                "comparison_rule": meta.get("comparison_rule"),
            },
            indent=2,
            default=str,
        ),
        "```",
        "",
        "Editorial interpretation belongs in docs/month2/mission_baseline_consolidation_report.md.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--run-id", required=True, help="Unique id under results/baseline_consolidation/")
    ap.add_argument(
        "--raw-dir",
        default=None,
        help="Optional override for Arrow numerical directory (or set COMBUSTION_RAW_DIR)",
    )
    args = ap.parse_args()
    run(
        smoke=args.smoke,
        run_id=args.run_id,
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
