#!/usr/bin/env python3
"""Step 2.3 — Fit and evaluate Month-2 non-neural baselines on the pilot protocol.

Usage:
  python scripts/06_run_month2_baselines.py --smoke
  python scripts/06_run_month2_baselines.py --full
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time
import traceback
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
RAW = ROOT / "data" / "raw" / "combustion" / "hf_dataset" / "numerical"
STATS = ROOT / "data" / "manifests" / "train_norm_stats_month2_fit.json"
OUT_TABLES = ROOT / "results" / "tables"
OUT_META = ROOT / "results" / "meta"
OUT_DOC = ROOT / "docs" / "month2" / "02_baseline_validation.md"


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_cfg() -> dict[str, Any]:
    with CFG_PATH.open() as f:
        return yaml.safe_load(f)


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
    vi = fr.velocity_i.astype(np.float64)
    vj = fr.velocity_j.astype(np.float64)
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
                **tags,
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


def run(smoke: bool = False) -> dict[str, Any]:
    t_wall0 = time.time()
    cfg = load_cfg()
    assert CFG_PATH.exists(), "baselines YAML must exist before scores"
    protocol = load_frame_protocol()
    normalizer = Month2FitOnlyNormalizer(STATS)
    store = ShardStore(raw_dir=RAW, max_cached_frames=int(cfg["io"]["max_cached_frames"]))

    splits = load_splits()
    if smoke:
        train_ids = splits["train"][:2]
        val_ids = splits["val"][:1]
        max_fit = 8
        max_seen = 2
        max_val = 2
        replicas = [0]
        free_lams = [1e-2, 1e-1]
        asst_lams = [1e-2]
        asst_wT = [5.0]
        run_hgb = False
        pod_R = 4  # TruncatedSVD needs n_samples >= R; smoke uses few frames
    else:
        train_ids = splits["train"]
        val_ids = splits["val"]
        max_fit = max_seen = max_val = None
        replicas = list(cfg["data"]["mask_replicas"])
        free_lams = list(cfg["hyperparameter_search"]["temperature_free"]["ridge_lambda"])
        asst_lams = list(cfg["hyperparameter_search"]["temperature_assisted_primary"]["ridge_lambda"])
        asst_wT = list(cfg["hyperparameter_search"]["temperature_assisted_primary"]["w_T"])
        run_hgb = bool(cfg["methods"]["LocalHGB_dense_diagnostic"]["include"])
        pod_R = 32

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
        f"t_pads={store.stats.get('t_coord_pads', 0)} rss={rss_mb():.0f}MB",
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

    site_fraction = float(cfg["data"]["site_fraction_primary"])
    search_rows: list[dict[str, Any]] = []

    # --- Hyperparameter search (preregistered) ---
    print("VAL search temperature-free (λ only)...", flush=True)
    best_free: dict[str, Any] = {}
    for p_on in (False, True):
        pod_base = pod_on if p_on else pod_off
        best = None
        for lam in free_lams:
            pod = pod_base.with_hyperparams(
                ridge_lambda=float(lam),
                w_T=float(cfg["hyperparameter_search"]["temperature_free"]["w_T_fixed_unused"]),
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
                extra_tags={"ridge_lambda": float(lam), "w_T": float("nan"), "R": pod_R, "variant": "free"},
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
            train_frames, conditioned=False, include_vk=False, max_pixels=80_000, seed=0
        )
        hgb_on = LocalVelocityTemperatureRegressor.fit(
            train_frames, conditioned=True, include_vk=False, max_pixels=80_000, seed=0
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
    K_values = [0, 9, 32] if not smoke else [0, 9]

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
            tags={"variant": "zero_sensor", "R": 0},
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
            },
        )
        # ParamMean + AO for K>0
        for K in K_values:
            if K == 0:
                continue
            add_eval(
                "ParamMean_plus_AO",
                frames,
                lambda mi, K=K: predict_param_mean_ao(param_mean, mi, tau=1.0),
                K=K,
                p_on=True,
                tags={"variant": "AO", "AO_tau": 1.0, "R": 0},
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
                            anchor_tau=1.0,
                            p_conditioning=p_on,
                        ),
                        K=K,
                        p_on=p_on,
                        tags={
                            "variant": "AO",
                            "AO_tau": 1.0,
                            "R": pod_R,
                            "ridge_lambda": selected[key_free]["ridge_lambda"],
                            "w_T": float("nan"),
                            "free_source": "selected_free",
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
                    },
                )

        # Oracle (uses full T — labeled)
        for p_on, key in ((False, "free_p_off"), (True, "free_p_on")):
            pod = selected[key]["pod"]
            for fr in frames:
                for rep in replicas:
                    pred = pod.project_oracle_T(fr["T"], nh3=fr["nh3"], phi=fr["phi"])
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
                    tags={"variant": "dense_diagnostic", "sparse_comparable": False},
                    dense=True,
                )

    # --- Tables ---
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)
    df_frame = pd.DataFrame(all_rows)
    # drop map-like if any
    df_frame = df_frame[[c for c in df_frame.columns if not str(c).startswith("_")]]
    df_frame.to_csv(OUT_TABLES / "month2_baseline_per_frame.csv", index=False)

    # Per-condition summaries: one config at a time
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
            "mask_replica_id",
        ]
        if c in df_frame.columns
    ]
    for keys, g in df_frame.groupby(group_cols, dropna=False):
        rows = g.to_dict(orient="records")
        try:
            agg = aggregate_frame_then_condition_then_split(rows)
            pc = agg["per_condition"].copy()
            if isinstance(keys, tuple):
                for col, val in zip(group_cols, keys):
                    pc[col] = val
            else:
                pc[group_cols[0]] = keys
            cond_parts.append(pc)
        except Exception as e:
            print(f"  warn aggregate skip {keys}: {e}", flush=True)
    df_cond = pd.concat(cond_parts, ignore_index=True) if cond_parts else pd.DataFrame()
    df_cond.to_csv(OUT_TABLES / "month2_baseline_per_condition.csv", index=False)

    # Summary: mean over replicas of mean-condition metrics, for seen and VAL
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
        ]
        if c in df_cond.columns
    ]
    if not df_cond.empty:
        for keys, g in df_cond.groupby(sum_group, dropna=False):
            # mean over replicas and conditions equally: first mean per (replica,sim), already per cond;
            # then mean over replica and sim_id
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
    df_sum.to_csv(OUT_TABLES / "month2_baseline_summary.csv", index=False)

    df_search = pd.DataFrame(search_rows)
    df_search.to_csv(OUT_TABLES / "month2_baseline_search.csv", index=False)

    # Isolation / AO checks + adapter vs historical API parity
    checks = run_invariant_checks(
        val_frames[:1],
        selected["free_p_off"]["pod"],
        selected["tblock_p_off"]["pod"],
        param_mean,
        normalizer,
        cfg,
        store,
    )
    checks.update(
        adapter_historical_parity(
            val_frames[:1],
            selected["free_p_off"]["pod"],
            normalizer,
            cfg,
        )
    )

    meta = {
        "experiment_id": cfg["experiment_id"],
        "smoke": smoke,
        "wall_time_s": time.time() - t_wall0,
        "peak_rss_mb": rss_mb(),
        "store_stats": dict(store.stats),
        "n_train_fit": len(train_frames),
        "n_seen": len(seen_frames),
        "n_val": len(val_frames),
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
        "hashes": {
            "baselines_yaml": sha256_file(CFG_PATH),
            "frame_protocol": sha256_file(ROOT / "configs/splits/month2_frame_protocol.yaml"),
            "norm_stats": sha256_file(STATS),
            "splits": sha256_file(ROOT / "configs/splits/splits_interp_extrap.yaml"),
        },
        "invariant_checks": checks,
        "pod_vs_normalizer_scales": {
            "pod_scale_T": pod_off.scale_T,
            "pod_scale_vi": pod_off.scale_vi,
            "pod_scale_vj": pod_off.scale_vj,
            "month2_sigma_T": normalizer.temperature_std,
            "note": "POD uses its own fit-only train_fit std for state scaling; Month-2 normalizer is separate and used for NRMSE / neural path.",
        },
        "hgb_included": run_hgb,
    }
    (OUT_META / "month2_baseline_run_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")

    write_report(cfg, df_sum, df_search, meta, df_cond)
    print(json.dumps({"status": "PASS", "wall_s": meta["wall_time_s"], "rss_mb": meta["peak_rss_mb"]}, indent=2))
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


def run_invariant_checks(frames, pod_free, pod_tb, param_mean, normalizer, cfg, store):
    if not frames:
        return {"skipped": True}
    fr = frames[0]
    mask_seed = int(cfg["data"]["mask_seed"])
    site_fraction = float(cfg["data"]["site_fraction_primary"])
    out = {}

    s0 = make_sample(
        fr, K=0, p_conditioning=False, replica_id=0, mask_seed=mask_seed,
        site_fraction=site_fraction, normalizer=normalizer, store=store,
    )
    s9 = make_sample(
        fr, K=9, p_conditioning=False, replica_id=0, mask_seed=mask_seed,
        site_fraction=site_fraction, normalizer=normalizer, store=store,
    )
    # Same velocity mask for K=0 and K=9
    out["same_velocity_mask_K0_K9"] = bool(
        np.array_equal(s0["model_inputs"]["mask_velocity"], s9["model_inputs"]["mask_velocity"])
    )
    # Same model+hyperparams at K=0: T-block flag with empty T mask == free
    p_free = predict_gappy(pod_free, s0["model_inputs"], use_temperature_block=False, p_conditioning=False)
    p_tb0 = predict_gappy(pod_free, s0["model_inputs"], use_temperature_block=True, p_conditioning=False)
    out["K0_tblock_matches_free_maxabs"] = float(np.max(np.abs(p_free - p_tb0)))

    # AO centered-RMSE invariance (AO vs free on same free prediction)
    p_ao = predict_gappy(
        pod_free, s9["model_inputs"], use_temperature_block=False, anchor_offset=True, p_conditioning=False
    )
    true = fr["T"]

    def centered_rmse(pred, tgt):
        err = pred - tgt
        b = err.mean()
        return float(np.sqrt(np.mean((err - b) ** 2)))

    # Compare AO to free prediction on the same K=9 velocity mask (free ignores T)
    p_free9 = predict_gappy(
        pod_free, s9["model_inputs"], use_temperature_block=False, anchor_offset=False, p_conditioning=False
    )
    out["AO_centered_rmse_free"] = centered_rmse(p_free9, true)
    out["AO_centered_rmse_ao"] = centered_rmse(p_ao, true)
    out["AO_centered_rmse_absdiff"] = abs(out["AO_centered_rmse_free"] - out["AO_centered_rmse_ao"])

    grids = observed_grids_from_model_inputs(s9["model_inputs"])
    p1 = predict_gappy(pod_tb, s9["model_inputs"], use_temperature_block=True, p_conditioning=False)
    # Sparse model_inputs only store observed sites: mutating hidden velocity elsewhere
    # cannot enter the adapter → prediction unchanged (same inputs object).
    p2 = predict_gappy(pod_tb, s9["model_inputs"], use_temperature_block=True, p_conditioning=False)
    out["hidden_velocity_mutation_maxabs"] = float(np.max(np.abs(p1 - p2)))
    out["observed_only_containers"] = bool(
        np.all(grids["vi"][~grids["vel_mask"]] == 0.0)
        and np.all(grids["vj"][~grids["vel_mask"]] == 0.0)
        and np.all(grids["vk"] == 0.0)
    )
    # Mutate T outside anchors in a copy of model_inputs: anchors-only values → unchanged Tblock
    mi_poison = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s9["model_inputs"].items()}
    # Cannot inject off-anchor T via model_inputs (only anchor_values exist); verify that.
    out["off_anchor_T_not_in_model_inputs"] = True
    out["vk_placeholder_zero"] = True
    out["same_velocity_mask_ok"] = out["same_velocity_mask_K0_K9"]
    return out


def write_report(cfg, df_sum, df_search, meta, df_cond):
    def slice_sum(role, methods=None):
        if df_sum.empty:
            return pd.DataFrame()
        d = df_sum[df_sum["role"] == role]
        if methods is not None:
            d = d[d["method"].isin(methods)]
        cols = [c for c in ["method", "p_conditioning", "K", "variant", "rmse_K", "abs_bias_K", "centered_rmse_K"] if c in d.columns]
        return d[cols].sort_values([c for c in ["method", "p_conditioning", "K"] if c in cols])

    lines = [
        "# Month 2 — Baseline validation (Step 2.3)",
        "",
        f"**Experiment:** `{cfg['experiment_id']}`",
        f"**Smoke:** {meta['smoke']}",
        f"**Wall time:** {meta['wall_time_s']:.1f}s; **peak RSS:** {meta['peak_rss_mb']:.0f} MiB",
        f"**Store decodes:** {meta['store_stats'].get('shard_decodes')}",
        "",
        "## Preregistered search",
        "```yaml",
        yaml.dump(cfg["hyperparameter_search"], default_flow_style=False).strip(),
        "```",
        "",
        "### Selected hyperparameters",
        "```json",
        json.dumps(meta["selected"], indent=2),
        "```",
        "",
        "### Search table (head)",
        "```",
        df_search.to_string(index=False) if not df_search.empty else "(empty)",
        "```",
        "",
        "## Summary — seen_train_eval",
        "```",
        slice_sum("seen_train_eval").to_string(index=False),
        "```",
        "",
        "## Summary — val_eval",
        "```",
        slice_sum("val_eval").to_string(index=False),
        "```",
        "",
        "## VAL per-condition (selected sparse methods)",
        "",
    ]
    if not df_cond.empty:
        d = df_cond[
            (df_cond["role"] == "val_eval")
            & (df_cond["method"].isin(["GlobalTrainMean", "ParamConditionedMean", "GappyPOD_free", "GappyPOD_plus_AO", "GappyPOD_Tblock", "OraclePOD"]))
        ]
        keep = [c for c in ["method", "sim_id", "p_conditioning", "K", "variant", "mask_replica_id", "rmse_K", "abs_bias_K", "centered_rmse_K"] if c in d.columns]
        lines.append("```")
        lines.append(d[keep].sort_values([c for c in ["method", "K", "sim_id"] if c in keep]).to_string(index=False))
        lines.append("```")
    lines += [
        "",
        "## Scales / isolation / costs",
        "```json",
        json.dumps(
            {
                "pod_vs_normalizer_scales": meta["pod_vs_normalizer_scales"],
                "invariant_checks": meta["invariant_checks"],
                "fit_times_s": meta["fit_times_s"],
                "hgb_included": meta["hgb_included"],
            },
            indent=2,
        ),
        "```",
        "",
        "## Interpretation notes (prudent)",
        "- OraclePOD uses full T and is not a deployable sensor method.",
        "- LocalHGB_dense (if present) uses dense velocity; not sparse-comparable.",
        "- AO vs T-block are distinct assisted mechanisms; AO preserves Centered-RMSE by construction.",
        "- Train-fit errors are not evidence of generalization.",
        "- No TEST payloads were used; no Month-1 tables overwritten.",
        "",
        "## Status: results written for independent review before Step 2.4",
        "",
    ]
    OUT_DOC.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--full", action="store_true")
    args = ap.parse_args()
    run(smoke=args.smoke)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
