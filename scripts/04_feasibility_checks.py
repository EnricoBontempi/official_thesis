#!/usr/bin/env python3
"""Step 1.5 — Inexpensive velocity→temperature feasibility checks (CPU-friendly).

Usage:
    python scripts/04_feasibility_checks.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import yaml
from scipy import stats as sp_stats
from sklearn.feature_selection import mutual_info_regression

from combustion_recon.data.masks import (
    build_layout_mask,
    stable_seed,
    temperature_anchor_mask,
)
from combustion_recon.evaluation.metrics import evaluate_sample
from combustion_recon.models.baselines import (
    GappyPODReconstructor,
    GlobalTrainMeanPredictor,
    LocalVelocityTemperatureRegressor,
    ParamConditionedMeanPredictor,
    apply_anchor_offset,
)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "configs" / "splits" / "splits_interp_extrap.yaml"
OC = ROOT / "data" / "manifests" / "operating_conditions.yaml"
STATS = ROOT / "data" / "manifests" / "train_norm_stats.json"
RAW = ROOT / "data" / "raw" / "combustion" / "hf_dataset" / "numerical"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures" / "month1_feasibility"

# Temporal protocol: fit on first 75%, buffer ≥2τ_c, evaluate seen on last 20%
FRAME_FIT = np.arange(0, 1500, 40)  # 38 frames/traj → 760
FRAME_SEEN = np.arange(1600, 2001, 40)  # 11 frames/traj → 220
FRAME_EVAL = np.arange(0, 2001, 50)  # 41 frames for val/interp/extrap
SEED = 20261114
SUMMARY_METRICS = [
    "rmse_K",
    "mae_K",
    "mean_signed_error_K",
    "abs_bias_K",
    "centered_rmse_K",
    "high_rmse_K",
    "jet_rmse_K",
    "non_high_rmse_K",
    "high_iou",
    "high_dice",
    "top25pct_iou",
    "mean_level_abs_error_K",
    "p95_abs_error_K",
    "r2",
]


def parse_sim_id(sim_id: str) -> tuple[int, float]:
    m = re.match(r"(\d+)NH3_(\d+\.?\d*)\.h5", sim_id)
    return int(m.group(1)), float(m.group(2))


def load_splits() -> dict[str, list[str]]:
    with SPLITS.open() as f:
        cfg = yaml.safe_load(f)
    return {k: list(cfg[k]["sim_ids"]) for k in ("train", "val", "test_interp", "test_extrap")}


def shard_map() -> dict[str, Path]:
    with OC.open() as f:
        oc = yaml.safe_load(f)
    return {c["sim_id"]: RAW / c["shard_filename"] for c in oc["conditions"]}


def load_traj(path: Path) -> dict[str, Any]:
    with pa.memory_map(str(path), "r") as mm:
        table = pa.ipc.open_stream(mm).read_all()
    row = {c: table.column(c)[0].as_py() for c in table.column_names}
    shape = (int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"]))
    nchan = int(row["numerical_channels"])
    numerical = np.frombuffer(row["numerical"], dtype=np.float32).reshape(*shape, nchan)
    x = np.frombuffer(row["x"], dtype=np.float64)
    y = np.frombuffer(row["y"], dtype=np.float64)
    return {"numerical": numerical, "x": x, "y": y, "sim_id": row["sim_id"]}


def mesh(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(x, y, indexing="xy")


def frame_pack(traj: dict[str, Any], fi: int, nh3: float, phi: float) -> dict[str, Any]:
    num = traj["numerical"][fi]
    vi, vj, vk, T = num[..., 11], num[..., 12], num[..., 13], num[..., 10]
    umag = np.sqrt(vi * vi + vj * vj + vk * vk)
    x2d, y2d = mesh(traj["x"], traj["y"])
    return {
        "vi": vi.astype(np.float64),
        "vj": vj.astype(np.float64),
        "vk": vk.astype(np.float64),
        "umag": umag.astype(np.float64),
        "T": T.astype(np.float64),
        "x2d": x2d,
        "y2d": y2d,
        "x": traj["x"],
        "y": traj["y"],
        "nh3": nh3,
        "phi": phi,
        "sim_id": traj["sim_id"],
        "frame": int(fi),
    }


def kinematic_fields(vi, vj, x, y):
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    dvi_dy, dvi_dx = np.gradient(vi, dy, dx)
    dvj_dy, dvj_dx = np.gradient(vj, dy, dx)
    div_preferred = dvj_dx + dvi_dy
    div_alt = dvi_dx + dvj_dy
    vort = np.abs(dvi_dx - dvj_dy)
    strain = np.sqrt(dvi_dx**2 + dvi_dy**2 + dvj_dx**2 + dvj_dy**2)
    return div_preferred, div_alt, vort, strain


def corr_block(a: np.ndarray, b: np.ndarray, max_mi: int = 8000, seed: int = 0) -> dict[str, float]:
    aa = a.ravel().astype(np.float64)
    bb = b.ravel().astype(np.float64)
    pear = float(np.corrcoef(aa, bb)[0, 1])
    spear = float(sp_stats.spearmanr(aa, bb).correlation)
    rng = np.random.default_rng(seed)
    if aa.size > max_mi:
        idx = rng.choice(aa.size, size=max_mi, replace=False)
        aa_s, bb_s = aa[idx], bb[idx]
    else:
        aa_s, bb_s = aa, bb
    mi = float(
        mutual_info_regression(aa_s.reshape(-1, 1), bb_s, random_state=seed, n_neighbors=5)[0]
    )
    return {"pearson": pear, "spearman": spear, "mutual_info": mi}


def eval_prediction(pred, true, sim_id, split, sigma_T) -> dict[str, Any]:
    return evaluate_sample(pred, true, train_sigma_T=sigma_T, sim_id=sim_id, split=split)


def mask_seed(sim_id: str, frame: int, tag: str = "vel") -> int:
    return stable_seed(SEED, sim_id, frame, tag)


def extract_frame_bank(sim_ids: list[str], frames: np.ndarray, smap: dict[str, Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sid in sim_ids:
        print(f"  load {sid}", flush=True)
        traj = load_traj(smap[sid])
        nh3, phi = parse_sim_id(sid)
        for fi in frames:
            if int(fi) < traj["numerical"].shape[0]:
                out.append(frame_pack(traj, int(fi), nh3 / 100.0, phi))
        del traj
    return out


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    smap = shard_map()
    sigma_T = float(json.loads(STATS.read_text())["fields"]["Temperature"]["std"])
    rng = np.random.default_rng(SEED)

    # ========== Check 1 (skip slow MI loop if CSV already present) ==========
    check1_csv = TABLES / "month1_check1_kinematic_correlations.csv"
    amb_ids = ["0NH3_1.h5", "40NH3_1.h5", "80NH3_1.05.h5"]

    def _ambiguity_and_front_sanity():
        amb_curves = {}
        front_pearson = None
        for sid in amb_ids:
            if sid not in smap:
                continue
            print(f"  amb/fig07 {sid}", flush=True)
            traj = load_traj(smap[sid])
            nh3, phi = parse_sim_id(sid)
            frames = FRAME_FIT[::2]
            packs = [frame_pack(traj, int(fi), nh3 / 100.0, phi) for fi in frames]
            umag = np.stack([p["umag"] for p in packs])
            T = np.stack([p["T"] for p in packs])
            um = umag.ravel()
            tt = T.ravel()
            bins = np.linspace(0, float(np.quantile(um, 0.99)), 16)
            inds = np.digitize(um, bins) - 1
            centers, means = [], []
            for b in range(len(bins) - 1):
                m = inds == b
                if m.sum() < 50:
                    continue
                centers.append(0.5 * (bins[b] + bins[b + 1]))
                means.append(float(tt[m].mean()))
            amb_curves[sid] = (np.array(centers), np.array(means), nh3)
            if sid == "0NH3_1.h5":
                fr = packs[len(packs) // 2]
                div, _, _, _ = kinematic_fields(fr["vi"], fr["vj"], traj["x"], traj["y"])
                dTy, dTx = np.gradient(fr["T"])
                gT = np.sqrt(dTx**2 + dTy**2)
                front_pearson = float(np.corrcoef(np.abs(div).ravel(), gT.ravel())[0, 1])
            del traj, packs
        return amb_curves, front_pearson

    print("\n=== Check 1: kinematic coupling ===", flush=True)
    if check1_csv.exists():
        print(f"  Reusing {check1_csv.name} (skip MI loop)", flush=True)
        df1 = pd.read_csv(check1_csv)
        amb_curves, front_pearson = _ambiguity_and_front_sanity()
    else:

        def run_check1_streaming(train_ids):
            rows = []
            pooled = {k: [] for k in ("vi", "vj", "vk", "umag", "div", "div_alt", "vort", "strain", "T")}
            amb_curves = {}
            for sid in train_ids:
                print(f"  check1 {sid}", flush=True)
                traj = load_traj(smap[sid])
                nh3, phi = parse_sim_id(sid)
                frames = FRAME_FIT[::2]
                packs = [frame_pack(traj, fi, nh3 / 100.0, phi) for fi in frames]
                vi = np.stack([p["vi"] for p in packs])
                vj = np.stack([p["vj"] for p in packs])
                vk = np.stack([p["vk"] for p in packs])
                umag = np.stack([p["umag"] for p in packs])
                T = np.stack([p["T"] for p in packs])
                divs, diva, vorts, strains = [], [], [], []
                for p in packs:
                    d, da, vo, st = kinematic_fields(p["vi"], p["vj"], traj["x"], traj["y"])
                    divs.append(d)
                    diva.append(da)
                    vorts.append(vo)
                    strains.append(st)
                div = np.stack(divs)
                div_alt = np.stack(diva)
                vort = np.stack(vorts)
                strain = np.stack(strains)
                for name, arr in [
                    ("vi", vi),
                    ("vj", vj),
                    ("vk", vk),
                    ("umag", umag),
                    ("div_preferred", div),
                    ("div_alt", div_alt),
                    ("vort", vort),
                    ("strain", strain),
                ]:
                    c = corr_block(arr, T, seed=stable_seed(sid, "corr"))
                    rows.append({"scope": "within_condition", "sim_id": sid, "feature": name, **c})
                for k, arr in [
                    ("vi", vi),
                    ("vj", vj),
                    ("vk", vk),
                    ("umag", umag),
                    ("div", div),
                    ("div_alt", div_alt),
                    ("vort", vort),
                    ("strain", strain),
                    ("T", T),
                ]:
                    pooled[k].append(arr.ravel()[::80])
                if (nh3 in (0, 40) and abs(phi - 1.0) < 0.06) or (
                    nh3 == 80 and abs(phi - 1.05) < 1e-6
                ):
                    um = umag.ravel()
                    tt = T.ravel()
                    bins = np.linspace(0, float(np.quantile(um, 0.99)), 16)
                    inds = np.digitize(um, bins) - 1
                    centers, means = [], []
                    for b in range(len(bins) - 1):
                        m = inds == b
                        if m.sum() < 50:
                            continue
                        centers.append(0.5 * (bins[b] + bins[b + 1]))
                        means.append(float(tt[m].mean()))
                    amb_curves[sid] = (np.array(centers), np.array(means), nh3)
                del traj, packs, vi, vj, vk, umag, T, div, div_alt, vort, strain
            Tp = np.concatenate(pooled["T"])
            for name in ("vi", "vj", "vk", "umag", "div", "div_alt", "vort", "strain"):
                c = corr_block(np.concatenate(pooled[name]), Tp, seed=1)
                rows.append({"scope": "pooled_train", "sim_id": "ALL_TRAIN", "feature": name, **c})
            return pd.DataFrame(rows), amb_curves

        df1, amb_curves = run_check1_streaming(splits["train"])
        df1.to_csv(check1_csv, index=False)
        _, front_pearson = _ambiguity_and_front_sanity()
    print(df1[df1.scope == "pooled_train"][["feature", "pearson", "spearman", "mutual_info"]], flush=True)
    if front_pearson is not None:
        print(f"  front sanity Pearson(|div|, |∇T|) on 0NH3_1 = {front_pearson:.4f}", flush=True)

    # ========== Frame banks (one load per trajectory) ==========
    print("\nExtracting frame banks...", flush=True)

    def extract_dual(sim_ids, frames_a, frames_b):
        bank_a, bank_b = [], []
        for sid in sim_ids:
            print(f"  load {sid}", flush=True)
            traj = load_traj(smap[sid])
            nh3, phi = parse_sim_id(sid)
            nT = traj["numerical"].shape[0]
            for fi in frames_a:
                if int(fi) < nT:
                    bank_a.append(frame_pack(traj, int(fi), nh3 / 100.0, phi))
            for fi in frames_b:
                if int(fi) < nT:
                    bank_b.append(frame_pack(traj, int(fi), nh3 / 100.0, phi))
            del traj
        return bank_a, bank_b

    train_frames, seen_frames = extract_dual(splits["train"], FRAME_FIT, FRAME_SEEN)
    val_frames = extract_frame_bank(splits["val"], FRAME_EVAL, smap)
    interp_frames = extract_frame_bank(splits["test_interp"], FRAME_EVAL, smap)
    extrap_frames = extract_frame_bank(splits["test_extrap"], FRAME_EVAL, smap)
    print(
        f"Frames: train_fit={len(train_frames)} seen={len(seen_frames)} "
        f"val={len(val_frames)} interp={len(interp_frames)} extrap={len(extrap_frames)}",
        flush=True,
    )

    cond_means, cond_ops, all_train_T = [], [], []
    by_sid: dict[str, list] = {}
    for fr in train_frames:
        by_sid.setdefault(fr["sim_id"], []).append(fr["T"])
    for sid, Ts in by_sid.items():
        stack = np.stack(Ts, axis=0)
        cond_means.append(stack.mean(axis=0))
        nh3, phi = parse_sim_id(sid)
        cond_ops.append((nh3 / 100.0, phi))
        all_train_T.extend(list(stack[::2]))

    print("Fitting mean predictors...", flush=True)
    global_mean = GlobalTrainMeanPredictor.fit(all_train_T)
    param_mean = ParamConditionedMeanPredictor.fit(cond_means, cond_ops, ridge_alpha="auto")
    print(
        f"  ParamMean selected basis={param_mean.selected_basis} "
        f"α={param_mean.selected_ridge_alpha:g} LOCO RMSE={param_mean.loco_rmse_K:.2f} K",
        flush=True,
    )

    print("Fitting local regressors (2D & 3D velocity)...", flush=True)
    fit_idx = rng.choice(len(train_frames), size=min(300, len(train_frames)), replace=False)
    fit_frames = [train_frames[i] for i in fit_idx]
    local_unc_3 = LocalVelocityTemperatureRegressor.fit(
        fit_frames, conditioned=False, max_pixels=60_000, seed=SEED, include_vk=True
    )
    local_cond_3 = LocalVelocityTemperatureRegressor.fit(
        fit_frames, conditioned=True, max_pixels=60_000, seed=SEED + 1, include_vk=True
    )
    local_cond_2 = LocalVelocityTemperatureRegressor.fit(
        fit_frames, conditioned=True, max_pixels=60_000, seed=SEED + 3, include_vk=False
    )

    print("Fitting Gappy POD on ALL train-fit frames (2-comp & 3-comp)...", flush=True)
    pod_base_3 = GappyPODReconstructor.fit(
        train_frames,
        R=32,
        ridge_lambda=1e-3,
        w_T=20.0,
        use_param_mean=False,
        max_train_frames=None,
        n_velocity_components=3,
        seed=SEED,
    )
    pod_base_2 = GappyPODReconstructor.fit(
        train_frames,
        R=32,
        ridge_lambda=1e-3,
        w_T=20.0,
        use_param_mean=False,
        max_train_frames=None,
        n_velocity_components=2,
        seed=SEED + 5,
    )
    pod_cond_base = GappyPODReconstructor.fit(
        train_frames,
        R=32,
        ridge_lambda=1e-3,
        w_T=20.0,
        use_param_mean=True,
        param_mean_predictor=param_mean,
        max_train_frames=None,
        n_velocity_components=3,
        seed=SEED + 2,
    )

    # Select λ, w_T on val (conditioned POD, random_5pct + K=4 assisted)
    print("Selecting λ, w_T on val...", flush=True)
    best = None
    for lam in (1e-4, 1e-3, 1e-2):
        for wT in (5.0, 20.0, 50.0):
            pod_try = pod_cond_base.with_hyperparams(ridge_lambda=lam, w_T=wT)
            errs = []
            for fr in val_frames[::4]:
                vel_mask = build_layout_mask(
                    "random_5pct", seed=mask_seed(fr["sim_id"], fr["frame"], "vel")
                )
                t_mask = temperature_anchor_mask(
                    4, seed=mask_seed(fr["sim_id"], fr["frame"], "temp"), nested=True
                )
                pred = pod_try.reconstruct(
                    fr["vi"],
                    fr["vj"],
                    fr["vk"],
                    vel_mask=vel_mask,
                    T_obs=fr["T"],
                    T_mask=t_mask,
                    nh3=fr["nh3"],
                    phi=fr["phi"],
                    use_temperature_block=True,
                )
                errs.append(np.sqrt(np.mean((pred - fr["T"]) ** 2)))
            score = float(np.mean(errs))
            print(f"  λ={lam:g} w_T={wT:g} val RMSE={score:.2f}", flush=True)
            if best is None or score < best[0]:
                best = (score, lam, wT)
    assert best is not None
    _, best_lam, best_wT = best
    print(f"Selected λ={best_lam:g}, w_T={best_wT:g}", flush=True)

    pod_unc = pod_base_3.with_hyperparams(ridge_lambda=best_lam, w_T=best_wT)
    pod_unc_2 = pod_base_2.with_hyperparams(ridge_lambda=best_lam, w_T=best_wT)
    pod_cond = pod_cond_base.with_hyperparams(ridge_lambda=best_lam, w_T=best_wT)

    # Also try R=64 conditioned briefly on val temp-free
    print("Confirming R on val (temp-free 5%)...", flush=True)
    pod64 = GappyPODReconstructor.fit(
        train_frames,
        R=64,
        ridge_lambda=best_lam,
        w_T=best_wT,
        use_param_mean=True,
        param_mean_predictor=param_mean,
        max_train_frames=None,
        n_velocity_components=3,
        seed=SEED + 7,
    )
    def _val_tf_rmse(pod):
        errs = []
        for fr in val_frames[::4]:
            mask = build_layout_mask("random_5pct", seed=mask_seed(fr["sim_id"], fr["frame"], "vel"))
            pred = pod.reconstruct(
                fr["vi"], fr["vj"], fr["vk"], vel_mask=mask, nh3=fr["nh3"], phi=fr["phi"]
            )
            errs.append(np.sqrt(np.mean((pred - fr["T"]) ** 2)))
        return float(np.mean(errs))

    r32 = _val_tf_rmse(pod_cond)
    r64 = _val_tf_rmse(pod64)
    print(f"  R=32 val RMSE={r32:.2f}  R=64 val RMSE={r64:.2f}", flush=True)
    best_R = 32
    if r64 < r32:
        best_R = 64
        pod_cond = pod64
        # rebuild unconditioned at R=64 for fairness
        pod_unc = GappyPODReconstructor.fit(
            train_frames,
            R=64,
            ridge_lambda=best_lam,
            w_T=best_wT,
            use_param_mean=False,
            max_train_frames=None,
            n_velocity_components=3,
            seed=SEED,
        )
        pod_unc_2 = GappyPODReconstructor.fit(
            train_frames,
            R=64,
            ridge_lambda=best_lam,
            w_T=best_wT,
            use_param_mean=False,
            max_train_frames=None,
            n_velocity_components=2,
            seed=SEED + 5,
        )
    print(f"Selected R={best_R}", flush=True)

    # ========== Evaluations ==========
    metric_rows: list[dict[str, Any]] = []

    def add_rows(results, method, task_mode, layout, K=0, n_vel_comp=3):
        for r in results:
            d = {k: v for k, v in r.items() if k != "_spatial_bias_map"}
            d.update(
                {
                    "method": method,
                    "task_mode": task_mode,
                    "layout": layout,
                    "K_temp_anchors": K,
                    "n_velocity_components": n_vel_comp,
                }
            )
            metric_rows.append(d)

    eval_sets = [
        ("seen_train_eval", seen_frames),
        ("val", val_frames),
        ("test_interp", interp_frames),
        ("test_extrap", extrap_frames),
    ]

    print("\n=== Check 2: mean predictors ===", flush=True)
    for split_name, frames in eval_sets:
        res_g, res_p = [], []
        for fr in frames:
            res_g.append(eval_prediction(global_mean.predict(), fr["T"], fr["sim_id"], split_name, sigma_T))
            res_p.append(
                eval_prediction(param_mean.predict(fr["nh3"], fr["phi"]), fr["T"], fr["sim_id"], split_name, sigma_T)
            )
        add_rows(res_g, "GlobalTrainMean", "temperature_free", "none", 0)
        add_rows(res_p, "ParamConditionedMean", "temperature_free", "none", 0)

    # ParamMean + Anchor Offset K sweep
    print("\n=== ParamMean + Anchor Offset ===", flush=True)
    for K in (0, 1, 4, 9, 16, 32):
        for split_name, frames in eval_sets:
            res = []
            for fr in frames:
                t_mask = temperature_anchor_mask(
                    K, seed=mask_seed(fr["sim_id"], fr["frame"], "temp"), nested=True
                )
                T0 = param_mean.predict(fr["nh3"], fr["phi"])
                if K > 0:
                    pred, _ = apply_anchor_offset(T0, fr["T"], t_mask, tau=1.0)
                else:
                    pred = T0
                res.append(eval_prediction(pred, fr["T"], fr["sim_id"], split_name, sigma_T))
            add_rows(res, "ParamMean_AnchorOffset", "temperature_assisted", "none", K)
        print(f"  done K={K}", flush=True)

    print("\n=== Check 3: local regressors ===", flush=True)
    for split_name, frames in eval_sets:
        res_u, res_c3, res_c2 = [], [], []
        for fr in frames:
            pred_u = local_unc_3.predict_field(
                fr["vi"], fr["vj"], fr["vk"], fr["umag"], fr["x2d"], fr["y2d"]
            )
            pred_c3 = local_cond_3.predict_field(
                fr["vi"], fr["vj"], fr["vk"], fr["umag"], fr["x2d"], fr["y2d"], nh3=fr["nh3"], phi=fr["phi"]
            )
            pred_c2 = local_cond_2.predict_field(
                fr["vi"], fr["vj"], fr["vk"], fr["umag"], fr["x2d"], fr["y2d"], nh3=fr["nh3"], phi=fr["phi"]
            )
            res_u.append(eval_prediction(pred_u, fr["T"], fr["sim_id"], split_name, sigma_T))
            res_c3.append(eval_prediction(pred_c3, fr["T"], fr["sim_id"], split_name, sigma_T))
            res_c2.append(eval_prediction(pred_c2, fr["T"], fr["sim_id"], split_name, sigma_T))
        add_rows(res_u, "LocalHGB_Unconditioned", "temperature_free", "full_grid_velocity", 0, 3)
        add_rows(res_c3, "LocalHGB_Conditioned", "temperature_free", "full_grid_velocity", 0, 3)
        add_rows(res_c2, "LocalHGB_Conditioned_2comp", "temperature_free", "full_grid_velocity", 0, 2)

    print("\n=== Check 4a: Gappy POD sparsity sweep ===", flush=True)
    sparsity_layouts = [
        "random_1pct",
        "random_2pct",
        "random_5pct",
        "random_10pct",
        "random_20pct",
        "missing_shear_layer_box",
        "axial_line_mid",
        "radial_line_mid",
    ]
    for layout in sparsity_layouts:
        for method_name, pod, nvc in [
            ("GappyPOD_Unconditioned", pod_unc, 3),
            ("GappyPOD_Unconditioned_2comp", pod_unc_2, 2),
            ("GappyPOD_Conditioned", pod_cond, 3),
        ]:
            for split_name, frames in eval_sets:
                use = frames if layout.startswith("random") else frames[::2]
                res = []
                for fr in use:
                    mask = build_layout_mask(
                        layout, seed=mask_seed(fr["sim_id"], fr["frame"], layout)
                    )
                    pred = pod.reconstruct(
                        fr["vi"],
                        fr["vj"],
                        fr["vk"],
                        vel_mask=mask,
                        nh3=fr["nh3"],
                        phi=fr["phi"],
                    )
                    res.append(eval_prediction(pred, fr["T"], fr["sim_id"], split_name, sigma_T))
                add_rows(res, method_name, "temperature_free", layout, 0, nvc)
        print(f"  done {layout}", flush=True)

    # Oracle POD temperature projection floor (full-T, Check 4a companion)
    print("\n=== Oracle POD T projection floor ===", flush=True)
    for method_name, pod, nvc in [
        ("OraclePOD_Unconditioned", pod_unc, 3),
        ("OraclePOD_Conditioned", pod_cond, 3),
    ]:
        for split_name, frames in eval_sets:
            res = []
            for fr in frames:
                pred = pod.project_oracle_T(fr["T"], nh3=fr["nh3"], phi=fr["phi"])
                res.append(eval_prediction(pred, fr["T"], fr["sim_id"], split_name, sigma_T))
            add_rows(res, method_name, "temperature_free", "oracle_full_T", 0, nvc)
        print(f"  done {method_name}", flush=True)

    print("\n=== Check 4b: temperature-assisted (weighted T-block + AnchorOffset) ===", flush=True)
    for K in (0, 1, 4, 9, 16, 32):
        for method_name, pod, mode, nvc in [
            ("GappyPOD_Conditioned", pod_cond, "weighted_T_block", 3),
            ("GappyPOD_Unconditioned", pod_unc, "weighted_T_block", 3),
            ("GappyPOD_Conditioned_AnchorOffset", pod_cond, "anchor_offset", 3),
            ("GappyPOD_Unconditioned_AnchorOffset", pod_unc, "anchor_offset", 3),
        ]:
            for split_name, frames in eval_sets:
                res = []
                for fr in frames:  # same frames + same mask seed as Check 4a @ random_5pct
                    vel_mask = build_layout_mask(
                        "random_5pct",
                        seed=mask_seed(fr["sim_id"], fr["frame"], "random_5pct"),
                    )
                    t_mask = temperature_anchor_mask(
                        K, seed=mask_seed(fr["sim_id"], fr["frame"], "temp"), nested=True
                    )
                    if mode == "anchor_offset":
                        pred = pod.reconstruct(
                            fr["vi"],
                            fr["vj"],
                            fr["vk"],
                            vel_mask=vel_mask,
                            T_obs=fr["T"] if K > 0 else None,
                            T_mask=t_mask if K > 0 else None,
                            nh3=fr["nh3"],
                            phi=fr["phi"],
                            use_temperature_block=False,
                            anchor_offset=(K > 0),
                        )
                    else:
                        pred = pod.reconstruct(
                            fr["vi"],
                            fr["vj"],
                            fr["vk"],
                            vel_mask=vel_mask,
                            T_obs=fr["T"] if K > 0 else None,
                            T_mask=t_mask if K > 0 else None,
                            nh3=fr["nh3"],
                            phi=fr["phi"],
                            use_temperature_block=(K > 0),
                            anchor_offset=False,
                        )
                    res.append(eval_prediction(pred, fr["T"], fr["sim_id"], split_name, sigma_T))
                add_rows(res, method_name, "temperature_assisted", "random_5pct", K, nvc)
        print(f"  done K={K}", flush=True)

    df_metrics = pd.DataFrame(metric_rows)
    df_metrics.to_csv(TABLES / "month1_check2_3_4_per_condition_metrics.csv", index=False)

    # Per-condition summary (every sim_id visible)
    cols = [c for c in SUMMARY_METRICS if c in df_metrics.columns]
    group_keys = [
        "task_mode",
        "method",
        "layout",
        "K_temp_anchors",
        "n_velocity_components",
        "split",
        "sim_id",
    ]
    df_per_cond = df_metrics.groupby(group_keys, as_index=False)[cols].mean()
    df_per_cond.to_csv(TABLES / "month1_per_condition_summary.csv", index=False)

    summary_rows = []
    for task_mode in ("temperature_free", "temperature_assisted"):
        sub = df_metrics[df_metrics.task_mode == task_mode]
        if sub.empty:
            continue
        g = sub.groupby(
            ["task_mode", "method", "layout", "K_temp_anchors", "n_velocity_components", "split"],
            as_index=False,
        )[cols].mean()
        summary_rows.append(g)
    df_sum = pd.concat(summary_rows, ignore_index=True)
    df_sum.to_csv(TABLES / "month1_feasibility_split_summary.csv", index=False)

    print("\n=== Key summary (temp-free / assisted) ===", flush=True)
    show = df_sum[
        (
            (df_sum.layout.isin(["none", "full_grid_velocity", "random_5pct"]))
            & (df_sum.task_mode == "temperature_free")
            & (df_sum.n_velocity_components == 3)
        )
        | (
            (df_sum.task_mode == "temperature_assisted")
            & (df_sum.K_temp_anchors.isin([0, 4, 16, 32]))
            & (df_sum.method.str.contains("AnchorOffset|ParamMean|GappyPOD_Conditioned$"))
        )
    ]
    cols_show = [
        "task_mode",
        "method",
        "layout",
        "K_temp_anchors",
        "split",
        "rmse_K",
        "high_rmse_K",
        "jet_rmse_K",
        "mean_signed_error_K",
        "centered_rmse_K",
        "high_iou",
        "top25pct_iou",
    ]
    print(show[cols_show].to_string(index=False), flush=True)

    print("\nGenerating figures...", flush=True)
    _make_figures(smap, df1, amb_curves, df_sum, pod_unc, pod_cond)

    (TABLES / "month1_feasibility_meta.json").write_text(
        json.dumps(
            {
                "best_pod_R": best_R,
                "best_ridge_lambda": best_lam,
                "best_w_T": best_wT,
                "param_mean_basis": param_mean.selected_basis,
                "param_mean_ridge_alpha": param_mean.selected_ridge_alpha,
                "param_mean_loco_rmse_K": param_mean.loco_rmse_K,
                "front_div_gradT_pearson": front_pearson,
                "sigma_T": sigma_T,
                "n_train_fit_frames": len(train_frames),
                "n_seen_frames": len(seen_frames),
                "frame_fit": FRAME_FIT.tolist(),
                "frame_seen": FRAME_SEEN.tolist(),
                "frame_eval": FRAME_EVAL.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    print("Done.", flush=True)


def _make_figures(smap, df1, amb_curves, df_sum, pod_unc, pod_cond):
    # fig07
    sid0 = "0NH3_1.h5"
    traj0 = load_traj(smap[sid0])
    fr0 = frame_pack(traj0, 1000, 0.0, 1.0)
    div, _, vort, _ = kinematic_fields(fr0["vi"], fr0["vj"], traj0["x"], traj0["y"])
    del traj0
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    # Symmetric RdBu for divergence about zero
    div_lim = float(np.nanpercentile(np.abs(div), 99))
    panels = [
        (axes[0, 0], fr0["T"], "T [K]", "inferno", None, None),
        (axes[0, 1], fr0["umag"], r"$|\mathbf{u}|$ [m/s]", "viridis", None, None),
        (axes[0, 2], div, r"$\nabla\cdot u$ (preferred)", "RdBu_r", -div_lim, div_lim),
        (axes[1, 0], vort, r"$|\omega_z|$", "magma", None, None),
    ]
    for ax, data, title, cmap, vmin, vmax in panels:
        im = ax.imshow(data, origin="lower", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1, 1]
    pooled = df1[df1.scope == "pooled_train"].set_index("feature")
    feats = ["vi", "vj", "vk", "umag", "div", "vort", "strain"]
    feat_labels, pearsons = [], []
    for f in feats:
        key = f if f in pooled.index else ("div_preferred" if f == "div" else f)
        if key in pooled.index:
            feat_labels.append(f)
            pearsons.append(float(pooled.loc[key, "pearson"]))
    ax.barh(feat_labels, pearsons, color="#456990")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Pooled Pearson corr with T")
    ax.set_xlabel("r")
    ax = axes[1, 2]
    for sid, (c, m, nh3) in amb_curves.items():
        ax.plot(c, m, marker="o", ms=3, label=f"{nh3}% NH$_3$")
    ax.set_xlabel(r"$|\mathbf{u}|$ [m/s]")
    ax.set_ylabel(r"$\mathbb{E}[T \mid |u|]$ [K]")
    ax.set_title("Cross-condition thermal ambiguity")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Check 1 — kinematic coupling & ambiguity", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGS / "fig07_kinematic_coupling_and_ambiguity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # fig08 — 2x2 panels
    methods_fig8 = [
        ("GlobalTrainMean", "none", "temperature_free", 3),
        ("ParamConditionedMean", "none", "temperature_free", 3),
        ("LocalHGB_Unconditioned", "full_grid_velocity", "temperature_free", 3),
        ("LocalHGB_Conditioned", "full_grid_velocity", "temperature_free", 3),
        ("GappyPOD_Unconditioned", "random_5pct", "temperature_free", 3),
        ("GappyPOD_Conditioned", "random_5pct", "temperature_free", 3),
    ]
    splits_order = ["seen_train_eval", "val", "test_interp", "test_extrap"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

    def _ys(method, layout, mode, nvc, met):
        out = []
        for sp in splits_order:
            sub = df_sum[
                (df_sum.method == method)
                & (df_sum.layout == layout)
                & (df_sum.task_mode == mode)
                & (df_sum.n_velocity_components == nvc)
                & (df_sum.split == sp)
                & (df_sum.K_temp_anchors == 0)
            ]
            out.append(float(sub[met].mean()) if len(sub) else np.nan)
        return out

    # (1) Full-field RMSE
    ax = axes[0, 0]
    for method, layout, mode, nvc in methods_fig8:
        ax.plot(splits_order, _ys(method, layout, mode, nvc, "rmse_K"), marker="o", label=method, lw=1.4)
    ax.set_title("(1) Full-field RMSE [K]")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=5.5, loc="best")

    # (2) High-T RMSE & Jet RMSE (conditioned vs unconditioned Gappy + Local)
    ax = axes[0, 1]
    for method, layout, mode, nvc, met, ls in [
        ("LocalHGB_Conditioned", "full_grid_velocity", "temperature_free", 3, "high_rmse_K", "-"),
        ("LocalHGB_Conditioned", "full_grid_velocity", "temperature_free", 3, "jet_rmse_K", "--"),
        ("GappyPOD_Conditioned", "random_5pct", "temperature_free", 3, "high_rmse_K", "-"),
        ("GappyPOD_Conditioned", "random_5pct", "temperature_free", 3, "jet_rmse_K", "--"),
        ("GappyPOD_Unconditioned", "random_5pct", "temperature_free", 3, "high_rmse_K", ":"),
        ("ParamConditionedMean", "none", "temperature_free", 3, "high_rmse_K", "-."),
    ]:
        label = f"{method.split('_')[0]}/{met.replace('_K','')}"
        ax.plot(splits_order, _ys(method, layout, mode, nvc, met), ls=ls, marker="o", label=label, lw=1.3)
    ax.set_title("(2) High-T RMSE vs Jet/Front RMSE [K]")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=5.5, loc="best")

    # (3) Bias vs centered pattern RMSE
    ax = axes[1, 0]
    for method, layout, mode, nvc in methods_fig8:
        ax.plot(
            splits_order,
            _ys(method, layout, mode, nvc, "mean_signed_error_K"),
            marker="o",
            ls="-",
            label=f"{method} bias",
            lw=1.15,
        )
        ax.plot(
            splits_order,
            _ys(method, layout, mode, nvc, "centered_rmse_K"),
            marker="s",
            ls="--",
            label=f"{method} ctrRMSE",
            lw=1.0,
            alpha=0.85,
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("(3) Mean signed bias vs centered pattern RMSE [K]")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=4.5, loc="best", ncol=2)

    # (4) High-T IoU and relative top-25% core IoU
    ax2 = axes[1, 1]
    for method, layout, mode, nvc in methods_fig8:
        ax2.plot(
            splits_order,
            _ys(method, layout, mode, nvc, "high_iou"),
            marker="o",
            ls="-",
            label=f"{method} highIoU",
            lw=1.2,
        )
        ax2.plot(
            splits_order,
            _ys(method, layout, mode, nvc, "top25pct_iou"),
            marker="^",
            ls="--",
            label=f"{method} top25",
            lw=1.0,
            alpha=0.85,
        )
    ax2.set_title("(4) High-T IoU ($T\\geq2100$K) & top-25% core IoU")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=4.5, loc="best", ncol=2)

    fig.suptitle("Unconditioned vs conditioned transfer (corrected MAP Gappy @5%)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGS / "fig08_unconditioned_vs_conditioned_transfer.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # fig09
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    cover = {
        "random_1pct": 1,
        "random_2pct": 2,
        "random_5pct": 5,
        "random_10pct": 10,
        "random_20pct": 20,
    }
    ax = axes[0]
    for split in ("val", "test_interp", "test_extrap"):
        for method, ls in [("GappyPOD_Unconditioned", "-"), ("GappyPOD_Conditioned", "--")]:
            xs, ys = [], []
            for layout, pct in cover.items():
                sub = df_sum[
                    (df_sum.method == method)
                    & (df_sum.layout == layout)
                    & (df_sum.split == split)
                    & (df_sum.task_mode == "temperature_free")
                    & (df_sum.n_velocity_components == 3)
                ]
                if len(sub):
                    xs.append(pct)
                    ys.append(float(sub.rmse_K.mean()))
            ax.plot(xs, ys, ls=ls, marker="o", label=f"{split}/{method.split('_')[1][:4]}")
    ax.set_xlabel("Velocity coverage [%]")
    ax.set_ylabel("RMSE [K]")
    ax.set_title("(a) Temperature-free sparsity (corrected)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)

    ax = axes[1]
    for split in ("val", "test_interp", "test_extrap"):
        for method, ls, met in [
            ("GappyPOD_Conditioned_AnchorOffset", "-", "rmse_K"),
            ("GappyPOD_Conditioned_AnchorOffset", "--", "abs_bias_K"),
            ("GappyPOD_Conditioned_AnchorOffset", ":", "high_rmse_K"),
            ("ParamMean_AnchorOffset", "-.", "rmse_K"),
        ]:
            xs, ys = [], []
            for K in (0, 1, 4, 9, 16, 32):
                sub = df_sum[
                    (df_sum.method == method)
                    & (df_sum.task_mode == "temperature_assisted")
                    & (df_sum.K_temp_anchors == K)
                    & (df_sum.split == split)
                ]
                if len(sub) and met in sub.columns:
                    xs.append(K)
                    ys.append(float(sub[met].mean()))
            short = "Gappy+AO" if "Gappy" in method else "Param+AO"
            ax.plot(xs, ys, ls=ls, marker="o", label=f"{split}/{short}/{met}", lw=1.2)
    ax.set_xlabel("K nested temperature anchors")
    ax.set_ylabel("RMSE / |bias| / High-T RMSE [K]")
    ax.set_title("(b) Anchor-calibrated K sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig09_sparsity_and_temp_assisted_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # fig10
    examples = [
        ("test_interp", "60NH3_0.95.h5", 1000),
        ("test_extrap", "20NH3_0.75.h5", 1000),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(14, 6.2))
    for r, (_split, sid, fi) in enumerate(examples):
        nh3, phi = parse_sim_id(sid)
        traj = load_traj(smap[sid])
        fr = frame_pack(traj, fi, nh3 / 100.0, phi)
        del traj
        vel_mask = build_layout_mask(
            "random_5pct", seed=mask_seed(sid, fi, "random_5pct")
        )
        t_mask = temperature_anchor_mask(4, seed=mask_seed(sid, fi, "temp"), nested=True)
        pred_unc = pod_unc.reconstruct(
            fr["vi"], fr["vj"], fr["vk"], vel_mask=vel_mask, nh3=fr["nh3"], phi=fr["phi"]
        )
        pred_cond = pod_cond.reconstruct(
            fr["vi"], fr["vj"], fr["vk"], vel_mask=vel_mask, nh3=fr["nh3"], phi=fr["phi"]
        )
        pred_asst = pod_cond.reconstruct(
            fr["vi"],
            fr["vj"],
            fr["vk"],
            vel_mask=vel_mask,
            T_obs=fr["T"],
            T_mask=t_mask,
            nh3=fr["nh3"],
            phi=fr["phi"],
            use_temperature_block=False,
            anchor_offset=True,
        )
        t_vmin = float(fr["T"].min())
        t_vmax = float(fr["T"].max())
        panels = [
            (fr["T"], "True T", False),
            (pred_unc, "Gappy uncond", False),
            (pred_cond, "Gappy cond (TF)", False),
            (pred_asst, "Cond + K=4 AO", False),
            (pred_asst - fr["T"], "AO error", True),
        ]
        for c, (data, title, is_err) in enumerate(panels):
            ax = axes[r, c]
            if is_err:
                lim = float(np.nanpercentile(np.abs(data), 98))
                im = ax.imshow(
                    data, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal"
                )
            else:
                im = ax.imshow(
                    data, origin="lower", cmap="inferno", vmin=t_vmin, vmax=t_vmax, aspect="equal"
                )
            ax.set_title(f"{sid}\n{title}" if c == 0 else title, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Qualitative reconstructions — interp vs extrap (vel 5%, corrected)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGS / "fig10_qualitative_reconstructions_interp_extrap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
