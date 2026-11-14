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
    temperature_anchor_mask,
)
from combustion_recon.evaluation.metrics import evaluate_sample
from combustion_recon.models.baselines import (
    GappyPODReconstructor,
    GlobalTrainMeanPredictor,
    LocalVelocityTemperatureRegressor,
    ParamConditionedMeanPredictor,
)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "configs" / "splits" / "splits_interp_extrap.yaml"
OC = ROOT / "data" / "manifests" / "operating_conditions.yaml"
STATS = ROOT / "data" / "manifests" / "train_norm_stats.json"
RAW = ROOT / "data" / "raw" / "combustion" / "hf_dataset" / "numerical"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures" / "month1_feasibility"

FRAME_MAIN = np.arange(0, 2001, 50)  # 41 frames
FRAME_SEEN = np.arange(25, 2000, 100)  # ~20 frames held-out times on train trajs
SEED = 20261114


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
    """Physical derivatives. Preferred: vj along x (axial), vi along y (transverse)."""
    # np.gradient(axis=1) ~ ∂/∂col ~ ∂/∂x ; axis=0 ~ ∂/∂row ~ ∂/∂y
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    dvi_dy, dvi_dx = np.gradient(vi, dy, dx)
    dvj_dy, dvj_dx = np.gradient(vj, dy, dx)
    div_preferred = dvj_dx + dvi_dy  # ∂vj/∂x + ∂vi/∂y
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


# -------------------- Evaluation helpers --------------------
def eval_prediction(pred, true, sim_id, split, sigma_T) -> dict[str, Any]:
    return evaluate_sample(pred, true, train_sigma_T=sigma_T, sim_id=sim_id, split=split)


def extract_frame_bank(sim_ids: list[str], frames: np.ndarray, smap: dict[str, Path]) -> list[dict[str, Any]]:
    """Load each trajectory once, keep only selected frames, free the rest."""
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

    # ========== Check 1 (loads train trajs one-by-one) ==========
    print("\n=== Check 1: kinematic coupling ===", flush=True)
    # Minimal cache for check1: load inside run_check1
    def run_check1_streaming(train_ids):
        rows = []
        pooled = {k: [] for k in ("vi", "vj", "vk", "umag", "div", "div_alt", "vort", "strain", "T")}
        amb_curves = {}
        for sid in train_ids:
            print(f"  check1 {sid}", flush=True)
            traj = load_traj(smap[sid])
            nh3, phi = parse_sim_id(sid)
            frames = FRAME_MAIN[::2]
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
                c = corr_block(arr, T, seed=hash(sid) % (2**32))
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
            if (nh3 in (0, 40) and abs(phi - 1.0) < 0.06) or (nh3 == 80 and abs(phi - 1.05) < 1e-6):
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
    df1.to_csv(TABLES / "month1_check1_kinematic_correlations.csv", index=False)
    print(df1[df1.scope == "pooled_train"][["feature", "pearson", "spearman", "mutual_info"]], flush=True)

    # ========== Prepare frame banks (one traj at a time) ==========
    print("\nExtracting frame banks...", flush=True)
    train_frames = extract_frame_bank(splits["train"], FRAME_MAIN, smap)
    seen_frames = extract_frame_bank(splits["train"], FRAME_SEEN, smap)
    val_frames = extract_frame_bank(splits["val"], FRAME_MAIN, smap)
    interp_frames = extract_frame_bank(splits["test_interp"], FRAME_MAIN, smap)
    extrap_frames = extract_frame_bank(splits["test_extrap"], FRAME_MAIN, smap)
    print(
        f"Frames: train={len(train_frames)} seen={len(seen_frames)} "
        f"val={len(val_frames)} interp={len(interp_frames)} extrap={len(extrap_frames)}",
        flush=True,
    )

    # Condition mean fields on train
    cond_means = []
    cond_ops = []
    all_train_T_for_global = []
    by_sid: dict[str, list] = {}
    for fr in train_frames:
        by_sid.setdefault(fr["sim_id"], []).append(fr["T"])
    for sid, Ts in by_sid.items():
        stack = np.stack(Ts, axis=0)
        cond_means.append(stack.mean(axis=0))
        nh3, phi = parse_sim_id(sid)
        cond_ops.append((nh3 / 100.0, phi))
        all_train_T_for_global.extend([stack[i] for i in range(0, len(stack), 2)])

    print("Fitting mean predictors...", flush=True)
    global_mean = GlobalTrainMeanPredictor.fit(all_train_T_for_global)
    param_mean = ParamConditionedMeanPredictor.fit(cond_means, cond_ops, ridge_alpha=1e-2)

    print("Fitting local regressors...", flush=True)
    # subsample frames for fitting
    fit_idx = rng.choice(len(train_frames), size=min(300, len(train_frames)), replace=False)
    fit_frames = [train_frames[i] for i in fit_idx]
    local_unc = LocalVelocityTemperatureRegressor.fit(
        fit_frames, conditioned=False, max_pixels=60_000, seed=SEED
    )
    local_cond = LocalVelocityTemperatureRegressor.fit(
        fit_frames, conditioned=True, max_pixels=60_000, seed=SEED + 1
    )

    print("Fitting Gappy POD (R=32/64)...", flush=True)
    # Select R on val with random_5pct temperature_free
    pod_candidates = {}
    for R in (32, 64):
        pod = GappyPODReconstructor.fit(
            train_frames,
            R=R,
            ridge_lambda=1e-2,
            use_param_mean=False,
            max_train_frames=350,
            seed=SEED,
        )
        errs = []
        for fr in val_frames[::4]:
            mask = build_layout_mask("random_5pct", seed=SEED + fr["frame"])
            pred = pod.reconstruct(fr["vi"], fr["vj"], fr["vk"], vel_mask=mask, nh3=fr["nh3"], phi=fr["phi"])
            errs.append(np.sqrt(np.mean((pred - fr["T"]) ** 2)))
        pod_candidates[R] = (pod, float(np.mean(errs)))
        print(f"  R={R} val RMSE={pod_candidates[R][1]:.2f}", flush=True)
    best_R = min(pod_candidates, key=lambda r: pod_candidates[r][1])
    print(f"Selected R={best_R}", flush=True)
    pod_unc = pod_candidates[best_R][0]
    pod_cond = GappyPODReconstructor.fit(
        train_frames,
        R=best_R,
        ridge_lambda=1e-2,
        use_param_mean=True,
        param_mean_predictor=param_mean,
        max_train_frames=350,
        seed=SEED + 2,
    )

    # ========== Checks 2–4 evaluations ==========
    metric_rows = []

    def add_rows(results, method, task_mode, layout, K=0):
        for r in results:
            d = {k: v for k, v in r.items() if k != "_spatial_bias_map"}
            d.update(
                {
                    "method": method,
                    "task_mode": task_mode,
                    "layout": layout,
                    "K_temp_anchors": K,
                }
            )
            metric_rows.append(d)

    eval_sets = [
        ("seen_train_eval", seen_frames),
        ("val", val_frames),
        ("test_interp", interp_frames),
        ("test_extrap", extrap_frames),
    ]

    # --- Check 2: mean predictors ---
    print("\n=== Check 2: mean predictors ===", flush=True)
    for split_name, frames in eval_sets:
        res_g, res_p = [], []
        # one prediction per condition (same for all frames) but score each frame
        for fr in frames:
            pred_g = global_mean.predict()
            pred_p = param_mean.predict(fr["nh3"], fr["phi"])
            res_g.append(eval_prediction(pred_g, fr["T"], fr["sim_id"], split_name, sigma_T))
            res_p.append(eval_prediction(pred_p, fr["T"], fr["sim_id"], split_name, sigma_T))
        add_rows(res_g, "GlobalTrainMean", "temperature_free", "none", 0)
        add_rows(res_p, "ParamConditionedMean", "temperature_free", "none", 0)

    # --- Check 3: local regressors (full-grid velocity oracle — upper bound on local map) ---
    print("\n=== Check 3: local regressors ===", flush=True)
    for split_name, frames in eval_sets:
        res_u, res_c = [], []
        for fr in frames[::1]:
            pred_u = local_unc.predict_field(
                fr["vi"], fr["vj"], fr["vk"], fr["umag"], fr["x2d"], fr["y2d"]
            )
            pred_c = local_cond.predict_field(
                fr["vi"],
                fr["vj"],
                fr["vk"],
                fr["umag"],
                fr["x2d"],
                fr["y2d"],
                nh3=fr["nh3"],
                phi=fr["phi"],
            )
            res_u.append(eval_prediction(pred_u, fr["T"], fr["sim_id"], split_name, sigma_T))
            res_c.append(eval_prediction(pred_c, fr["T"], fr["sim_id"], split_name, sigma_T))
        add_rows(res_u, "LocalHGB_Unconditioned", "temperature_free", "full_grid_velocity", 0)
        add_rows(res_c, "LocalHGB_Conditioned", "temperature_free", "full_grid_velocity", 0)

    # --- Check 4a: sparsity sweep temperature_free ---
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
        for method_name, pod in [
            ("GappyPOD_Unconditioned", pod_unc),
            ("GappyPOD_Conditioned", pod_cond),
        ]:
            for split_name, frames in eval_sets:
                # subsample frames for expensive layouts
                use = frames if layout.startswith("random") else frames[::2]
                res = []
                for fr in use:
                    mask = build_layout_mask(layout, seed=SEED + fr["frame"] + hash(fr["sim_id"]) % 1000)
                    pred = pod.reconstruct(
                        fr["vi"],
                        fr["vj"],
                        fr["vk"],
                        vel_mask=mask,
                        nh3=fr["nh3"],
                        phi=fr["phi"],
                    )
                    res.append(eval_prediction(pred, fr["T"], fr["sim_id"], split_name, sigma_T))
                add_rows(res, method_name, "temperature_free", layout, 0)
        print(f"  done {layout}", flush=True)

    # --- Check 4b: temperature-assisted K sweep at random_5pct velocity ---
    print("\n=== Check 4b: temperature-assisted K sweep ===", flush=True)
    for K in (0, 1, 4, 9, 16, 32):
        for method_name, pod in [
            ("GappyPOD_Unconditioned", pod_unc),
            ("GappyPOD_Conditioned", pod_cond),
        ]:
            for split_name, frames in eval_sets:
                res = []
                for fr in frames[::2]:
                    vel_mask = build_layout_mask(
                        "random_5pct", seed=SEED + fr["frame"] + hash(fr["sim_id"]) % 1000
                    )
                    if K > 0:
                        t_mask = temperature_anchor_mask(
                            K, seed=SEED + 17 + fr["frame"], exclude=~vel_mask
                        )
                    else:
                        t_mask = np.zeros_like(vel_mask)
                    pred = pod.reconstruct(
                        fr["vi"],
                        fr["vj"],
                        fr["vk"],
                        vel_mask=vel_mask,
                        T_obs=fr["T"] if K > 0 else None,
                        T_mask=t_mask if K > 0 else None,
                        nh3=fr["nh3"],
                        phi=fr["phi"],
                    )
                    res.append(eval_prediction(pred, fr["T"], fr["sim_id"], split_name, sigma_T))
                add_rows(res, method_name, "temperature_assisted", "random_5pct", K)
        print(f"  done K={K}", flush=True)

    df_metrics = pd.DataFrame(metric_rows)
    df_metrics.to_csv(TABLES / "month1_check2_3_4_per_condition_metrics.csv", index=False)

    # Split summary (separate task modes)
    summary_rows = []
    for task_mode in ("temperature_free", "temperature_assisted"):
        sub = df_metrics[df_metrics.task_mode == task_mode]
        if sub.empty:
            continue
        g = (
            sub.groupby(["task_mode", "method", "layout", "K_temp_anchors", "split"], as_index=False)[
                ["rmse_K", "mae_K", "mean_signed_error_K", "high_iou", "high_rmse_K", "mean_level_abs_error_K", "p95_abs_error_K", "r2"]
            ]
            .mean()
        )
        summary_rows.append(g)
    df_sum = pd.concat(summary_rows, ignore_index=True)
    df_sum.to_csv(TABLES / "month1_feasibility_split_summary.csv", index=False)
    print("\n=== Split summary (key methods @ random_5pct / none) ===", flush=True)
    show = df_sum[
        ((df_sum.layout.isin(["none", "full_grid_velocity", "random_5pct"])) & (df_sum.task_mode == "temperature_free"))
        | ((df_sum.task_mode == "temperature_assisted") & (df_sum.K_temp_anchors.isin([0, 4, 16])))
    ]
    print(show.to_string(index=False), flush=True)

    # ========== Figures ==========
    print("\nGenerating figures...", flush=True)
    _make_figures(
        smap,
        df1,
        amb_curves,
        df_sum,
        pod_unc,
        pod_cond,
    )

    # Save selected R
    (TABLES / "month1_feasibility_meta.json").write_text(
        json.dumps({"best_pod_R": best_R, "sigma_T": sigma_T, "n_train_frames": len(train_frames)}, indent=2)
        + "\n"
    )
    print("Done.", flush=True)


def _make_figures(
    smap: dict[str, Path],
    df1: pd.DataFrame,
    amb_curves: dict,
    df_sum: pd.DataFrame,
    pod_unc: GappyPODReconstructor,
    pod_cond: GappyPODReconstructor,
):
    # fig07
    sid0 = "0NH3_1.h5"
    traj0 = load_traj(smap[sid0])
    fr0 = frame_pack(traj0, 1000, 0.0, 1.0)
    div, _, vort, _ = kinematic_fields(fr0["vi"], fr0["vj"], traj0["x"], traj0["y"])
    del traj0
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    ims = [
        (axes[0, 0], fr0["T"], "T [K]", "inferno"),
        (axes[0, 1], fr0["umag"], r"$|\mathbf{u}|$ [m/s]", "viridis"),
        (axes[0, 2], div, r"$\nabla\cdot u$ (preferred)", "RdBu_r"),
        (axes[1, 0], vort, r"$|\omega_z|$", "magma"),
    ]
    for ax, data, title, cmap in ims:
        im = ax.imshow(data, origin="lower", cmap=cmap, aspect="equal")
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

    # fig08 transfer comparison
    methods_fig8 = [
        "GlobalTrainMean",
        "ParamConditionedMean",
        "LocalHGB_Unconditioned",
        "LocalHGB_Conditioned",
        "GappyPOD_Unconditioned",
        "GappyPOD_Conditioned",
    ]
    splits_order = ["seen_train_eval", "val", "test_interp", "test_extrap"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    metrics_ax = [
        ("rmse_K", "RMSE [K]"),
        ("mean_signed_error_K", "Mean signed bias [K]"),
        ("high_iou", "High-T IoU"),
    ]
    for ax, (met, title) in zip(axes, metrics_ax):
        for method in methods_fig8:
            ys = []
            for sp in splits_order:
                if method.startswith("GappyPOD"):
                    sub = df_sum[
                        (df_sum.method == method)
                        & (df_sum.layout == "random_5pct")
                        & (df_sum.task_mode == "temperature_free")
                        & (df_sum.split == sp)
                    ]
                elif method.startswith("Local"):
                    sub = df_sum[
                        (df_sum.method == method)
                        & (df_sum.layout == "full_grid_velocity")
                        & (df_sum.split == sp)
                    ]
                else:
                    sub = df_sum[
                        (df_sum.method == method) & (df_sum.layout == "none") & (df_sum.split == sp)
                    ]
                ys.append(float(sub[met].mean()) if len(sub) else np.nan)
            ax.plot(splits_order, ys, marker="o", label=method.replace("_", "\n"), lw=1.5)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=6, loc="best")
    fig.suptitle("Unconditioned vs conditioned transfer (temp-free; Gappy@5%)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig08_unconditioned_vs_conditioned_transfer.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # fig09 sparsity + K curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
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
                ]
                if len(sub):
                    xs.append(pct)
                    ys.append(float(sub.rmse_K.mean()))
            ax.plot(xs, ys, ls=ls, marker="o", label=f"{split}/{method.split('_')[1][:4]}")
    ax.set_xlabel("Velocity coverage [%]")
    ax.set_ylabel("RMSE [K]")
    ax.set_title("(a) Temperature-free sparsity sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)

    ax = axes[1]
    for split in ("val", "test_interp", "test_extrap"):
        for method, ls in [("GappyPOD_Unconditioned", "-"), ("GappyPOD_Conditioned", "--")]:
            xs, ys = [], []
            for K in (0, 1, 4, 9, 16, 32):
                sub = df_sum[
                    (df_sum.method == method)
                    & (df_sum.task_mode == "temperature_assisted")
                    & (df_sum.K_temp_anchors == K)
                    & (df_sum.split == split)
                ]
                if len(sub):
                    xs.append(K)
                    ys.append(float(sub.rmse_K.mean()))
            ax.plot(xs, ys, ls=ls, marker="o", label=f"{split}/{method.split('_')[1][:4]}")
    ax.set_xlabel("K temperature anchors")
    ax.set_ylabel("RMSE [K]")
    ax.set_title("(b) Temperature-assisted @ vel 5%")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGS / "fig09_sparsity_and_temp_assisted_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # fig10 qualitative
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
        vel_mask = build_layout_mask("random_5pct", seed=SEED)
        t_mask = temperature_anchor_mask(4, seed=SEED + 99)
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
        )
        panels = [
            (fr["T"], "True T"),
            (pred_unc, "Gappy uncond"),
            (pred_cond, "Gappy cond"),
            (pred_asst, "Cond + K=4 T"),
            (pred_cond - fr["T"], "Cond error"),
        ]
        for c, (data, title) in enumerate(panels):
            ax = axes[r, c]
            cmap = "RdBu_r" if "error" in title.lower() else "inferno"
            if "error" in title.lower():
                lim = float(np.nanpercentile(np.abs(data), 98))
                im = ax.imshow(data, origin="lower", cmap=cmap, vmin=-lim, vmax=lim, aspect="equal")
            else:
                im = ax.imshow(data, origin="lower", cmap=cmap, aspect="equal")
            ax.set_title(f"{sid}\n{title}" if c == 0 else title, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Qualitative reconstructions — interp vs extrap (vel 5%)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGS / "fig10_qualitative_reconstructions_interp_extrap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
