#!/usr/bin/env python3
"""Local CPU: fit/persist GappyPOD, gate vs historical 5%, eval nested 1%/20%.

Explicit deterministic train_fit POD refit (frozen λ from 5% selection).
Does NOT retune lambda. Does NOT touch Engaging S5 runs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys_path_insert = ROOT / "src"
import sys

sys.path.insert(0, str(sys_path_insert))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.data.month2_protocol import sha256_file  # noqa: E402
from combustion_recon.evaluation.month2_evaluator import (  # noqa: E402
    aggregate_frame_then_condition_then_split,
    evaluate_frame_prediction,
)
from combustion_recon.models.baselines import (  # noqa: E402
    GappyPODReconstructor,
    ParamConditionedMeanPredictor,
)
from combustion_recon.models.month2_baseline_adapters import predict_gappy  # noqa: E402

H = W = 128
COV_KEY = {0.01: "sites_1pct", 0.05: "sites_5pct", 0.20: "sites_20pct", 0.2: "sites_20pct"}


def _limit_blas_threads(n: int) -> None:
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(k, str(n))


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        return float("nan")
    return float("nan")


def _denormalize_velocity_stack(
    field_name: str,
    src: np.ndarray,
    normalizer: Month2FitOnlyNormalizer,
    *,
    chunk: int = 64,
) -> np.ndarray:
    """Denormalize mmap/float32 stack → owned float64 without a full intermediate copy."""
    s = normalizer.fields[field_name]
    mean = float(s["mean"])
    std = float(s["std"])
    n = int(src.shape[0])
    out = np.empty(src.shape, dtype=np.float64)
    for start in range(0, n, int(chunk)):
        stop = min(n, start + int(chunk))
        # One chunk float64 temp only (not a second full-role copy).
        out[start:stop] = np.asarray(src[start:stop], dtype=np.float64) * std + mean
    return out


def _load_role_physical(
    cache_root: Path,
    role: str,
    normalizer: Month2FitOnlyNormalizer,
    *,
    chunk: int = 64,
) -> list[dict[str, Any]]:
    """Load one role's physical fields.

    Peak RAM is dominated by the three owned float64 stacks for this role only
    (~0.28 GiB for train_fit). Source arrays are mmap'd; denorm is chunked so we
    never hold float32+float64 full copies of all roles at once.
    """
    rd = cache_root / role
    frames_meta = json.loads((rd / "frames.json").read_text())
    vi_src = np.load(rd / "vi_norm.npy", mmap_mode="r")
    vj_src = np.load(rd / "vj_norm.npy", mmap_mode="r")
    t_src = np.load(rd / "target_T_K.npy", mmap_mode="r")
    if vi_src.shape != vj_src.shape or vi_src.shape != t_src.shape:
        raise ValueError(f"{role}: vi/vj/T shape mismatch {vi_src.shape} {vj_src.shape} {t_src.shape}")
    if vi_src.shape[1:] != (H, W):
        raise ValueError(f"{role}: expected (*,{H},{W}), got {vi_src.shape}")

    print(
        f"  load_role={role} n={len(frames_meta)} rss_gb={_rss_gb():.2f} (mmap+chunked denorm)",
        flush=True,
    )
    vi = _denormalize_velocity_stack("Velocity_i", vi_src, normalizer, chunk=chunk)
    vj = _denormalize_velocity_stack("Velocity_j", vj_src, normalizer, chunk=chunk)
    # target_T_K is already physical Kelvin in the dense cache.
    t_K = np.empty(t_src.shape, dtype=np.float64)
    n = int(t_src.shape[0])
    for start in range(0, n, int(chunk)):
        stop = min(n, start + int(chunk))
        t_K[start:stop] = np.asarray(t_src[start:stop], dtype=np.float64)
    # Drop mmap handles before building frame list.
    del vi_src, vj_src, t_src

    out: list[dict[str, Any]] = []
    zero_vk = np.zeros((H, W), dtype=np.float64)
    for i, meta in enumerate(frames_meta):
        nh3 = float(meta.get("nh3_fraction", meta.get("nh3", 0.0)))
        phi = float(meta.get("phi", meta.get("equivalence_ratio", 1.0)))
        out.append(
            {
                "role": role,
                "cache_index": i,
                "sim_id": str(meta["sim_id"]),
                "frame": int(meta["frame_index"]),
                "nh3": nh3,
                "phi": phi,
                # Views into contiguous owned stacks (same order as frames.json).
                "vi": vi[i],
                "vj": vj[i],
                "vk": zero_vk,
                "T": t_K[i],
            }
        )
    print(
        f"  load_role={role} done rss_gb={_rss_gb():.2f} "
        f"vi_GiB={vi.nbytes / 1024**3:.3f}",
        flush=True,
    )
    return out


def _load_mask_index(mask_root: Path, role: str) -> dict[str, Any]:
    """Load nested sites once; lookup returns row views (no per-row copies)."""
    path = mask_root / f"{role}_nested_sites.npz"
    z = np.load(path, allow_pickle=True)
    sites_1 = np.array(z["sites_1pct"], dtype=np.int64, copy=True)
    sites_5 = np.array(z["sites_5pct"], dtype=np.int64, copy=True)
    sites_20 = np.array(z["sites_20pct"], dtype=np.int64, copy=True)
    frame = np.asarray(z["frame_index"], dtype=np.int64)
    rep = np.asarray(z["replica_id"], dtype=np.int64)
    hash_5 = [str(x) for x in z["hash_5pct"]]
    seeds = np.asarray(z["velocity_mask_seed"], dtype=np.int64)
    sim = [str(x) for x in z["sim_id"]]
    z.close()
    row_of: dict[tuple[str, int, int], int] = {}
    for i in range(len(sim)):
        row_of[(sim[i], int(frame[i]), int(rep[i]))] = i
    return {
        "sites_1pct": sites_1,
        "sites_5pct": sites_5,
        "sites_20pct": sites_20,
        "hash_5pct": hash_5,
        "velocity_mask_seed": seeds,
        "row_of": row_of,
    }


def _mask_sites(masks: dict[str, Any], key: tuple[str, int, int], site_key: str) -> np.ndarray:
    row = masks["row_of"].get(key)
    if row is None:
        raise KeyError(f"missing mask entry {key}")
    return masks[site_key][row]


def _model_inputs_from_sites(
    fr: dict[str, Any],
    sites: np.ndarray,
    *,
    p_conditioning: bool,
) -> dict[str, Any]:
    sites = np.asarray(sites, dtype=np.int64).ravel()
    mask = np.zeros((H, W), dtype=bool)
    if sites.size:
        mask.ravel()[sites] = True
        ys = sites // W
        xs = sites % W
        vals = np.stack([fr["vi"][ys, xs], fr["vj"][ys, xs]], axis=1).astype(np.float64)
    else:
        vals = np.zeros((0, 2), dtype=np.float64)
    mi: dict[str, Any] = {
        "velocity_values_physical": vals,
        "velocity_site_index": sites,
        "mask_velocity": mask,
        "mask_temperature": np.zeros((H, W), dtype=bool),
        "anchor_site_index": np.zeros((0,), dtype=np.int64),
        "anchor_values_physical": np.zeros((0,), dtype=np.float64),
        "K": 0,
        "site_fraction": float(sites.size) / float(H * W),
        "task_mode": "temperature_free",
    }
    if p_conditioning:
        mi["op_params"] = {
            "nh3_fraction": float(fr["nh3"]),
            "phi": float(fr["phi"]),
            "equivalence_ratio": float(fr["phi"]),
        }
    return mi


def _pod_to_npz(pod: GappyPODReconstructor) -> dict[str, Any]:
    return {
        "mean_vi": pod.mean_vi,
        "mean_vj": pod.mean_vj,
        "mean_vk": pod.mean_vk,
        "mean_T": pod.mean_T,
        "modes": pod.modes,
        "modal_variances": pod.modal_variances,
        "n_pix": pod.n_pix,
        "R": pod.R,
        "ridge_lambda": pod.ridge_lambda,
        "w_T": pod.w_T,
        "scale_vi": pod.scale_vi,
        "scale_vj": pod.scale_vj,
        "scale_vk": pod.scale_vk,
        "scale_T": pod.scale_T,
        "use_param_mean": pod.use_param_mean,
        "n_velocity_components": pod.n_velocity_components,
        "global_mean_T": pod.global_mean_T,
        "eps_var": pod.eps_var,
    }


def _pod_from_npz(d: dict[str, Any], param_mean: ParamConditionedMeanPredictor | None) -> GappyPODReconstructor:
    return GappyPODReconstructor(
        mean_vi=np.asarray(d["mean_vi"]),
        mean_vj=np.asarray(d["mean_vj"]),
        mean_vk=np.asarray(d["mean_vk"]),
        mean_T=np.asarray(d["mean_T"]),
        modes=np.asarray(d["modes"]),
        modal_variances=np.asarray(d["modal_variances"]),
        n_pix=int(d["n_pix"]),
        R=int(d["R"]),
        ridge_lambda=float(d["ridge_lambda"]),
        w_T=float(d["w_T"]),
        scale_vi=float(d["scale_vi"]),
        scale_vj=float(d["scale_vj"]),
        scale_vk=float(d["scale_vk"]),
        scale_T=float(d["scale_T"]),
        use_param_mean=bool(d["use_param_mean"]),
        n_velocity_components=int(d["n_velocity_components"]),
        param_mean_predictor=param_mean if d["use_param_mean"] else None,
        global_mean_T=np.asarray(d["global_mean_T"]) if d.get("global_mean_T") is not None else None,
        eps_var=float(d.get("eps_var", 1e-8)),
    )


def _evaluate_coverage(
    *,
    frames_by_role: dict[str, list[dict[str, Any]]],
    masks_by_role: dict[str, dict[str, Any]],
    pod_off: GappyPODReconstructor,
    pod_on: GappyPODReconstructor,
    coverage: float,
    replicas: list[int],
    roles: list[str],
    sigma_T: float,
    layout_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    site_key = COV_KEY[float(coverage)]
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for role in roles:
        frames = frames_by_role[role]
        masks = masks_by_role[role]
        for p_on, pod, lam in (
            (False, pod_off, float(pod_off.ridge_lambda)),
            (True, pod_on, float(pod_on.ridge_lambda)),
        ):
            role_rows: list[dict[str, Any]] = []
            per_rep_pc: list[pd.DataFrame] = []
            for rid in replicas:
                rep_rows: list[dict[str, Any]] = []
                for fr in frames:
                    key = (fr["sim_id"], int(fr["frame"]), int(rid))
                    sites = _mask_sites(masks, key, site_key)
                    mi = _model_inputs_from_sites(fr, sites, p_conditioning=p_on)
                    pred = predict_gappy(pod, mi, p_conditioning=p_on)
                    row = evaluate_frame_prediction(
                        pred,
                        fr["T"],
                        train_sigma_T=sigma_T,
                        sim_id=fr["sim_id"],
                        split="train" if role == "train_fit" else ("val" if role == "val_eval" else "seen"),
                        role=role,
                        frame_index=int(fr["frame"]),
                        method="GappyPOD_free",
                        variant="free",
                        p_conditioning=p_on,
                        K=0,
                        site_fraction=float(coverage),
                        model_seed=0,
                        mask_replica_id=int(rid),
                        layout_id=layout_id,
                        task_mode="temperature_free",
                        ridge_lambda=lam,
                        R=int(pod.R),
                    )
                    rep_rows.append(row)
                    role_rows.append(row)
                    rows.append(row)
                # Homogeneous per replica (matches month2 baseline summary construction).
                agg = aggregate_frame_then_condition_then_split(rep_rows)
                pc = agg["per_condition"].copy()
                pc["mask_replica_id"] = int(rid)
                per_rep_pc.append(pc)
            pc_all = pd.concat(per_rep_pc, ignore_index=True)
            mean_rmse = float(pc_all["rmse_K"].mean()) if not pc_all.empty else float("nan")
            mean_centered = (
                float(pc_all["centered_rmse_K"].mean())
                if "centered_rmse_K" in pc_all.columns
                else float("nan")
            )
            mean_bias = (
                float(pc_all["abs_bias_K"].mean()) if "abs_bias_K" in pc_all.columns else float("nan")
            )
            summary_rows.append(
                {
                    "method": "GappyPOD_free",
                    "role": role,
                    "p_conditioning": p_on,
                    "K": 0,
                    "site_fraction": float(coverage),
                    "variant": "free",
                    "ridge_lambda": lam,
                    "R": int(pod.R),
                    "rmse_K": mean_rmse,
                    "abs_bias_K": mean_bias,
                    "centered_rmse_K": mean_centered,
                    "n_condition_rows": int(len(pc_all)) if not pc_all.empty else 0,
                    "n_frame_rows": len(role_rows),
                }
            )
    return rows, summary_rows


def _gate_vs_historical(
    summary_5: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    atol_rmse_K: float,
) -> dict[str, Any]:
    hist = historical[
        (historical["method"] == "GappyPOD_free")
        & (historical["K"] == 0)
        & (np.isclose(historical["site_fraction"], 0.05))
    ].copy()
    checks = []
    ok = True
    for _, h in hist.iterrows():
        m = summary_5[
            (summary_5["role"] == h["role"])
            & (summary_5["p_conditioning"] == bool(h["p_conditioning"]))
            & np.isclose(summary_5["site_fraction"], 0.05)
        ]
        if len(m) != 1:
            checks.append({"role": h["role"], "p_conditioning": bool(h["p_conditioning"]), "ok": False, "error": "missing_or_duplicate"})
            ok = False
            continue
        row = m.iloc[0]
        d_rmse = float(row["rmse_K"] - h["rmse_K"])
        d_cent = float(row["centered_rmse_K"] - h["centered_rmse_K"])
        passed = abs(d_rmse) <= atol_rmse_K and abs(d_cent) <= atol_rmse_K
        checks.append(
            {
                "role": h["role"],
                "p_conditioning": bool(h["p_conditioning"]),
                "historical_rmse_K": float(h["rmse_K"]),
                "refit_rmse_K": float(row["rmse_K"]),
                "delta_rmse_K": d_rmse,
                "historical_centered_rmse_K": float(h["centered_rmse_K"]),
                "refit_centered_rmse_K": float(row["centered_rmse_K"]),
                "delta_centered_rmse_K": d_cent,
                "atol_rmse_K": atol_rmse_K,
                "ok": passed,
            }
        )
        ok = ok and passed
    return {"ok": ok, "n_checks": len(checks), "checks": checks}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/month2_sparse_gappypod_nested.yaml",
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "data/processed/month2_dense_unet_cache",
    )
    p.add_argument(
        "--mask-root",
        type=Path,
        default=ROOT / "data/processed/month2_sparse_eval_masks",
    )
    p.add_argument(
        "--stats",
        type=Path,
        default=ROOT / "data/manifests/train_norm_stats_month2_fit.json",
    )
    p.add_argument(
        "--historical-summary",
        type=Path,
        default=ROOT / "results/tables/month2_baseline_summary.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/month2/sparse_gappypod_nested_1_20",
    )
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--gate-atol-rmse-K", type=float, default=0.05)
    p.add_argument("--authorize-refit", action="store_true")
    p.add_argument("--git-sha", type=str, default="")
    p.add_argument(
        "--equiv-sample",
        type=int,
        default=8,
        help="Compare chunked mmap denorm vs legacy full-load on N train frames (0=skip).",
    )
    args = p.parse_args()

    _limit_blas_threads(int(args.threads))
    t_wall0 = time.perf_counter()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    art_dir = out / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    if not bool(args.authorize_refit) and not bool(cfg.get("refit_pod_basis_authorized")):
        report = {
            "status": "blocked_refit_not_authorized",
            "error": "Pass --authorize-refit or set refit_pod_basis_authorized: true",
        }
        (out / "blocked_refit_not_authorized.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 2

    # Preflight data
    for path in (
        args.cache_root / "train_fit" / "vi_norm.npy",
        args.mask_root / "val_eval_nested_sites.npz",
        args.stats,
        args.historical_summary,
    ):
        if not Path(path).exists():
            report = {"status": "blocked_missing_local_data", "missing": str(path)}
            (out / "blocked_missing_local_data.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
            return 2

    frozen = cfg["frozen_from_5pct"]
    lam_off = float(frozen["free_p_off_lambda"])
    lam_on = float(frozen["free_p_on_lambda"])
    R = int(frozen["R"])
    hist_manifest = json.loads((ROOT / frozen["source_manifest"]).read_text())
    pm_sel = hist_manifest["selected"]["param_mean"]

    normalizer = Month2FitOnlyNormalizer(args.stats)
    sigma_T = float(normalizer.temperature_std)

    print(
        f"Loading physical fields (role-lazy, chunked denorm); rss_gb={_rss_gb():.2f}",
        flush=True,
    )
    t0 = time.perf_counter()
    # Equivalence sample vs legacy full-array denorm (same formula, first N frames).
    if int(args.equiv_sample) > 0:
        rd = Path(args.cache_root) / "train_fit"
        n_eq = int(args.equiv_sample)
        vi_legacy = normalizer.denormalize_field(
            "Velocity_i", np.asarray(np.load(rd / "vi_norm.npy")[:n_eq], dtype=np.float64)
        )
        vj_legacy = normalizer.denormalize_field(
            "Velocity_j", np.asarray(np.load(rd / "vj_norm.npy")[:n_eq], dtype=np.float64)
        )
        t_legacy = np.asarray(np.load(rd / "target_T_K.npy")[:n_eq], dtype=np.float64)
        vi_new = _denormalize_velocity_stack(
            "Velocity_i", np.load(rd / "vi_norm.npy", mmap_mode="r")[:n_eq], normalizer, chunk=4
        )
        vj_new = _denormalize_velocity_stack(
            "Velocity_j", np.load(rd / "vj_norm.npy", mmap_mode="r")[:n_eq], normalizer, chunk=4
        )
        t_new = np.empty_like(t_legacy)
        t_src = np.load(rd / "target_T_K.npy", mmap_mode="r")
        for start in range(0, n_eq, 4):
            stop = min(n_eq, start + 4)
            t_new[start:stop] = np.asarray(t_src[start:stop], dtype=np.float64)
        eq = {
            "n": n_eq,
            "max_abs_vi": float(np.max(np.abs(vi_new - vi_legacy))),
            "max_abs_vj": float(np.max(np.abs(vj_new - vj_legacy))),
            "max_abs_T": float(np.max(np.abs(t_new - t_legacy))),
        }
        eq["ok"] = bool(eq["max_abs_vi"] == 0.0 and eq["max_abs_vj"] == 0.0 and eq["max_abs_T"] == 0.0)
        (out / "load_equivalence_sample.json").write_text(json.dumps(eq, indent=2) + "\n")
        print(f"  load_equivalence_sample ok={eq['ok']} {eq}", flush=True)
        if not eq["ok"]:
            report = {"status": "failed_load_equivalence", "eq": eq}
            (out / "failed_load_equivalence.json").write_text(json.dumps(report, indent=2) + "\n")
            return 5
        del vi_legacy, vj_legacy, t_legacy, vi_new, vj_new, t_new, t_src

    # Fit needs train_fit only; defer seen/val until after POD fit to bound peak RAM.
    train_frames = _load_role_physical(args.cache_root, "train_fit", normalizer)
    masks_by_role = {
        role: _load_mask_index(args.mask_root, role)
        for role in ("train_fit", "seen_train_eval", "val_eval")
    }
    t_load_train = time.perf_counter() - t0
    print(
        f"  n_train={len(train_frames)} masks_loaded "
        f"load_train+masks={t_load_train:.1f}s rss_gb={_rss_gb():.2f}",
        flush=True,
    )
    # ParamMean with historically selected basis/α (no retune).
    print(
        f"Fitting ParamMean basis={pm_sel['basis']} α={pm_sel['ridge_alpha']} (frozen selection)...",
        flush=True,
    )
    t0 = time.perf_counter()
    by_cond: dict[str, list[np.ndarray]] = {}
    ops: dict[str, tuple[float, float]] = {}
    for f in train_frames:
        by_cond.setdefault(f["sim_id"], []).append(f["T"])
        ops[f["sim_id"]] = (f["nh3"], f["phi"])
    cond_means = [np.mean(np.stack(v, 0), 0) for _, v in sorted(by_cond.items())]
    cond_ops = [ops[sid] for sid, _ in sorted(by_cond.items())]
    param_mean = ParamConditionedMeanPredictor.fit(
        cond_means,
        cond_ops,
        ridge_alpha=float(pm_sel["ridge_alpha"]),
        basis=str(pm_sel["basis"]),
    )
    t_pm = time.perf_counter() - t0
    print(f"  ParamMean LOCO={param_mean.loco_rmse_K:.4f}K ({t_pm:.1f}s)", flush=True)

    print(f"Fitting GappyPOD 2C R={R} p_off/p_on (seed=0)...", flush=True)
    t0 = time.perf_counter()
    pod_off = GappyPODReconstructor.fit(
        train_frames,
        R=R,
        ridge_lambda=1e-2,
        w_T=5.0,
        use_param_mean=False,
        n_velocity_components=2,
        seed=0,
    ).with_hyperparams(ridge_lambda=lam_off)
    pod_on = GappyPODReconstructor.fit(
        train_frames,
        R=R,
        ridge_lambda=1e-2,
        w_T=5.0,
        use_param_mean=True,
        param_mean_predictor=param_mean,
        n_velocity_components=2,
        seed=0,
    ).with_hyperparams(ridge_lambda=lam_on)
    t_fit = time.perf_counter() - t0
    print(
        f"  scales T/vi/vj={pod_off.scale_T:.6f}/{pod_off.scale_vi:.6f}/{pod_off.scale_vj:.6f} "
        f"fit={t_fit:.1f}s rss_gb={_rss_gb():.2f}",
        flush=True,
    )

    # Persist
    payload = {
        "format_id": "month2_gappypod_fitted_v1",
        "experiment_id": cfg.get("experiment_id"),
        "git_sha": args.git_sha or "",
        "R": R,
        "n_velocity_components": 2,
        "seed": 0,
        "frozen_lambdas": {"free_p_off": lam_off, "free_p_on": lam_on},
        "param_mean": {
            "coef": param_mean.coef,
            "H": param_mean.H,
            "W": param_mean.W,
            "ridge_alpha": param_mean.ridge_alpha,
            "basis": param_mean.basis,
            "selected_basis": param_mean.selected_basis,
            "selected_ridge_alpha": param_mean.selected_ridge_alpha,
            "loco_rmse_K": param_mean.loco_rmse_K,
        },
        "pod_p_off": _pod_to_npz(pod_off),
        "pod_p_on": _pod_to_npz(pod_on),
        "provenance": {
            "cache_root": str(args.cache_root),
            "stats_sha256": sha256_file(args.stats),
            "mask_manifest_sha256": sha256_file(args.mask_root / "manifest.json"),
            "historical_manifest": str(frozen["source_manifest"]),
            "refit_authorized": True,
            "retune_lambda": False,
            "source": "dense_unet_cache_denormalized_velocities_plus_target_T_K",
        },
    }
    art_path = art_dir / "gappypod_fitted_pod_p_off_on.npz"

    def _pack_pod(prefix: str, pod: GappyPODReconstructor) -> dict[str, Any]:
        d = _pod_to_npz(pod)
        out_d: dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}_{k}"
            if v is None:
                continue
            out_d[key] = np.asarray(v) if not np.isscalar(v) else np.asarray(v)
        return out_d

    meta_json = json.dumps(
        {
            "format_id": payload["format_id"],
            "experiment_id": payload["experiment_id"],
            "git_sha": payload["git_sha"],
            "R": R,
            "frozen_lambdas": payload["frozen_lambdas"],
            "provenance": payload["provenance"],
            "param_mean_meta": {
                "H": param_mean.H,
                "W": param_mean.W,
                "ridge_alpha": param_mean.ridge_alpha,
                "basis": param_mean.basis,
                "loco_rmse_K": param_mean.loco_rmse_K,
            },
        }
    )
    np.savez_compressed(
        art_path,
        meta_json=np.asarray(meta_json),
        param_mean_coef=param_mean.coef,
        **_pack_pod("off", pod_off),
        **_pack_pod("on", pod_on),
    )
    art_sha = sha256_file(art_path)
    print(f"  persisted {art_path} sha256={art_sha}", flush=True)

    # Reload round-trip
    z = np.load(art_path, allow_pickle=True)
    pm2 = ParamConditionedMeanPredictor(
        coef=np.asarray(z["param_mean_coef"]),
        H=param_mean.H,
        W=param_mean.W,
        ridge_alpha=param_mean.ridge_alpha,
        basis=param_mean.basis,
        selected_basis=param_mean.basis,
        selected_ridge_alpha=param_mean.ridge_alpha,
        loco_rmse_K=param_mean.loco_rmse_K,
    )

    def _extract(prefix: str) -> dict[str, Any]:
        keys = [
            "mean_vi",
            "mean_vj",
            "mean_vk",
            "mean_T",
            "modes",
            "modal_variances",
            "n_pix",
            "R",
            "ridge_lambda",
            "w_T",
            "scale_vi",
            "scale_vj",
            "scale_vk",
            "scale_T",
            "use_param_mean",
            "n_velocity_components",
            "global_mean_T",
            "eps_var",
        ]
        d: dict[str, Any] = {}
        for k in keys:
            arr = z[f"{prefix}_{k}"]
            d[k] = arr.item() if arr.shape == () else np.asarray(arr)
        return d

    pod_off_r = _pod_from_npz(_extract("off"), None)
    pod_on_r = _pod_from_npz(_extract("on"), pm2)
    reload_ok = bool(
        np.allclose(pod_off_r.modes, pod_off.modes)
        and np.allclose(pod_on_r.modes, pod_on.modes)
        and abs(pod_off_r.scale_T - pod_off.scale_T) < 1e-12
    )
    print(f"  reload_modes_ok={reload_ok}", flush=True)
    if not reload_ok:
        report = {"status": "failed_reload", "artifact": str(art_path)}
        (out / "failed_reload.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 3
    pod_off, pod_on = pod_off_r, pod_on_r

    # Scale gate vs historical POD scales
    hist_scales = hist_manifest["pod_vs_normalizer_scales"]
    scale_checks = {
        "scale_T": {
            "refit": float(pod_off.scale_T),
            "historical": float(hist_scales["pod_scale_T"]),
            "abs_diff": abs(float(pod_off.scale_T) - float(hist_scales["pod_scale_T"])),
        },
        "scale_vi": {
            "refit": float(pod_off.scale_vi),
            "historical": float(hist_scales["pod_scale_vi"]),
            "abs_diff": abs(float(pod_off.scale_vi) - float(hist_scales["pod_scale_vi"])),
        },
        "scale_vj": {
            "refit": float(pod_off.scale_vj),
            "historical": float(hist_scales["pod_scale_vj"]),
            "abs_diff": abs(float(pod_off.scale_vj) - float(hist_scales["pod_scale_vj"])),
        },
    }
    # Scales should match closely (fit-only train_fit std).
    scale_ok = all(v["abs_diff"] < 1e-4 for v in scale_checks.values())

    roles = list(cfg["evaluation"]["roles"])
    replicas = [int(x) for x in cfg["evaluation"]["replicas"]]
    layout_id = str(cfg["evaluation"]["layout_id"])

    print(
        f"Loading seen/val physical fields for eval; rss_gb={_rss_gb():.2f}",
        flush=True,
    )
    t_load_eval0 = time.perf_counter()
    frames_by_role: dict[str, list[dict[str, Any]]] = {
        "train_fit": train_frames,
        "seen_train_eval": _load_role_physical(args.cache_root, "seen_train_eval", normalizer),
        "val_eval": _load_role_physical(args.cache_root, "val_eval", normalizer),
    }
    t_load = t_load_train + (time.perf_counter() - t_load_eval0)
    print(
        f"  n_seen={len(frames_by_role['seen_train_eval'])} "
        f"n_val={len(frames_by_role['val_eval'])} "
        f"load_total={t_load:.1f}s rss_gb={_rss_gb():.2f}",
        flush=True,
    )

    print("Evaluating historical gate coverage 5%...", flush=True)
    t0 = time.perf_counter()
    rows_5, summary_5 = _evaluate_coverage(
        frames_by_role=frames_by_role,
        masks_by_role=masks_by_role,
        pod_off=pod_off,
        pod_on=pod_on,
        coverage=0.05,
        replicas=replicas,
        roles=roles,
        sigma_T=sigma_T,
        layout_id=layout_id,
    )
    t_5 = time.perf_counter() - t0
    summary_5_df = pd.DataFrame(summary_5)
    summary_5_df.to_csv(out / "summary_5pct_refit.csv", index=False)
    pd.DataFrame(rows_5).to_csv(out / "per_frame_5pct_refit.csv.gz", index=False, compression="gzip")
    print(f"  5% eval done in {t_5:.1f}s", flush=True)
    print(summary_5_df.to_string(index=False), flush=True)

    historical = pd.read_csv(args.historical_summary)
    gate = _gate_vs_historical(summary_5_df, historical, atol_rmse_K=float(args.gate_atol_rmse_K))
    gate["scale_checks"] = scale_checks
    gate["scale_ok"] = scale_ok
    gate["ok"] = bool(gate["ok"] and scale_ok)
    (out / "gate_5pct.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(f"  gate_5pct ok={gate['ok']} (atol={args.gate_atol_rmse_K} K)", flush=True)

    nested_summaries: list[dict[str, Any]] = []
    nested_status = "skipped_gate_failed"
    if gate["ok"]:
        print("Gate passed — evaluating nested 1% and 20%...", flush=True)
        for cov in (0.01, 0.20):
            t0 = time.perf_counter()
            rows_c, summary_c = _evaluate_coverage(
                frames_by_role=frames_by_role,
                masks_by_role=masks_by_role,
                pod_off=pod_off,
                pod_on=pod_on,
                coverage=cov,
                replicas=replicas,
                roles=roles,
                sigma_T=sigma_T,
                layout_id=layout_id,
            )
            dt = time.perf_counter() - t0
            tag = f"{int(cov*100)}pct"
            pd.DataFrame(summary_c).to_csv(out / f"summary_{tag}.csv", index=False)
            pd.DataFrame(rows_c).to_csv(out / f"per_frame_{tag}.csv.gz", index=False, compression="gzip")
            nested_summaries.extend(summary_c)
            print(f"  coverage={cov} done in {dt:.1f}s", flush=True)
            print(pd.DataFrame(summary_c).to_string(index=False), flush=True)
        nested_status = "completed"
        pd.DataFrame(nested_summaries).to_csv(out / "summary_nested_1_20.csv", index=False)
    else:
        (out / "nested_eval_skipped_gate_failed.json").write_text(
            json.dumps({"status": nested_status, "gate": gate}, indent=2) + "\n"
        )

    wall = time.perf_counter() - t_wall0
    report = {
        "status": nested_status if gate["ok"] else "gate_failed_after_fit",
        "experiment_id": cfg.get("experiment_id"),
        "git_sha": args.git_sha or "",
        "threads": int(args.threads),
        "wall_time_s": wall,
        "timings_s": {"load": t_load, "param_mean": t_pm, "pod_fit": t_fit, "eval_5pct": t_5},
        "artifact": {
            "path": str(art_path),
            "sha256": art_sha,
            "reload_ok": reload_ok,
        },
        "frozen_lambdas": {"free_p_off": lam_off, "free_p_on": lam_on},
        "param_mean": {
            "basis": param_mean.basis,
            "ridge_alpha": param_mean.ridge_alpha,
            "loco_rmse_K": param_mean.loco_rmse_K,
            "historical_loco_rmse_K": float(pm_sel["loco_rmse_K"]),
        },
        "gate_5pct": gate,
        "summary_5pct": summary_5,
        "summary_nested_1_20": nested_summaries,
        "retune_lambda": False,
        "refit_pod_basis_authorized": True,
        "s5_jobs_untouched": True,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "wall_time_s": wall, "gate_ok": gate["ok"], "artifact_sha256": art_sha}, indent=2))
    return 0 if gate["ok"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
