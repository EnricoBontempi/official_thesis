#!/usr/bin/env python3
"""Dense GappyPOD free reference at 100% coverage (Month-2 Step 2.4 U3).

Uses the dense cache + fit-only stats. Does not mutate sparse baseline contracts
or consolidation_v1 tables. Single all-True mask (no replica triplication).
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.dataset import load_splits, parse_sim_id
from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.data.month2_protocol import sha256_file
from combustion_recon.evaluation.month2_evaluator import (
    aggregate_frame_then_condition_then_split,
    evaluate_frame_prediction,
)
from combustion_recon.models.baselines import (
    GappyPODReconstructor,
    GlobalTrainMeanPredictor,
    ParamConditionedMeanPredictor,
)
from combustion_recon.models.month2_baseline_adapters import predict_gappy, predict_global_mean, predict_param_mean
from combustion_recon.training.dense_cache import (
    DEFAULT_CACHE_ROOT,
    DenseUNetCacheDataset,
    validate_dense_cache_manifest,
)

CFG_PATH = ROOT / "configs" / "experiments" / "month2_dense_linear.yaml"


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def assert_dense_linear_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg.get("experiment_id") != "month2_dense_linear_v1":
        raise ValueError(f"unexpected experiment_id={cfg.get('experiment_id')!r}")
    inputs = cfg.get("inputs") or {}
    if inputs.get("temporal_mode") != "single_frame":
        raise ValueError("temporal_mode must be single_frame")
    if inputs.get("task_mode") != "temperature_free":
        raise ValueError("task_mode must be temperature_free")
    if int(inputs.get("K", -1)) != 0:
        raise ValueError("K must be 0")
    if float(inputs.get("site_fraction", -1)) != 1.0:
        raise ValueError("site_fraction must be 1.0")
    if inputs.get("velocity_components") != "planar_2C":
        raise ValueError("velocity_components must be planar_2C")
    gp = cfg.get("gappy_pod") or {}
    if int(gp.get("R", -1)) != 32:
        raise ValueError("R must be 32 for this U3 contract")
    if int(gp.get("n_velocity_components", -1)) != 2:
        raise ValueError("n_velocity_components must be 2")
    return {
        "R": 32,
        "lambdas": [float(x) for x in gp["ridge_lambda_grid"]],
        "w_T": float(gp.get("w_T_fixed_unused", 5.0)),
        "seed": int(gp.get("seed", 0)),
        "p_values": [bool(x) for x in gp.get("p_conditioning", [False, True])],
    }


def dense_physical_frames_from_cache(
    role: str,
    *,
    cache_root: Path,
    normalizer: Month2FitOnlyNormalizer,
) -> list[dict[str, Any]]:
    """Materialize GappyPOD frame dicts from dense cache (denormalized velocities)."""
    ds = DenseUNetCacheDataset(role, cache_root=cache_root, p_conditioning=True)  # type: ignore[arg-type]
    out: list[dict[str, Any]] = []
    for i in range(len(ds)):
        meta = ds.frames[i]
        vi = normalizer.denormalize_field("Velocity_i", np.asarray(ds.vi[i], dtype=np.float64))
        vj = normalizer.denormalize_field("Velocity_j", np.asarray(ds.vj[i], dtype=np.float64))
        T = np.asarray(ds.t_K[i], dtype=np.float64)
        out.append(
            {
                "sim_id": str(meta["sim_id"]),
                "frame_index": int(meta["frame_index"]),
                "role": role,
                "split": str(meta["split"]),
                "nh3": float(meta["nh3_fraction"]),
                "phi": float(meta["phi"]),
                "vi": vi,
                "vj": vj,
                "vk": np.zeros_like(vi),
                "T": T,
            }
        )
    return out


def dense_model_inputs_from_frame(
    fr: dict[str, Any],
    *,
    p_conditioning: bool,
) -> dict[str, Any]:
    """Build Month-2 model_inputs for 100% planar velocity (single dense mask)."""
    H, W = fr["T"].shape
    mask_v = np.ones((H, W), dtype=bool)
    ys, xs = np.where(mask_v)
    sites = ys * W + xs
    vals = np.stack([fr["vi"][ys, xs], fr["vj"][ys, xs]], axis=1)
    mi: dict[str, Any] = {
        "mask_velocity": mask_v,
        "velocity_site_index": sites.astype(np.int64),
        "velocity_values_physical": vals.astype(np.float64),
        "velocity_values": vals.astype(np.float64),
        "mask_temperature": np.zeros((H, W), dtype=bool),
        "anchor_site_index": np.zeros((0,), dtype=np.int64),
        "anchor_values": np.zeros((0,), dtype=np.float64),
        "anchor_values_physical": np.zeros((0,), dtype=np.float64),
        "K": 0,
        "site_fraction": 1.0,
        "task_mode": "temperature_free",
    }
    if p_conditioning:
        mi["op_params"] = {
            "nh3_fraction": float(fr["nh3"]),
            "phi": float(fr["phi"]),
            "equivalence_ratio": float(fr["phi"]),
            "normalized": np.asarray([fr["nh3"], fr["phi"]], dtype=np.float64),
        }
    return mi


def evaluate_pod_on_frames(
    *,
    pod: GappyPODReconstructor,
    frames: list[dict[str, Any]],
    p_conditioning: bool,
    normalizer: Month2FitOnlyNormalizer,
    method: str,
    variant: str,
    ridge_lambda: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sigma = float(normalizer.temperature_std)
    for fr in frames:
        mi = dense_model_inputs_from_frame(fr, p_conditioning=p_conditioning)
        pred = predict_gappy(
            pod,
            mi,
            use_temperature_block=False,
            anchor_offset=False,
            p_conditioning=p_conditioning,
        )
        row = evaluate_frame_prediction(
            pred,
            fr["T"],
            train_sigma_T=sigma,
            sim_id=fr["sim_id"],
            split=fr["split"],
            role=fr["role"],
            frame_index=fr["frame_index"],
            method=method,
            variant=variant,
            p_conditioning=p_conditioning,
            K=0,
            site_fraction=1.0,
            model_seed=0,
            mask_replica_id=0,
            layout_id="dense_full_grid",
            task_mode="temperature_free",
            R=int(pod.R),
            ridge_lambda=float(ridge_lambda),
            w_T=float(pod.w_T),
        )
        rows.append(row)
    return rows


def selection_score(rows: list[dict[str, Any]]) -> float:
    agg = aggregate_frame_then_condition_then_split(rows)
    sel = agg["selection_mean_condition_rmse"]
    if sel is None or getattr(sel, "empty", True):
        raise RuntimeError("empty selection table")
    return float(sel["mean_condition_rmse_K"].iloc[0])


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CFG_PATH)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--run-id", type=str, default="dense_linear_u3_v1")
    p.add_argument("--out-root", type=Path, default=ROOT / "results" / "month2_dense_linear")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    contract = assert_dense_linear_contract(cfg)
    manifest_path = args.cache_root / "manifest.json"
    cache_manifest = json.loads(manifest_path.read_text())
    validate_dense_cache_manifest(
        cache_manifest,
        cache_root=args.cache_root,
        require_roles=("train_fit", "seen_train_eval", "val_eval"),
    )

    out_dir = args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    effective = {
        "experiment_id": cfg["experiment_id"],
        "run_id": args.run_id,
        "git_commit": _git_head(),
        "config_path": str(args.config.relative_to(ROOT)),
        "config_sha256": sha256_file(args.config),
        "cache_root": str(args.cache_root),
        "cache_data_fingerprints": cache_manifest.get("data_fingerprints")
        or cache_manifest.get("fingerprints"),
        "contract": contract,
        "device": "cpu",
        "overrides": {
            "run_id": args.run_id,
            "cache_root": str(args.cache_root),
        },
        "scientific_neural_training": False,
        "u4_authorized": False,
    }
    write_json(out_dir / "effective_config.json", effective)

    normalizer = Month2FitOnlyNormalizer()
    t0 = time.perf_counter()
    print("Loading dense cache frames (denormalized)...", flush=True)
    train_frames = dense_physical_frames_from_cache(
        "train_fit", cache_root=args.cache_root, normalizer=normalizer
    )
    seen_frames = dense_physical_frames_from_cache(
        "seen_train_eval", cache_root=args.cache_root, normalizer=normalizer
    )
    val_frames = dense_physical_frames_from_cache(
        "val_eval", cache_root=args.cache_root, normalizer=normalizer
    )
    assert len(train_frames) == 760 and len(seen_frames) == 220 and len(val_frames) == 123

    print("Fitting GlobalMean / ParamMean on full train_fit...", flush=True)
    global_mean = GlobalTrainMeanPredictor.fit([f["T"] for f in train_frames])
    by_cond: dict[str, list[np.ndarray]] = {}
    ops: dict[str, tuple[float, float]] = {}
    for f in train_frames:
        by_cond.setdefault(f["sim_id"], []).append(f["T"])
        ops[f["sim_id"]] = (f["nh3"], f["phi"])
    cond_means = [np.mean(np.stack(v, 0), 0) for _, v in sorted(by_cond.items())]
    cond_ops = [ops[sid] for sid, _ in sorted(by_cond.items())]
    # Match historical consolidation choice when possible; auto selects basis×α on full TRAIN.
    param_mean = ParamConditionedMeanPredictor.fit(
        cond_means, cond_ops, ridge_alpha=0.01, basis="piecewise_phi1"
    )

    print(f"Fitting GappyPOD dense 2C R={contract['R']}...", flush=True)
    pod_off = GappyPODReconstructor.fit(
        train_frames,
        R=contract["R"],
        ridge_lambda=1e-2,
        w_T=contract["w_T"],
        use_param_mean=False,
        n_velocity_components=2,
        seed=contract["seed"],
    )
    pod_on = GappyPODReconstructor.fit(
        train_frames,
        R=contract["R"],
        ridge_lambda=1e-2,
        w_T=contract["w_T"],
        use_param_mean=True,
        param_mean_predictor=param_mean,
        n_velocity_components=2,
        seed=contract["seed"],
    )

    search_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    all_eval_rows: list[dict[str, Any]] = []

    # Zero-sensor references (once)
    for role, frames in (
        ("train_fit", train_frames),
        ("seen_train_eval", seen_frames),
        ("val_eval", val_frames),
    ):
        for fr in frames:
            mi_on = dense_model_inputs_from_frame(fr, p_conditioning=True)
            pred_g = predict_global_mean(global_mean)
            pred_p = predict_param_mean(param_mean, mi_on)
            for method, pred, variant, p_on in (
                ("GlobalMean", pred_g, "p_off", False),
                ("ParamMean", pred_p, "p_on", True),
            ):
                all_eval_rows.append(
                    evaluate_frame_prediction(
                        pred,
                        fr["T"],
                        train_sigma_T=normalizer.temperature_std,
                        sim_id=fr["sim_id"],
                        split=fr["split"],
                        role=role,
                        frame_index=fr["frame_index"],
                        method=method,
                        variant=variant,
                        p_conditioning=p_on,
                        K=0,
                        site_fraction=1.0,
                        model_seed=0,
                        mask_replica_id=0,
                        layout_id="dense_full_grid",
                        task_mode="temperature_free",
                        R=None,
                        ridge_lambda=None,
                    )
                )

    print("VAL λ search (single dense mask)...", flush=True)
    for p_on in contract["p_values"]:
        base = pod_on if p_on else pod_off
        best = None
        for lam in contract["lambdas"]:
            pod = base.with_hyperparams(ridge_lambda=float(lam), w_T=contract["w_T"])
            rows = evaluate_pod_on_frames(
                pod=pod,
                frames=val_frames,
                p_conditioning=p_on,
                normalizer=normalizer,
                method="GappyPOD_dense_free",
                variant="p_on" if p_on else "p_off",
                ridge_lambda=float(lam),
            )
            score = selection_score(rows)
            search_rows.append(
                {
                    "p_conditioning": p_on,
                    "ridge_lambda": float(lam),
                    "val_mean_condition_rmse_K": score,
                    "R": contract["R"],
                    "site_fraction": 1.0,
                    "K": 0,
                    "mask_replicas": 1,
                }
            )
            print(
                f"  p_on={p_on} λ={lam:g} VAL mean-cond RMSE={score:.4f} K",
                flush=True,
            )
            if best is None or score < best[0]:
                best = (score, float(lam), pod)
        assert best is not None
        key = "p_on" if p_on else "p_off"
        selected[key] = {
            "ridge_lambda": best[1],
            "val_mean_condition_rmse_K": best[0],
            "pod": best[2],
        }

    # Final eval with selected λ on all roles (mask replica = 1 only)
    for key, p_on in (("p_off", False), ("p_on", True)):
        pod = selected[key]["pod"]
        lam = selected[key]["ridge_lambda"]
        for role, frames in (
            ("train_fit", train_frames),
            ("seen_train_eval", seen_frames),
            ("val_eval", val_frames),
        ):
            rows = evaluate_pod_on_frames(
                pod=pod,
                frames=frames,
                p_conditioning=p_on,
                normalizer=normalizer,
                method="GappyPOD_dense_free",
                variant=key,
                ridge_lambda=float(lam),
            )
            all_eval_rows.extend(rows)

    # Aggregate per config
    summaries = []
    df_all = pd.DataFrame(all_eval_rows)
    group_keys = ["method", "variant", "p_conditioning", "role", "ridge_lambda", "R"]
    for keys, sub in df_all.groupby(group_keys, dropna=False):
        rows = sub.to_dict(orient="records")
        agg = aggregate_frame_then_condition_then_split(rows)
        pc = agg["per_condition"]
        sel = agg["selection_mean_condition_rmse"]
        summaries.append(
            {
                "method": keys[0],
                "variant": keys[1],
                "p_conditioning": keys[2],
                "role": keys[3],
                "ridge_lambda": keys[4],
                "R": keys[5],
                "mean_condition_rmse_K": float(pc["rmse_K"].mean()) if not pc.empty else None,
                "per_condition": pc.to_dict(orient="records"),
                "selection": sel.to_dict(orient="records") if sel is not None and not sel.empty else [],
            }
        )

    elapsed = time.perf_counter() - t0
    pd.DataFrame(search_rows).to_csv(out_dir / "lambda_search.csv", index=False)
    df_all.to_csv(out_dir / "per_frame.csv.gz", index=False, compression="gzip")
    write_json(
        out_dir / "selection.json",
        {
            k: {
                "ridge_lambda": v["ridge_lambda"],
                "val_mean_condition_rmse_K": v["val_mean_condition_rmse_K"],
            }
            for k, v in selected.items()
        },
    )
    write_json(out_dir / "summary.json", summaries)
    write_json(
        out_dir / "run_card.json",
        {
            **effective,
            "elapsed_s": elapsed,
            "peak_rss_GB": _rss_gb(),
            "n_train": len(train_frames),
            "n_seen": len(seen_frames),
            "n_val": len(val_frames),
            "param_mean": {
                "basis": param_mean.basis,
                "ridge_alpha": param_mean.ridge_alpha,
                "n_conditions": len(cond_means),
            },
            "mask_policy": "single_dense_all_true",
            "historical_sparse_untouched": True,
            "consolidation_v1_untouched": True,
        },
    )
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), "selection": {
        k: {"lambda": v["ridge_lambda"], "val_rmse": v["val_mean_condition_rmse_K"]}
        for k, v in selected.items()
    }, "elapsed_s": elapsed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
