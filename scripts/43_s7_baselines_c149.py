#!/usr/bin/env python3
"""S7 baselines fitted on the C149 pool (GlobalTrainMean, ParamConditionedMean, GappyPOD R=32 p_on/off).

CPU only. Hyperparameters are frozen (configs/experiments/month2_s7_c149_baselines.yaml). Steps:
1. control: the same code fits on the A38 reference760 subset of the C149 cache and must reproduce the
   historical 5% baseline rows within 0.05 K (macro RMSE and centered RMSE) -- otherwise stop;
2. fit on the whole C149 pool (2980 frames) and persist the fitted objects with hashes;
3. evaluate roles reference760/seen220/VAL123, coverages 1/5/20%, mask replicas 0/1/2 with the persisted
   nested masks (mask_hash per row checked against the verified payload);
4. read everything back; COMPLETE.json only if the control, the counts and the hash checks all pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.data.month2_protocol import sha256_file  # noqa: E402
from combustion_recon.data.sparse_eval_masks import check_eval_mask_contract  # noqa: E402
from combustion_recon.evaluation.month2_evaluator import (  # noqa: E402
    aggregate_frame_then_condition_then_split,
    evaluate_frame_prediction,
)
from combustion_recon.models.baselines import (  # noqa: E402
    GappyPODReconstructor,
    GlobalTrainMeanPredictor,
    ParamConditionedMeanPredictor,
)
from combustion_recon.models.month2_baseline_adapters import (  # noqa: E402
    predict_gappy,
    predict_global_mean,
    predict_param_mean,
)
from combustion_recon.training.dense_cache import verify_cache_payload_hashes  # noqa: E402
from combustion_recon.training.scientific_epoch_sampler import cache_indices_for_pool  # noqa: E402

ROLES = ("train_fit", "seen_train_eval", "val_eval")
COVS = (0.01, 0.05, 0.20)
REPS = (0, 1, 2)
N_FRAMES = {"train_fit": 760, "seen_train_eval": 220, "val_eval": 123}
HASH_KEY = {0.01: "hash_1pct", 0.05: "hash_5pct", 0.2: "hash_20pct"}
SITE_KEY = {0.01: "sites_1pct", 0.05: "sites_5pct", 0.2: "sites_20pct"}
METHODS = (
    ("GlobalTrainMean", False, "zero_sensor"),
    ("ParamConditionedMean", True, "zero_sensor"),
    ("GappyPOD_free", False, "free"),
    ("GappyPOD_free", True, "free"),
)


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] s7base {msg}", flush=True)


def _mask_index(mask_root: Path, role: str) -> dict[tuple[str, int, int], dict[str, Any]]:
    z = np.load(mask_root / f"{role}_nested_sites.npz", allow_pickle=True)
    out = {}
    for i, (s, f, r) in enumerate(zip(z["sim_id"].tolist(), z["frame_index"], z["replica_id"])):
        key = (str(s), int(f), int(r))
        out[key] = {SITE_KEY[c]: np.asarray(z[SITE_KEY[c]][i], dtype=np.int64) for c in (0.01, 0.05, 0.2)}
        out[key].update({HASH_KEY[c]: str(z[HASH_KEY[c]][i]) for c in (0.01, 0.05, 0.2)})
    return out


def fit_all(frames: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    rec = cfg["recipes"]
    pm = rec["ParamConditionedMean"]
    g = rec["GappyPOD_free"]
    gm = GlobalTrainMeanPredictor.fit([f["T"] for f in frames])
    by_cond: dict[str, list[np.ndarray]] = {}
    ops: dict[str, tuple[float, float]] = {}
    for f in frames:
        by_cond.setdefault(f["sim_id"], []).append(f["T"])
        ops[f["sim_id"]] = (f["nh3"], f["phi"])
    conds = sorted(by_cond)
    param = ParamConditionedMeanPredictor.fit(
        [np.mean(np.stack(by_cond[c], 0), 0) for c in conds], [ops[c] for c in conds],
        ridge_alpha=float(pm["ridge_alpha"]), basis=str(pm["basis"]))
    common = dict(R=int(g["R"]), ridge_lambda=1e-2, w_T=float(g["w_T"]), n_velocity_components=int(g["n_velocity_components"]), seed=int(g["seed"]))
    pod_off = GappyPODReconstructor.fit(frames, use_param_mean=False, **common).with_hyperparams(ridge_lambda=float(g["ridge_lambda"]["p_off"]))
    pod_on = GappyPODReconstructor.fit(frames, use_param_mean=True, param_mean_predictor=param, **common).with_hyperparams(ridge_lambda=float(g["ridge_lambda"]["p_on"]))
    return {"GlobalTrainMean": gm, "ParamConditionedMean": param, "pod_off": pod_off, "pod_on": pod_on,
            "n_conditions": len(conds), "n_frames": len(frames)}


def evaluate(models, frames_by_role, masks, g15, *, covs, roles, reps, sigma_T, layout_id, tags, only_methods=None):
    rows_by_cfg: dict[tuple, list[dict[str, Any]]] = {}
    for role in roles:
        for method, p_on, variant in METHODS:
            if only_methods and (method, p_on) not in only_methods:
                continue
            for cov in covs:
                for rep in reps:
                    rows = []
                    for fr in frames_by_role[role]:
                        m = masks[role][(fr["sim_id"], int(fr["frame"]), int(rep))]
                        sites = m[SITE_KEY[round(cov, 2)]]
                        mi = g15._model_inputs_from_sites(fr, sites, p_conditioning=p_on)
                        if method == "GlobalTrainMean":
                            pred = predict_global_mean(models["GlobalTrainMean"])
                        elif method == "ParamConditionedMean":
                            pred = predict_param_mean(models["ParamConditionedMean"], mi)
                        else:
                            pod = models["pod_on"] if p_on else models["pod_off"]
                            pred = predict_gappy(pod, mi, p_conditioning=p_on)
                        extra = {}
                        if method == "GappyPOD_free":
                            extra = {"ridge_lambda": float((models["pod_on"] if p_on else models["pod_off"]).ridge_lambda), "R": 32}
                        row = evaluate_frame_prediction(
                            pred, fr["T"], train_sigma_T=sigma_T, sim_id=fr["sim_id"],
                            split={"train_fit": "train", "val_eval": "val"}.get(role, "seen"), role=role,
                            frame_index=int(fr["frame"]), method=method, variant=variant, p_conditioning=p_on, K=0,
                            site_fraction=float(cov), model_seed=0, mask_replica_id=int(rep), layout_id=layout_id,
                            task_mode="temperature_free", **extra)
                        row.pop("_spatial_bias_map", None)
                        row["mask_hash"] = m[HASH_KEY[round(cov, 2)]]
                        row.update(tags)
                        rows.append(row)
                    rows_by_cfg[(method, p_on, role, round(cov, 2), int(rep))] = rows
    return rows_by_cfg


def macro_table(rows_by_cfg: dict[tuple, list[dict[str, Any]]]) -> pd.DataFrame:
    out = []
    groups: dict[tuple, list] = {}
    for (method, p_on, role, cov, rep), rows in rows_by_cfg.items():
        agg = aggregate_frame_then_condition_then_split(rows)["per_condition"]
        groups.setdefault((method, p_on, role, cov), []).append(agg)
    for (method, p_on, role, cov), parts in sorted(groups.items()):
        pc = pd.concat(parts, ignore_index=True)
        out.append({"method": method, "p_conditioning": p_on, "role": role, "site_fraction": cov,
                    "rmse_K": float(pc["rmse_K"].mean()), "abs_bias_K": float(pc["abs_bias_K"].mean()),
                    "centered_rmse_K": float(pc["centered_rmse_K"].mean()), "n_replicas": len(parts)})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/experiments/month2_s7_c149_baselines.yaml")
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--cache-hash-file", type=Path, required=True)
    ap.add_argument("--expect-cache-aggregate-sha256", required=True)
    ap.add_argument("--mask-root", type=Path, required=True)
    ap.add_argument("--expect-mask-manifest-sha256", required=True)
    ap.add_argument("--expect-mask-aggregate-sha256", required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--expect-stats-sha256", required=True)
    ap.add_argument("--historical-summary", type=Path, default=ROOT / "results/tables/month2_baseline_summary.csv")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--eval-git-sha", required=True)
    ap.add_argument("--atol-K", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true", help="few frames per role, skips COMPLETE")
    args = ap.parse_args()
    if args.smoke != ("smoke" in args.out_root.name.lower()):
        raise SystemExit("--smoke requires an out-root named *smoke*; the scientific root must not be")
    cfg = yaml.safe_load(args.config.read_text())
    frozen = yaml.safe_load((ROOT / "configs/experiments/month2_sparse_gappypod_nested.yaml").read_text())["frozen_from_5pct"]
    g = cfg["recipes"]["GappyPOD_free"]
    if (g["ridge_lambda"]["p_off"], g["ridge_lambda"]["p_on"], g["R"]) != (frozen["free_p_off_lambda"], frozen["free_p_on_lambda"], frozen["R"]):
        raise SystemExit("baseline config differs from the frozen S6 selection")
    pm = cfg["recipes"]["ParamConditionedMean"]
    man = json.loads((ROOT / "results/meta/month2_baseline_run_manifest.json").read_text())["selected"]["param_mean"]
    if (pm["basis"], pm["ridge_alpha"]) != (man["basis"], man["ridge_alpha"]):
        raise SystemExit("ParamMean config differs from the historical selection")
    if sha256_file(args.stats) != args.expect_stats_sha256:
        raise SystemExit("normalizer sha mismatch")
    ver = verify_cache_payload_hashes(json.loads(args.cache_hash_file.read_text()), cache_root=args.cache_root)
    if not ver["ok"] or ver["actual_aggregate_sha256"] != args.expect_cache_aggregate_sha256:
        raise SystemExit(f"cache hash mismatch {ver['mismatches']}")
    check_eval_mask_contract(manifest_path=args.mask_root / "manifest.json", payload_root=args.mask_root,
                             expected_manifest_file_sha256=args.expect_mask_manifest_sha256,
                             expected_aggregate_payload_sha256=args.expect_mask_aggregate_sha256)
    _log("inputs verified (cache aggregate, masks, stats)")
    g15 = _load("s6gappy15", "scripts/15_gappypod_nested_coverage_eval.py")
    normalizer = Month2FitOnlyNormalizer(args.stats)
    sigma_T = float(normalizer.temperature_std)
    layout_id = cfg["evaluation"]["layout_id"]
    train_meta = json.loads((args.cache_root / "train_fit" / "frames.json").read_text())
    a38_idx = cache_indices_for_pool(train_meta, "A38")
    if len(a38_idx) != 760:
        raise SystemExit(f"A38 subset resolved {len(a38_idx)} frames")
    all_train = g15._load_role_physical(args.cache_root, "train_fit", normalizer)
    if len(all_train) != 2980 or len({f["sim_id"] for f in all_train}) != 20:
        raise SystemExit("C149 fit pool must be 2980 frames over 20 conditions")
    counts = pd.Series([f["sim_id"] for f in all_train]).value_counts()
    if not (counts == 149).all():
        raise SystemExit("C149 pool is not 149 frames per condition")
    masks = {role: _mask_index(args.mask_root, role) for role in ROLES}
    sub = 12 if args.smoke else None

    def role_frames(role: str, base: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if role == "train_fit":
            keep = set(a38_idx)
            frs = [f for f in base if f["cache_index"] in keep]
        else:
            frs = base
        if sub:
            frs = frs[:: max(1, len(frs) // sub)][:sub]
        return frs

    frames_by_role = {"train_fit": role_frames("train_fit", all_train)}
    for role in ("seen_train_eval", "val_eval"):
        frames_by_role[role] = role_frames(role, g15._load_role_physical(args.cache_root, role, normalizer))
    if not args.smoke:
        for role in ROLES:
            if len(frames_by_role[role]) != N_FRAMES[role]:
                raise SystemExit(f"{role}: {len(frames_by_role[role])} frames, expected {N_FRAMES[role]}")
    out = args.out_root
    out.mkdir(parents=True, exist_ok=False)

    # 1. control on the A38 subset of the same cache
    _log("control: fit on the A38 reference760 subset")
    a38_frames = [f for f in all_train if f["cache_index"] in set(a38_idx)]
    ctl_models = fit_all(a38_frames, cfg)
    ctl_rows = evaluate(ctl_models, frames_by_role, masks, g15, covs=(0.05,), roles=ROLES, reps=REPS, sigma_T=sigma_T,
                        layout_id=layout_id, tags={"fit_pool": "A38_subset_control"})
    ctl = macro_table(ctl_rows)
    hist = pd.read_csv(args.historical_summary)
    checks = []
    for _, h in hist[(hist["K"] == 0) & np.isclose(hist["site_fraction"], 0.05)
                      & hist["method"].isin(["GlobalTrainMean", "ParamConditionedMean", "GappyPOD_free"])].iterrows():
        m = ctl[(ctl.method == h["method"]) & (ctl.role == h["role"]) & (ctl.p_conditioning == bool(h["p_conditioning"]))]
        if len(m) != 1:
            checks.append({"method": h["method"], "role": h["role"], "ok": False, "error": "missing"})
            continue
        d = float(m.iloc[0]["rmse_K"] - h["rmse_K"])
        dc = float(m.iloc[0]["centered_rmse_K"] - h["centered_rmse_K"])
        checks.append({"method": h["method"], "role": h["role"], "p_conditioning": bool(h["p_conditioning"]),
                       "historical_rmse_K": float(h["rmse_K"]), "control_rmse_K": float(m.iloc[0]["rmse_K"]), "delta_rmse_K": d,
                       "delta_centered_rmse_K": dc, "ok": bool(abs(d) <= args.atol_K and abs(dc) <= args.atol_K)})
    control_ok = bool(len(checks) == 12 and all(c["ok"] for c in checks)) if not args.smoke else None
    (out / "control_a38_subset.json").write_text(json.dumps({"atol_K": args.atol_K, "ok": control_ok, "n_checks": len(checks), "checks": checks}, indent=2) + "\n")
    _log(f"control: n={len(checks)} ok={control_ok} max|delta|={max(abs(c.get('delta_rmse_K', 0)) for c in checks):.3e}")
    if control_ok is False:
        (out / "CONTROL_FAILED.json").write_text(json.dumps({"ok": False}) + "\n")
        return 3
    del ctl_models, ctl_rows

    # 2. fit on C149
    _log("fit on the C149 pool")
    models = fit_all(all_train, cfg)
    fit_keys = sorted((f["sim_id"], int(f["frame"])) for f in all_train)
    fit_sha = hashlib.sha256(json.dumps(fit_keys).encode()).hexdigest()
    art = out / "artifacts"
    art.mkdir()
    np.savez_compressed(art / "c149_fitted_baselines.npz",
                        global_mean=models["GlobalTrainMean"].mean_field,
                        param_coef=models["ParamConditionedMean"].coef,
                        **{f"pod_off_{k}": v for k, v in g15._pod_to_npz(models["pod_off"]).items() if v is not None},
                        **{f"pod_on_{k}": v for k, v in g15._pod_to_npz(models["pod_on"]).items() if v is not None})
    prov = {"experiment_id": cfg["experiment_id"], "config_sha256": sha256_file(args.config), "eval_git_sha": args.eval_git_sha,
            "fit_pool": "C149", "n_fit_frames": models["n_frames"], "n_conditions": models["n_conditions"],
            "fit_keys_sha256": fit_sha, "frames_per_condition": 149, "artifact_sha256": sha256_file(art / "c149_fitted_baselines.npz"),
            "normalizer_sha256": args.expect_stats_sha256, "cache_aggregate_sha256": args.expect_cache_aggregate_sha256,
            "eval_mask_manifest_file_sha256": args.expect_mask_manifest_sha256,
            "eval_mask_aggregate_payload_sha256": args.expect_mask_aggregate_sha256,
            "scales_T_vi_vj": [models["pod_off"].scale_T, models["pod_off"].scale_vi, models["pod_off"].scale_vj],
            "param_mean_loco_rmse_K": float(models["ParamConditionedMean"].loco_rmse_K),
            "frozen": {"R": g["R"], "lambda": g["ridge_lambda"], "param_mean": pm, "w_T": g["w_T"], "seed": g["seed"]},
            "a38_artifacts_untouched": True, "control_ok": control_ok, "smoke": bool(args.smoke)}
    (out / "fit_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")

    # 3. evaluation
    _log("evaluation: 4 methods x 3 coverages x 3 roles x 3 replicas")
    rows_by_cfg = evaluate(models, frames_by_role, masks, g15, covs=COVS, roles=ROLES, reps=REPS, sigma_T=sigma_T,
                           layout_id=layout_id, tags={"fit_pool": "C149", "pool_id": "C149", "n_fit_frames": 2980,
                                                       "fit_keys_sha256": fit_sha, "eval_git_sha": args.eval_git_sha})
    n_csv = 0
    for (method, p_on, role, cov, rep), rows in rows_by_cfg.items():
        name = f"{method}_{'p_on' if p_on else 'p_off'}"
        path = out / "per_frame" / name / f"cov{cov:.2f}" / role / f"rep{rep}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        n_csv += 1
    macro = macro_table(rows_by_cfg)
    macro["fit_pool"] = "C149"
    macro.to_csv(out / "summary_baselines_c149.csv", index=False)
    # 4. readback + mask hash check against the payload
    issues = []
    for (method, p_on, role, cov, rep), _ in rows_by_cfg.items():
        name = f"{method}_{'p_on' if p_on else 'p_off'}"
        fr = pd.read_csv(out / "per_frame" / name / f"cov{cov:.2f}" / role / f"rep{rep}.csv")
        if not args.smoke and len(fr) != N_FRAMES[role]:
            issues.append(f"{name}/{cov}/{role}/{rep}: {len(fr)} rows")
        if not np.isfinite(fr["rmse_K"].to_numpy()).all():
            issues.append(f"{name}/{cov}/{role}/{rep}: non-finite rmse")
        for s, f, h in zip(fr["sim_id"], fr["frame_index"], fr["mask_hash"]):
            if masks[role][(str(s), int(f), int(rep))][HASH_KEY[cov]] != h:
                issues.append(f"{name}/{cov}/{role}/{rep}: mask_hash differs from the payload")
                break
    expect_csv = len(METHODS) * len(COVS) * len(ROLES) * len(REPS)
    if n_csv != expect_csv:
        issues.append(f"{n_csv} csv, expected {expect_csv}")
    if issues:
        (out / "BASELINES_FAILED.json").write_text(json.dumps({"ok": False, "issues": issues[:50]}, indent=2) + "\n")
        _log(f"FAILED {issues[:5]}")
        return 3
    marker = "SMOKE_OK.json" if args.smoke else "COMPLETE.json"
    (out / marker).write_text(json.dumps({"ok": True, "n_csv": n_csv, "control_ok": control_ok, "fit_keys_sha256": fit_sha}, indent=2) + "\n")
    _log(f"done {marker}")
    print(macro[macro.site_fraction == 0.05].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
