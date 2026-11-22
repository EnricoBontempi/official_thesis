#!/usr/bin/env python3
"""K corrections and frozen T-block on the verified C149 baseline objects.

No refit. The npz written by the S7 baseline job is reloaded after its sha256
is checked. T-block uses the historical A38 selection (lambda, w_T) = 
(0.001, 0.5) for p_off and (0.01, 0.5) for p_on, read back from the frozen
manifest. The w_T=5 stored on the free fit is not the T-block weight.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.evaluation.month2_evaluator import evaluate_frame_prediction  # noqa: E402
from combustion_recon.evaluation.temperature_k import (  # noqa: E402
    C149_BASELINE_ARTIFACT_SHA256,
    GATE_ATOL_K,
    K_VALUES,
    N_FRAMES,
    SWEEP_ROLES,
    field_row,
    gate_free_against_reference,
    layout_by_replica,
    load_frozen_manifest,
    load_historical_tblock,
    prediction_sha256,
    rows_for_historical_t9,
    rows_for_prediction,
    sha256_file,
    sites_for_k,
)
from combustion_recon.models.baselines import (  # noqa: E402
    GlobalTrainMeanPredictor,
    ParamConditionedMeanPredictor,
)
from combustion_recon.models.month2_baseline_adapters import (  # noqa: E402
    predict_gappy,
    predict_global_mean,
    predict_param_mean,
)

MANIFEST = ROOT / "configs/observations/month2_temperature_anchors_nested_v1.json"
HISTORICAL = ROOT / "results/meta/month2_baseline_run_manifest.json"
H = W = 128


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] kbase {message}", flush=True)


def _with_anchors(model_inputs: dict[str, Any], true_T: np.ndarray, sites: np.ndarray) -> dict[str, Any]:
    out = dict(model_inputs)
    sites = np.asarray(sites, dtype=np.int64).ravel()
    mask = np.zeros((H, W), dtype=bool)
    if sites.size:
        mask.ravel()[sites] = True
    out["mask_temperature"] = mask
    out["anchor_site_index"] = sites
    out["anchor_values_physical"] = np.asarray(true_T, dtype=np.float64).ravel()[sites]
    out["K"] = int(sites.size)
    out["task_mode"] = "temperature_assisted" if sites.size else "temperature_free"
    return out


def _load_models(artifact: Path, g15):
    if sha256_file(artifact) != C149_BASELINE_ARTIFACT_SHA256:
        raise SystemExit(f"C149 baseline artifact sha256 is not {C149_BASELINE_ARTIFACT_SHA256}")
    stored = np.load(artifact)
    coef = np.asarray(stored["param_coef"])
    if coef.shape != (6, H * W):
        raise SystemExit(f"ParamMean coef shape {coef.shape} is not (6, {H * W})")
    param = ParamConditionedMeanPredictor(
        coef=coef, H=H, W=W, ridge_alpha=0.01, basis="piecewise_phi1",
        selected_basis="piecewise_phi1", selected_ridge_alpha=0.01,
    )
    def pod(prefix: str):
        payload = {key[len(prefix):]: stored[key] for key in stored.files if key.startswith(prefix)}
        return g15._pod_from_npz(payload, param if bool(payload["use_param_mean"]) else None)
    pod_off, pod_on = pod("pod_off_"), pod("pod_on_")
    if not np.isclose(pod_off.ridge_lambda, 0.001) or not np.isclose(pod_on.ridge_lambda, 0.01):
        raise SystemExit(f"artifact lambdas {pod_off.ridge_lambda}, {pod_on.ridge_lambda} are not the frozen free pair")
    if not np.isclose(pod_off.w_T, 5.0) or not np.isclose(pod_on.w_T, 5.0):
        raise SystemExit("artifact w_T is not the free-fit value 5; refusing to guess which weight was stored")
    mean = GlobalTrainMeanPredictor(mean_field=np.asarray(stored["global_mean"]))
    if mean.mean_field.shape != (H, W):
        raise SystemExit("global mean field is not 128x128")
    return {"GlobalTrainMean": mean, "ParamConditionedMean": param, "pod_off": pod_off, "pod_on": pod_on}


def _reference(root: Path, method: str, p_on: bool, role: str, replica: int) -> Path:
    name = f"{method}_{'p_on' if p_on else 'p_off'}"
    return root / "per_frame" / name / "cov0.05" / role / f"rep{replica}.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--expect-stats-sha256", required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--s7-baseline-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--eval-git-sha", required=True)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--smoke-frames", type=int, default=4)
    args = parser.parse_args()
    smoke = args.mode == "smoke"
    if smoke != ("smoke" in args.out_root.name.lower()):
        raise SystemExit("smoke mode needs an out-root whose name contains 'smoke'; full must not")
    if sha256_file(args.stats) != args.expect_stats_sha256:
        raise SystemExit("normalizer sha mismatch")
    manifest = load_frozen_manifest(MANIFEST)
    layouts = layout_by_replica(manifest)
    tblock = load_historical_tblock(HISTORICAL)
    g15 = _load("gappy15_k", "scripts/15_gappypod_nested_coverage_eval.py")
    base = _load("s7base_k", "scripts/43_s7_baselines_c149.py")
    normalizer = Month2FitOnlyNormalizer(args.stats)
    sigma = float(normalizer.temperature_std)
    models = _load_models(args.artifact, g15)
    tblock_models = {
        False: models["pod_off"].with_hyperparams(**tblock["p_off"]),
        True: models["pod_on"].with_hyperparams(**tblock["p_on"]),
    }
    out = args.out_root
    out.mkdir(parents=True, exist_ok=False)
    masks = {role: base._mask_index(args.mask_root, role) for role in SWEEP_ROLES}
    issues: list[str] = []
    gates: list[dict[str, Any]] = []
    methods = (
        ("GlobalTrainMean", False),
        ("ParamConditionedMean", True),
        ("GappyPOD_free", False),
        ("GappyPOD_free", True),
    )
    for role in SWEEP_ROLES:
        frames = g15._load_role_physical(args.cache_root, role, normalizer)
        if smoke:
            frames = frames[: int(args.smoke_frames)]
        elif len(frames) != N_FRAMES[role]:
            raise SystemExit(f"{role} has {len(frames)} frames, expected {N_FRAMES[role]}")
        _log(f"{role} frames={len(frames)}")
        for method, p_on in methods:
            for replica in (0, 1, 2):
                layout = layouts[replica]
                order = np.asarray(layout["site_order_flat"], dtype=np.int64)
                rows: list[dict[str, Any]] = []
                t_rows: list[dict[str, Any]] = []
                for fr in frames:
                    mask = masks[role][(fr["sim_id"], int(fr["frame"]), int(replica))]
                    vel_sites = mask["sites_5pct"]
                    mi = g15._model_inputs_from_sites(fr, vel_sites, p_conditioning=p_on)
                    if method == "GlobalTrainMean":
                        pred = predict_global_mean(models["GlobalTrainMean"])
                    elif method == "ParamConditionedMean":
                        pred = predict_param_mean(models["ParamConditionedMean"], mi)
                    else:
                        pred = predict_gappy(models["pod_on"] if p_on else models["pod_off"], mi, p_conditioning=p_on)
                    s7_row = evaluate_frame_prediction(
                        pred, fr["T"], train_sigma_T=sigma, sim_id=fr["sim_id"],
                        split="val" if role == "val_eval" else "seen", role=role,
                        frame_index=int(fr["frame"]), method=method, variant="p_on" if p_on else "p_off",
                        p_conditioning=p_on, K=0, site_fraction=0.05, model_seed=0,
                        mask_replica_id=int(replica), layout_id="random_sparse_primary",
                        task_mode="temperature_free", mask_hash=mask["hash_5pct"],
                    )
                    identity = {
                        "run_id": "C149-baselines",
                        "method": method,
                        "family": "baseline",
                        "p_conditioning": bool(p_on),
                        "model_seed": 0,
                        "checkpoint": "fitted",
                        "role": role,
                        "mask_replica_id": int(replica),
                        "sim_id": str(fr["sim_id"]),
                        "frame_index": int(fr["frame"]),
                        "velocity_mask_hash": mask["hash_5pct"],
                        "t_order_sha256": layout["order_sha256"],
                        "prediction_sha256": prediction_sha256(pred),
                        "s7_rmse_K": float(s7_row["rmse_K"]),
                        "artifact_sha256": C149_BASELINE_ARTIFACT_SHA256,
                    }
                    rows.extend(rows_for_prediction(
                        pred, fr["T"], order, identity=identity,
                        layout_id=f"nested_r{replica}", layout_kind="primary",
                    ))
                    rows.extend(rows_for_historical_t9(pred, fr["T"], identity=identity))
                    if method == "GappyPOD_free":
                        pod = tblock_models[p_on]
                        for k in K_VALUES:
                            sites = sites_for_k(order, k)
                            assisted = _with_anchors(mi, fr["T"], sites)
                            # Off-anchor temperature must not enter the container.
                            if k and np.any(np.asarray(fr["T"], dtype=np.float64).ravel()[sites] != assisted["anchor_values_physical"]):
                                raise SystemExit("anchor values were not the true temperature")
                            field = predict_gappy(pod, assisted, use_temperature_block=True, p_conditioning=p_on)
                            if k == 0 and not np.array_equal(field, pred):
                                delta = float(np.max(np.abs(np.asarray(field) - np.asarray(pred))))
                                issues.append(f"K=0 T-block differs from free by {delta} ({role} rep{replica} p_on={p_on})")
                            t_rows.append(field_row(
                                field, fr["T"], sites, order, identity=identity,
                                layout_id=f"nested_r{replica}", layout_kind="primary", k=k,
                                treatment="tblock", level_recipe="historical_tblock",
                                delta_applied=float("nan"),
                            ))
                stem = f"{method}_{'p_on' if p_on else 'p_off'}"
                dest = out / "per_frame" / stem / role / f"rep{replica}.csv"
                dest.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(dest, index=False)
                reread = pd.read_csv(dest)
                reference = _reference(args.s7_baseline_root, method, p_on, role, replica)
                ref = pd.read_csv(reference)
                if smoke:
                    keep = set(zip(reread["sim_id"].astype(str), reread["frame_index"].astype(int)))
                    ref = ref[ref.apply(lambda row: (str(row.sim_id), int(row.frame_index)) in keep, axis=1)]
                free = reread[(reread.treatment == "free") & (reread.K == 0) & (reread.layout_kind == "primary")].copy()
                free["sim_id"] = free["sim_id"].astype(str)
                free["frame_index"] = free["frame_index"].astype(int)
                gate = gate_free_against_reference(
                    free, ref, reference_path=str(reference), reference_sha256=sha256_file(reference)
                )
                per_frame = gate.pop("per_frame", None)
                gate.update({"method": stem, "role": role, "replica": replica})
                gates.append(gate)
                (out / "gate").mkdir(exist_ok=True)
                (out / "gate" / f"{stem}_{role}_rep{replica}.json").write_text(json.dumps(gate, indent=2) + "\n")
                if per_frame is not None:
                    per_frame.to_csv(out / "gate" / f"{stem}_{role}_rep{replica}_frames.csv", index=False)
                if t_rows:
                    t_dest = out / "per_frame" / f"GappyPOD_tblock_{'p_on' if p_on else 'p_off'}" / role / f"rep{replica}.csv"
                    t_dest.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(t_rows).to_csv(t_dest, index=False)
                _log(f"{stem} {role} rep{replica} gate={gate.get('ok')} macro_delta={gate.get('macro_delta_K')}")
    ok = all(item.get("ok") for item in gates) and not issues
    marker = "SMOKE_OK.json" if smoke and ok else "COMPLETE.json" if ok else "GATE_FAILED.json"
    (out / "provenance.json").write_text(json.dumps({
        "artifact_sha256": C149_BASELINE_ARTIFACT_SHA256,
        "tblock": tblock,
        "artifact_w_T_not_used_for_tblock": 5.0,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "eval_git_sha": args.eval_git_sha,
        "no_refit": True,
        "gates": gates,
        "issues": issues,
    }, indent=2) + "\n")
    (out / marker).write_text(json.dumps({"ok": ok, "n_issues": len(issues)}, indent=2) + "\n")
    _log(f"done {marker} issues={issues[:5]}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
