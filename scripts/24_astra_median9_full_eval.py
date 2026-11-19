#!/usr/bin/env python3
"""Median9 second pass on the full Astra seen/VAL set. No training.

One base prediction per method, frame and replica. From it: free, oracle, AO9
(tau = 1), unshrunk mean9, and the ordinary median9. T-only stays the historical
constant and does not receive mean9 or median9.

The primary free/oracle/AO9 tables are the replay reference. A missing or
out-of-tolerance replay does not write scientific COMPLETE.json. The diagnostic
tables stay. Full fields are not saved.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
from combustion_recon.data.month2_observations import mask_hash  # noqa: E402
from combustion_recon.data.month2_protocol import sha256_file  # noqa: E402
from combustion_recon.data.sparse_eval_masks import (  # noqa: E402
    sites_to_mask,
    verify_payload_against_manifest,
)
from combustion_recon.evaluation.astra_level import (  # noqa: E402
    prediction_tables,
    primary_level_replay,
    t9_site_indices,
    t_only_constant_row,
)

ROLES = ("seen_train_eval", "val_eval")
REPLICAS = (0, 1, 2)
COV = 0.05
LEVEL_TREATMENTS = ("free", "oracle", "ao9", "mean9", "median9")
CONFIG_ID = "month2_astra_median9_second_pass"


def _load_s18():
    path = ROOT / "scripts" / "18_astra_level_structure_eval.py"
    spec = importlib.util.spec_from_file_location("astra_level_eval18", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def _require_sha(path: Path, expected: str, label: str) -> str:
    got = sha256_file(path)
    if got != expected:
        raise SystemExit(f"{label} hash mismatch {got} != {expected}")
    return got


def _verify_masks(mask_root: Path, expected_manifest_sha: str) -> dict[str, Any]:
    man_path = mask_root / "manifest.json"
    manifest_sha = _require_sha(man_path, expected_manifest_sha, "mask manifest")
    report = verify_payload_against_manifest(payload_root=mask_root, manifest_path=man_path)
    if not report["ok"]:
        raise SystemExit(f"mask payload hash mismatch {report['mismatches']}")
    checked = 0
    for role in ROLES:
        path = mask_root / f"{role}_nested_sites.npz"
        with np.load(path, allow_pickle=True) as z:
            stored = [str(x) for x in z["hash_5pct"]]
            sites = np.asarray(z["sites_5pct"])
            for i, expected in enumerate(stored):
                got = mask_hash(sites_to_mask(np.asarray(sites[i], dtype=np.int64)))
                if got != expected:
                    raise SystemExit(f"mask hash mismatch {role} row {i}: {got} != {expected}")
                checked += 1
    manifest = json.loads(man_path.read_text())
    return {
        "mask_manifest_sha256": manifest_sha,
        "mask_aggregate_payload_sha256": manifest.get("aggregate_payload_sha256"),
        "mask_payload_files_sha256": manifest.get("payload_files_sha256"),
        "mask_payload_verified": True,
        "mask_rows_hash_checked": checked,
    }


def _verify_cache(cache_root: Path, expected_aggregate: str, identity_files: dict | None) -> dict[str, Any]:
    from combustion_recon.training.dense_cache import (
        compute_cache_payload_hashes,
        validate_dense_cache_manifest,
    )

    manifest = json.loads((cache_root / "manifest.json").read_text())
    validate_dense_cache_manifest(manifest, cache_root=cache_root, require_roles=("train_fit",) + ROLES)
    payload = compute_cache_payload_hashes(cache_root)
    if payload.get("aggregate_sha256") != expected_aggregate:
        raise SystemExit(
            f"cache aggregate {payload.get('aggregate_sha256')} != {expected_aggregate}"
        )
    if identity_files:
        for rel, got in (payload.get("files") or {}).items():
            if rel in identity_files and identity_files[rel] != got:
                raise SystemExit(f"cache file hash mismatch {rel}")
    return {
        "cache_aggregate_sha256": payload.get("aggregate_sha256"),
        "cache_files_sha256": payload.get("files"),
    }


def _mask_for(masks: dict[str, Any], key: tuple[str, int, int]) -> tuple[np.ndarray, str]:
    row = masks["row_of"].get(key)
    if row is None:
        raise SystemExit(f"missing mask entry {key}")
    sites = np.sort(np.asarray(masks["sites_5pct"][row], dtype=np.int64))
    stored = str(masks["hash_5pct"][row])
    if mask_hash(sites_to_mask(sites)) != stored:
        raise SystemExit(f"consumed mask hash mismatch {key}")
    return sites, stored


def _iter_role_frames(cache_root: Path, role: str, normalizer: Month2FitOnlyNormalizer):
    """One physical frame at a time. Velocity denorm matches the stacked path."""
    rd = cache_root / role
    frames_meta = json.loads((rd / "frames.json").read_text())
    vi_src = np.load(rd / "vi_norm.npy", mmap_mode="r")
    vj_src = np.load(rd / "vj_norm.npy", mmap_mode="r")
    t_src = np.load(rd / "target_T_K.npy", mmap_mode="r")
    vi_s = normalizer.fields["Velocity_i"]
    vj_s = normalizer.fields["Velocity_j"]
    zero_vk = np.zeros(vi_src.shape[1:], dtype=np.float64)
    for i, meta in enumerate(frames_meta):
        yield {
            "cache_index": i,
            "sim_id": str(meta["sim_id"]),
            "frame": int(meta["frame_index"]),
            "nh3": float(meta.get("nh3_fraction", meta.get("nh3", 0.0))),
            "phi": float(meta.get("phi", meta.get("equivalence_ratio", 1.0))),
            "vi": np.asarray(vi_src[i], dtype=np.float64) * float(vi_s["std"]) + float(vi_s["mean"]),
            "vj": np.asarray(vj_src[i], dtype=np.float64) * float(vj_s["std"]) + float(vj_s["mean"]),
            "vk": zero_vk,
            "T": np.asarray(t_src[i], dtype=np.float64),
        }


def _coverage_ok(df: pd.DataFrame, methods: list[str], expected: dict[str, int]) -> bool:
    level = set(LEVEL_TREATMENTS)
    for method in methods:
        for role, n_frames in expected.items():
            for rep in REPLICAS:
                sub = df[
                    (df["method"] == method) & (df["role"] == role) & (df["replica_id"] == rep)
                ]
                if set(sub["treatment"]) != level or len(sub) != n_frames * len(level):
                    return False
    return True


def _residual_coverage_ok(residuals: pd.DataFrame, methods: list[str], expected: dict[str, int]) -> bool:
    if len(residuals) == 0 or set(residuals["method"]) != set(methods):
        return False
    for method in methods:
        for role, n_frames in expected.items():
            for rep in REPLICAS:
                sub = residuals[
                    (residuals["method"] == method)
                    & (residuals["role"] == role)
                    & (residuals["replica_id"] == rep)
                ]
                if sub.empty:
                    return False
                counts = sub.groupby(["sim_id", "frame_index"]).size()
                if len(counts) != n_frames or int(counts.min()) != 9 or int(counts.max()) != 9:
                    return False
    return True


def _t_only_coverage_ok(df: pd.DataFrame, method: str, expected: dict[str, int]) -> bool:
    for role, n_frames in expected.items():
        for rep in REPLICAS:
            sub = df[(df["method"] == method) & (df["role"] == role) & (df["replica_id"] == rep)]
            if set(sub["treatment"]) != {"t_only_const"} or len(sub) != n_frames:
                return False
            if sub["delta_median9_K"].notna().any() or sub["delta_mean9_K"].notna().any():
                return False
    return True


def _load_primary(primary_root: Path, folder: str) -> tuple[pd.DataFrame | None, dict[str, str], list[str]]:
    files: dict[str, str] = {}
    missing: list[str] = []
    frames: list[pd.DataFrame] = []
    for role in ROLES:
        path = primary_root / folder / f"per_frame_{role}.csv.gz"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
            continue
        files[str(path)] = sha256_file(path)
        frames.append(pd.read_csv(path))
    if missing or not frames:
        return None, files, missing
    return pd.concat(frames, ignore_index=True), files, []


def _jsonable_replay(replay: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in replay.items() if k != "frames"}
    return out


def _finish(
    out_dir: Path,
    metrics: pd.DataFrame,
    residuals: pd.DataFrame,
    *,
    primary: pd.DataFrame | None,
    primary_files: dict[str, str],
    missing_primary: list[str],
    methods: list[str],
    expected: dict[str, int],
    atol: float,
    t_only_method: str | None,
) -> int:
    for role in ROLES:
        metrics[metrics.role == role].to_csv(
            out_dir / f"per_frame_{role}.csv.gz", index=False, compression="gzip"
        )
        if len(residuals):
            residuals[residuals.role == role].to_csv(
                out_dir / f"anchor_residuals_{role}.csv.gz", index=False, compression="gzip"
            )
    coverage = _coverage_ok(metrics, methods, expected)
    residual_ok = _residual_coverage_ok(residuals, methods, expected)
    t_only_ok = True if t_only_method is None else _t_only_coverage_ok(metrics, t_only_method, expected)
    replays: dict[str, Any] = {}
    frame_parts: list[pd.DataFrame] = []
    if primary is None or len(primary) == 0:
        executed = False
        validated = False
        reason = "missing_or_empty_primary_reference"
    else:
        ref = primary.copy()
        if "method" not in ref.columns:
            if "run_id" not in ref.columns:
                executed = False
                validated = False
                reason = "primary_reference_has_no_method"
                ref = None
            else:
                ref["method"] = ref["run_id"].astype(str)
        level = None if ref is None else primary_level_replay(
            metrics, ref, atol_K=atol, roles=ROLES, replicas=REPLICAS, expected_frames=expected,
            compare_delta_applied=True,
        )
        if level is not None:
            replays["free_oracle_ao9"] = _jsonable_replay(level)
            part = level["frames"].copy()
            part.insert(0, "gate", "free_oracle_ao9")
            frame_parts.append(part)
            executed = bool(level["replay_executed"])
            validated = bool(level["validated"])
            reason = level["reason"]
            if t_only_method is not None:
                t_only = primary_level_replay(
                    metrics, ref, atol_K=atol, treatments=("t_only_const",),
                    roles=ROLES, replicas=REPLICAS, expected_frames=expected,
                    compare_delta_applied=False,
                )
                replays["t_only_const"] = _jsonable_replay(t_only)
                part = t_only["frames"].copy()
                part.insert(0, "gate", "t_only_const")
                frame_parts.append(part)
                executed = executed and bool(t_only["replay_executed"])
                validated = validated and bool(t_only["validated"])
                if not t_only["validated"]:
                    reason = t_only["reason"] or reason
        else:
            executed = False
            validated = False
    if not coverage or not residual_ok or not t_only_ok:
        validated = False
        if executed:
            reason = "diagnostic_coverage"
    if frame_parts:
        pd.concat(frame_parts, ignore_index=True).to_csv(
            out_dir / "replay_primary_frames.csv.gz", index=False, compression="gzip"
        )
    gate = {
        "scientific_validated": bool(executed and validated),
        "replay_executed": bool(executed),
        "reason": None if executed and validated else reason,
        "coverage_ok": coverage,
        "residual_coverage_ok": residual_ok,
        "t_only_coverage_ok": t_only_ok,
        "atol_K": atol,
        "primary_files_sha256": primary_files,
        "missing_primary": missing_primary,
        "replays": replays,
        "note": (
            "AO9 is sum(resid)/(9+tau). mean9 is sum(resid)/9. median9 is the fifth "
            "ordered residual. A median-versus-AO9 difference is not only robustness. "
            "T-only is not given mean9 or median9."
        ),
    }
    _write_json(out_dir / "replay_primary.json", gate)
    _write_json(out_dir / "GATE.json", {
        "scientific_validated": gate["scientific_validated"],
        "replay_executed": gate["replay_executed"],
        "reason": gate["reason"],
    })
    if gate["scientific_validated"]:
        _write_json(out_dir / "COMPLETE.json", {
            "scientific_validated": True,
            "rows": int(len(metrics)),
            "residual_rows": int(len(residuals)),
            "scope": "seen220_val123_replicas0_1_2",
            "not_the_example_frame_package": True,
        })
        print(json.dumps({"status": "astra_median9_validated", "rows": int(len(metrics))}), flush=True)
        return 0
    print(json.dumps({"status": "ASTRA_MEDIAN9_NOT_VALIDATED", "reason": gate["reason"]}), flush=True)
    return 3


def run_neural(args, cfg) -> int:
    import torch
    from torch.utils.data import DataLoader

    from combustion_recon.data.sparse_eval_masks import load_nested_sites
    from combustion_recon.evaluation.month2_evaluator import denormalize_temperature
    from combustion_recon.models.glu_reconstructor import make_synthetic_query_grid
    from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset, collate_sparse_batch
    from combustion_recon.training.sparse_scientific_loop import load_sparse_scientific_checkpoint

    s18 = _load_s18()
    s17 = s18._load_script("s17_median9", "17_s6_sparse_eval_and_swap.py")
    spec = cfg["neural_checkpoints"][args.run_id]
    run_dir = Path(args.runs_root) / args.run_id
    ck_path = run_dir / "checkpoints" / "best.pt"
    ck_sha = _require_sha(ck_path, spec["best_sha256"], "best.pt")
    identity = json.loads((run_dir / "run_identity.json").read_text())
    if identity.get("git_sha") != cfg["inputs"]["train_git_sha"]:
        raise SystemExit(f"run git_sha {identity.get('git_sha')} != frozen train sha")
    cache_root, mask_root = Path(args.cache_root), Path(args.mask_root)
    stats_sha = _require_sha(Path(args.stats), cfg["inputs"]["stats_sha256"], "stats")
    mask_info = _verify_masks(mask_root, cfg["inputs"]["mask_manifest_sha256"])
    cache_info = _verify_cache(
        cache_root, cfg["inputs"]["cache_aggregate_sha256"],
        (identity.get("cache_payload_sha256") or {}).get("files") or {},
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("neural median9 requires CUDA")
    out_run = Path(args.out_root) / args.run_id
    out_run.mkdir(parents=True, exist_ok=False)
    prec = s18._setup_fp32(torch)
    normalizer = Month2FitOnlyNormalizer(Path(args.stats))
    sites9 = t9_site_indices()
    model, _ = s17._load_model(spec["family"], cache_root, device)
    payload = load_sparse_scientific_checkpoint(ck_path, model=model, require_resumable=True, restore_rng=False)
    model.to(device).eval()
    p_on = bool(spec["p_conditioning"])
    _write_json(out_run / "provenance.json", {
        "script": "scripts/24_astra_median9_full_eval.py",
        "mode": "neural",
        "experiment_id": cfg["experiment_id"],
        "astra_git_sha": args.git_sha,
        "config_sha256": sha256_file(Path(args.config)),
        "run_id": args.run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "job_id_s5": spec["job_id"],
        "best_pt": str(ck_path),
        "best_pt_sha256": ck_sha,
        "checkpoint_epoch": int((payload.get("epoch_state") or {}).get("epoch", -1)),
        "train_git_sha": identity.get("git_sha"),
        "stats_sha256": stats_sha,
        "t9_sites": sites9.tolist(),
        "tau": 1.0,
        "treatments": list(LEVEL_TREATMENTS),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "fields_saved": False,
        **mask_info,
        **cache_info,
        **prec,
    })
    rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    for role in ROLES:
        ds = SparseUNetCacheDataset(
            role,  # type: ignore[arg-type]
            cache_root=cache_root,
            mask_mode="evaluation",
            site_fraction=COV,
            eval_mask_seed=int(cfg["evaluation"]["eval_mask_seed"]),
            layout_id=cfg["evaluation"]["layout_id"],
            p_conditioning=p_on,
            normalizer=normalizer,
            eval_mask_payload_root=mask_root,
        )
        if len(ds) != cfg["evaluation"]["expected_frames"][role]:
            raise SystemExit(f"{role}: {len(ds)} frames")
        query = make_synthetic_query_grid(ds.x, ds.y).to(device) if spec["family"] == "adapted_glu" else None
        for rep in REPLICAS:
            ds.replica_id, ds.site_fraction, ds.mask_mode = rep, COV, "evaluation"
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                collate_fn=collate_sparse_batch,
            )
            t0 = time.perf_counter()
            for batch in loader:
                if spec["family"] == "adapted_glu":
                    s17._attach_p(batch, ds)
                with torch.no_grad():
                    pred_n = s17._forward(
                        model, spec["family"], batch, p_conditioning=p_on,
                        query_coords_phys=query, device=device,
                    )
                true_K = batch["target_T_K"].numpy()
                for i, meta in enumerate(batch["metadata"]):
                    loaded = load_nested_sites(
                        role=role, sim_id=str(meta["sim_id"]),
                        frame_index=int(meta["frame_index"]), replica_id=rep,
                        payload_root=mask_root,
                    )
                    vsites = np.sort(np.asarray(loaded["sites_5pct"], dtype=np.int64))
                    if str(loaded["hash_5pct"]) != str(meta["mask_hash"]):
                        raise SystemExit(
                            f"mask hash mismatch {role}/{meta['sim_id']}/{meta['frame_index']}/r{rep}"
                        )
                    if mask_hash(sites_to_mask(vsites)) != str(meta["mask_hash"]):
                        raise SystemExit(f"mask payload does not match consumed sites {meta['mask_hash']}")
                    pred_K = np.asarray(denormalize_temperature(pred_n[i, 0], normalizer), np.float64)
                    tru = np.asarray(true_K[i, 0], np.float64)
                    identity_row = {
                        "method": args.run_id,
                        "run_id": args.run_id,
                        "family": spec["family"],
                        "p_conditioning": p_on,
                        "role": role,
                        "replica_id": rep,
                        "sim_id": str(meta["sim_id"]),
                        "frame_index": int(meta["frame_index"]),
                        "cache_index": int(meta["cache_index"]),
                        "mask_hash": str(meta["mask_hash"]),
                        "n_velocity_sites": int(vsites.size),
                        "n_t9_on_velocity_sites": int(np.intersect1d(vsites, sites9).size),
                    }
                    metric_rows, resid = prediction_tables(pred_K, tru, sites9, identity_row)
                    rows.extend(metric_rows)
                    residual_rows.extend(resid)
                    del pred_K, tru
            timing.append({"role": role, "replica_id": rep, "wall_s": time.perf_counter() - t0})
            print(f"[{args.run_id}] {role} rep{rep} done {timing[-1]['wall_s']:.1f}s", flush=True)
    pd.DataFrame(timing).to_csv(out_run / "timing.csv", index=False)
    primary, files, missing = _load_primary(Path(args.primary_root), args.run_id)
    if primary is not None and "run_id" in primary.columns:
        primary = primary[primary["run_id"].astype(str) == args.run_id]
    return _finish(
        out_run, pd.DataFrame(rows), pd.DataFrame(residual_rows),
        primary=primary, primary_files=files, missing_primary=missing,
        methods=[args.run_id], expected=cfg["evaluation"]["expected_frames"],
        atol=float(cfg["replay_gate"]["macro_rmse_atol_K"]), t_only_method=None,
    )


def run_baselines(args, cfg) -> int:
    s18 = _load_s18()
    s15 = s18._load_script("s15_median9", "15_gappypod_nested_coverage_eval.py")
    from combustion_recon.models.month2_baseline_adapters import predict_gappy

    cache_root, mask_root = Path(args.cache_root), Path(args.mask_root)
    art = Path(args.gappy_artifact)
    art_sha = _require_sha(art, cfg["inputs"]["gappypod_artifact_sha256"], "gappy artifact")
    stats_sha = _require_sha(Path(args.stats), cfg["inputs"]["stats_sha256"], "stats")
    mask_info = _verify_masks(mask_root, cfg["inputs"]["mask_manifest_sha256"])
    cache_info = _verify_cache(cache_root, cfg["inputs"]["cache_aggregate_sha256"], None)
    out = Path(args.out_root) / "baselines"
    out.mkdir(parents=True, exist_ok=False)
    pod_off, pod_on, param_mean, meta = s18._load_gappy_artifact(art, s15)
    normalizer = Month2FitOnlyNormalizer(Path(args.stats))
    sites9 = t9_site_indices()
    t_src = np.load(cache_root / "train_fit" / "target_T_K.npy", mmap_mode="r")
    acc = np.zeros(t_src.shape[1:], dtype=np.float64)
    for start in range(0, t_src.shape[0], 64):
        acc += np.asarray(t_src[start : start + 64], dtype=np.float64).sum(axis=0)
    global_mean_field = acc / t_src.shape[0]
    del t_src, acc
    _write_json(out / "provenance.json", {
        "script": "scripts/24_astra_median9_full_eval.py",
        "mode": "baselines",
        "experiment_id": cfg["experiment_id"],
        "astra_git_sha": args.git_sha,
        "config_sha256": sha256_file(Path(args.config)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gappy_artifact": str(art),
        "gappy_artifact_sha256": art_sha,
        "gappy_meta": {k: meta[k] for k in ("format_id", "experiment_id", "git_sha", "R", "frozen_lambdas")},
        "stats_sha256": stats_sha,
        "t9_sites": sites9.tolist(),
        "tau": 1.0,
        "treatments": list(LEVEL_TREATMENTS),
        "t_only": "historical_t_only_const",
        "refit": False,
        "fields_saved": False,
        **mask_info,
        **cache_info,
    })
    rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for role in ROLES:
        n_meta = len(json.loads((cache_root / role / "frames.json").read_text()))
        if n_meta != cfg["evaluation"]["expected_frames"][role]:
            raise SystemExit(f"{role}: {n_meta} frames")
        masks = s15._load_mask_index(mask_root, role)
        for fr in _iter_role_frames(cache_root, role, normalizer):
            tru = fr["T"]
            for rep in REPLICAS:
                vsites, mhash = _mask_for(masks, (fr["sim_id"], int(fr["frame"]), rep))
                base = {
                    "role": role,
                    "replica_id": rep,
                    "sim_id": fr["sim_id"],
                    "frame_index": int(fr["frame"]),
                    "cache_index": int(fr["cache_index"]),
                    "mask_hash": mhash,
                    "n_velocity_sites": int(vsites.size),
                    "n_t9_on_velocity_sites": int(np.intersect1d(vsites, sites9).size),
                }
                predictions: list[tuple[str, bool, np.ndarray]] = []
                for name, pod, p_on in (("GappyPOD_p_off", pod_off, False), ("GappyPOD_p_on", pod_on, True)):
                    mi = s15._model_inputs_from_sites(fr, vsites, p_conditioning=p_on)
                    predictions.append((name, p_on, np.asarray(predict_gappy(pod, mi, p_conditioning=p_on), np.float64)))
                predictions.append(("ParamMean_p_on", True, np.asarray(param_mean.predict(fr["nh3"], fr["phi"]), np.float64)))
                predictions.append(("GlobalMean", False, global_mean_field))
                for name, p_on, pred in predictions:
                    identity_row = {**base, "method": name, "p_conditioning": p_on}
                    metric_rows, resid = prediction_tables(pred, tru, sites9, identity_row)
                    rows.extend(metric_rows)
                    residual_rows.extend(resid)
                rows.append(t_only_constant_row(
                    tru, sites9, {**base, "method": "T_only_const", "p_conditioning": False}
                ))
            del tru
        del masks
        print(f"[baselines] {role} done", flush=True)
    primary, files, missing = _load_primary(Path(args.primary_root), "baselines")
    return _finish(
        out, pd.DataFrame(rows), pd.DataFrame(residual_rows),
        primary=primary, primary_files=files, missing_primary=missing,
        methods=list(cfg["baselines_matched"]["methods"]),
        expected=cfg["evaluation"]["expected_frames"],
        atol=float(cfg["replay_gate"]["macro_rmse_atol_K"]),
        t_only_method=cfg["baselines_matched"]["t_only_method"],
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("neural", "baselines"), required=True)
    p.add_argument("--run-id", default=None)
    p.add_argument("--config", type=Path, default=ROOT / "configs/experiments/month2_astra_median9_second_pass.yaml")
    p.add_argument("--runs-root", type=Path, default=None)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--primary-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--mask-root", type=Path, required=True)
    p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--gappy-artifact", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--git-sha", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg.get("experiment_id") != CONFIG_ID:
        raise SystemExit(f"refusing config {cfg.get('experiment_id')}")
    if args.mode == "neural":
        if not args.run_id or args.run_id not in cfg["neural_checkpoints"] or not args.runs_root:
            raise SystemExit("--run-id and --runs-root required")
        return run_neural(args, cfg)
    if not args.gappy_artifact:
        raise SystemExit("--gappy-artifact required")
    return run_baselines(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
