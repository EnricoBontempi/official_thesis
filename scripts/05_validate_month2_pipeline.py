#!/usr/bin/env python3
"""Step 2.2 — Build Month-2 manifests/stats and validate the data pipeline on CPU.

Usage:
    python scripts/05_validate_month2_pipeline.py --write-artifacts
    python scripts/05_validate_month2_pipeline.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from torch.utils.data import DataLoader

from combustion_recon.data.collate import collate_month2_batch
from combustion_recon.data.dataset import load_splits
from combustion_recon.data.freeze import FreezeGateError
from combustion_recon.data.month2_dataset import InputIsolationProbe, Month2ReconDataset
from combustion_recon.data.month2_manifests import (
    build_frames_manifest,
    build_observations_manifest,
    write_json,
)
from combustion_recon.data.month2_normalizer import (
    Month2FitOnlyNormalizer,
    compute_month2_fit_stats,
)
from combustion_recon.data.month2_protocol import (
    assert_fit_seen_disjoint,
    frames_for_role,
    load_frame_protocol,
    sha256_file,
)
from combustion_recon.data.shard_access import ShardStore
from combustion_recon.evaluation.month2_evaluator import (
    aggregate_frame_then_condition_then_split,
    denormalize_temperature,
    evaluate_frame_prediction,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
RAW = ROOT / "data" / "raw" / "combustion" / "hf_dataset" / "numerical"
FRAME_PROTOCOL = ROOT / "configs" / "splits" / "month2_frame_protocol.yaml"
PILOT = ROOT / "configs" / "experiments" / "month2_pilot.yaml"
SPLITS = ROOT / "configs" / "splits" / "splits_interp_extrap.yaml"
STATS_OUT = MANIFESTS / "train_norm_stats_month2_fit.json"
FRAMES_OUT = MANIFESTS / "month2_frames.json"
OBS_OUT = MANIFESTS / "month2_observations.json"
REPORT_OUT = ROOT / "docs" / "month2" / "01_pipeline_validation.md"
MONTH1_STATS = MANIFESTS / "train_norm_stats.json"


def rss_mb() -> float:
    # Linux: ru_maxrss is kilobytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def write_stats() -> dict[str, Any]:
    protocol = load_frame_protocol(FRAME_PROTOCOL)
    assert_fit_seen_disjoint(protocol)
    splits = load_splits(SPLITS)
    train_ids = list(splits["train"])
    frames = frames_for_role(protocol, "train_fit")
    store = ShardStore(raw_dir=RAW, max_cached_frames=64)
    t0 = time.time()
    rss0 = rss_mb()
    stats = compute_month2_fit_stats(
        sim_ids=train_ids,
        frame_indices=frames,
        store=store,
        protocol_path=FRAME_PROTOCOL,
        splits_path=SPLITS,
        provenance_mode="production",
    )
    stats["meta"]["wall_time_s"] = time.time() - t0
    stats["meta"]["peak_rss_mb_during_fit_approx"] = rss_mb()
    stats["meta"]["rss_mb_before_fit"] = rss0
    stats["meta"]["host"] = {
        "cwd": str(ROOT),
        "raw_dir": str(RAW),
        "raw_exists": RAW.exists(),
        "n_shards": len(list(RAW.glob("data-*-of-00030.arrow"))) if RAW.exists() else 0,
        "engaging_verified": False,
        "note": "Local WSL/host used for pipeline validation; Engaging GPU/alloc not verified.",
    }
    write_json(STATS_OUT, stats)
    # Safety: never overwrite Month-1 stats
    assert MONTH1_STATS.exists()
    return stats


def write_manifests() -> tuple[dict[str, Any], dict[str, Any]]:
    frames = build_frames_manifest(protocol_path=FRAME_PROTOCOL, pilot_path=PILOT)
    obs = build_observations_manifest(protocol_path=FRAME_PROTOCOL, pilot_path=PILOT)
    write_json(FRAMES_OUT, frames)
    write_json(OBS_OUT, obs)
    return frames, obs


def independent_stats_check(stats: dict[str, Any], n_conditions: int = 2, n_frames: int = 3) -> dict[str, Any]:
    """Recompute mean/std on a tiny subset and compare to streaming stats on same subset."""
    sim_ids = list(stats["meta"]["sim_ids"][:n_conditions])
    frames = list(stats["meta"]["frame_indices"][:n_frames])
    store = ShardStore(raw_dir=RAW, max_cached_frames=128)
    vals_T = []
    for sid in sim_ids:
        for fi in frames:
            fr = store.get_frame(sid, fi)
            vals_T.append(fr.temperature.reshape(-1).astype(np.float64))
    arr = np.concatenate(vals_T)
    subset = compute_month2_fit_stats(
        sim_ids=sim_ids,
        frame_indices=frames,
        store=ShardStore(raw_dir=RAW, max_cached_frames=128),
        provenance_mode="debug_subset",  # explicit non-canonical frame subset for I/O check only
    )
    return {
        "subset_sim_ids": sim_ids,
        "subset_frames": frames,
        "direct_mean": float(arr.mean()),
        "direct_std_ddof1": float(arr.std(ddof=1)),
        "stream_mean": float(subset["fields"]["Temperature"]["mean"]),
        "stream_std": float(subset["fields"]["Temperature"]["std"]),
        "mean_abs_diff": abs(float(arr.mean()) - float(subset["fields"]["Temperature"]["mean"])),
        "std_abs_diff": abs(float(arr.std(ddof=1)) - float(subset["fields"]["Temperature"]["std"])),
    }


def validate_real_batch(norm: Month2FitOnlyNormalizer) -> dict[str, Any]:
    store = ShardStore(raw_dir=RAW, max_cached_frames=256)
    ds = Month2ReconDataset(
        "train_fit",
        task_mode="temperature_free",
        K=0,
        p_conditioning=False,
        normalizer=norm,
        store=store,
        max_cached_frames=256,
    )
    # Compare one sample against raw shard decode
    sample = ds[0]
    ident = ds.get_identity(0)
    raw = store.get_frame(ident["sim_id"], ident["frame_index"])
    T = sample["targets"]["temperature"]
    coord_ok = np.allclose(sample["metadata"]["x"], raw.x) and np.allclose(
        sample["metadata"]["y"], raw.y
    )
    t_ok = np.allclose(T, raw.temperature.astype(np.float64))
    # Observed velocity values match raw at mask sites
    mask = sample["model_inputs"]["mask_velocity"]
    vi = sample["metadata"]["_hidden_full_velocity_i"]
    sites = sample["model_inputs"]["velocity_site_index"]
    H, W = T.shape
    ys, xs = sites // W, sites % W
    vel_ok = np.allclose(
        sample["model_inputs"]["velocity_values_physical"][:, 0], vi[ys, xs]
    ) and np.allclose(
        sample["model_inputs"]["velocity_values_physical"][:, 1],
        sample["metadata"]["_hidden_full_velocity_j"][ys, xs],
    )

    # DataLoader with workers
    rss_before = rss_mb()
    t0 = time.time()
    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda xs: collate_month2_batch(xs, as_torch=True),
    )
    batch = next(iter(loader))
    elapsed = time.time() - t0
    rss_after = rss_mb()

    # Evaluator smoke with a trivial "prediction" = domain mean
    pred = np.full_like(T, float(norm.temperature_mean))
    row = evaluate_frame_prediction(
        pred,
        T,
        train_sigma_T=norm.temperature_std,
        sim_id=ident["sim_id"],
        split="train",
        role="train_fit",
        frame_index=ident["frame_index"],
        mask_temperature=sample["model_inputs"]["mask_temperature"],
    )
    agg = aggregate_frame_then_condition_then_split([row])

    # TEST access denied
    test_denied = False
    try:
        tds = Month2ReconDataset(
            "test_interp_eval",
            normalizer=norm,
            store=store,
            allow_test_payloads=False,
        )
        _ = tds[0]
    except FreezeGateError:
        test_denied = True

    # Denorm roundtrip
    Tn = sample["targets"]["temperature_normalized"]
    T_back = denormalize_temperature(Tn, norm)
    denorm_ok = np.allclose(T_back, T, atol=1e-4, rtol=1e-5)

    return {
        "sample0_sim_id": ident["sim_id"],
        "sample0_frame": ident["frame_index"],
        "n_train_fit": len(ds),
        "coord_match_raw": bool(coord_ok),
        "temperature_match_raw": bool(t_ok),
        "velocity_sites_match_raw": bool(vel_ok),
        "denorm_roundtrip_ok": bool(denorm_ok),
        "batch_keys": sorted(batch["model_inputs"].keys()),
        "batch_velocity_valid_shape": list(batch["model_inputs"]["velocity_valid"].shape),
        "dataloader_workers": 2,
        "dataloader_batch_wall_s": elapsed,
        "rss_mb_before_loader": rss_before,
        "rss_mb_after_loader": rss_after,
        "store_stats": dict(store.stats),
        "compact_cache_frames": store.cached_frame_count,
        "compact_cache_bytes_est": store.estimate_compact_cache_bytes(),
        "test_access_denied_without_freeze": test_denied,
        "evaluator_rmse_mean_predictor": row["rmse_K"],
        "selection_table_rows": int(len(agg["selection_mean_condition_rmse"])),
        "p_off_has_op_params": "op_params" in sample["model_inputs"],
        "vk_in_model_inputs": any(
            "velocity_k" in str(k).lower() for k in sample["model_inputs"]
        ),
    }


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Month 2 — Pipeline validation (Step 2.2)",
        "",
        f"**Date:** {payload['date']}",
        "**Host:** local workspace (WSL). Engaging GPU/allocation **not verified**.",
        f"**Commit base expected:** `{payload.get('base_commit', 'unknown')}`",
        "",
        "## Environment",
        f"- CPU count: {payload['env']['cpu_count']}",
        f"- RAM available (from /proc or free): {payload['env'].get('mem_available_gb', 'n/a')} GiB",
        f"- Disk free on workspace: {payload['env'].get('disk_free_gb', 'n/a')} GiB",
        f"- Raw shards: `{RAW}` exists={payload['env']['raw_exists']}, n={payload['env']['n_shards']}",
        f"- Peak RSS observed during this script (MB): {payload['env']['peak_rss_mb']:.1f}",
        "",
        "## Artifacts written",
    ]
    for p in payload["artifacts"]:
        lines.append(f"- `{p}`")
    lines += [
        "",
        "## Frame counts",
        "```json",
        json.dumps(payload["frame_counts"], indent=2),
        "```",
        "",
        "## Fit-only stats",
        f"- Path: `data/manifests/train_norm_stats_month2_fit.json`",
        f"- Conditions: {payload['stats_meta']['n_conditions']}",
        f"- Frames/condition: {payload['stats_meta']['n_frames_per_condition']}",
        f"- Temperature mean/std (K): {payload['stats_fields']['Temperature']['mean']:.6f} / {payload['stats_fields']['Temperature']['std']:.6f}",
        f"- Month-1 stats file preserved: {payload['month1_stats_preserved']}",
        f"- Independent subset mean/std abs diff: {payload['independent_check']['mean_abs_diff']:.3e} / {payload['independent_check']['std_abs_diff']:.3e}",
        "",
        "## Real-data batch check",
        "```json",
        json.dumps(payload["real_batch"], indent=2),
        "```",
        "",
        "## Memory / I/O notes",
        "- Arrow full-trajectory decode peaks near ~2.1 GiB per shard (float32 numerical blob).",
        "- Month-2 store keeps only compact T+vi+vj(+vk) frames in an LRU cache; full arrays are discarded after extract.",
        f"- Compact cache estimate after validation: {payload['real_batch']['compact_cache_bytes_est']} bytes "
        f"({payload['real_batch']['compact_cache_frames']} frames).",
        "- No persistent heavy conversion of the 59 GiB dataset was created.",
        "",
        "## Limits / not claimed",
        "- Engaging access/GPU allocation not verified in this step.",
        "- No baseline/POD/ParamMean fitting, no neural training.",
        "- U-Net/GLU forward isolation still to be repeated when models exist (probe only).",
        "- Pseudo-extrap folds not fitted; fold lock still open.",
        "",
        f"## Overall status: **{payload['status']}**",
        "",
    ]
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-artifacts", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    if not args.write_artifacts and not args.validate_only:
        args.write_artifacts = True

    env = {
        "cpu_count": os.cpu_count(),
        "raw_exists": RAW.exists(),
        "n_shards": len(list(RAW.glob("data-*-of-00030.arrow"))) if RAW.exists() else 0,
        "peak_rss_mb": rss_mb(),
    }
    try:
        import psutil  # type: ignore

        env["mem_available_gb"] = round(psutil.virtual_memory().available / (1024**3), 2)
        env["disk_free_gb"] = round(psutil.disk_usage(str(ROOT)).free / (1024**3), 2)
    except Exception:
        # fallback without psutil
        env["mem_available_gb"] = None
        env["disk_free_gb"] = None

    if not RAW.exists() or env["n_shards"] < 30:
        print("ERROR: real shards unavailable; refusing to claim full Step 2.2 validation.")
        return 2

    if args.write_artifacts:
        print("Writing fit-only stats (streaming TRAIN-fit frames)...")
        stats = write_stats()
        print("Writing manifests...")
        frames_m, obs_m = write_manifests()
    else:
        stats = json.loads(STATS_OUT.read_text())
        frames_m = json.loads(FRAMES_OUT.read_text())
        obs_m = json.loads(OBS_OUT.read_text())

    # Ensure Month-1 untouched
    month1_preserved = MONTH1_STATS.exists()
    month1_sha = sha256_file(MONTH1_STATS)

    print("Independent stats subset check...")
    indep = independent_stats_check(stats)
    print("Real batch / dataloader / TEST gate validation...")
    norm = Month2FitOnlyNormalizer(STATS_OUT)
    real = validate_real_batch(norm)
    env["peak_rss_mb"] = rss_mb()

    ok = (
        real["coord_match_raw"]
        and real["temperature_match_raw"]
        and real["velocity_sites_match_raw"]
        and real["denorm_roundtrip_ok"]
        and real["test_access_denied_without_freeze"]
        and (not real["p_off_has_op_params"])
        and (not real["vk_in_model_inputs"])
        and indep["mean_abs_diff"] < 1e-6
        and indep["std_abs_diff"] < 1e-6
        and frames_m["counts"]["train_fit"]["n_samples"] == 760
        and frames_m["counts"]["seen_train_eval"]["n_samples"] == 220
        and frames_m["counts"]["val_eval"]["n_samples"] == 123
        and all(v.get("nested_K_ok") for v in obs_m["verified_examples"])
    )
    payload = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "base_commit": "5bce7142d7dbaf9e5ea6bc1e3c8dffe75e066ce9",
        "env": env,
        "artifacts": [
            str(STATS_OUT.relative_to(ROOT)),
            str(FRAMES_OUT.relative_to(ROOT)),
            str(OBS_OUT.relative_to(ROOT)),
            str(REPORT_OUT.relative_to(ROOT)),
        ],
        "frame_counts": frames_m["counts"],
        "stats_meta": stats["meta"],
        "stats_fields": stats["fields"],
        "month1_stats_preserved": month1_preserved,
        "month1_stats_sha256": month1_sha,
        "independent_check": indep,
        "real_batch": real,
        "status": "PASS" if ok else "FAIL",
    }
    write_report(payload)
    summary_path = MANIFESTS / "month2_pipeline_validation_summary.json"
    write_json(summary_path, payload)
    print(json.dumps({"status": payload["status"], "summary": str(summary_path)}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
