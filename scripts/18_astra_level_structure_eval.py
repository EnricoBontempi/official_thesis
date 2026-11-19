#!/usr/bin/env python3
"""Astra A2 — free / oracle-offset / AO9 on frozen predictions. No training.

``--mode neural --run-id <S5 run>``: one forward per frame/replica on the frozen best.pt,
then free/oracle/AO9 computed from that same prediction.
``--mode baselines``: GappyPOD p_off/p_on, ParamMean (p_on), GlobalMean, T-only constant.

Reuses scripts/17 (model loading / forward) and scripts/15 (Gappy loaders) by import.
Output directory must not exist (non-overwritable).
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
from combustion_recon.evaluation.astra_level import (  # noqa: E402
    AO_TAU,
    anchor_mask_from_sites,
    frame_metrics,
    t9_site_indices,
    treat_frame,
)

ROLES = ("seen_train_eval", "val_eval")
REPLICAS = (0, 1, 2)
COV = 0.05


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def _setup_fp32(torch) -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "precision": "fp32_no_amp_no_tf32_no_compile",
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch": torch.__version__,
    }


def _macro_rmse(df: pd.DataFrame, col: str = "all_rmse_K") -> float:
    """frame -> condition -> mean over conditions (one replica)."""
    return float(df.groupby("sim_id")[col].mean().mean())


def _replay(rows: pd.DataFrame, ref: dict[tuple[str, int], float], atol: float) -> list[dict[str, Any]]:
    out = []
    free = rows[rows["treatment"] == "free"]
    for (role, rid), g in free.groupby(["role", "replica_id"]):
        mine = _macro_rmse(g)
        r = ref.get((str(role), int(rid)))
        out.append(
            {
                "role": role,
                "replica_id": int(rid),
                "astra_free_macro_rmse_K": mine,
                "reference_macro_rmse_K": r,
                "delta_K": None if r is None else mine - r,
                "ok": None if r is None else bool(abs(mine - r) <= atol),
            }
        )
    return out


# ---------------------------------------------------------------- neural ---------------
def run_neural(args, cfg) -> int:
    import torch
    from torch.utils.data import DataLoader

    s17 = _load_script("s17", "17_s6_sparse_eval_and_swap.py")
    from combustion_recon.evaluation.month2_evaluator import denormalize_temperature
    from combustion_recon.models.glu_reconstructor import make_synthetic_query_grid
    from combustion_recon.data.sparse_eval_masks import load_nested_sites
    from combustion_recon.training.dense_cache import (
        compute_cache_payload_hashes,
        validate_dense_cache_manifest,
    )
    from combustion_recon.training.sparse_dataset import (
        SparseUNetCacheDataset,
        collate_sparse_batch,
    )
    from combustion_recon.training.sparse_scientific_loop import load_sparse_scientific_checkpoint

    rid_name = args.run_id
    spec = cfg["neural_checkpoints"][rid_name]
    run_dir = Path(args.runs_root) / rid_name
    ck_path = run_dir / "checkpoints" / "best.pt"
    ck_sha = sha256_file(ck_path)
    if ck_sha != spec["best_sha256"]:
        raise SystemExit(f"best.pt hash mismatch {ck_sha} != {spec['best_sha256']}")
    identity = json.loads((run_dir / "run_identity.json").read_text())
    if identity.get("git_sha") != cfg["train_git_sha"]:
        raise SystemExit(f"run git_sha {identity.get('git_sha')} != {cfg['train_git_sha']}")

    cache_root, mask_root = Path(args.cache_root), Path(args.mask_root)
    manifest = json.loads((cache_root / "manifest.json").read_text())
    validate_dense_cache_manifest(manifest, cache_root=cache_root, require_roles=("train_fit",) + ROLES)
    payload_hashes = compute_cache_payload_hashes(cache_root)
    exp_files = (identity.get("cache_payload_sha256") or {}).get("files") or {}
    for rel, got in (payload_hashes.get("files") or {}).items():
        if rel in exp_files and exp_files[rel] != got:
            raise SystemExit(f"cache file hash mismatch {rel}")
    mask_manifest = json.loads((mask_root / "manifest.json").read_text())

    out_run = Path(args.out_root) / rid_name
    out_run.mkdir(parents=True, exist_ok=False)  # non-overwritable
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    prec = _setup_fp32(torch)

    normalizer = Month2FitOnlyNormalizer(Path(args.stats))
    sigma = float(normalizer.temperature_std)
    sites9 = t9_site_indices()
    model, _ = s17._load_model(spec["family"], cache_root, device)
    payload = load_sparse_scientific_checkpoint(
        ck_path, model=model, require_resumable=True, restore_rng=False
    )
    model.to(device).eval()
    ck_epoch = int((payload.get("epoch_state") or {}).get("epoch", -1))
    p_on = bool(spec["p_conditioning"])

    _write_json(
        out_run / "provenance.json",
        {
            "script": "scripts/18_astra_level_structure_eval.py",
            "astra_git_sha": args.git_sha,
            "config_sha256": sha256_file(Path(args.config)),
            "run_id": rid_name,
            "job_id_s5": spec["job_id"],
            "best_pt": str(ck_path),
            "best_pt_sha256": ck_sha,
            "checkpoint_epoch": ck_epoch,
            "train_git_sha": identity.get("git_sha"),
            "cache_aggregate_sha256": payload_hashes.get("aggregate_sha256"),
            "mask_manifest_sha256": sha256_file(mask_root / "manifest.json"),
            "mask_aggregate_payload_sha256": mask_manifest.get("aggregate_payload_sha256"),
            "stats_sha256": sha256_file(Path(args.stats)),
            "t9_sites": sites9.tolist(),
            "tau": AO_TAU,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            **prec,
        },
    )

    rows: list[dict[str, Any]] = []
    saved_fields: dict[str, np.ndarray] = {}
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
        if args.limit_frames:  # smoke only; never used for scientific output
            ds.indices = ds.indices[: int(args.limit_frames)]
        q = None
        if spec["family"] == "adapted_glu":
            q = make_synthetic_query_grid(ds.x, ds.y).to(device)
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
                        query_coords_phys=q, device=device,
                    )
                true_K = batch["target_T_K"].numpy()
                for i, meta in enumerate(batch["metadata"]):
                    # persisted masks must be the ones consumed (no silent seed fallback)
                    loaded = load_nested_sites(
                        role=role, sim_id=str(meta["sim_id"]),
                        frame_index=int(meta["frame_index"]), replica_id=rep,
                        payload_root=mask_root,
                    )
                    vsites = np.sort(np.asarray(loaded["sites_5pct"], dtype=np.int64))
                    if str(loaded["hash_5pct"]) != str(meta["mask_hash"]):
                        raise SystemExit(f"mask hash mismatch {role}/{meta['sim_id']}/{meta['frame_index']}/r{rep}")
                    pred_K = np.asarray(denormalize_temperature(pred_n[i, 0], normalizer), np.float64)
                    tru = np.asarray(true_K[i, 0], np.float64)
                    base = {
                        "run_id": rid_name, "family": spec["family"], "p_conditioning": p_on,
                        "role": role, "replica_id": rep, "sim_id": str(meta["sim_id"]),
                        "frame_index": int(meta["frame_index"]), "cache_index": int(meta["cache_index"]),
                        "mask_hash": str(meta["mask_hash"]), "n_velocity_sites": int(vsites.size),
                        "n_t9_on_velocity_sites": int(np.intersect1d(vsites, sites9).size),
                    }
                    for r in treat_frame(pred_K, tru, sites9, tau=AO_TAU):
                        rows.append({**base, **r})
                    if role == "val_eval" and rep == 0 and int(meta["frame_index"]) == 1000:
                        saved_fields[f"{meta['sim_id']}__pred_K"] = pred_K.astype(np.float32)
                        saved_fields[f"{meta['sim_id']}__true_K"] = tru.astype(np.float32)
            timing.append({"role": role, "replica_id": rep, "wall_s": time.perf_counter() - t0})
            print(f"[{rid_name}] {role} rep{rep} done {timing[-1]['wall_s']:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    for role in ROLES:
        df[df.role == role].to_csv(out_run / f"per_frame_{role}.csv.gz", index=False, compression="gzip")
    if saved_fields:
        np.savez_compressed(out_run / "val_frame1000_rep0_fields.npz", t9_sites=sites9, **saved_fields)
    pd.DataFrame(timing).to_csv(out_run / "timing.csv", index=False)

    # replay: free K=0 vs S6 per-replica best@5% mean-condition RMSE
    ref: dict[tuple[str, int], float] = {}
    ref_path = Path(args.s6_ref_root) / rid_name / "summary_all_settings.csv"
    if ref_path.exists():
        s6 = pd.read_csv(ref_path)
        s6 = s6[
            (s6.checkpoint == "best") & np.isclose(s6.site_fraction.astype(float), COV)
            & (s6.replica_id.astype(str).isin(["0", "1", "2"]))
        ]
        for _, r in s6.iterrows():
            ref[(str(r["role"]), int(r["replica_id"]))] = float(r["mean_condition_rmse_K"])
    rp = _replay(df, ref, float(cfg["replay_gate"]["free_k0_macro_rmse_atol_K"]))
    _write_json(
        out_run / "replay_k0.json",
        {"reference_file": str(ref_path), "reference_sha256": sha256_file(ref_path) if ref_path.exists() else None,
         "atol_K": cfg["replay_gate"]["free_k0_macro_rmse_atol_K"], "checks": rp,
         "all_ok": bool(rp and all(c["ok"] for c in rp))},
    )
    _write_json(out_run / "COMPLETE.json", {"rows": len(df), "run_id": rid_name})
    print(json.dumps({"status": "astra_neural_complete", "run": rid_name, "rows": len(df)}), flush=True)
    return 0


# ---------------------------------------------------------------- baselines -------------
def _load_gappy_artifact(path: Path, s15):
    from combustion_recon.models.baselines import ParamConditionedMeanPredictor

    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    pm = meta["param_mean_meta"]
    param_mean = ParamConditionedMeanPredictor(
        coef=np.asarray(z["param_mean_coef"]), H=int(pm["H"]), W=int(pm["W"]),
        ridge_alpha=float(pm["ridge_alpha"]), basis=str(pm["basis"]),
        selected_basis=str(pm["basis"]), selected_ridge_alpha=float(pm["ridge_alpha"]),
        loco_rmse_K=float(pm["loco_rmse_K"]),
    )
    keys = ["mean_vi", "mean_vj", "mean_vk", "mean_T", "modes", "modal_variances", "n_pix", "R",
            "ridge_lambda", "w_T", "scale_vi", "scale_vj", "scale_vk", "scale_T", "use_param_mean",
            "n_velocity_components", "global_mean_T", "eps_var"]

    def extract(prefix: str) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k in keys:
            name = f"{prefix}_{k}"
            if name not in z.files:
                continue
            arr = z[name]
            d[k] = arr.item() if arr.shape == () else np.asarray(arr)
        return d

    return (
        s15._pod_from_npz(extract("off"), None),
        s15._pod_from_npz(extract("on"), param_mean),
        param_mean,
        meta,
    )


def run_baselines(args, cfg) -> int:
    s15 = _load_script("s15", "15_gappypod_nested_coverage_eval.py")
    from combustion_recon.models.baselines import GlobalTrainMeanPredictor
    from combustion_recon.models.month2_baseline_adapters import predict_gappy

    out = Path(args.out_root) / "baselines"
    out.mkdir(parents=True, exist_ok=False)
    cache_root, mask_root = Path(args.cache_root), Path(args.mask_root)
    art = Path(args.gappy_artifact)
    art_sha = sha256_file(art)
    if art_sha != cfg["baselines_matched"]["gappypod_artifact_sha256"]:
        raise SystemExit(f"gappy artifact hash mismatch {art_sha}")
    pod_off, pod_on, param_mean, meta = _load_gappy_artifact(art, s15)
    normalizer = Month2FitOnlyNormalizer(Path(args.stats))
    sites9 = t9_site_indices()
    am9 = anchor_mask_from_sites(sites9)

    # GlobalMean: fit-only train_fit mean field (chunked, never all fields in RAM)
    t_src = np.load(cache_root / "train_fit" / "target_T_K.npy", mmap_mode="r")
    acc = np.zeros(t_src.shape[1:], dtype=np.float64)
    for s in range(0, t_src.shape[0], 64):
        acc += np.asarray(t_src[s : s + 64], dtype=np.float64).sum(axis=0)
    global_mean_field = acc / t_src.shape[0]
    _ = GlobalTrainMeanPredictor  # object type used historically; mean is identical by definition

    _write_json(out / "provenance.json", {
        "script": "scripts/18_astra_level_structure_eval.py", "mode": "baselines",
        "astra_git_sha": args.git_sha, "config_sha256": sha256_file(Path(args.config)),
        "gappy_artifact": str(art), "gappy_artifact_sha256": art_sha,
        "gappy_meta": {k: meta[k] for k in ("format_id", "experiment_id", "git_sha", "R", "frozen_lambdas")},
        "mask_manifest_sha256": sha256_file(mask_root / "manifest.json"),
        "stats_sha256": sha256_file(Path(args.stats)),
        "hist_summary_sha256": sha256_file(Path(args.hist_summary)),
        "t9_sites": sites9.tolist(), "tau": AO_TAU, "refit": False,
    })

    rows: list[dict[str, Any]] = []
    for role in ROLES:
        frames = s15._load_role_physical(cache_root, role, normalizer)
        masks = s15._load_mask_index(mask_root, role)
        if len(frames) != cfg["evaluation"]["expected_frames"][role]:
            raise SystemExit(f"{role}: {len(frames)} frames")
        if args.limit_frames:  # smoke only
            frames = frames[: int(args.limit_frames)]
        for rep in REPLICAS:
            for fr in frames:
                key = (fr["sim_id"], int(fr["frame"]), rep)
                vsites = np.sort(np.asarray(s15._mask_sites(masks, key, "sites_5pct"), dtype=np.int64))
                base = {
                    "role": role, "replica_id": rep, "sim_id": fr["sim_id"],
                    "frame_index": int(fr["frame"]), "cache_index": int(fr["cache_index"]),
                    "n_velocity_sites": int(vsites.size),
                    "n_t9_on_velocity_sites": int(np.intersect1d(vsites, sites9).size),
                }
                tru = np.asarray(fr["T"], np.float64)
                methods: list[tuple[str, bool, np.ndarray]] = []
                for name, pod, p_on in (("GappyPOD_p_off", pod_off, False), ("GappyPOD_p_on", pod_on, True)):
                    mi = s15._model_inputs_from_sites(fr, vsites, p_conditioning=p_on)
                    methods.append((name, p_on, np.asarray(predict_gappy(pod, mi, p_conditioning=p_on), np.float64)))
                methods.append(("ParamMean_p_on", True, np.asarray(param_mean.predict(fr["nh3"], fr["phi"]), np.float64)))
                methods.append(("GlobalMean", False, global_mean_field))
                for name, p_on, pred in methods:
                    for r in treat_frame(pred, tru, sites9, tau=AO_TAU):
                        rows.append({**base, "method": name, "p_conditioning": p_on, **r})
                # T-only constant: only the nine observed T values enter the field
                const = np.full_like(tru, float(np.mean(tru.ravel()[sites9])))
                r = {"treatment": "t_only_const", "delta_applied_K": float("nan")}
                r.update(frame_metrics(const, tru, am9))
                rows.append({**base, "method": "T_only_const", "p_conditioning": False, **r})
        del frames, masks

    df = pd.DataFrame(rows)
    for role in ROLES:
        df[df.role == role].to_csv(out / f"per_frame_{role}.csv.gz", index=False, compression="gzip")

    # replay vs historical K=0 5% baseline summary
    hist = pd.read_csv(args.hist_summary)
    hist = hist[(hist.K == 0) & np.isclose(hist.site_fraction, COV)]
    hmap = {
        "GappyPOD_p_off": ("GappyPOD_free", False), "GappyPOD_p_on": ("GappyPOD_free", True),
        "ParamMean_p_on": ("ParamConditionedMean", True), "GlobalMean": ("GlobalTrainMean", False),
    }
    checks = []
    free = df[df.treatment == "free"]
    for m, (hm, hp) in hmap.items():
        for role in ROLES:
            g = free[(free.method == m) & (free.role == role)]
            mine = float(np.mean([_macro_rmse(gr) for _, gr in g.groupby("replica_id")]))
            h = hist[(hist.method == hm) & (hist.role == role) & (hist.p_conditioning == hp)]
            ref = float(h["rmse_K"].iloc[0]) if len(h) == 1 else None
            checks.append({"method": m, "role": role, "astra_macro_rmse_K": mine, "historical_rmse_K": ref,
                           "delta_K": None if ref is None else mine - ref,
                           "ok": None if ref is None else bool(abs(mine - ref) <= cfg["replay_gate"]["free_k0_macro_rmse_atol_K"])})
    _write_json(out / "replay_k0.json", {"atol_K": cfg["replay_gate"]["free_k0_macro_rmse_atol_K"],
                                         "checks": checks, "all_ok": bool(all(c["ok"] for c in checks))})
    _write_json(out / "COMPLETE.json", {"rows": len(df)})
    print(json.dumps({"status": "astra_baselines_complete", "rows": len(df)}), flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("neural", "baselines"), required=True)
    p.add_argument("--run-id", default=None)
    p.add_argument("--config", type=Path, default=ROOT / "configs/experiments/month2_astra_level_structure.yaml")
    p.add_argument("--runs-root", type=Path, default=None)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--mask-root", type=Path, required=True)
    p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--s6-ref-root", type=Path, default=None)
    p.add_argument("--gappy-artifact", type=Path, default=None)
    p.add_argument("--hist-summary", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit-frames", type=int, default=0, help="smoke test only")
    p.add_argument("--git-sha", required=True, help="SHA of the committed tip actually running")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.mode == "neural":
        if not args.run_id or args.run_id not in cfg["neural_checkpoints"] or not args.runs_root:
            raise SystemExit("--run-id (frozen S5 id) and --runs-root required")
        return run_neural(args, cfg)
    if not (args.gappy_artifact and args.hist_summary):
        raise SystemExit("--gappy-artifact and --hist-summary required")
    return run_baselines(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
