#!/usr/bin/env python3
"""Dense U-Net reconstructor utilities (Month-2 Step 2.4).

Authorized in the first Cursor mission for:
  - cache build (U1)
  - tiny TRAIN overfit diagnostic (U2)

Scientific full training (U4), profiling (U3), and VAL-guided selection are
NOT authorized by default. Pass explicit flags; refuse TEST payloads.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.data.month2_protocol import load_yaml, sha256_file
from combustion_recon.models.baselines import ParamConditionedMeanPredictor
from combustion_recon.models.unet import build_dense_unet_from_config
from combustion_recon.training.dense_cache import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DENSE_CFG,
    DenseUNetCacheDataset,
    assert_dense_scientific_contract,
    build_dense_cache,
    subset_indices_for_overfit,
)
from combustion_recon.training.trainer import (
    DenseUNetTrainer,
    TrainConfig,
    load_checkpoint,
    make_loader,
    save_checkpoint,
    set_seed,
    write_json,
)


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def cmd_build_cache(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    roles = tuple(args.roles.split(",")) if args.roles else ("train_fit", "seen_train_eval", "val_eval")
    for r in roles:
        if r.startswith("test"):
            raise SystemExit("Refusing TEST roles in dense cache build")
    manifest = build_dense_cache(
        cache_root=args.cache_root,
        roles=roles,  # type: ignore[arg-type]
        overwrite=args.overwrite,
        verify_n=args.verify_n,
    )
    out = {
        "status": "ok",
        "elapsed_s": time.perf_counter() - t0,
        "peak_rss_GB": _rss_gb(),
        "counts": manifest["counts"],
        "bytes_total": manifest["bytes_total"],
        "cache_root": str(args.cache_root),
        "git_commit": manifest.get("git_commit"),
        "moments_train_fit": manifest.get("moments_train_fit"),
    }
    print(json.dumps(out, indent=2))
    if args.out_json:
        write_json(Path(args.out_json), {**out, "manifest": manifest})
    return 0


def _param_mean_on_subset(
    ds: DenseUNetCacheDataset,
    indices: list[int],
) -> dict[str, Any]:
    """Fit ParamMean on condition means of the tiny subset (diagnostic only)."""
    by_cond: dict[tuple[str, float, float], list[np.ndarray]] = {}
    for i in indices:
        row = ds.frames[i]
        key = (str(row["sim_id"]), float(row["nh3_fraction"]), float(row["phi"]))
        by_cond.setdefault(key, []).append(np.asarray(ds.t_K[i], dtype=np.float64))
    mean_fields = [np.mean(np.stack(v, axis=0), axis=0) for v in by_cond.values()]
    op_params = [(nh3, phi) for (_, nh3, phi) in by_cond]
    pm = ParamConditionedMeanPredictor.fit(
        mean_fields,
        op_params,
        ridge_alpha=0.01,
        basis="piecewise_phi1",
    )
    rmses = []
    for i in indices:
        row = ds.frames[i]
        pred = pm.predict(float(row["nh3_fraction"]), float(row["phi"]))
        T = np.asarray(ds.t_K[i], dtype=np.float64)
        rmses.append(float(np.sqrt(np.mean((pred - T) ** 2))))
    return {
        "param_mean_rmse_K_mean": float(np.mean(rmses)),
        "param_mean_rmse_K_per_frame": rmses,
        "n_conditions_fit": len(mean_fields),
        "basis": pm.basis,
        "ridge_alpha": pm.ridge_alpha,
    }


def cmd_overfit(args: argparse.Namespace) -> int:
    if args.scientific:
        raise SystemExit("Scientific training is not authorized in this mission (STOP A / U2 only)")
    cache_root = Path(args.cache_root)
    if not (cache_root / "manifest.json").exists():
        print("Cache missing; building train_fit/seen/val...", flush=True)
        build_dense_cache(cache_root=cache_root, overwrite=False)

    set_seed(args.seed)
    normalizer = Month2FitOnlyNormalizer()
    full = DenseUNetCacheDataset("train_fit", cache_root=cache_root, p_conditioning=not args.p_off)
    frames = full.frames
    indices = subset_indices_for_overfit(
        frames,
        n_frames=args.n_frames,
        min_conditions=2,
        max_conditions=4,
        seed=args.seed,
    )
    ds = DenseUNetCacheDataset(
        "train_fit",
        cache_root=cache_root,
        p_conditioning=not args.p_off,
        indices=indices,
    )
    loader = make_loader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=True, seed=args.seed)

    model = build_dense_unet_from_config(init_seed=args.seed, base_channels=args.base_channels)
    n_params = model.count_parameters()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_updates=args.max_updates,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        wall_time_cap_s=args.wall_minutes * 60.0,
    )
    trainer = DenseUNetTrainer(model, cfg=cfg, normalizer=normalizer)

    # Initial loss / RMSE on subset before updates.
    trainer.eval_mode()
    with torch.no_grad():
        losses = []
        rmses_K = []
        for batch in make_loader(ds, batch_size=len(ds), shuffle=False, seed=args.seed):
            x = batch["input_tensor"].to(trainer.device)
            y = batch["target_T_norm"].to(trainer.device)
            pred = model(x)
            losses.append(float(torch.mean((pred - y) ** 2).cpu()))
            pred_K = normalizer.denormalize_field("Temperature", pred.cpu().numpy())
            true_K = batch["target_T_K"].numpy()
            rmses_K.append(float(np.sqrt(np.mean((pred_K - true_K) ** 2))))
    initial = {"mse_norm": float(np.mean(losses)), "rmse_K": float(np.mean(rmses_K))}

    t0 = time.perf_counter()
    rss0 = _rss_gb()
    result = trainer.train_updates(loader, max_updates=args.max_updates, log_every=args.log_every)
    elapsed = time.perf_counter() - t0
    rss1 = _rss_gb()

    trainer.eval_mode()
    with torch.no_grad():
        losses = []
        rmses_K = []
        preds = []
        trues = []
        for batch in make_loader(ds, batch_size=len(ds), shuffle=False, seed=args.seed):
            x = batch["input_tensor"].to(trainer.device)
            y = batch["target_T_norm"].to(trainer.device)
            pred = model(x)
            losses.append(float(torch.mean((pred - y) ** 2).cpu()))
            pred_K = normalizer.denormalize_field("Temperature", pred.cpu().numpy())
            true_K = batch["target_T_K"].numpy()
            rmses_K.append(float(np.sqrt(np.mean((pred_K - true_K) ** 2))))
            preds.append(pred_K)
            trues.append(true_K)
    final = {"mse_norm": float(np.mean(losses)), "rmse_K": float(np.mean(rmses_K))}

    pm = _param_mean_on_subset(full, indices)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "overfit_last.pt"
    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        state=trainer.state,
        cfg=cfg,
        extra={
            "indices": indices,
            "frame_ids": [full.frames[i] for i in indices],
            "n_params": n_params,
            "cache_root": str(cache_root),
            "experiment_id": "month2_dense_unet_v1_overfit_u2",
            "p_conditioning": not args.p_off,
        },
    )

    # Reload coherence check
    model2 = build_dense_unet_from_config(init_seed=999, base_channels=args.base_channels)
    load_checkpoint(ckpt_path, model=model2, map_location="cpu", restore_rng=False)
    model2.eval()
    with torch.no_grad():
        batch = next(iter(make_loader(ds, batch_size=len(ds), shuffle=False, seed=args.seed)))
        p1 = model(batch["input_tensor"].to(trainer.device)).detach().cpu()
        p2 = model2(batch["input_tensor"]).detach().cpu()
        reload_max_abs = float((p1.cpu() - p2).abs().max())

    summary = {
        "status": "ok",
        "phase": "U2_overfit_diagnostic",
        "scientific_training": False,
        "device": device,
        "n_params": n_params,
        "base_channels": args.base_channels,
        "indices": indices,
        "frames": [full.frames[i] for i in indices],
        "conditions": sorted({full.frames[i]["sim_id"] for i in indices}),
        "updates": result["updates"],
        "samples_seen": result["samples_seen"],
        "stopped_reason": result["stopped_reason"],
        "initial": initial,
        "final": final,
        "train_result": result,
        "param_mean_same_subset": pm,
        "elapsed_s": elapsed,
        "peak_rss_GB": max(rss0, rss1),
        "reload_max_abs_pred_diff": reload_max_abs,
        "checkpoint": str(ckpt_path),
        "learning_demonstrated": bool(final["mse_norm"] < 0.5 * initial["mse_norm"]),
    }
    write_json(out_dir / "overfit_summary.json", summary)
    # Optional maps for a couple of frames (compressed).
    if preds:
        np.savez_compressed(
            out_dir / "overfit_maps.npz",
            pred_T_K=preds[0],
            true_T_K=trues[0],
            indices=np.asarray(indices, dtype=np.int64),
        )
    print(json.dumps({k: summary[k] for k in summary if k != "frames"}, indent=2))
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """U3 cost profiling: 1–2 TRAIN epochs + timed eval; VAL metrics not used for choices."""
    if args.scientific:
        raise SystemExit("Scientific U4 training is not authorized in profiling")
    dense_cfg = load_yaml(args.dense_config)
    contract = assert_dense_scientific_contract(dense_cfg)
    cache_root = Path(args.cache_root)
    if not (cache_root / "manifest.json").exists():
        raise SystemExit(f"Missing dense cache at {cache_root}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    # Fresh weights — never load overfit checkpoint.
    model = build_dense_unet_from_config(
        init_seed=args.seed, base_channels=args.base_channels
    )
    n_params = model.count_parameters()
    train_ds = DenseUNetCacheDataset(
        "train_fit", cache_root=cache_root, p_conditioning=not args.p_off
    )
    val_ds = DenseUNetCacheDataset(
        "val_eval", cache_root=cache_root, p_conditioning=not args.p_off
    )
    train_loader = make_loader(
        train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    val_loader = make_loader(
        val_ds, batch_size=args.batch_size, shuffle=False, seed=args.seed
    )
    updates_per_epoch = int(np.ceil(len(train_ds) / max(1, args.batch_size)))
    max_updates = int(args.epochs * updates_per_epoch)
    cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_updates=max_updates,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        wall_time_cap_s=args.wall_minutes * 60.0,
    )
    normalizer = Month2FitOnlyNormalizer()
    trainer = DenseUNetTrainer(model, cfg=cfg, normalizer=normalizer)

    effective = {
        "phase": "U3_profiling",
        "scientific_training": False,
        "u4_authorized": False,
        "dense_contract": contract,
        "dense_config_path": str(Path(args.dense_config)),
        "dense_config_sha256": sha256_file(Path(args.dense_config)),
        "overrides": {
            "base_channels": args.base_channels,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": device,
            "p_off": bool(args.p_off),
            "wall_minutes": args.wall_minutes,
        },
        "n_params": n_params,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "updates_per_epoch": updates_per_epoch,
        "max_updates": max_updates,
        "torch_threads": int(torch.get_num_threads()),
        "note_val_metrics": (
            "VAL timing only; metrics recorded for cost accounting, "
            "NOT used to choose width/lr/budget"
        ),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "effective_config.json", effective)

    rss0 = _rss_gb()
    t0 = time.perf_counter()
    train_result = trainer.train_updates(
        train_loader, max_updates=max_updates, log_every=max(1, updates_per_epoch // 2)
    )
    t_train = time.perf_counter() - t0
    rss1 = _rss_gb()

    t1 = time.perf_counter()
    # Timed eval for cost only — do not use scores for architecture/lr decisions.
    eval_out = trainer.evaluate_loader(
        val_loader,
        role="val_eval",
        method="DenseUNet_profile",
        variant="p_off" if args.p_off else "p_on",
        model_seed=args.seed,
    )
    t_eval = time.perf_counter() - t1
    rss2 = _rss_gb()

    # Projection helpers (cost only).
    sec_per_update = t_train / max(1, train_result["updates"])
    projected = {
        "sec_per_update": sec_per_update,
        "sec_per_train_epoch": sec_per_update * updates_per_epoch,
        "sec_per_val_eval_pass": t_eval,
        "candidate_E_max_200_train_only_h": (sec_per_update * updates_per_epoch * 200) / 3600.0,
        "candidate_E_max_200_with_eval_every_10_h": (
            sec_per_update * updates_per_epoch * 200 + t_eval * (200 // 10 + 1)
        )
        / 3600.0,
    }
    # Stability signal on TRAIN loss only (not VAL).
    hist = trainer.history
    train_stability = {
        "first_loss": train_result.get("first_loss"),
        "last_loss": train_result.get("last_loss"),
        "finite_throughout": all(
            np.isfinite(h["loss"]) and np.isfinite(h["grad_norm_pre_clip"]) for h in hist
        ),
        "clip_frequency": train_result.get("clip_frequency"),
        "stopped_reason": train_result.get("stopped_reason"),
    }
    # Budget proposal (recorded; U4 not launched).
    propose_b = args.base_channels
    propose_batch = args.batch_size
    propose_device = device
    # Keep b=32/batch=8 unless concrete resource pressure.
    wall_hours_200 = projected["candidate_E_max_200_with_eval_every_10_h"]
    if wall_hours_200 > 48 and propose_b == 32:
        # Only suggest narrowing if projected cost is extreme on this host.
        alt_note = "Consider b=16 if wall budget < projected; not applied automatically"
    else:
        alt_note = "Retain b=32, batch=8 on measured CPU profile unless GPU becomes available"

    proposal = {
        "base_channels": propose_b,
        "batch_size": propose_batch,
        "device": propose_device,
        "E_max": 200,
        "wall_time_cap_h": max(24.0, wall_hours_200 * 1.25),
        "lr": args.lr,
        "eval_every_epochs": 10,
        "rationale": alt_note,
        "val_metrics_used_for_choice": False,
    }

    # Extract VAL RMSE only as a logged number, flagged non-decisive.
    sel = eval_out["aggregation"]["selection_mean_condition_rmse"]
    val_rmse = None
    if sel is not None and not getattr(sel, "empty", True):
        val_rmse = float(sel["mean_condition_rmse_K"].iloc[0])

    summary = {
        "status": "ok",
        "effective_config": effective,
        "train_result": train_result,
        "train_wall_s": t_train,
        "eval_wall_s": t_eval,
        "peak_rss_GB": max(rss0, rss1, rss2),
        "projected": projected,
        "train_stability": train_stability,
        "budget_proposal_for_u4": proposal,
        "val_mean_condition_rmse_K_logged_not_for_selection": val_rmse,
        "overfit_weights_reused": False,
    }
    write_json(out_dir / "profile_summary.json", summary)
    # Do not persist profiling weights as a scientific checkpoint.
    print(json.dumps(summary, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-cache", help="Build dense U-Net cache (U1)")
    b.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    b.add_argument("--roles", type=str, default="train_fit,seen_train_eval,val_eval")
    b.add_argument("--overwrite", action="store_true")
    b.add_argument("--verify-n", type=int, default=3)
    b.add_argument("--out-json", type=Path, default=None)
    b.set_defaults(func=cmd_build_cache)

    o = sub.add_parser("overfit", help="Tiny TRAIN overfit diagnostic (U2)")
    o.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    o.add_argument("--out-dir", type=Path, default=ROOT / "results" / "month2_dense_unet" / "u2_overfit")
    o.add_argument("--n-frames", type=int, default=8)
    o.add_argument("--max-updates", type=int, default=500)
    o.add_argument("--wall-minutes", type=float, default=30.0)
    o.add_argument("--batch-size", type=int, default=8)
    o.add_argument("--lr", type=float, default=1e-3)
    o.add_argument("--weight-decay", type=float, default=1e-4)
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--base-channels", type=int, default=32)
    o.add_argument("--device", type=str, default="auto")
    o.add_argument("--p-off", action="store_true")
    o.add_argument("--log-every", type=int, default=25)
    o.add_argument(
        "--scientific",
        action="store_true",
        help="Refused: scientific training not part of U2",
    )
    o.set_defaults(func=cmd_overfit)

    pr = sub.add_parser("profile", help="U3 cost profiling (1–2 TRAIN epochs); not U4")
    pr.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    pr.add_argument("--dense-config", type=Path, default=DEFAULT_DENSE_CFG)
    pr.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "month2_dense_unet" / "u3_profile",
    )
    pr.add_argument("--epochs", type=float, default=2.0)
    pr.add_argument("--batch-size", type=int, default=8)
    pr.add_argument("--base-channels", type=int, default=32)
    pr.add_argument("--lr", type=float, default=1e-3)
    pr.add_argument("--weight-decay", type=float, default=1e-4)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--device", type=str, default="auto")
    pr.add_argument("--p-off", action="store_true")
    pr.add_argument("--wall-minutes", type=float, default=120.0)
    pr.add_argument("--scientific", action="store_true")
    pr.set_defaults(func=cmd_profile)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
