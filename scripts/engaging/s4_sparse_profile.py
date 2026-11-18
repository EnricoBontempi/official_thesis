#!/usr/bin/env python3
"""S4 diagnostic GPU profiling for sparse U-Net or adapted GLU.

Does NOT authorize S5 scientific training. Fresh diagnostic weights only.
On OOM/error: write diagnostics and exit nonzero (no automatic retries).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.data.month2_protocol import sha256_file  # noqa: E402
from combustion_recon.models.glu_reconstructor import (  # noqa: E402
    AdaptedDeterministicGLU,
    build_glu_from_config,
    make_synthetic_query_grid,
)
from combustion_recon.models.unet import build_sparse_unet_from_config  # noqa: E402
from combustion_recon.training.dense_cache import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    compute_cache_payload_hashes,
    validate_dense_cache_manifest,
    verify_cache_payload_hashes,
)
from combustion_recon.training.epoch_loop import apply_tf32_policy  # noqa: E402
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset  # noqa: E402
from combustion_recon.training.sparse_scientific_loop import (  # noqa: E402
    SparseScientificConfig,
    SparseScientificTrainer,
    save_sparse_scientific_checkpoint,
)
from combustion_recon.training.trainer import mse_loss, set_seed, write_json  # noqa: E402

CPU_GPU_ATOL = 1e-4
CPU_GPU_RTOL = 1e-4


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_vram_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("sparse_unet", "adapted_glu"), required=True)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/month2_sparse_s4_profiling.yaml",
    )
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--mask-root", type=Path, default=ROOT / "data/processed/month2_sparse_eval_masks")
    p.add_argument("--stats", type=Path, default=ROOT / "data/manifests/train_norm_stats_month2_fit.json")
    p.add_argument("--expect-cache-hashes", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--git-sha", type=str, default="")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_setup0 = time.perf_counter()
    report: dict = {
        "status": "starting",
        "family": args.family,
        "profiling_diagnostic_only": True,
        "scientific_training_authorized": False,
        "weights_reusable_for_science": False,
        "tolerances": {"cpu_gpu_atol": CPU_GPU_ATOL, "cpu_gpu_rtol": CPU_GPU_RTOL},
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for S4 Engaging profiling")

        cfg_yaml = yaml.safe_load(Path(args.config).read_text())
        if not bool(cfg_yaml.get("profiling_diagnostic_only")):
            raise RuntimeError("profiling config must set profiling_diagnostic_only: true")
        if bool(cfg_yaml.get("scientific_training_authorized")):
            raise RuntimeError("profiling must not authorize scientific training")
        prof = cfg_yaml["profiling"]

        tf32 = apply_tf32_policy(allow_tf32=False)
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()

        cache_root = Path(args.cache_root)
        manifest = json.loads((cache_root / "manifest.json").read_text())
        validate_dense_cache_manifest(
            manifest,
            cache_root=cache_root,
            require_roles=("train_fit", "seen_train_eval", "val_eval"),
        )
        payload_hashes = compute_cache_payload_hashes(cache_root)
        verify_info = {"ok": True, "note": "computed_only"}
        if args.expect_cache_hashes:
            expected = json.loads(Path(args.expect_cache_hashes).read_text())
            verify_info = verify_cache_payload_hashes(expected, cache_root=cache_root)
            if not verify_info["ok"]:
                raise RuntimeError(f"cache hash mismatch: {verify_info['mismatches']}")

        mask_manifest = Path(args.mask_root) / "manifest.json"
        if not mask_manifest.exists():
            raise RuntimeError(f"missing mask payload manifest: {mask_manifest}")

        set_seed(int(prof["seed"]))
        normalizer = Month2FitOnlyNormalizer(args.stats)

        if args.family == "sparse_unet":
            model = build_sparse_unet_from_config(init_seed=int(prof["seed"]))
        else:
            model = build_glu_from_config(
                domain_geometry_path=cache_root / "geometry.npz",
            )

        # Fixed-weight CPU/GPU compare on a few TRAIN frames
        train_ds = SparseUNetCacheDataset(
            "train_fit",
            cache_root=cache_root,
            mask_mode="training",
            site_fraction=float(prof["site_fraction_train"]),
            train_mask_seed=int(prof["train_mask_seed"]),
            p_conditioning=True,
            normalizer=normalizer,
            eval_mask_payload_root=args.mask_root,
        )
        train_ds.set_epoch(0)
        n_cmp = min(int(prof["cpu_gpu_compare"]["n_frames"]), len(train_ds))
        cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Build one batch for compare
        from combustion_recon.training.sparse_dataset import collate_sparse_batch

        samples = [train_ds[i] for i in range(n_cmp)]
        batch = collate_sparse_batch(samples)
        p_rows = [
            [float(train_ds.p_norm[int(m["cache_index"]), 0]), float(train_ds.p_norm[int(m["cache_index"]), 1])]
            for m in batch["metadata"]
        ]
        batch["p_norm_tensor"] = torch.tensor(p_rows, dtype=torch.float32)

        queries = make_synthetic_query_grid(train_ds.x, train_ds.y)

        def forward_unet(m, b, dev):
            return m(b["input_tensor"].to(dev))

        def forward_glu(m, b, dev):
            tokens = b["glu_tokens"]
            coords = torch.stack(
                [torch.from_numpy(np.asarray(t["coords_phys"], dtype=np.float32)) for t in tokens], 0
            ).to(dev)
            vals = torch.stack(
                [torch.from_numpy(np.asarray(t["values_norm"], dtype=np.float32)) for t in tokens], 0
            ).to(dev)
            sites = torch.stack(
                [torch.from_numpy(np.asarray(t["site_index"], dtype=np.int64)) for t in tokens], 0
            ).to(dev)
            p = b["p_norm_tensor"].to(dev)
            Y = queries.to(dev).unsqueeze(0).expand(coords.shape[0], -1, -1)
            return m(
                coords, vals, sites, p, Y,
                p_conditioning=True,
                decode_chunked=True,
                chunk_size=int(prof["decode_chunk_size"]),
                return_flat=False,
            )

        fwd = forward_unet if args.family == "sparse_unet" else forward_glu

        if args.family == "sparse_unet":
            m_cpu = build_sparse_unet_from_config(init_seed=int(prof["seed"]))
        else:
            m_cpu = build_glu_from_config(domain_geometry_path=cache_root / "geometry.npz")
        m_cpu.load_state_dict(cpu_state)
        m_cpu.eval()
        with torch.no_grad():
            pred_cpu = fwd(m_cpu, batch, torch.device("cpu")).cpu()

        model.load_state_dict(cpu_state)
        model.to(device)
        model.eval()
        with torch.no_grad():
            _sync()
            pred_gpu = fwd(model, batch, device).detach().cpu()
            _sync()
        abs_diff = (pred_cpu - pred_gpu).abs()
        cpu_gpu = {
            "max_abs": float(abs_diff.max()),
            "mean_abs": float(abs_diff.mean()),
            "allclose": bool(
                torch.allclose(pred_cpu, pred_gpu, rtol=CPU_GPU_RTOL, atol=CPU_GPU_ATOL)
            ),
            "atol": CPU_GPU_ATOL,
            "rtol": CPU_GPU_RTOL,
            "n_frames": n_cmp,
            "note": "No universal CPU/GPU bit-identity claim",
        }

        # Finite train step smoke
        model.train()
        model.zero_grad(set_to_none=True)
        pred = fwd(model, batch, device)
        y = batch["target_T_norm"].to(device)
        loss = mse_loss(pred, y)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss on smoke step")
        loss.backward()
        if not all((p.grad is None) or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("non-finite gradients on smoke step")

        setup_s = time.perf_counter() - t_setup0

        # Technical warmup steps (not measured)
        opt = torch.optim.AdamW(model.parameters(), lr=float(prof["lr"]), weight_decay=float(prof["weight_decay"]))
        warm = int(prof["technical_warmup_steps"])
        for _ in range(warm):
            opt.zero_grad(set_to_none=True)
            pred = fwd(model, batch, device)
            mse_loss(pred, batch["target_T_norm"].to(device)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(prof["grad_clip_norm"]))
            opt.step()
        _sync()

        # Measured: 2 TRAIN epochs via scientific trainer (short overrides)
        val_ds = SparseUNetCacheDataset(
            "val_eval",
            cache_root=cache_root,
            mask_mode="evaluation",
            site_fraction=0.05,
            eval_mask_seed=int(prof["eval_mask_seed"]),
            layout_id=str(prof["layout_id"]),
            p_conditioning=True,
            normalizer=normalizer,
            eval_mask_payload_root=args.mask_root,
        )
        sci_cfg = SparseScientificConfig(
            n_epochs=int(prof["measured_train_epochs"]),
            batch_size=int(prof["batch_size"]),
            lr=float(prof["lr"]),
            weight_decay=float(prof["weight_decay"]),
            warmup_updates=max(1, warm),
            plateau_factor=0.5,
            plateau_patience_evals=5,
            plateau_threshold_rel=0.001,
            plateau_min_lr=1e-5,
            grad_clip_norm=float(prof["grad_clip_norm"]),
            seed=int(prof["seed"]),
            shuffle_seed=int(prof["seed"]),
            device="cuda",
            eval_every_epochs=1,
            wall_time_cap_s=2 * 3600,
            p_conditioning=True,
            base_channels=32,
            early_stop_min_epochs=10_000,
            early_stop_patience_evals=15,
            early_stop_rel_improvement=0.001,
            family=args.family,
            site_fraction=float(prof["site_fraction_train"]),
            train_mask_seed=int(prof["train_mask_seed"]),
            eval_mask_seed=int(prof["eval_mask_seed"]),
            layout_id=str(prof["layout_id"]),
            val_replicas=tuple(int(x) for x in prof["val_replicas"]),
            decode_chunk_size=int(prof["decode_chunk_size"]),
            decode_chunked_eval=bool(prof["decode_chunked_eval"]),
            microbatch_size=None if prof.get("microbatch_size") in (None, "null") else int(prof["microbatch_size"]),
            eval_train_fit=False,
        )
        # Fresh model for measured loop (diagnostic seed)
        set_seed(int(prof["seed"]))
        if args.family == "sparse_unet":
            model2 = build_sparse_unet_from_config(init_seed=int(prof["seed"])).to(device)
            q = None
        else:
            model2 = build_glu_from_config(domain_geometry_path=cache_root / "geometry.npz").to(device)
            q = queries

        trainer = SparseScientificTrainer(
            model2,
            cfg=sci_cfg,
            normalizer=normalizer,
            n_train_samples=len(train_ds),
            query_coords_phys=q,
        )

        t_compute0 = time.perf_counter()
        _sync()
        summary = trainer.fit(
            train_ds,
            val_ds=val_ds,
            out_dir=out_dir / "measured_short_loop",
            run_identity={
                "run_id": f"s4_profile_{args.family}",
                "profiling_diagnostic_only": True,
                "family": args.family,
                "git_sha": args.git_sha,
                "cache_payload_sha256": payload_hashes,
                "mask_payload_sha256": sha256_file(mask_manifest),
                "stats_sha256": sha256_file(Path(args.stats)),
            },
        )
        _sync()
        train_compute_s = time.perf_counter() - t_compute0

        # VAL timing already inside fit (eval every epoch). Record separately inference coverages.
        infer = {}
        model2.eval()
        with torch.no_grad():
            for cov in prof["inference_coverages"]:
                ds_inf = SparseUNetCacheDataset(
                    "val_eval",
                    cache_root=cache_root,
                    mask_mode="evaluation",
                    site_fraction=float(cov),
                    eval_mask_seed=int(prof["eval_mask_seed"]),
                    layout_id=str(prof["layout_id"]),
                    replica_id=0,
                    p_conditioning=True,
                    normalizer=normalizer,
                    eval_mask_payload_root=args.mask_root,
                )
                # latency batch=1 on first frame
                item = ds_inf[0]
                b1 = collate_sparse_batch([item])
                b1["p_norm_tensor"] = torch.tensor(
                    [[float(ds_inf.p_norm[int(item["metadata"]["cache_index"]), 0]),
                      float(ds_inf.p_norm[int(item["metadata"]["cache_index"]), 1])]],
                    dtype=torch.float32,
                )
                _sync()
                t0 = time.perf_counter()
                _ = fwd(model2, b1, device)
                _sync()
                lat_ms = (time.perf_counter() - t0) * 1000.0
                # full role pass timing (batch 8)
                t1 = time.perf_counter()
                out = trainer.evaluate_dataset(
                    ds_inf,
                    role="val_eval",
                    replica_id=0,
                    method=args.family,
                    variant="p_on",
                    model_seed=int(prof["seed"]),
                    batch_size=int(prof["batch_size"]),
                )
                _sync()
                infer[str(cov)] = {
                    "latency_batch1_ms": lat_ms,
                    "full_role_s": time.perf_counter() - t1,
                    "mean_condition_rmse_K": out["mean_condition_rmse_K"],
                    "note": "rmse is diagnostic only; not for architecture selection",
                }

        # Checkpoint cost
        ckpt_path = out_dir / "diagnostic_last.pt"
        _sync()
        t_ck0 = time.perf_counter()
        save_sparse_scientific_checkpoint(
            ckpt_path,
            model=model2,
            optimizer=trainer.optimizer,
            plateau_scheduler=trainer.plateau_scheduler,
            state=trainer.state,
            cfg=sci_cfg,
            checkpoint_kind="sparse_scientific_epoch_complete_v1",
            loader_generator=trainer.loader_generator,
            eval_history=trainer.eval_history,
            curves=trainer.curves,
            clip_events=trainer.clip_events,
            extra={"profiling_diagnostic_only": True},
        )
        _sync()
        ckpt_s = time.perf_counter() - t_ck0

        glu_break = None
        if args.family == "adapted_glu" and isinstance(model2, AdaptedDeterministicGLU):
            # Coarse encoder/decoder timing on one batch
            model2.eval()
            with torch.no_grad():
                tokens = batch["glu_tokens"]
                coords = torch.stack(
                    [torch.from_numpy(np.asarray(t["coords_phys"], dtype=np.float32)) for t in tokens], 0
                ).to(device)
                vals = torch.stack(
                    [torch.from_numpy(np.asarray(t["values_norm"], dtype=np.float32)) for t in tokens], 0
                ).to(device)
                sites = torch.stack(
                    [torch.from_numpy(np.asarray(t["site_index"], dtype=np.int64)) for t in tokens], 0
                ).to(device)
                p = batch["p_norm_tensor"].to(device)
                Y = queries.to(device).unsqueeze(0).expand(coords.shape[0], -1, -1)
                _sync(); t_e0 = time.perf_counter()
                # Use public forward once; breakdown approximates via CUDA events around full call
                # plus a second call with decode_chunked for decoder-dominant cost.
                _ = model2(coords, vals, sites, p, Y, decode_chunked=False, return_flat=True)
                _sync(); t_full = time.perf_counter() - t_e0
                _sync(); t_c0 = time.perf_counter()
                _ = model2(coords, vals, sites, p, Y, decode_chunked=True, chunk_size=512, return_flat=True)
                _sync(); t_chunk = time.perf_counter() - t_c0
            glu_break = {
                "full_forward_s": t_full,
                "chunked_forward_s": t_chunk,
                "geometry": model2.geometry_meta,
                "neighbor_sort": "topk_distance_then_site_index in ThesisSoftDomainAdaptiveReconstructor",
                "note": "Coarse timing; encoder/decoder not isolated to separate CUDA graphs",
            }

        report.update(
            {
                "status": "completed_diagnostic",
                "gpu_name": gpu_name,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "git_sha": args.git_sha,
                "tf32_policy": tf32,
                "setup_s": setup_s,
                "queue_s_external": None,
                "compute_s": {
                    "measured_train_loop_including_val": train_compute_s,
                    "checkpoint_write_s": ckpt_s,
                },
                "cpu_gpu_compare": cpu_gpu,
                "train_summary": summary,
                "inference": infer,
                "peak_vram_gb": _peak_vram_gb(),
                "peak_rss_gb": _rss_gb(),
                "checkpoint_bytes": ckpt_path.stat().st_size if ckpt_path.exists() else None,
                "cache_verify": verify_info,
                "mask_manifest_sha256": sha256_file(mask_manifest),
                "glu_breakdown": glu_break,
                "note": (
                    "Diagnostic only. VAL/inference RMSE must not select architecture, "
                    "LR, or variant. No S5 authorization."
                ),
            }
        )
        write_json(out_dir / "profile_report.json", report)
        print(json.dumps({k: report[k] for k in ("status", "family", "gpu_name", "peak_vram_gb", "setup_s")}, indent=2))
        return 0

    except torch.cuda.OutOfMemoryError as exc:
        report["status"] = "oom"
        report["error"] = str(exc)
        report["peak_vram_gb"] = _peak_vram_gb()
        report["peak_rss_gb"] = _rss_gb()
        report["traceback"] = traceback.format_exc()
        write_json(out_dir / "profile_oom.json", report)
        print(json.dumps(report, indent=2)[:2000])
        return 10
    except Exception as exc:  # noqa: BLE001
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["peak_vram_gb"] = _peak_vram_gb()
        report["peak_rss_gb"] = _rss_gb()
        report["traceback"] = traceback.format_exc()
        write_json(out_dir / "profile_error.json", report)
        print(json.dumps(report, indent=2)[:2000])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
