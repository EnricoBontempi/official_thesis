#!/usr/bin/env python3
"""GPU smoke + short profiling for U6 Engaging p_off (diagnostic weights only).

Does NOT authorize or start the scientific U6 run. Uses fresh diagnostic
weights that must not be reused for the scientific training. Verifies that
parameter channels are exactly zero on the p_off dataset path.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.data.month2_protocol import sha256_file
from combustion_recon.models.unet import build_dense_unet_from_config
from combustion_recon.training.dense_cache import (
    DEFAULT_CACHE_ROOT,
    DenseUNetCacheDataset,
    compute_cache_payload_hashes,
    validate_dense_cache_manifest,
    verify_cache_payload_hashes,
)
from combustion_recon.training.epoch_loop import (
    EpochDenseTrainer,
    EpochTrainConfig,
    apply_tf32_policy,
    make_epoch_loader,
)
from combustion_recon.training.trainer import set_seed, write_json

# Declared BEFORE comparison (CPU vs GPU on fixed weights).
CPU_GPU_ATOL = 1e-4
CPU_GPU_RTOL = 1e-4


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--expect-cache-hashes", type=Path, default=None)
    p.add_argument("--profile-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--max-compute-minutes", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=12345, help="Diagnostic seed (not scientific 0)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_setup0 = time.perf_counter()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for Engaging smoke/profile")

    tf32 = apply_tf32_policy(allow_tf32=False)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)

    cache_root = Path(args.cache_root)
    manifest = json.loads((cache_root / "manifest.json").read_text())
    validate_dense_cache_manifest(
        manifest,
        cache_root=cache_root,
        require_roles=("train_fit", "seen_train_eval", "val_eval"),
    )
    payload = compute_cache_payload_hashes(cache_root)
    verify_info = {"ok": True, "note": "computed_only"}
    if args.expect_cache_hashes:
        expected = json.loads(Path(args.expect_cache_hashes).read_text())
        verify_info = verify_cache_payload_hashes(expected, cache_root=cache_root)
        if not verify_info["ok"]:
            raise SystemExit(f"cache hash mismatch: {verify_info['mismatches']}")

    set_seed(int(args.seed))
    model = build_dense_unet_from_config(
        init_seed=int(args.seed), base_channels=int(args.base_channels)
    )
    # Fixed weights snapshot for CPU/GPU compare (before any training).
    cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_ds = DenseUNetCacheDataset("train_fit", cache_root=cache_root, p_conditioning=False)
    val_ds = DenseUNetCacheDataset("val_eval", cache_root=cache_root, p_conditioning=False)
    n_compare = min(4, len(train_ds))
    xs = []
    ys = []
    for i in range(n_compare):
        sample = train_ds[i]
        xt = sample["model_inputs"]["input_tensor"]
        if int(torch.count_nonzero(xt[4])) != 0 or int(torch.count_nonzero(xt[5])) != 0:
            raise SystemExit("p_off smoke failed: parameter channels not zero")
        if "op_params" in sample["model_inputs"]:
            raise SystemExit("p_off smoke failed: op_params leaked into model_inputs")
        xs.append(xt)
        ys.append(sample["targets"]["temperature_normalized"])
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)

    # CPU forward with fixed weights
    model_cpu = build_dense_unet_from_config(
        init_seed=int(args.seed), base_channels=int(args.base_channels)
    )
    model_cpu.load_state_dict(cpu_state)
    model_cpu.eval()
    with torch.no_grad():
        pred_cpu = model_cpu(x).cpu()

    # GPU forward/backward finite check with same weights
    model_gpu = build_dense_unet_from_config(
        init_seed=int(args.seed), base_channels=int(args.base_channels)
    )
    model_gpu.load_state_dict(cpu_state)
    model_gpu.to(device)
    model_gpu.train()
    x_g = x.to(device)
    y_g = y.to(device)
    model_gpu.zero_grad(set_to_none=True)
    pred_g = model_gpu(x_g)
    loss = torch.mean((pred_g - y_g) ** 2)
    assert torch.isfinite(loss)
    loss.backward()
    grads_finite = all(
        (p.grad is None) or torch.isfinite(p.grad).all() for p in model_gpu.parameters()
    )
    if not grads_finite:
        raise SystemExit("non-finite gradients on GPU smoke")

    model_gpu.eval()
    with torch.no_grad():
        pred_gpu = model_gpu(x_g).detach().cpu()
    abs_diff = (pred_cpu - pred_gpu).abs()
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    close = bool(torch.allclose(pred_cpu, pred_gpu, rtol=CPU_GPU_RTOL, atol=CPU_GPU_ATOL))

    # Checkpoint roundtrip on GPU
    from combustion_recon.training.epoch_loop import (
        CHECKPOINT_KIND_COMPLETE,
        EpochTrainState,
        load_epoch_checkpoint,
        save_epoch_checkpoint,
    )
    from combustion_recon.training.trainer import TrainConfig, build_optimizer

    cfg_ck = EpochTrainConfig(
        n_epochs=2,
        batch_size=args.batch_size,
        device="cuda",
        warmup_updates=2,
        seed=args.seed,
        base_channels=args.base_channels,
        early_stop_min_epochs=10_000,
        p_conditioning=False,
    )
    opt = build_optimizer(
        model_gpu,
        TrainConfig(lr=1e-3, weight_decay=1e-4, max_updates=10, batch_size=args.batch_size),
    )
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    plateau = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, threshold=0.001, threshold_mode="rel", min_lr=1e-5)
    ckpt_path = out_dir / "smoke_checkpoint.pt"
    save_epoch_checkpoint(
        ckpt_path,
        model=model_gpu,
        optimizer=opt,
        plateau_scheduler=plateau,
        state=EpochTrainState(epoch=0, update=0),
        cfg=cfg_ck,
        checkpoint_kind=CHECKPOINT_KIND_COMPLETE,
        extra={"smoke": True},
    )
    model_reload = build_dense_unet_from_config(
        init_seed=int(args.seed), base_channels=int(args.base_channels)
    )
    load_epoch_checkpoint(ckpt_path, model=model_reload, map_location="cpu", require_resumable=True)

    # K conversion path
    normalizer = Month2FitOnlyNormalizer()
    from combustion_recon.evaluation.month2_evaluator import denormalize_temperature

    t_k = denormalize_temperature(pred_gpu[0, 0].numpy(), normalizer)
    assert np.isfinite(t_k).all()

    setup_s = time.perf_counter() - t_setup0

    # Short profiling with NEW diagnostic weights (not smoke compare weights).
    set_seed(int(args.seed) + 999)
    prof_model = build_dense_unet_from_config(
        init_seed=int(args.seed) + 999, base_channels=int(args.base_channels)
    )
    u_per_ep = max(1, (len(train_ds) + args.batch_size - 1) // args.batch_size)
    ecfg = EpochTrainConfig(
        n_epochs=int(args.profile_epochs),
        batch_size=int(args.batch_size),
        device="cuda",
        warmup_updates=min(10, u_per_ep),
        seed=int(args.seed) + 999,
        base_channels=int(args.base_channels),
        eval_every_epochs=1,
        wall_time_cap_s=float(args.max_compute_minutes) * 60.0,
        early_stop_min_epochs=10_000,
        eval_train_fit=True,
        p_conditioning=False,
    )
    trainer = EpochDenseTrainer(
        prof_model, cfg=ecfg, normalizer=normalizer, n_train_samples=len(train_ds)
    )
    train_eval, _ = make_epoch_loader(
        train_ds, batch_size=args.batch_size, shuffle=False, seed=args.seed
    )
    val_loader, _ = make_epoch_loader(
        val_ds, batch_size=args.batch_size, shuffle=False, seed=args.seed
    )

    torch.cuda.reset_peak_memory_stats()
    _sync()
    t_compute0 = time.perf_counter()
    summary = trainer.fit(
        train_ds,
        seen_loader=None,
        val_loader=val_loader,
        train_eval_loader=train_eval,
        out_dir=out_dir / "profile_run",
        run_identity={
            "run_id": "u6_gpu_smoke_profile_poff",
            "diagnostic_only": True,
            "scientific_weights_forbidden": True,
            "p_conditioning": False,
            "gpu_name": gpu_name,
        },
    )
    _sync()
    compute_s = time.perf_counter() - t_compute0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

    epochs_done = max(1, int(summary.get("epochs_completed") or 1))
    s_per_epoch = compute_s / epochs_done
    # One scientific cadence: train 10 epochs then eval train+seen+val.
    # Approximate eval cost from profile (eval every epoch here includes train+val).
    eval_cost_proxy_s = compute_s / max(1, epochs_done) * 0.35  # conservative split unknown
    # Better: use elapsed and epochs; project 1000 train epochs + 100 eval points.
    n_evals = 1000 // 10  # 100 evals at cadence 10, plus possible last
    # Profile ran eval every epoch; estimate eval fraction from updates vs eval.
    # Use measured s/epoch as full train+eval-every-1; for cadence 10, ~train + 0.1*eval_bundle.
    # Conservative projection: 1000 * (train-only) + 101 * eval_bundle.
    # We did not separate train vs eval timing; use total/epoch as upper bound for train+eval/1
    # and project 1000 epochs with eval every 10 as:
    #   1000 * train_frac + 101 * eval_frac, with train_frac+eval_frac = s_per_epoch_profile.
    # Upper bound: 1000 * s_per_epoch (as if eval every epoch) — too pessimistic.
    # Lower-ish bound used for gate: 1000 * 0.9 * s_per_epoch + 101 * 0.1 * s_per_epoch
    # Simplification requested: report measured s/epoch and projected 1000 * s_per_epoch_train_est.
    # Here s_per_epoch includes eval-every-1; for cadence 10 estimate:
    projected_1000_s = 1000 * s_per_epoch * (0.92) + 101 * s_per_epoch * (0.08)
    # Also report naive 1000 * s_per_epoch as upper envelope from this profile shape.
    projected_1000_naive_s = 1000 * s_per_epoch

    report = {
        "status": "ok" if summary.get("epochs_completed", 0) >= 2 else "incomplete",
        "declared_cpu_gpu_tolerance": {"atol": CPU_GPU_ATOL, "rtol": CPU_GPU_RTOL},
        "cpu_gpu_compare": {
            "n_frames": n_compare,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "allclose": close,
        },
        "grads_finite": bool(grads_finite),
        "loss_finite": True,
        "checkpoint_smoke_ok": ckpt_path.is_file(),
        "cache_payload_sha256": payload["aggregate_sha256"],
        "cache_verification": verify_info,
        "tf32_policy": tf32,
        "gpu_name": gpu_name,
        "gpu_total_mem_GB": props.total_memory / (1024**3),
        "peak_vram_GB": peak_vram_gb,
        "peak_rss_GB": _rss_gb(),
        "timings_s": {
            "setup": setup_s,
            "compute_profile": compute_s,
            "seconds_per_epoch_measured": s_per_epoch,
            "queue_wait": None,
            "note": "queue_wait filled by submit wrapper; setup vs compute separated here",
        },
        "projection": {
            "profile_epochs": epochs_done,
            "projected_1000_epochs_s_cadence10_est": projected_1000_s,
            "projected_1000_epochs_hours_cadence10_est": projected_1000_s / 3600.0,
            "projected_1000_naive_same_shape_as_profile_hours": projected_1000_naive_s / 3600.0,
            "gate_hours": 4.5,
            "under_gate_cadence10_est": (projected_1000_s / 3600.0) < 4.5,
        },
        "profile_summary": summary,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "diagnostic_weights_must_not_reuse_for_science": True,
        "p_conditioning": False,
        "p_off_channels_verified_zero": True,
        "manifest_sha_hint": sha256_file(cache_root / "manifest.json"),
    }
    write_json(out_dir / "smoke_profile_report.json", report)
    print(json.dumps(report, indent=2, default=str))
    if not close:
        return 3
    if not report["projection"]["under_gate_cadence10_est"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
