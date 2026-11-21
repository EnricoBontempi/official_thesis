#!/usr/bin/env python3
"""Sparse scientific training entrypoint (U-Net / adapted GLU).

Refuses to run unless BOTH:
  - candidate YAML has scientific_training_authorized: true
  - CLI --authorize-scientific-s5 OR --authorize-scientific-opus is passed

S5 candidates also require s5_submission_authorized; Opus candidates require
opus_submission_authorized. Do not reuse S4/S5 profiling or prior-run weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.data.month2_protocol import sha256_file  # noqa: E402
from combustion_recon.data.sparse_eval_masks import check_eval_mask_contract  # noqa: E402
from combustion_recon.models.glu_reconstructor import (  # noqa: E402
    build_glu_from_config,
    make_synthetic_query_grid,
)
from combustion_recon.models.unet import build_sparse_unet_from_config  # noqa: E402
from combustion_recon.training.dense_cache import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    OPUS_C149_CACHE_ID,
    cache_validation_kwargs,
    compute_cache_payload_hashes,
    validate_dense_cache_manifest,
    verify_cache_payload_hashes,
)
from combustion_recon.training.scientific_epoch_sampler import (  # noqa: E402
    assert_pool_nesting,
    cache_indices_for_pool,
)
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset  # noqa: E402
from combustion_recon.training.sparse_scientific_loop import (  # noqa: E402
    SparseScientificTrainer,
    assert_sparse_scientific_candidate_contract,
    config_from_candidate,
)
from combustion_recon.training.trainer import set_seed, write_json  # noqa: E402


def _state_dict_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key in sorted(model.state_dict()):
        h.update(key.encode())
        t = model.state_dict()[key].detach().cpu().contiguous()
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _log(phase: str, **fields) -> None:
    """Timestamped, flushed phase line (stdout is tee'd by the launcher)."""
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] PHASE {phase} {extra}".rstrip(), flush=True)


def _smoke_subset(n: int, k: int) -> list[int]:
    """Deterministic spread of k indices over n frames (smoke only)."""
    k = min(int(k), int(n))
    return sorted({int(round(i * (n - 1) / max(1, k - 1))) for i in range(k)}) if k > 1 else [0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--authorize-scientific-s5",
        action="store_true",
        help="Authorize an S5 candidate (YAML s5_submission_authorized must be true)",
    )
    p.add_argument(
        "--authorize-scientific-opus",
        action="store_true",
        help="Authorize an Opus candidate (YAML opus_submission_authorized must be true)",
    )
    p.add_argument(
        "--authorize-scientific-s7",
        action="store_true",
        help="Authorize an S7 candidate (YAML s7_submission_authorized must be true)",
    )
    p.add_argument("--expect-mask-manifest-sha256", type=str, default=None)
    p.add_argument("--expect-mask-aggregate-sha256", type=str, default=None)
    p.add_argument("--expect-stats-sha256", type=str, default=None)
    p.add_argument("--expect-cache-aggregate-sha256", type=str, default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Non-scientific smoke: tiny epochs and eval subsets; --out-root must contain 'smoke'",
    )
    p.add_argument("--smoke-presentations", type=int, default=48)
    p.add_argument("--smoke-epochs", type=int, default=4)
    p.add_argument("--smoke-eval-every", type=int, default=2)
    p.add_argument("--smoke-eval-frames", type=int, default=6)
    p.add_argument(
        "--smoke-fault-at",
        type=str,
        default=None,
        help="smoke only: 'E:B' hard-exits (os._exit(137)) before batch B of 0-based epoch E",
    )
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
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
    p.add_argument("--expect-cache-hashes", type=Path, default=None)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--wall-hours", type=float, default=None, help="Optional override wall cap")
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--allow-extend-max-epochs", action="store_true")
    p.add_argument(
        "--extend-n-epochs",
        type=int,
        default=None,
        help="Resume only: raise n_epochs above the candidate YAML (needs --allow-extend-max-epochs). "
        "Every other training key stays locked by the resume check.",
    )
    p.add_argument("--git-sha", type=str, default="")
    args = p.parse_args()

    auth_flags = [
        args.authorize_scientific_s5,
        args.authorize_scientific_opus,
        args.authorize_scientific_s7,
    ]
    if not any(auth_flags):
        raise SystemExit(
            "Refusing sparse scientific training: pass --authorize-scientific-s5, "
            "--authorize-scientific-opus or --authorize-scientific-s7 only after Enrico "
            "authorizes the wave."
        )
    if sum(bool(f) for f in auth_flags) != 1:
        raise SystemExit("pass exactly one --authorize-scientific-* flag")
    if args.smoke and "smoke" not in str(args.out_root):
        raise SystemExit(f"--smoke requires 'smoke' in --out-root, got {args.out_root}")
    if args.authorize_scientific_s7 and not args.smoke:
        for need in ("expect_mask_manifest_sha256", "expect_mask_aggregate_sha256",
                     "expect_stats_sha256", "expect_cache_hashes",
                     "expect_cache_aggregate_sha256"):
            if not getattr(args, need):
                raise SystemExit(f"S7 scientific launch requires --{need.replace('_', '-')}")
        if args.wall_hours is None:
            raise SystemExit("S7 launch requires an explicit --wall-hours (per attempt)")

    if args.extend_n_epochs is not None:
        if args.resume_from is None or not args.allow_extend_max_epochs:
            raise SystemExit("--extend-n-epochs requires --resume-from and --allow-extend-max-epochs")
        if args.smoke:
            raise SystemExit("--extend-n-epochs is not valid with --smoke")

    cand = yaml.safe_load(Path(args.config).read_text())
    meta = assert_sparse_scientific_candidate_contract(cand)
    if not bool(cand.get("scientific_training_authorized")):
        raise SystemExit(
            "candidate YAML scientific_training_authorized must be true before launch"
        )
    if args.authorize_scientific_s7 and cand.get("s7_submission_authorized") is not True:
        raise SystemExit(
            "S7 launch requires s7_submission_authorized: true in the candidate YAML"
        )
    if args.authorize_scientific_s5 and cand.get("s5_submission_authorized") is not True:
        raise SystemExit(
            "S5 launch requires s5_submission_authorized: true in the candidate YAML"
        )
    if args.authorize_scientific_opus and cand.get("opus_submission_authorized") is not True:
        raise SystemExit(
            "Opus launch requires opus_submission_authorized: true in the candidate YAML"
        )
    if cand.get("s5_submission_authorized") is False and args.authorize_scientific_s5:
        raise SystemExit("candidate marks s5_submission_authorized: false")

    out_root = Path(args.out_root)
    run_id = args.run_id or (cand.get("artifacts") or {}).get("run_id")
    if not run_id:
        raise SystemExit("run_id required")
    out_dir = out_root / run_id
    resuming = args.resume_from is not None
    # Launchers may pre-create the run dir with slurm_context/stdout only.
    launcher_only = {"slurm_context.txt", "stdout.log", "dep_preflight.json", "init_manifest.json"}
    if resuming:
        # Resume is only ever from this run's own last.pt at an epoch boundary.
        rp = Path(args.resume_from).resolve()
        if rp != (out_dir / "checkpoints" / "last.pt").resolve():
            raise SystemExit(
                f"resume must use {out_dir / 'checkpoints' / 'last.pt'}, got {rp} "
                "(never best.pt, best_seen_diagnostic.pt or partial_interrupt.pt)"
            )
        for need in ("run_identity.json", "init_manifest.json", "effective_config.json"):
            if not (out_dir / need).is_file():
                raise SystemExit(f"resume refused: {out_dir / need} missing")
        prev_ident = json.loads((out_dir / "run_identity.json").read_text())
        if prev_ident.get("run_id") != run_id:
            raise SystemExit("resume refused: run_id differs from existing run_identity.json")
    elif out_dir.exists():
        existing = {p.name for p in out_dir.iterdir()}
        blocked = existing - launcher_only
        if blocked:
            raise SystemExit(
                f"refuse overwrite existing run dir {out_dir}: unexpected entries {sorted(blocked)}"
            )
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
    _log("start", run_id=run_id, resume=resuming, pid=os.getpid(), smoke=args.smoke)

    cache_root = Path(args.cache_root)
    manifest = json.loads((cache_root / "manifest.json").read_text())
    wall_s = None if args.wall_hours is None else float(args.wall_hours) * 3600.0
    overrides = None
    if args.smoke:
        overrides = {
            "presentations_per_scientific_epoch": int(args.smoke_presentations),
            "n_epochs": int(args.smoke_epochs),
            "eval_every_epochs": int(args.smoke_eval_every),
            # Cross the warmup boundary inside the tiny smoke so the post-warmup
            # scheduler path and its counters are exercised (non-scientific).
            "warmup_updates": 10,
        }
    if args.extend_n_epochs is not None:
        overrides = {**(overrides or {}), "n_epochs": int(args.extend_n_epochs)}
    cfg = config_from_candidate(
        cand, device=args.device, wall_time_cap_s=wall_s, overrides=overrides
    )
    pool_id = cfg.pool_id or "A38"

    cache_kwargs = cache_validation_kwargs(pool_id)
    if pool_id == "C149" and manifest.get("cache_id") != OPUS_C149_CACHE_ID:
        raise SystemExit(
            f"C149 requires cache_id={OPUS_C149_CACHE_ID}, got {manifest.get('cache_id')}"
        )
    validate_dense_cache_manifest(
        manifest,
        cache_root=cache_root,
        require_roles=("train_fit", "seen_train_eval", "val_eval"),
        **cache_kwargs,
    )
    _log("cache_hashing_begin", cache_root=cache_root)
    payload_hashes = compute_cache_payload_hashes(cache_root)
    if args.expect_cache_hashes:
        expected = json.loads(Path(args.expect_cache_hashes).read_text())
        verify = verify_cache_payload_hashes(expected, cache_root=cache_root)
        if not verify["ok"]:
            raise SystemExit(f"cache hash mismatch: {verify['mismatches']}")
    if (
        args.expect_cache_aggregate_sha256
        and payload_hashes.get("aggregate_sha256") != args.expect_cache_aggregate_sha256
    ):
        raise SystemExit(
            "cache aggregate mismatch: "
            f"{payload_hashes.get('aggregate_sha256')} != {args.expect_cache_aggregate_sha256}"
        )
    _log("cache_hashes_ok", aggregate=payload_hashes.get("aggregate_sha256"))

    stats_sha = sha256_file(Path(args.stats))
    if args.expect_stats_sha256 and stats_sha != args.expect_stats_sha256:
        raise SystemExit(f"stats hash mismatch: {stats_sha} != {args.expect_stats_sha256}")
    mask_contract = None
    if args.expect_mask_manifest_sha256 or args.expect_mask_aggregate_sha256:
        if not (args.expect_mask_manifest_sha256 and args.expect_mask_aggregate_sha256):
            raise SystemExit("pass both mask manifest-file and aggregate-payload expectations")
        try:
            mask_contract = check_eval_mask_contract(
                manifest_path=Path(args.mask_root) / "manifest.json",
                payload_root=Path(args.mask_root),
                expected_manifest_file_sha256=args.expect_mask_manifest_sha256,
                expected_aggregate_payload_sha256=args.expect_mask_aggregate_sha256,
            )
        except Exception as exc:  # noqa: BLE001 - EvalMaskContractError carries details
            raise SystemExit(f"eval mask contract failed: {type(exc).__name__}: {exc} {getattr(exc, 'details', '')}")
        _log("mask_contract_ok", **mask_contract)

    set_seed(int(cfg.seed))
    normalizer = Month2FitOnlyNormalizer(args.stats)

    # Pool selection on the train_fit role of the provided cache.
    train_frames = json.loads((cache_root / "train_fit" / "frames.json").read_text())
    pool_report = assert_pool_nesting(train_frames)
    if pool_id == "C149" and not all(
        v.get("c149_fully_in_cache") for v in pool_report["conditions"].values()
    ):
        raise SystemExit(
            "C149 not fully present in train_fit cache; refuse silent partial pool"
        )
    train_indices = cache_indices_for_pool(train_frames, pool_id)
    expected_n = {"B19": 380, "A38": 760, "C149": 2980}[pool_id]
    if len(train_indices) != expected_n:
        raise SystemExit(
            f"pool {pool_id} resolved {len(train_indices)} cache rows; expected {expected_n}"
        )
    # Diagnostic TRAIN metric always on historical reference760 (A38 indices).
    ref760_indices = cache_indices_for_pool(train_frames, "A38")
    if len(ref760_indices) != 760:
        raise SystemExit(f"reference760 resolved {len(ref760_indices)} != 760")

    require_payload = bool(cfg.require_eval_mask_payload)
    train_ds = SparseUNetCacheDataset(
        "train_fit",
        cache_root=cache_root,
        indices=train_indices,
        mask_mode="training",
        site_fraction=float(cfg.site_fraction),
        train_mask_seed=int(cfg.train_mask_seed),
        p_conditioning=bool(cfg.p_conditioning),
        normalizer=normalizer,
        eval_mask_payload_root=args.mask_root,
        require_eval_mask_payload=require_payload,
        **cache_kwargs,
    )
    # Historical 760-frame reference for TRAIN diagnostic (B19/C149 included).
    train_eval_ds = SparseUNetCacheDataset(
        "train_fit",
        cache_root=cache_root,
        indices=ref760_indices,
        mask_mode="evaluation",
        site_fraction=float(cfg.site_fraction),
        eval_mask_seed=int(cfg.eval_mask_seed),
        layout_id=str(cfg.layout_id),
        replica_id=0,
        p_conditioning=bool(cfg.p_conditioning),
        normalizer=normalizer,
        eval_mask_payload_root=args.mask_root,
        require_eval_mask_payload=require_payload,
        **cache_kwargs,
    )
    val_ds = SparseUNetCacheDataset(
        "val_eval",
        cache_root=cache_root,
        mask_mode="evaluation",
        site_fraction=float(cfg.site_fraction),
        eval_mask_seed=int(cfg.eval_mask_seed),
        layout_id=str(cfg.layout_id),
        p_conditioning=bool(cfg.p_conditioning),
        normalizer=normalizer,
        eval_mask_payload_root=args.mask_root,
        require_eval_mask_payload=require_payload,
        **cache_kwargs,
    )
    seen_ds = SparseUNetCacheDataset(
        "seen_train_eval",
        cache_root=cache_root,
        mask_mode="evaluation",
        site_fraction=float(cfg.site_fraction),
        eval_mask_seed=int(cfg.eval_mask_seed),
        layout_id=str(cfg.layout_id),
        replica_id=0,
        p_conditioning=bool(cfg.p_conditioning),
        normalizer=normalizer,
        eval_mask_payload_root=args.mask_root,
        require_eval_mask_payload=require_payload,
        **cache_kwargs,
    )

    if args.smoke:
        # Non-scientific: evaluate on a deterministic spread of real frames only.
        for _ds in (train_eval_ds, val_ds, seen_ds):
            _ds.indices = [_ds.indices[i] for i in _smoke_subset(len(_ds.indices), args.smoke_eval_frames)]
        _log("smoke_eval_subsets", train_eval=len(train_eval_ds), val=len(val_ds), seen=len(seen_ds))
    _log("datasets_ready", train=len(train_ds), val=len(val_ds), seen=len(seen_ds), p_conditioning=cfg.p_conditioning)

    if cfg.family == "sparse_unet":
        model = build_sparse_unet_from_config(init_seed=int(cfg.seed))
        queries = None
    else:
        model = build_glu_from_config(domain_geometry_path=cache_root / "geometry.npz")
        queries = make_synthetic_query_grid(train_ds.x, train_ds.y)

    init_sha = _state_dict_sha256(model)
    init_manifest = {
        "run_id": run_id,
        "family": cfg.family,
        "p_conditioning": bool(cfg.p_conditioning),
        "model_seed": int(cfg.seed),
        "shuffle_seed": int(cfg.shuffle_seed),
        "pool_id": pool_id,
        "pool_size": len(train_ds),
        "presentations_per_scientific_epoch": cfg.presentations_per_scientific_epoch,
        "init_state_dict_sha256": init_sha,
        "git_sha": args.git_sha,
        "config_sha256": sha256_file(Path(args.config)),
        "warm_start": False,
        "note": "hash of freshly initialized weights before any optimizer step",
    }
    if resuming:
        # Keep the original fresh-init manifest; record that this process rebuilt the model.
        prev_init = json.loads((out_dir / "init_manifest.json").read_text())
        if prev_init.get("init_state_dict_sha256") != init_sha:
            raise SystemExit(
                "resume refused: rebuilt fresh-init hash differs from init_manifest.json "
                f"({init_sha} != {prev_init.get('init_state_dict_sha256')})"
            )
    else:
        write_json(out_dir / "init_manifest.json", init_manifest)
    _log("model_ready", family=cfg.family, init_sha=init_sha[:16],
         n_params=sum(p_.numel() for p_ in model.parameters()))

    trainer = SparseScientificTrainer(
        model,
        cfg=cfg,
        normalizer=normalizer,
        n_train_samples=len(train_ds),
        query_coords_phys=queries,
    )
    identity = {
        "run_id": run_id,
        "family": cfg.family,
        "p_conditioning": bool(cfg.p_conditioning),
        "scientific_training_authorized": True,
        "authorization_flag": (
            "authorize-scientific-s7"
            if args.authorize_scientific_s7
            else "authorize-scientific-opus"
            if args.authorize_scientific_opus
            else "authorize-scientific-s5"
        ),
        "refuse_reuse_s4_profiling_checkpoints": True,
        "refuse_warm_start": True,
        "git_sha": args.git_sha,
        "config_path": str(args.config),
        "config_sha256": sha256_file(Path(args.config)),
        "protocol_sha256": sha256_file(ROOT / "configs/experiments/month2_sparse_protocol.yaml"),
        "scientific_candidate_sha256": sha256_file(
            ROOT / "configs/experiments/month2_sparse_scientific_candidate.yaml"
        ),
        "model_config_sha256": sha256_file(
            ROOT
            / (
                "configs/models/unet_sparse.yaml"
                if cfg.family == "sparse_unet"
                else "configs/models/glu_minimal.yaml"
            )
        ),
        "stats_sha256": stats_sha,
        "cache_payload_sha256": payload_hashes,
        # Legacy key: SHA-256 of the manifest FILE bytes (not the payload aggregate).
        "mask_payload_sha256": sha256_file(Path(args.mask_root) / "manifest.json"),
        "eval_mask_manifest_file_sha256": (
            mask_contract["eval_mask_manifest_file_sha256"] if mask_contract else None
        ),
        "eval_mask_aggregate_payload_sha256": (
            mask_contract["eval_mask_aggregate_payload_sha256"] if mask_contract else None
        ),
        "slurm": {
            k: os.environ.get(k)
            for k in (
                "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT", "SLURM_JOB_QOS",
                "SLURM_RESTART_COUNT", "SLURMD_NODENAME",
            )
        },
        "non_scientific_smoke": bool(args.smoke),
        "expect_cache_hashes_path": str(args.expect_cache_hashes) if args.expect_cache_hashes else None,
        "pool_id": pool_id,
        "pool_size": len(train_ds),
        "pool_nesting_report": pool_report,
        "presentations_per_scientific_epoch": cfg.presentations_per_scientific_epoch,
        "init_state_dict_sha256": init_sha,
        "precision": "fp32",
        "amp_enabled": False,
        "torch_compile": False,
        "allow_tf32": False,
        "require_eval_mask_payload": require_payload,
        "operational": cand.get("operational_s7")
        or cand.get("operational_from_s5_recipe")
        or cand.get("operational_from_s4_profiling_candidate")
        or cand.get("operational_provisional_pending_s4_profiling"),
        "contract_meta": meta,
    }
    if not resuming:
        write_json(out_dir / "effective_config.json", {"candidate": cand, "runtime_cfg": cfg.__dict__})
    else:
        # Preserve the original; log what this attempt consumed next to it.
        n_prev = len(list(out_dir.glob("effective_config_resume_*.json")))
        (out_dir / f"run_identity_before_resume_{n_prev + 1}.json").write_text(
            (out_dir / "run_identity.json").read_text()
        )
        write_json(
            out_dir / f"effective_config_resume_{n_prev + 1}.json",
            {"candidate": cand, "runtime_cfg": cfg.__dict__},
        )
        prev_summary = out_dir / "summary.json"
        if prev_summary.exists():
            n_s = len(list(out_dir.glob("summary_before_resume_*.json")))
            prev_summary.rename(out_dir / f"summary_before_resume_{n_s + 1}.json")
    trainer.stream_trace_path = out_dir / "stream_trace.jsonl"
    if args.smoke_fault_at:
        if not args.smoke:
            raise SystemExit("--smoke-fault-at is only valid with --smoke")
        f_epoch, f_batch = (int(x) for x in args.smoke_fault_at.split(":"))

        def _fault(epoch: int, batches_done: int) -> None:
            if epoch == f_epoch and batches_done == f_batch:
                _log("smoke_fault_injected", epoch=epoch, batches_done=batches_done)
                os._exit(137)

        trainer.fault_hook = _fault
    _log("fit_begin", resume_from=args.resume_from, n_epochs=cfg.n_epochs, wall_cap_s=cfg.wall_time_cap_s)
    summary = trainer.fit(
        train_ds,
        val_ds=val_ds,
        seen_ds=seen_ds,
        train_eval_ds=train_eval_ds,
        out_dir=out_dir,
        run_identity=identity,
        resume_path=args.resume_from,
        allow_extend_max_epochs=bool(args.allow_extend_max_epochs),
    )
    _log("fit_end", status=summary.get("status"), epochs=summary.get("epochs_completed"),
         elapsed_s=round(float(summary.get("elapsed_s") or 0), 1))
    print(json.dumps({"run_id": run_id, "status": summary.get("status"), "out_dir": str(out_dir)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
