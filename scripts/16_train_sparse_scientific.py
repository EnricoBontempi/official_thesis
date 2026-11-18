#!/usr/bin/env python3
"""Sparse scientific training entrypoint (U-Net / adapted GLU).

Refuses to run unless BOTH:
  - candidate YAML has scientific_training_authorized: true
  - CLI --authorize-scientific-s5 is passed

Prepared S5 candidates ship with authorization false. Do not reuse S4
profiling checkpoints. Operational walltime/mem/microbatch may remain provisional.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.data.month2_protocol import sha256_file  # noqa: E402
from combustion_recon.models.glu_reconstructor import (  # noqa: E402
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
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset  # noqa: E402
from combustion_recon.training.sparse_scientific_loop import (  # noqa: E402
    SparseScientificTrainer,
    assert_sparse_scientific_candidate_contract,
    config_from_candidate,
)
from combustion_recon.training.trainer import set_seed, write_json  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--authorize-scientific-s5",
        action="store_true",
        help="Required explicit authorization; still needs YAML flag true",
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
    p.add_argument("--wall-hours", type=float, default=None, help="Optional override; provisional until S4 freeze")
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--allow-extend-max-epochs", action="store_true")
    p.add_argument("--git-sha", type=str, default="")
    args = p.parse_args()

    if not args.authorize_scientific_s5:
        raise SystemExit(
            "Refusing sparse scientific training: pass --authorize-scientific-s5 only after "
            "Enrico authorizes S5 screening (S4 profiling reviewed; no auto-launch)."
        )

    cand = yaml.safe_load(Path(args.config).read_text())
    meta = assert_sparse_scientific_candidate_contract(cand)
    if not bool(cand.get("scientific_training_authorized")):
        raise SystemExit(
            "candidate YAML scientific_training_authorized must be true before launch"
        )
    if cand.get("s5_submission_authorized") is False:
        raise SystemExit(
            "candidate marks s5_submission_authorized: false — update only after explicit decision"
        )

    out_root = Path(args.out_root)
    run_id = args.run_id or (cand.get("artifacts") or {}).get("run_id")
    if not run_id:
        raise SystemExit("run_id required")
    out_dir = out_root / run_id
    # Launchers may pre-create the run dir with slurm_context/stdout only.
    launcher_only = {"slurm_context.txt", "stdout.log", "dep_preflight.json"}
    if out_dir.exists():
        existing = {p.name for p in out_dir.iterdir()}
        blocked = existing - launcher_only
        if blocked:
            raise SystemExit(
                f"refuse overwrite existing run dir {out_dir}: unexpected entries {sorted(blocked)}"
            )
    else:
        out_dir.mkdir(parents=True, exist_ok=False)

    cache_root = Path(args.cache_root)
    manifest = json.loads((cache_root / "manifest.json").read_text())
    validate_dense_cache_manifest(
        manifest,
        cache_root=cache_root,
        require_roles=("train_fit", "seen_train_eval", "val_eval"),
    )
    payload_hashes = compute_cache_payload_hashes(cache_root)
    if args.expect_cache_hashes:
        expected = json.loads(Path(args.expect_cache_hashes).read_text())
        verify = verify_cache_payload_hashes(expected, cache_root=cache_root)
        if not verify["ok"]:
            raise SystemExit(f"cache hash mismatch: {verify['mismatches']}")

    wall_s = None if args.wall_hours is None else float(args.wall_hours) * 3600.0
    cfg = config_from_candidate(cand, device=args.device, wall_time_cap_s=wall_s)
    # Provisional microbatch from candidate remains null unless YAML set after profiling.
    set_seed(int(cfg.seed))
    normalizer = Month2FitOnlyNormalizer(args.stats)

    train_ds = SparseUNetCacheDataset(
        "train_fit",
        cache_root=cache_root,
        mask_mode="training",
        site_fraction=float(cfg.site_fraction),
        train_mask_seed=int(cfg.train_mask_seed),
        p_conditioning=bool(cfg.p_conditioning),
        normalizer=normalizer,
        eval_mask_payload_root=args.mask_root,
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
    )

    if cfg.family == "sparse_unet":
        model = build_sparse_unet_from_config(init_seed=int(cfg.seed))
        queries = None
    else:
        model = build_glu_from_config(domain_geometry_path=cache_root / "geometry.npz")
        queries = make_synthetic_query_grid(train_ds.x, train_ds.y)

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
        "refuse_reuse_s4_profiling_checkpoints": True,
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
        "stats_sha256": sha256_file(Path(args.stats)),
        "cache_payload_sha256": payload_hashes,
        "mask_payload_sha256": sha256_file(Path(args.mask_root) / "manifest.json"),
        "operational": cand.get("operational_from_s4_profiling_candidate")
        or cand.get("operational_provisional_pending_s4_profiling"),
        "contract_meta": meta,
        "refuse_s4_profile_checkpoints": True,
    }
    write_json(out_dir / "effective_config.json", {"candidate": cand, "runtime_cfg": cfg.__dict__})
    summary = trainer.fit(
        train_ds,
        val_ds=val_ds,
        seen_ds=seen_ds,
        out_dir=out_dir,
        run_identity=identity,
        resume_path=args.resume_from,
        allow_extend_max_epochs=bool(args.allow_extend_max_epochs),
    )
    print(json.dumps({"run_id": run_id, "status": summary.get("status"), "out_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
