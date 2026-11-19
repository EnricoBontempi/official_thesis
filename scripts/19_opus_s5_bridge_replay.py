#!/usr/bin/env python3
"""Inference replay of the two historical S5 U-Net best checkpoints at 5%.

This checks the current evaluator against the recorded VAL macro RMSE.
It does not retrain and it does not show that the training order matches.
Eval masks are read from the persisted payload; seed regeneration is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.training.dense_cache import cache_validation_kwargs
from combustion_recon.training.epoch_loop import apply_tf32_policy
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset
from combustion_recon.training.sparse_scientific_loop import load_sparse_scientific_checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module  # noqa: E402

_s6 = import_module("17_s6_sparse_eval_and_swap")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(name: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise SystemExit(f"REFUSE {name}: got {actual} expected {expected}")


def evaluate_run(
    spec: dict,
    *,
    run_root: Path,
    cache_root: Path,
    mask_root: Path,
    normalizer: Month2FitOnlyNormalizer,
    device: torch.device,
    batch_size: int,
) -> dict:
    run_dir = run_root / spec["run_id"]
    ckpt = run_dir / "checkpoints" / "best.pt"
    _require(f"{spec['run_id']} best.pt", _sha256(ckpt), spec["best_pt_sha256"])
    model, _queries = _s6._load_model("sparse_unet", cache_root, device)
    payload = load_sparse_scientific_checkpoint(
        ckpt, model=model, require_resumable=True, restore_rng=False
    )
    epoch = int((payload.get("epoch_state") or {}).get("epoch", -1))
    if epoch != int(spec["best_val_epoch"]):
        raise SystemExit(
            f"REFUSE {spec['run_id']}: checkpoint epoch {epoch} != {spec['best_val_epoch']}"
        )
    model.to(device)
    model.eval()
    cache_kwargs = cache_validation_kwargs("A38")
    roles = {
        "train_fit": [0],
        "seen_train_eval": [0],
        "val_eval": [0, 1, 2],
    }
    per_replica: dict[str, list[dict]] = {}
    for role, replicas in roles.items():
        dataset = SparseUNetCacheDataset(
            role,  # type: ignore[arg-type]
            cache_root=cache_root,
            mask_mode="evaluation",
            site_fraction=0.05,
            eval_mask_seed=20261116,
            layout_id="random_sparse_primary",
            replica_id=0,
            p_conditioning=bool(spec["p_conditioning"]),
            normalizer=normalizer,
            eval_mask_payload_root=mask_root,
            require_eval_mask_payload=True,
            **cache_kwargs,
        )
        per_replica[role] = []
        for replica in replicas:
            out = _s6.evaluate_setting(
                model=model,
                family="sparse_unet",
                p_conditioning=bool(spec["p_conditioning"]),
                ds=dataset,
                role=role,
                replica_id=replica,
                site_fraction=0.05,
                method="sparse_unet_bridge_replay",
                variant="best",
                checkpoint_kind="best.pt",
                run_id=spec["run_id"],
                normalizer=normalizer,
                query_coords_phys=None,
                device=device,
                batch_size=batch_size,
            )
            per_replica[role].append(
                {
                    "replica_id": replica,
                    "mean_condition_rmse_K": out["mean_condition_rmse_K"],
                    "n_frames": out["n_frames"],
                    "wall_s": out["wall_s"],
                }
            )
    val_scores = [float(row["mean_condition_rmse_K"]) for row in per_replica["val_eval"]]
    val_macro = sum(val_scores) / float(len(val_scores))
    expected = float(spec["best_val_metric"])
    delta = val_macro - expected
    return {
        "run_id": spec["run_id"],
        "checkpoint": str(ckpt),
        "checkpoint_sha256": spec["best_pt_sha256"],
        "checkpoint_epoch": epoch,
        "p_conditioning": bool(spec["p_conditioning"]),
        "site_fraction": 0.05,
        "per_replica": per_replica,
        "val_macro_rmse_K": val_macro,
        "expected_val_macro_rmse_K": expected,
        "val_delta_K": delta,
        "within_0p05_K": abs(delta) <= 0.05,
        "proves_training_equivalence": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-hashes", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    expected = json.loads(args.expected.read_text())
    _require("mask manifest", _sha256(args.mask_root / "manifest.json"), expected["mask_manifest_sha256"])
    _require("stats", _sha256(args.stats), expected["stats_sha256"])
    hashes = json.loads(args.cache_hashes.read_text())
    _require(
        "cache aggregate",
        str(hashes.get("aggregate_sha256")),
        expected["cache_aggregate_sha256"],
    )
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit("REFUSE: cuda is required for the bridge replay")
    device = torch.device(args.device)
    apply_tf32_policy(allow_tf32=False)
    normalizer = Month2FitOnlyNormalizer(args.stats)

    results = []
    for spec in expected["runs"]:
        results.append(
            evaluate_run(
                spec,
                run_root=args.run_root,
                cache_root=args.cache_root,
                mask_root=args.mask_root,
                normalizer=normalizer,
                device=device,
                batch_size=int(args.batch_size),
            )
        )
    passed = all(row["within_0p05_K"] for row in results)
    report = {
        "kind": "s5_unet_best_5pct_inference_replay",
        "training_equivalence": "not_tested",
        "tolerance_K": 0.05,
        "passed": passed,
        "runs": results,
        "identity": {
            "cache_root": str(args.cache_root),
            "cache_aggregate_sha256": expected["cache_aggregate_sha256"],
            "mask_root": str(args.mask_root),
            "mask_manifest_sha256": expected["mask_manifest_sha256"],
            "stats_sha256": expected["stats_sha256"],
            "allow_tf32": False,
            "require_eval_mask_payload": True,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": passed, "out": str(args.out)}, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
