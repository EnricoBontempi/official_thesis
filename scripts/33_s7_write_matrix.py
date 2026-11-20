#!/usr/bin/env python3
"""Write the ten S7 C149 run configs and the matrix JSON.

Each run config is the Opus C149 on/off seed-0 candidate with only these fields
changed: family, p_conditioning, seed == shuffle_seed, ids, authorization key and
paths. Nothing scientific is edited. The generator is committed so the diff
against ``configs/experiments/opus_O-C149-on-s0.yaml`` can be reproduced.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "configs" / "experiments" / "s7"
MATRIX = ROOT / "configs" / "experiments" / "month2_s7_c149_matrix.json"
TEMPLATE = ROOT / "configs" / "experiments" / "opus_O-C149-on-s0.yaml"

ENGAGING_HOME = "/home/bontempi"
OUT_ROOT = f"{ENGAGING_HOME}/official_thesis_s7_c149_runs"
SMOKE_ROOT = f"{ENGAGING_HOME}/official_thesis_s7_smoke_runs"
# Byte-identical copy (20 payload files, aggregate fa1c9358...) of the pool cache
# /orcd/pool/005/bontempi/official_thesis_data/month2_dense_unet_cache_c149. The pool automount
# hung on compute nodes during the S7 smoke (stat of /orcd/pool/005/bontempi/... timed out on
# node3004 and node4102), so the campaign reads the home copy; the hashes are re-verified at launch.
CACHE = f"{ENGAGING_HOME}/official_thesis_data/month2_dense_unet_cache_c149"
POOL_CACHE = "/orcd/pool/005/bontempi/official_thesis_data/month2_dense_unet_cache_c149"

SHARED = {
    "cache_root": CACHE,
    "cache_root_pool_original": POOL_CACHE,
    "cache_hash_file": f"{CACHE}/payload_hashes.json",
    "expect_cache_aggregate_sha256": "fa1c9358deb3bda0313cd74f9ab6855af87bc969c4069a463073d4c4120f03b4",
    "cache_id": "month2_dense_unet_cache_opus_c149_v1",
    "cache_counts": {"train_fit": 2980, "seen_train_eval": 220, "val_eval": 123},
    "mask_root": f"{ENGAGING_HOME}/projects/official_thesis_s5/data/processed/month2_sparse_eval_masks",
    "mask_manifest_file_sha256": "2e2d8ff6edabf62ad56702a155140d064c0b4014d06ee96debf52811efa5c75e",
    "mask_aggregate_payload_sha256": "7be78f35ffea2ebc598adce8cec39dc719339aac17ecadde3137f2064891ffeb",
    "stats_relpath": "data/manifests/train_norm_stats_month2_fit.json",
    "stats_sha256": "678263fb4cb2598350b8fd69a71ce4843f747aa61c19556cf7f0f9c5c3ceac0e",
}

NEW_RUNS = [
    # array index, run id, family, p_on, seed
    (0, "S7-C149-unet-on-s1", "sparse_unet", True, 1),
    (1, "S7-C149-unet-off-s1", "sparse_unet", False, 1),
    (2, "S7-C149-unet-on-s2", "sparse_unet", True, 2),
    (3, "S7-C149-unet-off-s2", "sparse_unet", False, 2),
    (4, "S7-C149-glu-on-s0", "adapted_glu", True, 0),
    (5, "S7-C149-glu-off-s0", "adapted_glu", False, 0),
    (6, "S7-C149-glu-on-s1", "adapted_glu", True, 1),
    (7, "S7-C149-glu-off-s1", "adapted_glu", False, 1),
    (8, "S7-C149-glu-on-s2", "adapted_glu", True, 2),
    (9, "S7-C149-glu-off-s2", "adapted_glu", False, 2),
]

REUSED = [
    {
        "run_id": "O-C149-on-s0",
        "family": "sparse_unet",
        "p_conditioning": True,
        "seed": 0,
        "training_job": 29645053,
        "training_sha": "d6332000491ba0e8983b9e7fa09b34badb191ea4",
        "training_dir": f"{ENGAGING_HOME}/official_thesis_opus_c149_attempt2_runs/O-C149-on-s0",
        "action": "reuse_not_relaunched",
    },
    {
        "run_id": "O-C149-off-s0",
        "family": "sparse_unet",
        "p_conditioning": False,
        "seed": 0,
        "training_job": 29645054,
        "training_sha": "d6332000491ba0e8983b9e7fa09b34badb191ea4",
        "training_dir": f"{ENGAGING_HOME}/official_thesis_opus_c149_attempt2_runs/O-C149-off-s0",
        "action": "reuse_not_relaunched",
    },
]


def build_config(run_id: str, family: str, p_on: bool, seed: int) -> dict:
    cfg = yaml.safe_load(TEMPLATE.read_text())
    cfg["experiment_id"] = f"month2_s7_{run_id}_v1"
    cfg["status"] = "s7_c149_free_authorized"
    cfg.pop("opus_submission_authorized", None)
    cfg["s7_submission_authorized"] = True
    cfg["family"] = family
    cfg["decision"] = {
        "by": "Enrico",
        "date": "2026-11-20",
        "summary": (
            "Enrico approved the temperature-free S7 comparison: U-Net masked and "
            "deterministic adapted GLU, p_on/p_off, seeds 0/1/2, C149 pool, same Opus "
            "recipe. Ten new trainings; two U-Net C149 seed-0 runs reused."
        ),
    }
    if family == "adapted_glu":
        cfg["model"]["path"] = "configs/models/glu_minimal.yaml"
    tc = cfg["training_candidate"]
    tc["family"] = family
    tc["p_conditioning"] = bool(p_on)
    tc["seed"] = int(seed)
    tc["shuffle_seed"] = int(seed)
    tc["init"] = f"fresh_seed{seed}_no_prior_weights_no_warm_start"
    # Per-attempt internal cap; the launcher always passes --wall-hours explicitly.
    tc["wall_time_cap_hours"] = 23.0
    cfg.pop("operational_from_s5_recipe", None)
    cfg["operational_s7"] = {
        "campaign": "S7-C149-free",
        "scientific_run_id": run_id,
        "resources": "decided at submission; see launcher EXPECT_* variables",
        "microbatch_size": None,
        "pool_frames_per_condition": 149,
        "pool_total_frames": 2980,
        "normalizer": "historical fit-only 760 frozen (not refit on C149)",
        "reference_unet_runs_reused": ["O-C149-on-s0", "O-C149-off-s0"],
    }
    cfg["artifacts"] = {
        "results_root_engaging": OUT_ROOT,
        "run_id": run_id,
        "scientific_run_id": run_id,
        "warm_start": False,
        "refuse_overwrite_existing_run_dir": True,
        "refuse_load_s5_or_wave1_checkpoints": True,
    }
    cfg["notes"] = [
        "S7 free campaign run. K=0, no temperature in the forward pass.",
        "Fresh init; shuffle_seed == model seed; train_mask_seed 20261118; eval_mask_seed 20261116.",
        "Scientific recipe identical to O-C149-{on,off}-s0 (Opus attempt2, SHA d633200).",
        "Preemption: resume only from this run's own last.pt at an epoch boundary.",
    ]
    return cfg


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for idx, run_id, family, p_on, seed in NEW_RUNS:
        cfg = build_config(run_id, family, p_on, seed)
        path = OUT_DIR / f"{run_id}.yaml"
        path.write_text(
            "# Generated by scripts/33_s7_write_matrix.py from the Opus C149 candidate.\n"
            + yaml.safe_dump(cfg, sort_keys=False)
        )
        runs.append(
            {
                "array_index": idx,
                "run_id": run_id,
                "family": family,
                "p_conditioning": p_on,
                "seed": seed,
                "shuffle_seed": seed,
                "config": f"configs/experiments/s7/{run_id}.yaml",
                "out_dir": f"{OUT_ROOT}/{run_id}",
            }
        )
    matrix = {
        "schema": "s7_c149_matrix_v1",
        "campaign": "S7-C149-free",
        "decision": "Enrico 2026-11-20; roadmapsparse.md S7; plan commit 5f8842f",
        "out_root": OUT_ROOT,
        "smoke_root": SMOKE_ROOT,
        "shared_inputs": SHARED,
        "new_runs": runs,
        "reused_runs": REUSED,
        "n_new_trainings": len(runs),
    }
    MATRIX.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(runs)} configs under {OUT_DIR} and {MATRIX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
