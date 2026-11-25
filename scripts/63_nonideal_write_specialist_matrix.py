#!/usr/bin/env python3
"""Write the 30 specialist training candidates, the run matrix, the protocol and the baseline manifest.

2 families (U-Net masked, adapted GLU) x p_on only x seeds 0/1/2 x regimes N1, N2, G1, G1-eq, N1G1.
The recipe is the frozen S7 one (templates: S7-C149-unet-on-s1 / S7-C149-glu-on-s0, seed overridden);
the only differences are the regime block and, for the GLU, microbatch_size 1 (gradient accumulation to batch 8).
Deterministic; refuses to overwrite.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from combustion_recon.data.nonideal_observations import file_sha256  # noqa: E402
from combustion_recon.training.nonideal_specialist import EVAL_SEED, REGIMES, SELECTION_SEED, TRAIN_SEED  # noqa: E402

CFG_DIR = ROOT / "configs/experiments/nonideal_specialist"
MATRIX = ROOT / "configs/experiments/nonideal_specialist_matrix.json"
PROTOCOL = ROOT / "configs/nonideal/specialist_protocol_v1.yaml"
BASELINES = ROOT / "configs/nonideal/specialist_baselines_v1.json"
SPEC = "configs/nonideal/degradation_spec_v1_12models.yaml"
S7_MATRIX = ROOT / "configs/experiments/month2_s7_c149_matrix.json"
EVAL_MATRIX = ROOT / "configs/experiments/month2_s7_c149_eval_matrix.json"
OUT_ROOT = "/home/bontempi/official_thesis_nonideal_spec_runs"
SMOKE_ROOT = "/home/bontempi/official_thesis_nonideal_spec_smoke_runs"
TEMPLATES = {"sparse_unet": "configs/experiments/s7/S7-C149-unet-on-s1.yaml", "adapted_glu": "configs/experiments/s7/S7-C149-glu-on-s0.yaml"}
FAM = {"sparse_unet": "unet", "adapted_glu": "glu"}
BASELINE_RUN = {("sparse_unet", 0): "O-C149-on-s0", ("sparse_unet", 1): "S7-C149-unet-on-s1", ("sparse_unet", 2): "S7-C149-unet-on-s2",
                ("adapted_glu", 0): "S7-C149-glu-on-s0", ("adapted_glu", 1): "S7-C149-glu-on-s1", ("adapted_glu", 2): "S7-C149-glu-on-s2"}
ORDER = [("adapted_glu", r, s) for r in REGIMES for s in (0, 1, 2)] + [("sparse_unet", r, s) for r in REGIMES for s in (0, 1, 2)]


def rid(fam, regime, seed):
    return f"NI-{FAM[fam]}-on-{regime.replace('-', '')}-s{seed}"


def main() -> int:
    for p in (MATRIX, PROTOCOL, BASELINES):
        if p.exists():
            raise SystemExit(f"refuse to overwrite {p}")
    CFG_DIR.mkdir(parents=True, exist_ok=False)
    spec_sha = file_sha256(ROOT / SPEC)
    s7 = json.loads(S7_MATRIX.read_text())
    ev = json.loads(EVAL_MATRIX.read_text())
    runs = []
    for i, (fam, regime, seed) in enumerate(ORDER):
        cand = yaml.safe_load((ROOT / TEMPLATES[fam]).read_text())
        run_id = rid(fam, regime, seed)
        cand["experiment_id"] = f"nonideal_specialist_{run_id}_v1"
        cand["status"] = "nonideal_specialist_authorized"
        cand.pop("s7_submission_authorized", None)
        cand.pop("operational_s7", None)
        cand["nonideal_submission_authorized"] = True
        cand["decision"] = {"by": "Enrico", "date": "2026-11-24",
                            "summary": "Specialist retraining on one synthetic degradation regime: 2 families x p_on x seeds 0/1/2 x regimes N1, N2, G1, G1-eq, N1G1 = 30 trainings from scratch; original clean selection kept."}
        tc = cand["training_candidate"]
        tc["seed"] = seed
        tc["shuffle_seed"] = seed
        tc["init"] = f"fresh_seed{seed}_no_prior_weights_no_warm_start"
        tc["microbatch_size"] = 1 if fam == "adapted_glu" else None
        cand["artifacts"]["results_root_engaging"] = OUT_ROOT
        cand["artifacts"]["run_id"] = run_id
        cand["artifacts"]["scientific_run_id"] = run_id
        cand["nonideal_specialist"] = {
            "regime": regime,
            "clean_fraction": 0,
            "degradation_spec": SPEC,
            "degradation_spec_sha256": spec_sha,
            "train_degradation_seed": TRAIN_SEED,
            "selection_degradation_seed": SELECTION_SEED,
            "evaluation_degradation_seed_never_used_for_training": EVAL_SEED,
            "realization_key": "(sim_id, frame_index, mask_epoch); no model seed, no family",
            "baseline_original_run_id": BASELINE_RUN[(fam, seed)],
            "selection": "original clean VAL 5% macro, 3 replicas (first absolute minimum); specialist VAL recorded only",
        }
        cand["notes"] = ["Specialist retraining, synthetic regime (not a PIV validation). From scratch, C149, fit-only760 normalizer.",
                         "GLU: microbatch 1 with gradient accumulation to effective batch 8 (95 updates/epoch).",
                         "Preemption: resume only from this run's own last.pt at an epoch boundary."]
        path = CFG_DIR / f"{run_id}.yaml"
        path.write_text(yaml.safe_dump(cand, sort_keys=False, width=100))
        runs.append({"array_index": i, "config": str(path.relative_to(ROOT)), "family": fam, "out_dir": f"{OUT_ROOT}/{run_id}",
                     "p_conditioning": True, "run_id": run_id, "seed": seed, "shuffle_seed": seed, "regime": regime})
    # smoke-only identity (R0) specialists: the specialist code path with nothing degraded (never part of the 30)
    smoke_dir = CFG_DIR / "smoke"
    smoke_dir.mkdir()
    smoke_runs = []
    for j, (fam, seed) in enumerate((("sparse_unet", 1), ("adapted_glu", 0))):
        cand = yaml.safe_load((CFG_DIR / f"{rid(fam, 'N1G1', seed)}.yaml").read_text())
        run_id = f"NI-{FAM[fam]}-on-R0smoke-s{seed}"
        cand["experiment_id"] = f"nonideal_specialist_{run_id}_v1"
        cand["nonideal_specialist"]["regime"] = "R0"
        cand["artifacts"]["run_id"] = run_id
        cand["artifacts"]["scientific_run_id"] = run_id
        (smoke_dir / f"{run_id}.yaml").write_text(yaml.safe_dump(cand, sort_keys=False, width=100))
        smoke_runs.append({"array_index": j, "config": f"configs/experiments/nonideal_specialist/smoke/{run_id}.yaml", "family": fam,
                           "out_dir": f"{SMOKE_ROOT}/{run_id}", "p_conditioning": True, "run_id": run_id, "seed": seed, "shuffle_seed": seed, "regime": "R0"})
    shared = {k: v for k, v in s7["shared_inputs"].items()}
    (ROOT / "configs/experiments/nonideal_specialist_smoke_matrix.json").write_text(json.dumps(
        {"schema": "nonideal_specialist_smoke_matrix_v1", "new_runs": smoke_runs, "out_root": SMOKE_ROOT, "shared_inputs": shared,
         "note": "smoke only: identity specialist path, never scientific"}, indent=1, sort_keys=True) + "\n")
    matrix = {"schema": "nonideal_specialist_matrix_v1", "campaign": "NI-specialist-retraining",
              "decision": "Enrico 2026-11-24: 2 families x p_on x 3 seeds x 5 regimes = 30 trainings (supersedes the earlier 'at most 6 retrainings' limit explicitly)",
              "n_new_trainings": len(runs), "new_runs": runs, "out_root": OUT_ROOT, "smoke_root": SMOKE_ROOT,
              "shared_inputs": shared, "degradation_spec": SPEC, "degradation_spec_sha256": spec_sha}
    MATRIX.write_text(json.dumps(matrix, indent=1, sort_keys=True) + "\n")
    protocol = {
        "format": "nonideal_specialist_protocol_v1", "status": "frozen_before_runs", "authorized_by": "Enrico, 2026-11-24",
        "matrix": {"families": ["sparse_unet", "adapted_glu"], "parameters": ["p_on"], "seeds": [0, 1, 2], "regimes": list(REGIMES),
                   "n_trainings": 30, "baseline": "original clean models of the same family/seed (not retrained)",
                   "excluded": ["mixed training", "p_off", "assisted", "new families", "T input", "thermocouples", "post hoc correction"]},
        "regime_definitions": "operators and levels frozen in " + SPEC + " (N1 0.5 m/s, N2 1.5 m/s, G1 union of discs 25%, G1-eq same count uniform, N1G1)",
        "training": {"from_scratch": True, "pool": "C149 (2980 frames)", "normalizer": "historical fit-only760, frozen",
                     "recipe": "S7: batch 8, 760 presentations and 95 updates/epoch, AdamW lr 1e-3 wd 1e-4 clip 1, warmup 950 updates, fp32, no AMP/TF32/compile",
                     "init_seed_shuffle_seed": "seed 0/1/2 as the originals", "train_masks": "historical TRAIN 5% recipe (epoch-dependent), degradation applied afterwards",
                     "regime_fixed_per_run": True, "clean_fraction": 0},
        "realizations": {"train_namespace_seed": TRAIN_SEED, "selection_namespace_seed": SELECTION_SEED, "evaluation_namespace_seed": EVAL_SEED,
                         "key": "(sim_id, frame_index, mask epoch)", "paired_across_families_and_seeds": True,
                         "pairing": "N1/N2 same standardized noise (N2 = 3 x N1); G1/N1G1 same region; G1-eq removes the same count uniformly"},
        "glu_variable_tokens": {"policy": "microbatch_size 1 + gradient accumulation to effective batch 8, for ALL five GLU specialists",
                                "loss": "per-microbatch MSE scaled by 1/8 (equal weight per sample)", "clip_and_step": "once per effective batch",
                                "counters": "update/warmup/scheduler counted per effective batch (95 per epoch)",
                                "not_done": ["truncation to the batch minimum", "padding", "filling tokens", "regrouping samples"],
                                "difference_from_original_glu": "original GLU training used one forward of batch 8 (microbatch null); numerics differ only by summation order"},
        "selection_and_stop": {"primary": "first absolute minimum of the clean VAL 5% macro RMSE, mean of 3 replicas (as S7)",
                               "scheduler_and_early_stop": "driven by the same clean metric as S7",
                               "min_epochs": 200, "patience_evals": 15, "rel_improvement": 0.001, "reset_on_lr_reduce": True, "eval_every": 10,
                               "initial_tranche_epochs": 1000, "single_continuation_to_epochs": 1200,
                               "after_1200_without_plateau": "budget_exhausted, results kept, no further automatic extension",
                               "keep": ["best", "last"], "specialist_val_metric": "recorded only on fixed SELECTION-namespace realizations; never used for scheduler, early stop or best",
                               "statement": "We compare specialist augmentations under the ORIGINAL selection, not the optimal checkpoint per scenario."},
        "evaluation": {"checkpoints": ["best", "last"], "cells": ["R0", "N1", "N2", "G1", "G1-eq", "N1G1"], "roles": ["seen_train_eval", "val_eval"], "replicas": [0, 1, 2],
                       "spec": SPEC, "batching": "U-Net 8, GLU 1 (as Block II)", "reference": "original clean models, Block II package",
                       "replay_gate": "R0 macro within 0.05 K of the training-loop metric at the evaluated epoch (val 3 replicas, seen replica 0)"},
        "stops": ["no automatic further training", "no mixed or new configurations after the result", "A/B/C suspended", "TEST closed"],
    }
    PROTOCOL.write_text(yaml.safe_dump(protocol, sort_keys=False, width=110))
    eruns = {r["run_id"]: r for r in ev["runs"]}
    pkg = ROOT / "results/nonideal/block2/package/package_manifest.json"
    base = {"format": "nonideal_specialist_baselines_v1", "block2_commit": "d4f2e3a58da9c95bb2259f2130cfbe41727df955",
            "block2_package_index": "results/nonideal/block2/package/INDEX.md",
            "block2_package_manifest_sha256": file_sha256(pkg),
            "block2_per_frame_csv": "/home/bontempi/official_thesis_nonideal_b2_pkg/package/tables/per_frame_all.csv.gz",
            "block2_per_frame_csv_sha256": "cb96687cf3614c5393bc9658ec3469980313413f495f32930fd2e810ffc070b3",
            "block2_eval_root": "/home/bontempi/official_thesis_nonideal_b2_eval", "spec": "configs/nonideal/degradation_spec_v1_12models.yaml",
            "baselines_not_retrained": True, "runs": {}}
    for (fam, seed), b in BASELINE_RUN.items():
        best = eruns[b]["checkpoints"]["best"]
        base["runs"][b] = {"family": fam, "model_seed": seed, "epoch": best["epoch"], "best_file_sha256": best["file_sha256"],
                           "best_weights_sha256": best["weights_sha256"], "training_dir": eruns[b]["training_dir"]}
    BASELINES.write_text(json.dumps(base, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(runs)} configs, matrix, protocol, baseline manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
