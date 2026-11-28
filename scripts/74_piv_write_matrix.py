#!/usr/bin/env python3
"""Write the 35 main-campaign training candidates (T1-T7) and the run matrix. Deterministic; refuses to overwrite.

Recipe = frozen S7 one (same templates as the specialists); only differences: the observation block and, for the GLU,
microbatch_size 1 with accumulation to batch 8 for ALL GLU cells (uniform recipe). Observation spec: v1.1 (hash recorded).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from combustion_recon.data.nonideal_observations import file_sha256  # noqa: E402
from combustion_recon.training.piv_training import CELLS, PIV_SPEC_REL  # noqa: E402
from combustion_recon.training.sparse_scientific_loop import assert_sparse_scientific_candidate_contract  # noqa: E402

CFG_DIR = ROOT / "configs/experiments/piv_campaign"
MATRIX = ROOT / "configs/experiments/piv_campaign_matrix.json"
OUT_ROOT = "/home/bontempi/official_thesis_piv_runs"
SMOKE_ROOT = "/home/bontempi/official_thesis_piv_smoke_runs"
TEMPLATES = {"sparse_unet": "configs/experiments/s7/S7-C149-unet-on-s1.yaml", "adapted_glu": "configs/experiments/s7/S7-C149-glu-on-s0.yaml"}
FAM = {"sparse_unet": "unet", "adapted_glu": "glu"}
GROUP = {"T1": "T1", "T2": "T2", "T3": "T3", "T4": "T4", "T5-ideal": "T5", "T5-piv": "T5", "T6-ideal": "T6", "T6-piv": "T6", "T7": "T7"}
SEEDS = {c: (range(5) if CELLS[c][0] == "sparse_unet" else range(3)) for c in CELLS}
ORDER = [c for c in ("T5-piv", "T5-ideal", "T6-piv", "T6-ideal", "T7", "T2", "T1", "T4", "T3")]


def main() -> int:
    if MATRIX.exists():
        raise SystemExit(f"refuse to overwrite {MATRIX}")
    CFG_DIR.mkdir(parents=True, exist_ok=False)
    spec_sha = file_sha256(ROOT / PIV_SPEC_REL)
    shared = json.loads((ROOT / "configs/experiments/nonideal_specialist_matrix.json").read_text())["shared_inputs"]
    runs = []
    for cell in ORDER:
        fam, obs = CELLS[cell]
        for seed in SEEDS[cell]:
            run_id = f"PIV-{cell}-{FAM[fam]}-s{seed}"
            cand = yaml.safe_load((ROOT / TEMPLATES[fam]).read_text())
            cand["experiment_id"] = f"piv_campaign_{run_id}_v1"
            cand["status"] = "piv_campaign_authorized"
            for k in ("s7_submission_authorized", "operational_s7"):
                cand.pop(k, None)
            cand["piv_submission_authorized"] = True
            cand["decision"] = {"by": "Enrico", "date": "2026-11-28", "summary": "Mission 02 main campaign T1-T7: 35 trainings from scratch (decisione_R2 + autonomy mandate); T8-T11 not authorised."}
            tc = cand["training_candidate"]
            tc["seed"] = seed; tc["shuffle_seed"] = seed
            tc["init"] = f"fresh_seed{seed}_no_prior_weights_no_warm_start"
            tc["microbatch_size"] = 1 if fam == "adapted_glu" else None
            cand["artifacts"]["results_root_engaging"] = OUT_ROOT
            cand["artifacts"]["run_id"] = run_id; cand["artifacts"]["scientific_run_id"] = run_id
            cand["piv_campaign"] = {"cell": cell, "group": GROUP[cell], "observation": obs, "spec": PIV_SPEC_REL, "spec_sha256": spec_sha,
                                    "train_role": "train", "selection_role": "selection",
                                    "selection": "VAL of the SAME observation, selection namespace, 3 replicas, first absolute minimum; last and curve always kept",
                                    "realization_key": "(namespace, role seed, operator, component, sim_id, frame, epoch|replica); no model seed, no family"}
            cand["notes"] = ["Mission 02 virtual PIV, main campaign. From scratch, C149 pool, fit-only760 normaliser.",
                             "GLU: microbatch 1, gradient accumulation to batch 8 (95 updates/epoch) in every GLU cell.",
                             "Preemption: resume only from this run's own last.pt at an epoch boundary."]
            assert_sparse_scientific_candidate_contract(cand)
            path = CFG_DIR / f"{run_id}.yaml"
            path.write_text(yaml.safe_dump(cand, sort_keys=False, width=100))
            runs.append({"array_index": len(runs), "config": str(path.relative_to(ROOT)), "family": fam, "out_dir": f"{OUT_ROOT}/{run_id}",
                         "p_conditioning": True, "run_id": run_id, "seed": seed, "shuffle_seed": seed, "cell": cell, "group": GROUP[cell], "observation": obs})
    assert len(runs) == 35, len(runs)
    MATRIX.write_text(json.dumps({"schema": "piv_campaign_matrix_v1", "campaign": "PIV-main-T1-T7", "n_new_trainings": 35, "new_runs": runs, "out_root": OUT_ROOT,
                                   "smoke_root": SMOKE_ROOT, "shared_inputs": shared, "observation_spec": PIV_SPEC_REL, "observation_spec_sha256": spec_sha,
                                   "conditional_groups_not_authorised": ["T8", "T9", "T10", "T11"]}, indent=1, sort_keys=True) + "\n")
    print("wrote", len(runs), "runs; spec", spec_sha)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
