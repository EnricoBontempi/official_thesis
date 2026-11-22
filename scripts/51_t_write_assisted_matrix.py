#!/usr/bin/env python3
"""Write the 12 assisted run candidates and the launcher matrix (temperature study).

Each candidate is the S7 free candidate of the same family (recipe template) with only the run identity, p and seeds
changed; the T-input is switched on by ``16_train_sparse_scientific.py --assisted`` (K, layouts, selection are in
``configs/experiments/month2_temperature_study_protocol.yaml``). Nothing else is altered.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = "/home/bontempi/official_thesis_t_assisted_runs"


def main() -> int:
    free = json.loads((ROOT / "configs/experiments/month2_s7_c149_matrix.json").read_text())
    rows, idx = [], 0
    outdir = ROOT / "configs/experiments/t_assisted"
    outdir.mkdir(exist_ok=True)
    for fam, short in (("sparse_unet", "unet"), ("adapted_glu", "glu")):
        tmpl = yaml.safe_load((ROOT / f"configs/experiments/s7/S7-C149-{short}-on-s1.yaml").read_text())
        for p in (True, False):
            for seed in (0, 1, 2):
                rid = f"TA-{short}-{'on' if p else 'off'}-s{seed}"
                c = copy.deepcopy(tmpl)
                c["experiment_id"] = f"month2_t_assisted_{rid}_v1"
                c["status"] = "t_assisted_study_authorized"
                c["decision"] = {"by": "Enrico", "date": "2026-11-21",
                                 "summary": "Enrico authorized the 12 temperature-assisted trainings (U-Net and GLU x p_on/off x seed 0/1/2, variable K); "
                                            "recipe template = S7 free candidate; T input via --assisted (see month2_temperature_study_protocol.yaml)."}
                tc = c["training_candidate"]
                tc["p_conditioning"] = p
                tc["seed"] = seed
                tc["shuffle_seed"] = seed
                c["artifacts"].update({"results_root_engaging": OUT_ROOT, "run_id": rid, "scientific_run_id": rid})
                c["notes"] = ["Temperature-assisted run: free S7 recipe template; K variable (train K uniform per optimizer update) and the "
                              "common-domain six-K selection criterion are applied by --assisted (new criterion vs S7).",
                              "Resume only from this run's own last.pt at an epoch boundary."]
                c["operational_s7"] = {"campaign": "T-assisted", "scientific_run_id": rid, "resources": "decided at submission",
                                       "pool_frames_per_condition": 149, "pool_total_frames": 2980,
                                       "normalizer": "historical fit-only 760 frozen (not refit on C149)"}
                path = outdir / f"{rid}.yaml"
                path.write_text(yaml.safe_dump(c, sort_keys=False))
                rows.append({"array_index": idx, "run_id": rid, "family": fam, "p_conditioning": p, "seed": seed, "shuffle_seed": seed,
                             "config": str(path.relative_to(ROOT)), "out_dir": f"{OUT_ROOT}/{rid}", "assisted": True})
                idx += 1
    matrix = {"campaign": "T-assisted", "schema": "month2_t_assisted_matrix_v1", "out_root": OUT_ROOT,
              "shared_inputs": free["shared_inputs"], "n_new_trainings": len(rows), "new_runs": rows,
              "protocol": "configs/experiments/month2_temperature_study_protocol.yaml",
              "anchor_manifest": "configs/observations/month2_temperature_anchors_nested_v1.json"}
    (ROOT / "configs/experiments/month2_t_assisted_matrix.json").write_text(json.dumps(matrix, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} candidates and the matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
