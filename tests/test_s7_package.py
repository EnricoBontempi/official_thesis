"""S7 aggregation: refusal of an incomplete matrix and the shape of the descriptive tables."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from combustion_recon.evaluation import s7_package as pkg
from combustion_recon.evaluation.opus_o5_plan import CHECKPOINTS, COVERAGES, ROLES


def _matrix():
    runs = []
    for fam, p, s in itertools.product(("sparse_unet", "adapted_glu"), (True, False), (0, 1, 2)):
        runs.append({"run_id": f"{fam}-{p}-{s}", "family": fam, "p_conditioning": p, "model_seed": s,
                     "reused_not_relaunched": False, "training_git_sha": "x", "status": "stopped_early_plateau",
                     "epochs_completed": 1, "extended_past_initial_tranche": False,
                     "checkpoints": {c: {"reuse_forward_from": None} for c in CHECKPOINTS}})
    shared = {"c149_cache_aggregate_sha256": "a", "c149_cache_id": "b", "stats_sha256": "c",
              "eval_mask_manifest_file_sha256": "d", "eval_mask_aggregate_payload_sha256": "e"}
    return {"runs": runs, "shared_inputs": shared}


def _frames(matrix):
    rng = np.random.default_rng(0)
    rows = []
    for e in matrix["runs"]:
        for ck, role, cov in itertools.product(CHECKPOINTS, ROLES, COVERAGES):
            for rep in (0, 1):
                for sim in ("a", "b"):
                    for fr in (0, 1):
                        row = {"run_id": e["run_id"], "family": e["family"], "p_conditioning": e["p_conditioning"],
                               "model_seed": e["model_seed"], "checkpoint_kind": ck, "epoch": 10, "role": role,
                               "coverage": cov, "replica": rep, "sim_id": sim, "frame_index": fr}
                        for m in pkg.ALL_METRICS:
                            row[m] = float(rng.uniform(1, 5))
                        rows.append(row)
    return pd.DataFrame(rows)


def test_incomplete_matrix_is_refused_not_aggregated(tmp_path):
    table, runs, issues = pkg.load_and_check(tmp_path, _matrix(), matrix_sha256="m", allowed_eval_shas=["s"])
    assert table.empty and len(runs) == 12
    assert sum("COMPLETE.json missing" in i for i in issues) == 12


def test_tables_have_one_row_per_setting_and_no_ranking():
    m = _matrix()
    t = pkg.build_tables(_frames(m))
    assert len(t["global_by_setting"]) == 12 * 3 * 3 * 3
    assert not {"rank", "winner"} & set(t["global_by_setting"].columns)
    s = pkg.family_p_summary(t["global_by_setting"])
    assert (s["rmse_K_n_seeds"] == 3).all() and len(s) == 2 * 2 * 3 * 3 * 3
    paired = pkg.paired_contrasts(t["per_condition"])
    assert set(paired["paired_summary"]["contrast"]) == {"family", "p"}
    assert paired["paired_summary"]["label"].nunique() == 12
    d = pkg.degradations(t["global_by_setting"], t["per_condition"])
    assert {"delta_cov01_minus_cov05_K", "delta_cov20_minus_cov05_K"} <= set(d["coverage_degradation"].columns)
    assert len(d["worst_val_condition"]) == 12 * 3 * 3


def test_duplicate_keys_and_mask_hash_conflicts_are_reported():
    m = _matrix()
    f = _frames(m)
    f["mask_hash"] = "h"
    f["role"] = f["role"].astype(str)
    dup = pd.concat([f, f.iloc[:1]], ignore_index=True)
    issues = pkg.cross_checks(dup, m, None)
    assert any("duplicated" in i for i in issues)
    f.loc[0, "mask_hash"] = "other"
    assert any("several mask hashes" in i for i in pkg.cross_checks(f, m, None))
