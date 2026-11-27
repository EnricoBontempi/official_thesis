#!/usr/bin/env python3
"""Macro diagnostics (bias, centered RMSE, regions, peak, IoU, ...) for the 30 specialists and their original models.

Re-aggregation of EXISTING per-frame CSVs only: no inference, no new evaluation, no GPU.
Reduction (same as the package): mean over frames within a condition, then uniform mean over conditions,
then mean over the three mask replicas. Region columns (g1in/g1out) are NaN where the G1 region is undefined
(R0, N1, N2) and are averaged with NaN skipped. Baselines = the six original p_on models named in
configs/nonideal/specialist_baselines_v1.json, checkpoint best, same cell/role/replica/observations.
Output columns: <metric>__spec, <metric>__base, <metric>__diff (= spec - base; NOTE: opposite sign of the 'gain').
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K", "rel_l2", "r2", "nrmse_train_sigma",
           "high_rmse_K", "high_mae_K", "high_iou", "high_dice", "high_rel_area_error", "non_high_rmse_K", "non_high_mae_K",
           "peak_rmse_K", "peak_mae_K", "peak_abs_error_K", "p95_abs_error_K", "jet_rmse_K", "jet_mae_K",
           "g1in_rmse_K", "g1out_rmse_K", "g1in_bias_K", "g1out_bias_K"]
CELLS = ["R0", "N1", "N2", "G1", "G1-eq", "N1G1"]
ROLES = ["seen_train_eval", "val_eval"]
KEYS = ["cell", "role"]


def macro(df: pd.DataFrame) -> pd.Series:
    per_cond = df.groupby(["replica", "condition"])[METRICS].mean()          # frames -> condition (NaN skipped)
    per_rep = per_cond.groupby("replica").mean()                              # conditions -> replica
    return per_rep.mean()                                                     # replicas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--baseline-per-frame", type=Path, required=True)
    ap.add_argument("--baselines-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    freeze = json.loads((a.dest / "freeze.json").read_text())["runs"]
    bj = json.loads(a.baselines_json.read_text())
    base_ids = {f"{v['family']}/{v['model_seed']}": k for k, v in bj["runs"].items()}
    assert len(base_ids) == 6, base_ids
    bdf = pd.read_csv(a.baseline_per_frame, usecols=lambda c: c in set(METRICS + ["run_id", "cell", "role", "replica", "condition", "checkpoint_kind", "variant"]))
    bdf = bdf[(bdf["checkpoint_kind"] == "best") & (bdf["variant"] == "p_on")]
    brows = {}
    for (rid, cell, role), g in bdf.groupby(["run_id", "cell", "role"]):
        brows[(rid, cell, role)] = macro(g)
    out = []
    for rid, r in sorted(freeze.items()):
        fam, seed = r["family"], r["seed"]
        base_id = base_ids[f"{fam}/{seed}"]
        for ck in ("best", "last"):
            for cell in CELLS:
                for role in ROLES:
                    reps = []
                    for k in (0, 1, 2):
                        p = a.dest / "eval" / rid / ck / "per_frame" / cell / role / f"rep{k}.csv"
                        d = pd.read_csv(p, usecols=lambda c: c in set(METRICS + ["condition", "replica"]))
                        reps.append(d)
                    s = macro(pd.concat(reps))
                    b = brows[(base_id, cell, role)]
                    row = {"run_id": rid, "family": fam, "model_seed": seed, "train_regime": r["regime"], "checkpoint_kind": ck,
                           "cell": cell, "role": role, "baseline_run_id": base_id}
                    for m in METRICS:
                        row[f"{m}__spec"], row[f"{m}__base"] = s[m], b[m]
                        row[f"{m}__diff"] = s[m] - b[m]
                    out.append(row)
    o = pd.DataFrame(out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    o.to_csv(a.out, index=False, float_format="%.6g", compression="gzip" if a.out.suffix == ".gz" else None)
    print("rows", len(o), "cols", o.shape[1])


if __name__ == "__main__":
    main()
