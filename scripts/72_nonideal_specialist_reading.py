#!/usr/bin/env python3
"""Reading tables for the 30 p_on specialists, derived only from tables/per_run_scenario.csv and training_diagnostics.csv.

Sign convention (same as the package): gain = RMSE_original - RMSE_specialist in K, same scenario, same observations
(positive: specialist lower). clean cost = RMSE_specialist - RMSE_original on R0 (positive: specialist worse).
Nothing here is a new evaluation and no checkpoint other than the protocol's best (and the protocol's last) is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REG = ["N1", "N2", "G1", "G1-eq", "N1G1"]
CELLS = ["R0", "N1", "N2", "G1", "G1-eq", "N1G1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.tables / "per_run_scenario.csv")
    a.out.mkdir(parents=True, exist_ok=True)
    d["is_diag"] = d["cell"] == d["train_regime"]
    # per run: own-scenario gain, clean cost, per checkpoint and role (the three seeds stay visible)
    diag = d[d["is_diag"]][["run_id", "family", "model_seed", "train_regime", "checkpoint_kind", "role", "base_rmse_K", "rmse_K", "gain_vs_baseline_K",
                             "baseline_deterioration_K", "specialist_deterioration_K", "fraction_of_baseline_deterioration_recovered"]]
    r0 = d[d["cell"] == "R0"][["run_id", "checkpoint_kind", "role", "clean_cost_K"]]
    diag = diag.merge(r0, on=["run_id", "checkpoint_kind", "role"])
    diag.to_csv(a.out / "own_scenario_by_run.csv", index=False, float_format="%.4f")

    def agg(g: pd.DataFrame, col: str) -> pd.Series:
        return pd.Series({"mean": g[col].mean(), "min": g[col].min(), "max": g[col].max(), "sd": g[col].std(ddof=1), "n_pos": int((g[col] > 0).sum()), "n": len(g)})

    rows = []
    for (fam, ck, ro, reg), g in diag.groupby(["family", "checkpoint_kind", "role", "train_regime"]):
        for col in ("gain_vs_baseline_K", "clean_cost_K"):
            s = agg(g, col)
            rows.append({"family": fam, "checkpoint": ck, "role": ro, "train_regime": reg, "quantity": col, **s.to_dict()})
    pd.DataFrame(rows).to_csv(a.out / "own_scenario_summary.csv", index=False, float_format="%.4f")
    # transfer matrix: mean gain over the three seeds, every train regime x eval cell
    t = d.groupby(["family", "checkpoint_kind", "role", "train_regime", "cell"])["gain_vs_baseline_K"].agg(["mean", "min", "max"]).reset_index()
    t.to_csv(a.out / "transfer_matrix_gain.csv", index=False, float_format="%.4f")
    # baseline deterioration per cell (original models, same observations): the size of the problem the specialists address
    b = d.drop_duplicates(["family", "model_seed", "cell", "role"]).groupby(["family", "role", "cell"])["baseline_deterioration_K"].mean().reset_index()
    b.to_csv(a.out / "baseline_deterioration.csv", index=False, float_format="%.4f")
    # descriptive decomposition of the 5x6 matrix of mean gains (train regime x eval scenario): sums of squares of row means,
    # column means and the remainder. It describes the 30 numbers; it is not a significance test.
    import numpy as np
    dec = []
    for (fam, ck, ro), g in t.groupby(["family", "checkpoint_kind", "role"]):
        M = np.array([[g[(g.train_regime == r) & (g.cell == c)]["mean"].iloc[0] for c in CELLS] for r in REG])
        gm = M.mean(); row = M.mean(1) - gm; col = M.mean(0) - gm
        ssr, ssc, sst = 6 * (row ** 2).sum(), 5 * (col ** 2).sum(), ((M - gm) ** 2).sum()
        prem = [M[i, CELLS.index(r)] - np.mean([M[i, k] for k in range(6) if k != CELLS.index(r)]) for i, r in enumerate(REG)]
        dec.append({"family": fam, "checkpoint": ck, "role": ro, "pct_variance_train_regime_rows": 100 * ssr / sst, "pct_variance_eval_scenario_columns": 100 * ssc / sst,
                    "pct_variance_remainder": 100 * (sst - ssr - ssc) / sst, "mean_own_scenario_premium_K": float(np.mean(prem)),
                    **{f"own_scenario_premium_K_{r}": float(v) for r, v in zip(REG, prem)}})
    pd.DataFrame(dec).to_csv(a.out / "transfer_row_vs_column_decomposition.csv", index=False, float_format="%.3f")
    print("wrote", sorted(p.name for p in a.out.iterdir()))


if __name__ == "__main__":
    main()
