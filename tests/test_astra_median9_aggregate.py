"""The median9 aggregate keeps both signs and leaves T-only outside the contrasts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "25_astra_median9_aggregate.py"
    spec = importlib.util.spec_from_file_location("astra_median9_aggregate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_contrasts_keep_shrink_and_estimator_separate_and_omit_t_only():
    mod = _load()
    pairs = mod.treatment_pairs(["s5_sparse_unet_pon_seed0", "T_only_const", "GlobalMean"])
    names = {p[0] for p in pairs}
    assert "median9_minus_ao9" in names
    assert "mean9_minus_ao9" in names
    assert "median9_minus_mean9" in names
    assert all(p[1] != "T_only_const" and p[3] != "T_only_const" for p in pairs)
    assert all(p[1] == p[3] for p in pairs)


def test_frame_signs_count_improvements_and_worsenings():
    mod = _load()
    rows = []
    for i, delta in enumerate((-2.0, 3.0, 0.0)):
        rows.append({
            "role": "val_eval", "comparison": "median9_minus_ao9",
            "model_A": "m", "treat_A": "median9", "treat_B": "ao9",
            "replica_id": 0, "sim_id": "s", "frame_index": i,
            "all_rmse_K": delta, "off_rmse_K": delta,
            "all_mae_K": delta, "off_mae_K": delta,
            "all_abs_bias_K": delta, "off_abs_bias_K": delta,
        })
    out = mod.frame_sign_counts(pd.DataFrame(rows))
    cell = out[(out.metric == "all_rmse_K") & (out.replica_id == 0)].iloc[0]
    assert int(cell.n_negative) == 1
    assert int(cell.n_positive) == 1
    assert int(cell.n_zero) == 1
    assert "winner" not in out.columns
