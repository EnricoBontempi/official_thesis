"""Gate for the median9 second pass: primary free/oracle/AO9, not a new estimator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combustion_recon.evaluation.astra_level import (
    prediction_tables,
    primary_level_replay,
    t9_site_indices,
    t_only_constant_row,
    treat_frame,
)


def _pair(n_sim=2, n_frame=2, rmse=10.0, applied=1.0, method="m"):
    rows = []
    for sim in range(n_sim):
        for frame in range(n_frame):
            for treatment, delta in (("free", 0.0), ("oracle", applied), ("ao9", applied * 0.9)):
                rows.append(
                    {
                        "method": method,
                        "role": "val_eval",
                        "replica_id": 0,
                        "sim_id": f"s{sim}",
                        "frame_index": frame,
                        "treatment": treatment,
                        "all_rmse_K": rmse + (0.0 if treatment == "free" else 0.2),
                        "delta_applied_K": delta,
                    }
                )
    return pd.DataFrame(rows)


def test_replay_passes_on_identical_rows_in_any_order():
    ref = _pair()
    cand = ref.sample(frac=1.0, random_state=0).reset_index(drop=True)
    out = primary_level_replay(
        cand, ref, roles=("val_eval",), replicas=(0,), expected_frames={"val_eval": 4}
    )
    assert out["replay_executed"] and out["validated"]
    assert out["max_abs_macro_rmse_K"] == 0.0
    assert out["max_abs_frame_rmse_K"] == 0.0
    assert len(out["frames"]) == len(ref)


def test_empty_reference_is_not_a_replay():
    out = primary_level_replay(
        _pair(), pd.DataFrame(), roles=("val_eval",), replicas=(0,), expected_frames={"val_eval": 4}
    )
    assert out["replay_executed"] is False
    assert out["validated"] is False
    assert out["reason"] == "empty_reference"


def test_macro_and_frame_tolerance_and_offset():
    ref = _pair()
    shifted = ref.copy()
    shifted["all_rmse_K"] = shifted["all_rmse_K"] + 0.1
    bad = primary_level_replay(
        shifted, ref, roles=("val_eval",), replicas=(0,), expected_frames={"val_eval": 4}
    )
    assert bad["replay_executed"] and not bad["validated"]
    assert bad["max_abs_macro_rmse_K"] == pytest.approx(0.1)

    dropped = ref.iloc[:-1]
    missing = primary_level_replay(
        dropped, ref, roles=("val_eval",), replicas=(0,), expected_frames={"val_eval": 4}
    )
    assert missing["replay_executed"] and not missing["validated"]

    offset = ref.copy()
    offset["delta_applied_K"] = offset["delta_applied_K"] + 1.0
    off = primary_level_replay(
        offset, ref, roles=("val_eval",), replicas=(0,), expected_frames={"val_eval": 4}
    )
    assert off["replay_executed"] and not off["validated"]
    assert off["max_abs_delta_applied_K"] == 1.0


def test_mean9_rows_are_outside_the_primary_gate():
    ref = _pair()
    extra = ref.copy()
    med = ref[ref.treatment == "free"].copy()
    med["treatment"] = "median9"
    med["delta_applied_K"] = 3.0
    cand = pd.concat([extra, med], ignore_index=True)
    out = primary_level_replay(
        cand, ref, roles=("val_eval",), replicas=(0,), expected_frames={"val_eval": 4}
    )
    assert out["validated"]
    assert set(out["frames"]["treatment"]) == {"free", "oracle", "ao9"}


def test_prediction_tables_keep_primary_rows_and_split_shrink():
    rng = np.random.default_rng(1)
    true = 1800.0 + rng.normal(0, 30, (128, 128))
    pred = true - 40.0
    sites = t9_site_indices()
    metrics, residuals = prediction_tables(pred, true, sites, {"method": "m"})
    primary = treat_frame(pred, true, sites)
    by = {row["treatment"]: row for row in metrics}
    assert set(by) == {"free", "oracle", "ao9", "mean9", "median9"}
    for old in primary:
        assert by[old["treatment"]]["all_rmse_K"] == old["all_rmse_K"]
        assert by[old["treatment"]]["delta_applied_K"] == old["delta_applied_K"]
    assert by["mean9"]["delta_applied_K"] == 40.0
    assert by["median9"]["delta_applied_K"] == 40.0
    assert by["ao9"]["delta_applied_K"] == 36.0
    assert by["mean9"]["all_centered_rmse_K"] == by["free"]["all_centered_rmse_K"]
    assert len(residuals) == 9
    assert [row["anchor_id"] for row in residuals] == list(range(9))
    assert sum(row["ao9_contribution_K"] for row in residuals) == by["ao9"]["delta_applied_K"]


def test_t_only_does_not_receive_median9():
    true = np.full((128, 128), 2000.0)
    true.ravel()[t9_site_indices()] = np.arange(9) + 2000.0
    row = t_only_constant_row(true, t9_site_indices(), {"method": "T_only_const"})
    assert row["treatment"] == "t_only_const"
    assert row["level_recipe"] == "historical_t_only_const"
    assert np.isnan(row["delta_median9_K"])
    assert np.isnan(row["delta_mean9_K"])
    assert np.isnan(row["delta_applied_K"])


def _population(rmse_shift=0.0):
    rows, residuals, ref_rows = [], [], []
    for role, sim in (("seen_train_eval", "a"), ("val_eval", "b")):
        for rep in (0, 1, 2):
            for treatment, delta in (
                ("free", 0.0), ("oracle", 2.0), ("ao9", 1.8), ("mean9", 2.0), ("median9", 1.5),
            ):
                rows.append({
                    "method": "m", "role": role, "replica_id": rep, "sim_id": sim,
                    "frame_index": 0, "treatment": treatment,
                    "all_rmse_K": 10.0 + rmse_shift, "delta_applied_K": delta,
                    "delta_median9_K": 1.5, "delta_mean9_K": 2.0,
                })
            for treatment, delta in (("free", 0.0), ("oracle", 2.0), ("ao9", 1.8)):
                ref_rows.append({
                    "run_id": "m", "role": role, "replica_id": rep, "sim_id": sim,
                    "frame_index": 0, "treatment": treatment,
                    "all_rmse_K": 10.0, "delta_applied_K": delta,
                })
            for anchor in range(9):
                residuals.append({
                    "method": "m", "role": role, "replica_id": rep, "sim_id": sim,
                    "frame_index": 0, "anchor_id": anchor,
                })
    return pd.DataFrame(rows), pd.DataFrame(residuals), pd.DataFrame(ref_rows)


def _load_eval():
    path = Path(__file__).resolve().parents[1] / "scripts" / "24_astra_median9_full_eval.py"
    spec = importlib.util.spec_from_file_location("astra_median9_full_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_missing_reference_keeps_tables_and_does_not_write_complete(tmp_path):
    mod = _load_eval()
    metrics, residuals, _ = _population()
    code = mod._finish(
        tmp_path, metrics, residuals, primary=None, primary_files={}, missing_primary=["x"],
        methods=["m"], expected={"seen_train_eval": 1, "val_eval": 1}, atol=0.05, t_only_method=None,
    )
    assert code == 3
    assert (tmp_path / "per_frame_val_eval.csv.gz").exists()
    assert (tmp_path / "anchor_residuals_seen_train_eval.csv.gz").exists()
    gate = json.loads((tmp_path / "GATE.json").read_text())
    assert gate["scientific_validated"] is False
    assert gate["replay_executed"] is False
    assert not (tmp_path / "COMPLETE.json").exists()


def test_matching_reference_writes_complete_and_a_shift_does_not(tmp_path):
    mod = _load_eval()
    metrics, residuals, ref = _population()
    ok = tmp_path / "ok"
    ok.mkdir()
    code = mod._finish(
        ok, metrics, residuals, primary=ref, primary_files={"ref": "abc"}, missing_primary=[],
        methods=["m"], expected={"seen_train_eval": 1, "val_eval": 1}, atol=0.05, t_only_method=None,
    )
    assert code == 0
    assert json.loads((ok / "COMPLETE.json").read_text())["scientific_validated"] is True
    bad_metrics, residuals, ref = _population(rmse_shift=0.2)
    bad = tmp_path / "bad"
    bad.mkdir()
    code = mod._finish(
        bad, bad_metrics, residuals, primary=ref, primary_files={"ref": "abc"}, missing_primary=[],
        methods=["m"], expected={"seen_train_eval": 1, "val_eval": 1}, atol=0.05, t_only_method=None,
    )
    assert code == 3
    assert (bad / "per_frame_seen_train_eval.csv.gz").exists()
    assert (bad / "replay_primary_frames.csv.gz").exists()
    assert not (bad / "COMPLETE.json").exists()


def test_sbatch_keeps_two_gpu_cap_and_advanced_qos():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "scripts" / "engaging"
    neural = (root / "astra_median9_neural.sbatch").read_text()
    base = (root / "astra_median9_baselines.sbatch").read_text()
    assert "#SBATCH --array=0-3%2" in neural
    assert "#SBATCH --qos=mit_amf_advanced_gpu" in neural
    assert "#SBATCH --partition=mit_normal_gpu" in neural
    assert "gpu:l40s:1" in neural
    assert "#SBATCH --qos=mit_amf_advanced_cpu" in base
    assert "#SBATCH --partition=mit_normal" in base
    for text in (neural, base):
        assert "preempt" not in text
        assert "mit_general" not in text
        assert "EXPECT_SHA" in text
