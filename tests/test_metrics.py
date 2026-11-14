"""Unit tests for the four evaluation metric families and per-condition tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from combustion_recon.evaluation.metrics import (
    evaluate_per_condition,
    evaluate_sample,
    field_errors,
    high_temperature_region_errors,
    spatial_bias,
    temperature_level_errors,
)


def test_perfect_reconstruction_field_errors():
    T = np.linspace(300, 2200, 128 * 128).reshape(128, 128)
    m = field_errors(T, T, train_sigma_T=300.0)
    assert m["rmse_K"] == 0.0
    assert m["mae_K"] == 0.0
    assert m["rel_l2"] == 0.0
    assert abs(m["r2"] - 1.0) < 1e-12


def test_constant_bias_mean_signed_error():
    true = np.ones((16, 16)) * 2000.0
    pred = true + 5.0
    b = spatial_bias(pred, true)
    assert abs(b["mean_signed_error_K"] - 5.0) < 1e-12
    assert np.allclose(b["spatial_bias_map"], 5.0)


def test_high_temperature_iou_and_conditional_rmse():
    true = np.full((32, 32), 1900.0)
    true[:10, :10] = 2150.0
    pred = true.copy()
    pred[:10, :10] = 2160.0  # still above 2100
    pred[10:15, 10:15] = 2150.0  # false positive high region
    out = high_temperature_region_errors(pred, true, t_high=2100.0)
    assert out["high_n_voxels"] == 100
    assert out["high_rmse_K"] > 0
    assert 0.0 < out["high_iou"] < 1.0
    # Jet region T<1800 is empty here → conditional RMSE is NaN by design
    assert np.isnan(out["jet_rmse_K"])
    assert out["frac_omega_high_true"] == pytest.approx(100 / (32 * 32))


def test_temperature_level_errors():
    true = np.array([1000.0, 2000.0, 2200.0, 2300.0])
    pred = np.array([1010.0, 1990.0, 2210.0, 2400.0])
    out = temperature_level_errors(pred, true)
    assert out["mean_level_abs_error_K"] >= 0
    assert out["peak_abs_error_K"] == abs(2400 - 2300)


def test_evaluate_per_condition_disaggregation():
    rng = np.random.default_rng(0)
    results = []
    for split, sim in [("val", "A"), ("val", "B"), ("test_interp", "C")]:
        true = 2000 + rng.normal(size=(8, 8))
        pred = true + rng.normal(scale=5.0, size=true.shape)
        results.append(
            evaluate_sample(pred, true, train_sigma_T=300.0, sim_id=sim, split=split)
        )
    df = evaluate_per_condition(results)
    assert isinstance(df, pd.DataFrame)
    assert set(df["sim_id"]) == {"A", "B", "C"}
    assert "rmse_K" in df.columns
    assert "high_iou" in df.columns
    assert "mean_level_abs_error_K" in df.columns
    # Must not collapse to a single pooled row
    assert len(df) == 3
