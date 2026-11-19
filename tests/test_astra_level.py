"""Astra level/structure: AO9, oracle offset, off-anchor, T9 layout contracts."""

from __future__ import annotations

import numpy as np
import pytest

from combustion_recon.evaluation.astra_level import (
    H,
    W,
    anchor_mask_from_sites,
    ao_correct,
    frame_metrics,
    oracle_delta,
    t9_site_indices,
    treat_frame,
)


def _fields(seed=0):
    rng = np.random.default_rng(seed)
    true = 1500.0 + 400.0 * rng.random((H, W))
    pred = true + rng.normal(0, 20.0, (H, W)) - 30.0
    return pred, true


def test_t9_layout_fixed_distinct_valid():
    s = t9_site_indices()
    assert s.shape == (9,) and len(set(s.tolist())) == 9
    assert sorted(s.tolist()) == s.tolist()
    rows, cols = s // W, s % W
    assert set(rows) == {32, 64, 95} and set(cols) == {32, 64, 95}


def test_ao_formula_sign_and_tau():
    pred, true = _fields()
    s = t9_site_indices()
    vals = true.ravel()[s]
    _, d = ao_correct(pred, vals, s, tau=1.0)
    expect = float(np.sum(vals - pred.ravel()[s]) / (9 + 1.0))
    assert d == pytest.approx(expect, rel=1e-12)
    assert d > 0  # pred is biased low -> positive correction
    _, d0 = ao_correct(pred, vals, s, tau=1e-12)  # no shrink -> mean residual
    assert d0 == pytest.approx(np.mean(vals - pred.ravel()[s]), rel=1e-9)
    assert abs(d) < abs(d0)  # shrink toward zero
    # known offset: pred = true - 40 everywhere -> d = 9*40/10
    _, dk = ao_correct(true - 40.0, vals, s)
    assert dk == pytest.approx(36.0)


def test_ao_k0_is_identity():
    pred, _ = _fields()
    out, d = ao_correct(pred, np.zeros(0), np.zeros(0, dtype=np.int64))
    assert d == 0.0 and np.array_equal(out, pred)


def test_ao_permutation_invariant():
    pred, true = _fields()
    s = t9_site_indices()
    p = np.random.default_rng(3).permutation(9)
    _, d1 = ao_correct(pred, true.ravel()[s], s)
    _, d2 = ao_correct(pred, true.ravel()[s][p], s[p])
    assert d1 == pytest.approx(d2, rel=1e-12)


def test_oracle_rmse_equals_centered_and_centered_translation_invariant():
    pred, true = _fields(1)
    am = anchor_mask_from_sites(t9_site_indices())
    free = frame_metrics(pred, true, am)
    d = oracle_delta(pred, true)
    orc = frame_metrics(pred + d, true, am)
    assert orc["all_rmse_K"] == pytest.approx(free["all_centered_rmse_K"], rel=1e-9)
    assert orc["all_bias_K"] == pytest.approx(0.0, abs=1e-9)
    for shift in (-77.0, 13.0):
        sh = frame_metrics(pred + shift, true, am)
        assert sh["all_centered_rmse_K"] == pytest.approx(free["all_centered_rmse_K"], rel=1e-9)
        assert sh["off_centered_rmse_K"] == pytest.approx(free["off_centered_rmse_K"], rel=1e-9)


def test_mutating_true_off_anchor_changes_oracle_not_free_or_ao():
    pred, true = _fields(2)
    s = t9_site_indices()
    am = anchor_mask_from_sites(s)
    true2 = true.copy()
    true2[~am] += 55.0
    r1 = {r["treatment"]: r for r in treat_frame(pred, true, s)}
    r2 = {r["treatment"]: r for r in treat_frame(pred, true2, s)}
    assert r1["ao9"]["delta_ao9_K"] == r2["ao9"]["delta_ao9_K"]
    assert r1["free"]["delta_applied_K"] == r2["free"]["delta_applied_K"] == 0.0
    assert r1["oracle"]["delta_applied_K"] != r2["oracle"]["delta_applied_K"]
    # corrected FIELD depends only on pred + anchors
    a1, _ = ao_correct(pred, true.ravel()[s], s)
    a2, _ = ao_correct(pred, true2.ravel()[s], s)
    assert np.array_equal(a1, a2)


def test_mutating_observed_anchor_changes_delta_by_formula():
    pred, true = _fields(4)
    s = t9_site_indices()
    v = true.ravel()[s].copy()
    _, d1 = ao_correct(pred, v, s)
    v[0] += 100.0
    _, d2 = ao_correct(pred, v, s)
    assert d2 - d1 == pytest.approx(100.0 / 10.0)


def test_ao_does_not_read_non_anchor_pixels_of_obs():
    # observation grid is NaN off-anchor inside ao_correct: result must be finite
    pred, true = _fields(5)
    s = t9_site_indices()
    out, d = ao_correct(pred, true.ravel()[s], s)
    assert np.isfinite(out).all() and np.isfinite(d)


def test_off_anchor_excludes_exactly_nine_sites_and_counts():
    pred, true = _fields(6)
    s = t9_site_indices()
    am = anchor_mask_from_sites(s)
    m = frame_metrics(pred, true, am)
    assert m["n_anchor"] == 9 and m["n_off_anchor"] == H * W - 9 == 16375
    # direct recompute of off-anchor RMSE
    e = (pred - true).ravel()
    keep = np.ones(H * W, bool)
    keep[s] = False
    assert m["off_rmse_K"] == pytest.approx(np.sqrt(np.mean(e[keep] ** 2)))


def test_empty_hot_regions_are_nan_with_counts():
    pred = np.full((H, W), 1000.0)
    true = np.full((H, W), 1000.0)
    m = frame_metrics(pred, true, anchor_mask_from_sites(t9_site_indices()))
    assert m["all_hot_n_true"] == 0 and m["all_hot_n_union"] == 0
    assert np.isnan(m["all_hot_iou"]) and np.isnan(m["all_hot_rmse_K"])


def test_hot_iou_changes_with_level_shift():
    true = np.full((H, W), 1900.0)
    true[:, 64:] = 2300.0
    pred = true - 250.0  # structure right, level wrong
    s = t9_site_indices()
    r = {x["treatment"]: x for x in treat_frame(pred, true, s)}
    assert r["free"]["all_hot_iou"] == 0.0
    assert r["oracle"]["all_hot_iou"] == pytest.approx(1.0)
    assert r["oracle"]["mse_recovery_vs_oracle"] == pytest.approx(1.0)


def test_treat_frame_has_three_treatments_single_prediction():
    pred, true = _fields(7)
    rows = treat_frame(pred, true, t9_site_indices())
    assert [r["treatment"] for r in rows] == ["free", "oracle", "ao9"]
    assert rows[0]["all_rmse_K"] == pytest.approx(np.sqrt(np.mean((pred - true) ** 2)))


def test_script18_compiles_and_config_consistent():
    import yaml
    from pathlib import Path
    import py_compile

    root = Path(__file__).resolve().parents[1]
    py_compile.compile(str(root / "scripts/18_astra_level_structure_eval.py"), doraise=True)
    cfg = yaml.safe_load((root / "configs/experiments/month2_astra_level_structure.yaml").read_text())
    assert cfg["anchors_T9"]["rows"] == [32, 64, 95] and cfg["offsets"]["tau"] == 1.0
    assert len(cfg["neural_checkpoints"]) == 4
    assert not cfg["scientific_training_authorized"]


def test_glu_poff_forward_ignores_p_values():
    """scripts/17 _forward must zero p for p_off GLU (isolation used by Astra)."""
    import importlib.util, sys
    from pathlib import Path
    import torch

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("s17t", root / "scripts/17_s6_sparse_eval_and_swap.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["s17t"] = mod; spec.loader.exec_module(mod)
    seen = {}

    class Stub(torch.nn.Module):
        def forward(self, coords, vals, sites, p, Y, **kw):
            seen["p"] = p.clone()
            return torch.zeros(coords.shape[0], 1, 4, 4)

    tok = {"coords_phys": np.zeros((3, 2)), "values_norm": np.zeros((3, 2)), "site_index": np.arange(3)}
    batch = {"glu_tokens": [tok, tok], "p_norm_tensor": torch.tensor([[1.5, -2.0], [0.3, 9.0]])}
    q = torch.zeros(16, 2)
    mod._forward(Stub(), "adapted_glu", batch, p_conditioning=False, query_coords_phys=q, device=torch.device("cpu"))
    assert torch.all(seen["p"] == 0)
    mod._forward(Stub(), "adapted_glu", batch, p_conditioning=True, query_coords_phys=q, device=torch.device("cpu"))
    assert torch.allclose(seen["p"], batch["p_norm_tensor"])
