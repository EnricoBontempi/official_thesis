"""K-sweep corrections: identity at K=0, even median, median9 reuse, frozen manifest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from combustion_recon.evaluation.astra_level import (
    H,
    W,
    delta_mean9,
    delta_median9,
    t9_site_indices,
)
from combustion_recon.evaluation.temperature_k import (
    MANIFEST_CONTENT_SHA256,
    MANIFEST_FILE_SHA256,
    PRIMARY_TREATMENTS,
    apply_correction,
    correction_deltas,
    gate_free_against_reference,
    load_frozen_manifest,
    load_historical_tblock,
    ordinary_median,
    per_frame_csv_paths,
    rows_for_historical_t9,
    rows_for_prediction,
    sites_for_k,
    summarize_sweep,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _fields(seed=0):
    rng = np.random.default_rng(seed)
    true = 1500.0 + 400.0 * rng.random((H, W))
    pred = true + rng.normal(0.0, 20.0, (H, W)) - 30.0
    return pred, true


def test_manifest_bytes_content_hash_and_generator():
    path = ROOT / "configs/observations/month2_temperature_anchors_nested_v1.json"
    manifest = load_frozen_manifest(path)
    assert manifest["manifest_content_sha256"] == MANIFEST_CONTENT_SHA256
    from combustion_recon.evaluation.temperature_k import sha256_file

    assert sha256_file(path) == MANIFEST_FILE_SHA256
    assert [item["replica_id"] for item in manifest["layouts"]] == [0, 1, 2]


def test_historical_tblock_pair_is_the_frozen_manifest():
    got = load_historical_tblock(ROOT / "results/meta/month2_baseline_run_manifest.json")
    assert got["p_off"] == {"ridge_lambda": 0.001, "w_T": 0.5}
    assert got["p_on"] == {"ridge_lambda": 0.01, "w_T": 0.5}


def test_k0_corrections_are_exactly_free_and_oracle_is_not():
    pred, true = _fields()
    deltas = correction_deltas(pred, true, np.zeros((0,), dtype=np.int64))
    assert deltas["ao"] == deltas["mean"] == deltas["median"] == 0.0
    assert deltas["oracle"] != 0.0
    for name in ("ao", "mean", "median"):
        assert np.array_equal(apply_correction(pred, deltas[name], identity=True), pred)


def test_even_median_is_mean_of_two_central_values_and_matches_numpy():
    residuals = np.array([1.0, 10.0, 3.0, 2.0])
    assert ordinary_median(residuals) == pytest.approx(2.5)
    assert ordinary_median(residuals) == pytest.approx(float(np.median(residuals)))
    odd = np.array([4.0, 1.0, 9.0])
    assert ordinary_median(odd) == pytest.approx(4.0)


def test_k1_mean_equals_median_and_ao_is_half():
    pred, true = _fields(1)
    manifest = load_frozen_manifest(ROOT / "configs/observations/month2_temperature_anchors_nested_v1.json")
    site = sites_for_k(manifest["layouts"][0]["site_order_flat"], 1)
    deltas = correction_deltas(pred, true, site)
    assert deltas["mean"] == pytest.approx(deltas["median"])
    assert deltas["ao"] == pytest.approx(deltas["mean"] / 2.0)


def test_uniform_offset_sign_and_k9_matches_median9():
    true = np.full((H, W), 1800.0)
    pred = true - 40.0
    sites = t9_site_indices()
    deltas = correction_deltas(pred, true, sites)
    resid = np.full(9, 40.0)
    assert deltas["mean"] == pytest.approx(delta_mean9(resid))
    assert deltas["median"] == pytest.approx(delta_median9(resid))
    assert deltas["mean"] == pytest.approx(40.0)
    assert deltas["median"] == pytest.approx(40.0)
    assert deltas["ao"] == pytest.approx(36.0)
    assert deltas["oracle"] == pytest.approx(40.0)
    assert deltas["ao"] > 0.0


def test_even_k_on_a_real_layout_matches_two_central_mean():
    pred, true = _fields(3)
    manifest = load_frozen_manifest(ROOT / "configs/observations/month2_temperature_anchors_nested_v1.json")
    sites = sites_for_k(manifest["layouts"][1]["site_order_flat"], 4)
    deltas = correction_deltas(pred, true, sites)
    resid = true.ravel()[sites] - pred.ravel()[sites]
    ordered = np.sort(resid)
    assert deltas["median"] == pytest.approx(0.5 * (ordered[1] + ordered[2]))
    assert deltas["mean"] == pytest.approx(float(resid.mean()))


def test_off_anchor_mutation_changes_oracle_only():
    pred, true = _fields(4)
    manifest = load_frozen_manifest(ROOT / "configs/observations/month2_temperature_anchors_nested_v1.json")
    order = np.asarray(manifest["layouts"][2]["site_order_flat"], dtype=np.int64)
    sites = sites_for_k(order, 16)
    mutated = true.copy()
    mask = np.ones(H * W, dtype=bool)
    mask[sites] = False
    mutated.ravel()[mask] += 80.0
    keep = correction_deltas(pred, true, sites)
    changed = correction_deltas(pred, mutated, sites)
    assert keep["ao"] == pytest.approx(changed["ao"])
    assert keep["mean"] == pytest.approx(changed["mean"])
    assert keep["median"] == pytest.approx(changed["median"])
    assert keep["oracle"] != pytest.approx(changed["oracle"])


def test_rows_cover_k_grid_and_k0_identity_and_centered():
    pred, true = _fields(5)
    manifest = load_frozen_manifest(ROOT / "configs/observations/month2_temperature_anchors_nested_v1.json")
    order = manifest["layouts"][0]["site_order_flat"]
    rows = rows_for_prediction(
        pred, true, order, identity={"sim_id": "0NH3_1.0", "frame_index": 3},
        layout_id="replica0", layout_kind="primary",
    )
    frame = pd.DataFrame(rows)
    assert set(frame["treatment"]) == set(PRIMARY_TREATMENTS)
    assert sorted(frame["K"].unique()) == [0, 1, 4, 9, 16, 32]
    k0 = frame[frame["K"] == 0]
    assert set(k0["delta_applied_K"].loc[k0["treatment"].isin(["free", "ao", "mean", "median"])]) == {0.0}
    assert int(k0.loc[k0["treatment"] == "free", "anchor_empty"].iloc[0]) == 1
    assert np.isnan(k0.loc[k0["treatment"] == "free", "anchor_rmse_K"].iloc[0])
    free = frame[frame["treatment"] == "free"].set_index("K")["full_centered_rmse_K"]
    for name in ("ao", "mean", "median", "oracle"):
        got = frame[frame["treatment"] == name].set_index("K")["full_centered_rmse_K"]
        assert np.allclose(got, free)
    assert (frame["common_n"] == H * W - 32).all()
    assert int(frame.loc[frame["K"] == 0, "off_n"].iloc[0]) == H * W
    assert int(frame.loc[frame["K"] == 32, "off_n"].iloc[0]) == H * W - 32


def test_historical_t9_is_not_a_primary_layout():
    pred, true = _fields(6)
    rows = rows_for_historical_t9(pred, true, identity={"sim_id": "a", "frame_index": 1})
    frame = pd.DataFrame(rows)
    assert set(frame["layout_kind"]) == {"t9_bridge"}
    assert set(frame["K"]) == {9}
    assert "t_only_const" in set(frame["treatment"])
    assert frame["common_n"].eq(0).all()
    assert np.isnan(frame["common_rmse_K"]).all()


def test_empty_reference_is_not_a_pass_and_per_frame_deltas_are_kept():
    empty = gate_free_against_reference(
        pd.DataFrame([{"sim_id": "a", "frame_index": 1, "s7_rmse_K": 1.0}]),
        pd.DataFrame(columns=["sim_id", "frame_index", "rmse_K"]),
        reference_path="ref.csv",
        reference_sha256="abc",
    )
    assert empty["ok"] is False and empty["replay_executed"] is False
    ref = pd.DataFrame(
        {
            "sim_id": ["a", "a", "b"],
            "frame_index": [0, 1, 0],
            "rmse_K": [10.0, 12.0, 20.0],
        }
    )
    cand = pd.DataFrame(
        {
            "sim_id": ["a", "a", "b"],
            "frame_index": [0, 1, 0],
            "s7_rmse_K": [10.02, 12.01, 20.04],
            # The sweep rows already carry full_rmse_K. The gate must not rename
            # s7_rmse_K onto that name: pandas then averages two columns.
            "full_rmse_K": [99.0, 99.0, 99.0],
        }
    )
    passed = gate_free_against_reference(cand, ref, reference_path="ref.csv", reference_sha256="abc")
    assert passed["ok"] is True
    assert passed["per_frame"]["delta_rmse_K"].abs().max() == pytest.approx(0.04)
    cand.loc[0, "s7_rmse_K"] = 30.0
    failed = gate_free_against_reference(cand, ref, reference_path="ref.csv", reference_sha256="abc")
    assert failed["ok"] is False and failed["replay_executed"] is True
    assert failed["per_frame"]["delta_rmse_K"].abs().max() > 0.05


def test_summary_keeps_seeds_worsenings_and_declares_empty_anchors():
    rows = []
    for seed, shift in ((0, 0.0), (1, 5.0)):
        for sim, base in (("a", 10.0), ("b", 20.0)):
            for treatment, extra in (("free", 0.0), ("mean", extra_mean := shift)):
                rows.append({
                    "method": "sparse_unet", "p_conditioning": True, "model_seed": seed,
                    "checkpoint": "best", "role": "val_eval", "mask_replica_id": 0,
                    "layout_id": "replica0", "layout_kind": "primary", "K": 0,
                    "treatment": treatment, "sim_id": sim, "frame_index": 1,
                    "full_rmse_K": base + extra, "full_bias_K": 1.0, "full_abs_bias_K": 1.0,
                    "full_centered_rmse_K": 8.0, "off_rmse_K": base + extra, "off_centered_rmse_K": 8.0,
                    "common_rmse_K": base + extra, "common_bias_K": 1.0, "common_centered_rmse_K": 8.0,
                    "anchor_rmse_K": float("nan"), "anchor_n": 0,
                    "full_hot_rmse_K": float("nan"), "full_hot_n": 0,
                    "full_peak_rmse_K": float("nan"), "full_peak_n": 0,
                    "full_jet_rmse_K": 3.0, "full_jet_n": 10,
                    "full_nonhot_rmse_K": 4.0, "full_nonhot_n": 10,
                })
    out = summarize_sweep(pd.DataFrame(rows))
    assert set(out["by_seed"]["model_seed"]) == {0, 1}
    worse = out["versus_free"]
    assert set(worse.loc[worse["model_seed"] == 1, "delta_full_vs_free_K"]) == {5.0}
    assert worse.loc[worse["model_seed"] == 1, "worse_full"].all()
    assert worse.loc[worse["model_seed"] == 0, "worse_full"].eq(False).all()
    assert np.isnan(out["by_seed"]["anchor_rmse_K"]).all()
    assert np.isnan(out["by_seed"]["full_hot_rmse_K"]).all()
    assert (out["by_seed"]["full_hot_n_empty"] == 2).all()


def test_per_frame_tables_include_a_run_directory_and_a_flat_baseline(tmp_path):
    nested = tmp_path / "neural" / "S7-C149-unet-on-s1" / "per_frame" / "best" / "val_eval"
    nested.mkdir(parents=True)
    (nested / "rep0.csv").write_text("a\n")
    flat = tmp_path / "baseline" / "per_frame" / "GlobalTrainMean_p_off" / "val_eval"
    flat.mkdir(parents=True)
    (flat / "rep0.csv").write_text("a\n")
    (tmp_path / "neural" / "S7-C149-unet-on-s1" / "gate").mkdir()
    (tmp_path / "neural" / "S7-C149-unet-on-s1" / "gate" / "best_val_eval_rep0_frames.csv").write_text("a\n")
    assert [path.name for path in per_frame_csv_paths(tmp_path / "neural")] == ["rep0.csv"]
    assert [path.name for path in per_frame_csv_paths(tmp_path / "baseline")] == ["rep0.csv"]


def test_protocol_text_names_the_frozen_manifest_and_is_not_a_site_list():
    text = (ROOT / "configs/experiments/month2_temperature_study_protocol.yaml").read_text()
    assert "configs/observations/month2_temperature_anchors_nested_v1.json" in text
    assert "historical_T9_grid: separate bridge" in text
    assert "site_order_flat" not in text
