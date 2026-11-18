"""Meaningful p_off isolation tests for U6 (channels zero; no p leakage into forward)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from combustion_recon.models.unet import build_dense_unet_from_config
from combustion_recon.training.dense_cache import (
    CHANNEL_ORDER,
    DenseUNetCacheDataset,
    build_dense_input_tensor,
    dense_fields_to_input_tensor,
)
from combustion_recon.training.epoch_loop import (
    assert_u4_candidate_contract,
    assert_u6_candidate_contract,
)
from combustion_recon.training.run_provenance import collect_run_provenance
from combustion_recon.training.trainer import set_seed

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "processed" / "month2_dense_unet_cache"
HAS_CACHE = (CACHE_ROOT / "manifest.json").exists()


def test_u6_candidate_contract_requires_poff():
    cand = yaml.safe_load(
        (ROOT / "configs/experiments/month2_dense_unet_u6_candidate.yaml").read_text()
    )
    out = assert_u6_candidate_contract(cand)
    assert out["p_conditioning"] is False
    assert out["variant"] == "p_off"
    assert out["n_epochs"] == 1000
    assert out["warmup_updates"] == 950

    broken = yaml.safe_load(
        (ROOT / "configs/experiments/month2_dense_unet_u6_candidate.yaml").read_text()
    )
    broken["training_candidate"]["p_conditioning"] = True
    with pytest.raises(ValueError, match="p_conditioning false"):
        assert_u6_candidate_contract(broken)


def test_u4_candidate_still_requires_pon():
    cand = yaml.safe_load(
        (ROOT / "configs/experiments/month2_dense_unet_u4_candidate.yaml").read_text()
    )
    out = assert_u4_candidate_contract(cand)
    assert out["p_conditioning"] is True


def test_provenance_records_sha_and_sources():
    prov = collect_run_provenance(
        extra_sources=["configs/experiments/month2_dense_unet_u6_candidate.yaml"]
    )
    assert prov["git_commit"]
    assert "scientific_source_sha256" in prov
    key = "configs/experiments/month2_dense_unet_u6_candidate.yaml"
    assert prov["scientific_source_sha256"].get(key)
    assert isinstance(prov["working_tree"]["clean"], bool)


def test_build_dense_input_tensor_poff_zeros_and_ignores_source_p():
    H = W = 16
    vi = np.random.randn(H, W).astype(np.float32)
    vj = np.random.randn(H, W).astype(np.float32)
    xs = np.linspace(-1, 1, W, dtype=np.float32)[None, :].repeat(H, 0)
    ys = np.linspace(-1, 1, H, dtype=np.float32)[:, None].repeat(W, 1)
    x0 = build_dense_input_tensor(
        vi_norm=vi,
        vj_norm=vj,
        x_scaled=xs,
        y_scaled=ys,
        nh3_norm=1.25,
        phi_norm=-0.7,
        p_conditioning=False,
    )
    x1 = build_dense_input_tensor(
        vi_norm=vi,
        vj_norm=vj,
        x_scaled=xs,
        y_scaled=ys,
        nh3_norm=-9.0,
        phi_norm=4.0,
        p_conditioning=False,
    )
    assert x0.shape == (6, H, W)
    assert list(CHANNEL_ORDER)[:2] == ["vi_norm", "vj_norm"]
    assert np.array_equal(x0, x1)
    assert np.count_nonzero(x0[4]) == 0 and np.count_nonzero(x0[5]) == 0
    assert np.array_equal(x0[0], vi) and np.array_equal(x0[1], vj)
    assert np.array_equal(x0[2], xs) and np.array_equal(x0[3], ys)


def test_poff_forward_invariant_to_source_p_mutation():
    set_seed(7)
    model = build_dense_unet_from_config(init_seed=7, base_channels=16)
    model.eval()
    H = W = 32
    base = {
        "vi_norm": np.random.randn(H, W).astype(np.float32),
        "vj_norm": np.random.randn(H, W).astype(np.float32),
        "x_scaled": np.zeros((H, W), dtype=np.float32),
        "y_scaled": np.ones((H, W), dtype=np.float32),
        "nh3_norm": 0.4,
        "phi_norm": -0.2,
        "temperature": np.full((H, W), 1800.0, dtype=np.float32),
        "temperature_normalized": np.zeros((H, W), dtype=np.float32),
        "velocity_k": np.random.randn(H, W).astype(np.float32),
        "metadata": {"sim_id": "synthetic", "nh3_fraction": 0.2, "phi": 1.0},
    }
    x0 = dense_fields_to_input_tensor(
        vi_norm=base["vi_norm"],
        vj_norm=base["vj_norm"],
        x_scaled=base["x_scaled"],
        y_scaled=base["y_scaled"],
        nh3_norm=base["nh3_norm"],
        phi_norm=base["phi_norm"],
        p_conditioning=False,
        temperature=base["temperature"],
        temperature_normalized=base["temperature_normalized"],
        velocity_k=base["velocity_k"],
        metadata=base["metadata"],
    )
    x1 = dense_fields_to_input_tensor(
        vi_norm=base["vi_norm"],
        vj_norm=base["vj_norm"],
        x_scaled=base["x_scaled"],
        y_scaled=base["y_scaled"],
        nh3_norm=base["nh3_norm"] + 3.0,
        phi_norm=base["phi_norm"] - 2.0,
        p_conditioning=False,
        temperature=base["temperature"] + 100.0,
        temperature_normalized=base["temperature_normalized"] + 1.0,
        velocity_k=base["velocity_k"] + 1.0,
        metadata={**base["metadata"], "nh3_fraction": 0.99, "phi": 0.5},
    )
    assert np.array_equal(x0, x1)
    with torch.no_grad():
        y0 = model(torch.from_numpy(x0)[None])
        y1 = model(torch.from_numpy(x1)[None])
    assert torch.allclose(y0, y1)
    # Positive: changing vi must move the p_off forward.
    x_vi = dense_fields_to_input_tensor(
        vi_norm=base["vi_norm"] + 0.8,
        vj_norm=base["vj_norm"],
        x_scaled=base["x_scaled"],
        y_scaled=base["y_scaled"],
        nh3_norm=base["nh3_norm"],
        phi_norm=base["phi_norm"],
        p_conditioning=False,
    )
    with torch.no_grad():
        y_vi = model(torch.from_numpy(x_vi)[None])
    assert not torch.allclose(y0, y_vi)


@pytest.mark.skipif(not HAS_CACHE, reason="dense cache required")
def test_cache_dataset_poff_zeros_and_keeps_vi_coords_target():
    ds_on = DenseUNetCacheDataset("val_eval", cache_root=CACHE_ROOT, p_conditioning=True)
    ds_off = DenseUNetCacheDataset("val_eval", cache_root=CACHE_ROOT, p_conditioning=False)
    item_on = ds_on[0]
    item_off = ds_off[0]
    x_on = item_on["model_inputs"]["input_tensor"]
    x_off = item_off["model_inputs"]["input_tensor"]
    assert torch.count_nonzero(x_off[4]) == 0
    assert torch.count_nonzero(x_off[5]) == 0
    assert torch.allclose(x_on[0], x_off[0])
    assert torch.allclose(x_on[1], x_off[1])
    assert torch.allclose(x_on[2], x_off[2])
    assert torch.allclose(x_on[3], x_off[3])
    assert not torch.allclose(x_on[4], x_off[4])
    assert "op_params" not in item_off["model_inputs"]
    assert "op_params" in item_on["model_inputs"]
    # Metadata may still carry condition IDs for eval/grouping; they must not enter tensor.
    assert "sim_id" in item_off["metadata"]
    assert torch.allclose(
        item_on["targets"]["temperature_normalized"],
        item_off["targets"]["temperature_normalized"],
    )
    # Mutating source p_norm arrays is not done here; channel construction ignores p when off.
    # Sanity: changing only nh3/phi in build path already covered above.


def test_cli_refuses_u6_without_authorization():
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/07_train_reconstructor.py"),
            "train-u6",
            "--run-id",
            "should_not_run_u6",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "authorize-scientific-u6" in (proc.stderr + proc.stdout)
