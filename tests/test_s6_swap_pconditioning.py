"""S6 swap path must expose p_conditioning to collate_sparse_batch (p_on and p_off)."""

from __future__ import annotations

import numpy as np
import torch

from combustion_recon.training.sparse_dataset import collate_sparse_batch
from combustion_recon.training.sparse_view import build_sparse_unet_input


def _fake_swapped_item(*, p_conditioning: bool, nh3: float = 1.5, phi: float = -0.7):
    """Mirror scripts/17 build_sparse_swapped_batch_item model_inputs contract."""
    sites = np.asarray([0, 1, 2], dtype=np.int64)
    vals = np.zeros((3, 2), dtype=np.float64)
    x_scaled = np.zeros((128, 128), dtype=np.float32)
    y_scaled = np.zeros((128, 128), dtype=np.float32)
    x = build_sparse_unet_input(
        sites=sites,
        velocity_values_norm=vals,
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        nh3_norm=nh3,
        phi_norm=phi,
        p_conditioning=p_conditioning,
    )
    y = np.zeros((1, 128, 128), dtype=np.float32)
    t_K = np.zeros((1, 128, 128), dtype=np.float32)
    return {
        "model_inputs": {
            "input_tensor": torch.from_numpy(np.array(x, dtype=np.float32, copy=True)),
            "p_conditioning": bool(p_conditioning),
        },
        "targets": {
            "temperature_normalized": torch.from_numpy(np.array(y, dtype=np.float32, copy=True)),
            "temperature": torch.from_numpy(np.array(t_K, dtype=np.float32, copy=True)),
            "temperature_donor": torch.from_numpy(np.array(t_K, dtype=np.float32, copy=True)),
        },
        "observation": None,
        "glu_tokens": None,
        "baseline_physical": None,
        "metadata": {"p_conditioning": bool(p_conditioning), "cache_index": 0},
    }


def test_swap_collate_p_on_keeps_param_channels():
    item = _fake_swapped_item(p_conditioning=True, nh3=1.5, phi=-0.7)
    batch = collate_sparse_batch([item])
    assert batch["p_conditioning"] is True
    assert torch.allclose(batch["input_tensor"][0, -2], torch.full((128, 128), 1.5))
    assert torch.allclose(batch["input_tensor"][0, -1], torch.full((128, 128), -0.7))


def test_swap_collate_p_off_zeros_param_channels():
    item = _fake_swapped_item(p_conditioning=False, nh3=1.5, phi=-0.7)
    batch = collate_sparse_batch([item])
    assert batch["p_conditioning"] is False
    assert torch.all(batch["input_tensor"][0, -2:] == 0)


def test_legacy_swap_item_missing_p_conditioning_raises():
    item = _fake_swapped_item(p_conditioning=True)
    del item["model_inputs"]["p_conditioning"]
    try:
        collate_sparse_batch([item])
    except KeyError as e:
        assert "p_conditioning" in str(e)
    else:
        raise AssertionError("expected KeyError for missing p_conditioning")
