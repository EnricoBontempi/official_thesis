"""U5 snapshot-swap tests: pairing + actual input transforms (no scientific run)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from combustion_recon.training.dense_cache import (
    CHANNEL_ORDER,
    DenseUNetCacheDataset,
    build_dense_input_tensor,
)
from combustion_recon.training.snapshot_swap import (
    apply_velocity_swap_to_input,
    build_swapped_input_from_cache,
    build_within_condition_swap_pairs,
    inverse_pairs,
    pairs_by_dest_cache_index,
    reconstruct_aligned_input_via_inverse_swap,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "processed" / "month2_dense_unet_cache"
HAS_CACHE = (CACHE_ROOT / "manifest.json").is_file()


def test_pairing_no_fixed_points_and_role_isolation():
    frames = []
    for sim in ("A.h5", "B.h5"):
        for k, fi in enumerate(range(0, 200, 40)):
            frames.append(
                {
                    "sim_id": sim,
                    "frame_index": fi,
                    "cache_index": k if sim == "A.h5" else 100 + k,
                    "role": "val_eval",
                }
            )
    pairs = build_within_condition_swap_pairs(frames, role="val_eval")
    assert len(pairs) == len(frames)
    for p in pairs:
        assert p["src_frame_index"] != p["dest_frame_index"]
        assert p["src_cache_index"] != p["dest_cache_index"]
        assert p["role"] == "val_eval"
    # Inverse is a perfect involution on cache indices for even n.
    inv = inverse_pairs(pairs)
    fwd = pairs_by_dest_cache_index(pairs)
    for p in inv:
        back = fwd[int(p["src_cache_index"])]
        assert int(back["src_cache_index"]) == int(p["dest_cache_index"])


def test_apply_velocity_swap_preserves_non_velocity_channels():
    rng = np.random.default_rng(0)
    dest = rng.normal(size=(6, 8, 8)).astype(np.float32)
    src_vel = rng.normal(size=(2, 8, 8)).astype(np.float32)
    out = apply_velocity_swap_to_input(dest, src_vel)
    assert np.allclose(out[0], src_vel[0])
    assert np.allclose(out[1], src_vel[1])
    assert np.allclose(out[2:], dest[2:])
    assert not np.shares_memory(out, dest)


@pytest.mark.skipif(not HAS_CACHE, reason="dense cache required")
def test_cache_swap_changes_only_velocity_and_is_invertible():
    ds = DenseUNetCacheDataset("val_eval", cache_root=CACHE_ROOT, p_conditioning=True)
    pairs = build_within_condition_swap_pairs(ds.frames, role="val_eval")
    lookup = pairs_by_dest_cache_index(pairs)
    # Use a few indices spanning conditions
    for i in (0, 40, 82):
        aligned = ds[i]
        x0 = aligned["model_inputs"]["input_tensor"].numpy()
        y0 = aligned["targets"]["temperature_normalized"].numpy()
        t0 = aligned["targets"]["temperature"].numpy()
        swapped = build_swapped_input_from_cache(ds, i, pair_lookup=lookup)
        xs = swapped["model_inputs"]["input_tensor"].numpy()
        # Velocity changed (unless degenerate — forbidden by pairing)
        assert not np.allclose(xs[0], x0[0]) or not np.allclose(xs[1], x0[1])
        # Coords + p unchanged
        assert np.array_equal(xs[2], x0[2])
        assert np.array_equal(xs[3], x0[3])
        assert np.array_equal(xs[4], x0[4])
        assert np.array_equal(xs[5], x0[5])
        # Targets unchanged
        assert np.array_equal(
            swapped["targets"]["temperature_normalized"].numpy(), y0
        )
        assert np.array_equal(swapped["targets"]["temperature"].numpy(), t0)
        # Channel order tag intact
        assert list(swapped["model_inputs"]["channel_order"]) == list(CHANNEL_ORDER)
        # Reconstruct aligned without mutating cache
        x_rec = reconstruct_aligned_input_via_inverse_swap(
            ds, i, forward_pair_lookup=lookup
        )
        assert np.allclose(x_rec, x0, rtol=0.0, atol=0.0)
        # Cache mmap unchanged vs fresh read
        x_again = ds[i]["model_inputs"]["input_tensor"].numpy()
        assert np.allclose(x_again, x0)


@pytest.mark.skipif(not HAS_CACHE, reason="dense cache required")
def test_swap_uses_donor_velocity_components_together():
    ds = DenseUNetCacheDataset("seen_train_eval", cache_root=CACHE_ROOT, p_conditioning=True)
    pairs = build_within_condition_swap_pairs(ds.frames, role="seen_train_eval")
    lookup = pairs_by_dest_cache_index(pairs)
    i = 5
    pair = lookup[int(ds.indices[i])]
    src = int(pair["src_cache_index"])
    swapped = build_swapped_input_from_cache(ds, i, pair_lookup=lookup)
    xs = swapped["model_inputs"]["input_tensor"].numpy()
    assert np.allclose(xs[0], ds.vi[src])
    assert np.allclose(xs[1], ds.vj[src])
    # Donor components stay paired: rebuilding from src vi/vj matches
    rebuilt = build_dense_input_tensor(
        vi_norm=np.asarray(ds.vi[src]),
        vj_norm=np.asarray(ds.vj[src]),
        x_scaled=ds.x_scaled,
        y_scaled=ds.y_scaled,
        nh3_norm=float(ds.p_norm[ds.indices[i], 0]),
        phi_norm=float(ds.p_norm[ds.indices[i], 1]),
        p_conditioning=True,
    )
    assert np.allclose(xs, rebuilt)
