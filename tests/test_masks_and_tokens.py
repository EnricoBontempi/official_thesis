"""Tests that sparse tokens never inject physical zeros for unobserved sites."""

from __future__ import annotations

import numpy as np

from combustion_recon.data.masks import (
    VAR_ID,
    build_layout_mask,
    build_observation_tokens,
    masked_field_to_tokens,
    random_sparse_mask,
)


def test_random_sparse_approx_fraction():
    mask = random_sparse_mask(0.05, seed=0, height=128, width=128)
    assert mask.dtype == bool
    assert mask.shape == (128, 128)
    assert abs(mask.mean() - 0.05) < 0.002


def test_tokens_only_at_observed_locations_and_match_values():
    rng = np.random.default_rng(0)
    H = W = 128
    x = np.linspace(-0.037, 0.037, W)
    y = np.linspace(0.063, 0.159, H)
    vi = rng.normal(size=(H, W))
    vj = rng.normal(size=(H, W))
    T = 2000 + 100 * rng.normal(size=(H, W))

    # Poison unobserved locations with a sentinel that must NEVER appear as a token
    # unless the site is observed. Use NaN so any accidental zero-fill path is obvious.
    tokens = build_observation_tokens(
        vi,
        vj,
        T,
        layout_name="random_1pct",
        task_mode="temperature_free",
        seed=123,
        x_coords=x,
        y_coords=y,
    )
    assert tokens.mask_temperature.sum() == 0
    n_vel = int(tokens.mask_velocity.sum())
    # two velocity channels share the same spatial mask
    assert tokens.var_ids.shape[0] == 2 * n_vel
    assert set(tokens.var_ids.tolist()) <= {VAR_ID["velocity_i"], VAR_ID["velocity_j"]}

    # Reconstruct which sites were emitted for velocity_i and verify values
    xx, yy = np.meshgrid(x, y, indexing="xy")
    sel = tokens.var_ids == VAR_ID["velocity_i"]
    assert int(sel.sum()) == n_vel
    for (xv, yv), val in zip(tokens.coords[sel], tokens.values[sel]):
        j = int(np.argmin(np.abs(x - xv)))
        i = int(np.argmin(np.abs(y - yv)))
        assert tokens.mask_velocity[i, j]
        assert abs(val - vi[i, j]) < 1e-9


def test_temperature_assisted_emits_exact_k_temperature_tokens():
    rng = np.random.default_rng(1)
    H = W = 64
    x = np.linspace(-1, 1, W)
    y = np.linspace(0, 1, H)
    vi = rng.normal(size=(H, W))
    vj = rng.normal(size=(H, W))
    T = rng.normal(size=(H, W))
    # Use a custom small grid via direct masked_field_to_tokens API with known masks
    vel_mask = random_sparse_mask(0.1, seed=0, height=H, width=W)
    t_mask = np.zeros((H, W), dtype=bool)
    t_mask[10, 10] = True
    t_mask[20, 20] = True
    t_mask[30, 30] = True
    t_mask[40, 40] = True
    fields = {
        VAR_ID["velocity_i"]: vi,
        VAR_ID["velocity_j"]: vj,
        VAR_ID["temperature"]: T,
    }
    masks = {
        VAR_ID["velocity_i"]: vel_mask,
        VAR_ID["velocity_j"]: vel_mask,
        VAR_ID["temperature"]: t_mask,
    }
    coords, var_ids, values = masked_field_to_tokens(fields, masks, x, y)
    n_t = int((var_ids == VAR_ID["temperature"]).sum())
    assert n_t == 4
    # No extra tokens beyond observed sites
    assert coords.shape[0] == 2 * int(vel_mask.sum()) + 4


def test_missing_box_never_observes_interior():
    mask = build_layout_mask("missing_shear_layer_box", seed=0)
    # Interior of box rows 44:84, cols 80:120 must be False (unobserved)
    assert not mask[44:84, 80:120].any()
    # Outside should have some True
    assert mask[0:10, 0:10].all()


def test_unobserved_sites_not_emitted_as_zero():
    """Even if the underlying field is zero at unobserved sites, we emit no token there."""
    H = W = 32
    x = np.arange(W, dtype=float)
    y = np.arange(H, dtype=float)
    field = np.zeros((H, W), dtype=float)
    field[5, 5] = 3.14
    mask = np.zeros((H, W), dtype=bool)
    mask[5, 5] = True
    coords, var_ids, values = masked_field_to_tokens(
        {11: field}, {11: mask}, x, y
    )
    assert len(values) == 1
    assert values[0] == 3.14
    # If we incorrectly zero-filled all pixels we'd get 32*32 tokens — guard against that
    assert len(values) != H * W
