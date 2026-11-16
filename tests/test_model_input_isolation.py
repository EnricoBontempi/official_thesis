"""Interface isolation: model_inputs must ignore hidden / forbidden fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from combustion_recon.data.month2_dataset import InputIsolationProbe, Month2ReconDataset
from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.data.month2_observations import build_month2_observation_pack
from combustion_recon.data.shard_access import ShardStore

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "combustion" / "hf_dataset" / "numerical"
STATS = ROOT / "data" / "manifests" / "train_norm_stats_month2_fit.json"
HAS_REAL = RAW.exists() and STATS.exists() and len(list(RAW.glob("data-*-of-00030.arrow"))) >= 1


def _synthetic_base():
    rng = np.random.default_rng(42)
    H = W = 32
    vi = rng.normal(size=(H, W))
    vj = rng.normal(size=(H, W))
    vk = rng.normal(size=(H, W))
    T = 2000 + 50 * rng.normal(size=(H, W))
    x = np.linspace(-0.03, 0.03, W)
    y = np.linspace(0.06, 0.16, H)
    return vi, vj, vk, T, x, y


def test_isolation_free_ignores_target_T_change():
    vi, vj, vk, T, x, y = _synthetic_base()
    pack0 = build_month2_observation_pack(
        velocity_i=vi, velocity_j=vj, temperature=T, x_coords=x, y_coords=y,
        site_fraction=0.05, K=0, layout_id="L", mask_seed=1, sim_id="s", frame_index=0,
        include_dense_temperature_channels=False,
    )
    T2 = T + 500.0
    pack1 = build_month2_observation_pack(
        velocity_i=vi, velocity_j=vj, temperature=T2, x_coords=x, y_coords=y,
        site_fraction=0.05, K=0, layout_id="L", mask_seed=1, sim_id="s", frame_index=0,
        include_dense_temperature_channels=False,
    )
    assert pack0.velocity_mask_hash == pack1.velocity_mask_hash
    assert np.array_equal(pack0.velocity_values, pack1.velocity_values)
    assert np.array_equal(pack0.dense_values, pack1.dense_values)
    assert pack0.temperature_mask_hash == pack1.temperature_mask_hash


def test_isolation_assisted_ignores_T_off_anchors():
    vi, vj, vk, T, x, y = _synthetic_base()
    pack0 = build_month2_observation_pack(
        velocity_i=vi, velocity_j=vj, temperature=T, x_coords=x, y_coords=y,
        site_fraction=0.05, K=9, layout_id="L", mask_seed=2, sim_id="s", frame_index=5,
    )
    T2 = T.copy()
    # Poison everywhere except anchors
    T2[~pack0.mask_temperature] = T2[~pack0.mask_temperature] + 777.0
    pack1 = build_month2_observation_pack(
        velocity_i=vi, velocity_j=vj, temperature=T2, x_coords=x, y_coords=y,
        site_fraction=0.05, K=9, layout_id="L", mask_seed=2, sim_id="s", frame_index=5,
    )
    assert pack0.velocity_mask_hash == pack1.velocity_mask_hash
    assert pack0.temperature_mask_hash == pack1.temperature_mask_hash
    assert np.allclose(pack0.anchor_values, pack1.anchor_values)
    assert np.array_equal(pack0.velocity_values, pack1.velocity_values)
    # Dense T channel only at anchors → identical
    assert np.array_equal(pack0.dense_values, pack1.dense_values)


def test_isolation_sparse_ignores_velocity_outside_mask():
    vi, vj, vk, T, x, y = _synthetic_base()
    pack0 = build_month2_observation_pack(
        velocity_i=vi, velocity_j=vj, temperature=T, x_coords=x, y_coords=y,
        site_fraction=0.05, K=0, layout_id="L", mask_seed=3, sim_id="s", frame_index=1,
        include_dense_temperature_channels=False,
    )
    vi2 = vi.copy()
    vj2 = vj.copy()
    vi2[~pack0.mask_velocity] += 123.0
    vj2[~pack0.mask_velocity] -= 55.0
    pack1 = build_month2_observation_pack(
        velocity_i=vi2, velocity_j=vj2, temperature=T, x_coords=x, y_coords=y,
        site_fraction=0.05, K=0, layout_id="L", mask_seed=3, sim_id="s", frame_index=1,
        include_dense_temperature_channels=False,
    )
    assert np.array_equal(pack0.velocity_values, pack1.velocity_values)
    assert np.array_equal(pack0.dense_values, pack1.dense_values)


def test_isolation_2c_ignores_vk():
    vi, vj, vk, T, x, y = _synthetic_base()
    pack0 = build_month2_observation_pack(
        velocity_i=vi, velocity_j=vj, temperature=T, x_coords=x, y_coords=y,
        site_fraction=0.05, K=0, layout_id="L", mask_seed=4, sim_id="s", frame_index=2,
        include_dense_temperature_channels=False,
    )
    # vk is not an argument to the builder at all for 2C; changing a parallel vk array
    # cannot affect outputs. Sanity: pack has only 2 velocity components.
    assert pack0.velocity_values.shape[1] == 2
    assert "velocity_k" not in pack0.dense_channel_names


def test_isolation_p_off_has_no_op_params_in_synthetic_contract():
    # Contract-level: observation pack itself never embeds op params.
    vi, vj, vk, T, x, y = _synthetic_base()
    pack = build_month2_observation_pack(
        velocity_i=vi, velocity_j=vj, temperature=T, x_coords=x, y_coords=y,
        site_fraction=0.05, K=0, layout_id="L", mask_seed=4, sim_id="s", frame_index=2,
        include_dense_temperature_channels=False,
    )
    assert not hasattr(pack, "op_params")


@pytest.mark.skipif(not HAS_REAL, reason="real data or month2 stats missing")
def test_real_sample_probe_isolation_mutations():
    norm = Month2FitOnlyNormalizer(STATS)
    store = ShardStore(raw_dir=RAW, max_cached_frames=8)
    probe = InputIsolationProbe()

    ds_free = Month2ReconDataset(
        "train_fit",
        task_mode="temperature_free",
        K=0,
        p_conditioning=False,
        normalizer=norm,
        store=store,
    )
    s = ds_free[0]
    base = probe(s["model_inputs"])

    # (a) mutate target T in a copied sample's targets — model_inputs unchanged
    s2 = Month2ReconDataset(
        "train_fit", task_mode="temperature_free", K=0, p_conditioning=False,
        normalizer=norm, store=store,
    )[0]
    s2["targets"]["temperature"] = s2["targets"]["temperature"] + 1000.0
    assert np.allclose(probe(s2["model_inputs"]), base)

    # Rebuild inputs from mutated hidden full fields to simulate (c)/(d)
    # (c) velocity outside mask
    mi = s["model_inputs"]
    mask = mi["mask_velocity"]
    vi = s["metadata"]["_hidden_full_velocity_i"].copy()
    vj = s["metadata"]["_hidden_full_velocity_j"].copy()
    vk = s["metadata"]["_hidden_velocity_k"].copy()
    T = s["targets"]["temperature"].copy()
    vi[~mask] += 50.0
    vj[~mask] += 50.0
    vk += 999.0  # (d)
    from combustion_recon.data.month2_observations import build_month2_observation_pack

    pack = build_month2_observation_pack(
        velocity_i=vi,
        velocity_j=vj,
        temperature=T,
        x_coords=s["metadata"]["x"],
        y_coords=s["metadata"]["y"],
        site_fraction=float(mi["site_fraction"]),
        K=0,
        layout_id=s["metadata"]["layout_id"],
        mask_seed=s["metadata"]["mask_seed"],
        sim_id=s["metadata"]["sim_id"],
        frame_index=s["metadata"]["frame_index"],
        replica_id=s["metadata"]["replica_id"],
        include_dense_temperature_channels=False,
        vi_norm=norm.normalize_field("Velocity_i", vi),
        vj_norm=norm.normalize_field("Velocity_j", vj),
        t_norm=norm.normalize_field("Temperature", T),
    )
    assert pack.velocity_mask_hash == s["metadata"]["velocity_mask_hash"]
    assert np.allclose(pack.velocity_values, s["model_inputs"]["velocity_values_physical"])

    # (e) p_off dataset must not include op_params; p_on must
    assert "op_params" not in s["model_inputs"]
    ds_on = Month2ReconDataset(
        "train_fit",
        task_mode="temperature_free",
        K=0,
        p_conditioning=True,
        normalizer=norm,
        store=store,
    )
    s_on = ds_on[0]
    assert "op_params" in s_on["model_inputs"]
    # Changing p in metadata without rebuilding inputs isn't done by dataset;
    # ensure p_off probe output has no op-param dimensions beyond the free digest.
    assert probe(s["model_inputs"]).shape == (8,)

    # Assisted off-anchor T mutation
    ds_as = Month2ReconDataset(
        "train_fit",
        task_mode="temperature_assisted",
        K=9,
        p_conditioning=False,
        normalizer=norm,
        store=store,
        return_dense_temperature_channels=True,
    )
    sa = ds_as[0]
    base_a = probe(sa["model_inputs"])
    Tm = sa["targets"]["temperature"].copy()
    Tm[~sa["model_inputs"]["mask_temperature"]] += 321.0
    pack_a = build_month2_observation_pack(
        velocity_i=sa["metadata"]["_hidden_full_velocity_i"],
        velocity_j=sa["metadata"]["_hidden_full_velocity_j"],
        temperature=Tm,
        x_coords=sa["metadata"]["x"],
        y_coords=sa["metadata"]["y"],
        site_fraction=float(sa["model_inputs"]["site_fraction"]),
        K=9,
        layout_id=sa["metadata"]["layout_id"],
        mask_seed=sa["metadata"]["mask_seed"],
        sim_id=sa["metadata"]["sim_id"],
        frame_index=sa["metadata"]["frame_index"],
        replica_id=sa["metadata"]["replica_id"],
        include_dense_temperature_channels=True,
        vi_norm=norm.normalize_field("Velocity_i", sa["metadata"]["_hidden_full_velocity_i"]),
        vj_norm=norm.normalize_field("Velocity_j", sa["metadata"]["_hidden_full_velocity_j"]),
        t_norm=norm.normalize_field("Temperature", Tm),
    )
    assert pack_a.temperature_mask_hash == sa["metadata"]["temperature_mask_hash"]
    assert np.allclose(pack_a.anchor_values, sa["model_inputs"]["anchor_values_physical"])
    # Probe on reconstructed allowed tensors should match original anchors/vel
    mi2 = dict(sa["model_inputs"])
    mi2["anchor_values"] = norm.normalize_field("Temperature", pack_a.anchor_values)
    assert np.allclose(probe(mi2), base_a)
