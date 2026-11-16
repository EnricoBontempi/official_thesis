"""Month-2 baseline adapter invariants (no full experiment)."""

from __future__ import annotations

import numpy as np

from combustion_recon.models.baselines import (
    GappyPODReconstructor,
    GlobalTrainMeanPredictor,
    apply_anchor_offset,
)
from combustion_recon.models.month2_baseline_adapters import (
    observed_grids_from_model_inputs,
    predict_gappy,
)


def _toy_pod():
    rng = np.random.default_rng(0)
    H = W = 16
    frames = []
    for i in range(12):
        frames.append(
            {
                "vi": rng.normal(size=(H, W)),
                "vj": rng.normal(size=(H, W)),
                "vk": np.zeros((H, W)),
                "T": 2000 + 20 * rng.normal(size=(H, W)),
                "nh3": 0.2,
                "phi": 1.0,
            }
        )
    return GappyPODReconstructor.fit(
        frames, R=4, ridge_lambda=0.01, w_T=5.0, n_velocity_components=2, seed=0
    )


def test_observed_grids_ignore_hidden_and_vk():
    H = W = 8
    mask = np.zeros((H, W), dtype=bool)
    mask[1, 1] = mask[2, 3] = True
    sites = np.array([1 * W + 1, 2 * W + 3], dtype=np.int64)
    vals = np.array([[1.0, 2.0], [3.0, 4.0]])
    tmask = np.zeros((H, W), dtype=bool)
    tmask[4, 4] = True
    mi = {
        "mask_velocity": mask,
        "velocity_site_index": sites,
        "velocity_values_physical": vals,
        "mask_temperature": tmask,
        "anchor_site_index": np.array([4 * W + 4]),
        "anchor_values_physical": np.array([2100.0]),
        "K": 1,
    }
    g = observed_grids_from_model_inputs(mi)
    assert g["vi"][1, 1] == 1.0 and g["vj"][2, 3] == 4.0
    assert g["vi"][0, 0] == 0.0
    assert np.all(g["vk"] == 0.0)
    assert g["T_obs"][4, 4] == 2100.0
    assert g["T_obs"][0, 0] == 0.0


def test_ao_preserves_centered_rmse():
    rng = np.random.default_rng(1)
    T0 = 2000 + rng.normal(size=(16, 16))
    true = 2000 + rng.normal(size=(16, 16))
    tmask = np.zeros_like(true, dtype=bool)
    tmask.ravel()[:9] = True
    T_obs = true.copy()
    T_ao, b = apply_anchor_offset(T0, T_obs, tmask, tau=1.0)
    assert b != 0.0

    def crmse(p, t):
        e = p - t
        return float(np.sqrt(np.mean((e - e.mean()) ** 2)))

    assert abs(crmse(T0, true) - crmse(T_ao, true)) < 1e-9


def test_gappy_k0_tblock_matches_free_and_hidden_vel_irrelevant():
    pod = _toy_pod()
    H = W = 16
    rng = np.random.default_rng(2)
    mask = np.zeros((H, W), dtype=bool)
    idx = rng.choice(H * W, size=20, replace=False)
    mask.ravel()[idx] = True
    vi = rng.normal(size=(H, W))
    vj = rng.normal(size=(H, W))
    ys, xs = np.where(mask)
    sites = ys * W + xs
    vals = np.stack([vi[ys, xs], vj[ys, xs]], axis=1)
    mi = {
        "mask_velocity": mask,
        "velocity_site_index": sites,
        "velocity_values_physical": vals,
        "mask_temperature": np.zeros((H, W), dtype=bool),
        "anchor_site_index": np.zeros((0,), dtype=np.int64),
        "anchor_values_physical": np.zeros((0,)),
        "K": 0,
    }
    p0 = predict_gappy(pod, mi, use_temperature_block=False, p_conditioning=False)
    p1 = predict_gappy(pod, mi, use_temperature_block=True, p_conditioning=False)
    assert np.allclose(p0, p1)

    # Mutating values outside mask cannot change inputs (inputs only store observed sites)
    vi2 = vi.copy()
    vi2[~mask] += 50
    vals2 = np.stack([vi2[ys, xs], vj[ys, xs]], axis=1)
    assert np.allclose(vals2, vals)
    mi2 = dict(mi)
    mi2["velocity_values_physical"] = vals2
    p2 = predict_gappy(pod, mi2, use_temperature_block=False, p_conditioning=False)
    assert np.allclose(p0, p2)


def test_global_mean_fit_smoke():
    fields = [np.ones((4, 4)) * k for k in (1.0, 3.0)]
    m = GlobalTrainMeanPredictor.fit(fields)
    assert np.allclose(m.predict(), 2.0)


def test_t_coord_pad_for_short_timeline_shard():
    """40NH3_1.2.h5 has shape_t=2001 but t_shape=2000; protocol uses index 2000."""
    from pathlib import Path

    from combustion_recon.data.shard_access import ShardStore

    raw = Path("data/raw/combustion/hf_dataset/numerical")
    if not (raw / "data-00000-of-00030.arrow").exists():
        return  # skip offline
    store = ShardStore(raw_dir=raw, max_cached_frames=4)
    sid = "40NH3_1.2.h5"
    store.prefetch(sid, [1999, 2000])
    f0 = store.get_frame(sid, 1999)
    f1 = store.get_frame(sid, 2000)
    assert f1.frame_index == 2000
    assert store.stats["t_coord_pads"] >= 1
    assert np.isfinite(f1.t)
    assert f1.t >= f0.t
