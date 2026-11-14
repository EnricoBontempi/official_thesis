"""Unit tests for Month 1 non-neural baselines (mask/token rules)."""

from __future__ import annotations

import numpy as np

from combustion_recon.data.masks import build_layout_mask, temperature_anchor_mask
from combustion_recon.models.baselines import (
    GappyPODReconstructor,
    GlobalTrainMeanPredictor,
    ParamConditionedMeanPredictor,
)


def _synthetic_frames(n: int = 12, H: int = 32, W: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-0.04, 0.04, W)
    y = np.linspace(0.06, 0.16, H)
    x2d, y2d = np.meshgrid(x, y, indexing="xy")
    frames = []
    for i in range(n):
        nh3 = float(i % 5) * 0.2
        phi = 0.8 + 0.1 * (i % 4)
        vi = rng.normal(size=(H, W))
        vj = 10.0 + rng.normal(size=(H, W))
        vk = 0.1 * rng.normal(size=(H, W))
        # T correlated weakly with |u| and strongly with nh3
        umag = np.sqrt(vi**2 + vj**2 + vk**2)
        T = 1500.0 + 400.0 * nh3 + 5.0 * umag + 20.0 * rng.normal(size=(H, W))
        frames.append(
            {
                "vi": vi,
                "vj": vj,
                "vk": vk,
                "umag": umag,
                "T": T,
                "x2d": x2d,
                "y2d": y2d,
                "nh3": nh3,
                "phi": phi,
                "sim_id": f"{int(nh3*100)}NH3_{phi:.2f}.h5",
                "frame": i,
            }
        )
    return frames


def test_global_train_mean_predictor_shape_and_value():
    frames = _synthetic_frames(8)
    Ts = [f["T"] for f in frames]
    model = GlobalTrainMeanPredictor.fit(Ts)
    pred = model.predict()
    assert pred.shape == frames[0]["T"].shape
    expected = np.mean(np.stack(Ts, axis=0), axis=0)
    np.testing.assert_allclose(pred, expected, rtol=1e-12)


def test_gappy_pod_temperature_free_ignores_unobserved_and_respects_mask():
    frames = _synthetic_frames(16, H=32, W=32)
    pod = GappyPODReconstructor.fit(
        frames, R=8, ridge_lambda=1e-2, use_param_mean=False, max_train_frames=16, seed=0
    )
    fr = frames[0]
    # Poison unobserved velocity sites with huge sentinels — must not leak into T recon
    vi = fr["vi"].copy()
    vj = fr["vj"].copy()
    vk = fr["vk"].copy()
    mask = build_layout_mask("random_5pct", seed=7, height=32, width=32)
    vi[~mask] = 1e6
    vj[~mask] = -1e6
    vk[~mask] = 1e6
    pred_poison = pod.reconstruct(vi, vj, vk, vel_mask=mask, nh3=fr["nh3"], phi=fr["phi"])
    pred_clean = pod.reconstruct(
        fr["vi"], fr["vj"], fr["vk"], vel_mask=mask, nh3=fr["nh3"], phi=fr["phi"]
    )
    np.testing.assert_allclose(pred_poison, pred_clean, rtol=1e-10, atol=1e-8)
    assert pred_clean.shape == fr["T"].shape
    assert np.isfinite(pred_clean).all()


def test_gappy_pod_temperature_assisted_uses_only_masked_T():
    frames = _synthetic_frames(16, H=32, W=32)
    param = ParamConditionedMeanPredictor.fit(
        [f["T"] for f in frames[:5]],
        [(f["nh3"], f["phi"]) for f in frames[:5]],
        ridge_alpha=1e-2,
    )
    pod = GappyPODReconstructor.fit(
        frames,
        R=8,
        ridge_lambda=1e-2,
        use_param_mean=True,
        param_mean_predictor=param,
        max_train_frames=16,
        seed=1,
    )
    fr = frames[3]
    vel_mask = build_layout_mask("random_5pct", seed=3, height=32, width=32)
    t_mask = temperature_anchor_mask(4, seed=11, height=32, width=32)
    T_obs = fr["T"].copy()
    T_poison = T_obs.copy()
    T_poison[~t_mask] = -9999.0  # must be ignored
    pred_a = pod.reconstruct(
        fr["vi"],
        fr["vj"],
        fr["vk"],
        vel_mask=vel_mask,
        T_obs=T_obs,
        T_mask=t_mask,
        nh3=fr["nh3"],
        phi=fr["phi"],
    )
    pred_b = pod.reconstruct(
        fr["vi"],
        fr["vj"],
        fr["vk"],
        vel_mask=vel_mask,
        T_obs=T_poison,
        T_mask=t_mask,
        nh3=fr["nh3"],
        phi=fr["phi"],
    )
    np.testing.assert_allclose(pred_a, pred_b, rtol=1e-10, atol=1e-8)


def test_gappy_pod_empty_mask_returns_mean_field():
    frames = _synthetic_frames(10, H=24, W=24)
    pod = GappyPODReconstructor.fit(frames, R=4, max_train_frames=10, seed=2)
    fr = frames[0]
    empty = np.zeros((24, 24), dtype=bool)
    pred = pod.reconstruct(fr["vi"], fr["vj"], fr["vk"], vel_mask=empty, nh3=0.0, phi=1.0)
    np.testing.assert_allclose(pred, pod.mean_T)
