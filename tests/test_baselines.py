"""Unit tests for Month 1 non-neural baselines (mask/token rules)."""

from __future__ import annotations

import numpy as np

from combustion_recon.data.masks import build_layout_mask, temperature_anchor_mask
from combustion_recon.models.baselines import (
    GappyPODReconstructor,
    GlobalTrainMeanPredictor,
    ParamConditionedMeanPredictor,
    apply_anchor_offset,
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


def test_gappy_pod_stores_modal_variances_and_anchor_offset_reduces_bias():
    frames = _synthetic_frames(20, H=32, W=32, seed=3)
    pod = GappyPODReconstructor.fit(
        frames, R=8, ridge_lambda=1e-3, w_T=20.0, max_train_frames=None, seed=0
    )
    assert pod.modal_variances.shape == (8,)
    assert np.all(pod.modal_variances > 0)
    fr = frames[-1]
    # Inject a large global T offset into the "true" field relative to mean
    T_shift = fr["T"] + 80.0
    vel_mask = build_layout_mask("random_5pct", seed=9, height=32, width=32)
    t_mask = temperature_anchor_mask(9, seed=12, height=32, width=32, nested=True)
    pred_tf = pod.reconstruct(
        fr["vi"], fr["vj"], fr["vk"], vel_mask=vel_mask, nh3=fr["nh3"], phi=fr["phi"]
    )
    pred_ao = pod.reconstruct(
        fr["vi"],
        fr["vj"],
        fr["vk"],
        vel_mask=vel_mask,
        T_obs=T_shift,
        T_mask=t_mask,
        nh3=fr["nh3"],
        phi=fr["phi"],
        use_temperature_block=False,
        anchor_offset=True,
    )
    bias_tf = float((pred_tf - T_shift).mean())
    bias_ao = float((pred_ao - T_shift).mean())
    assert abs(bias_ao) < abs(bias_tf)


def test_apply_anchor_offset_exact_on_constant_bias():
    T0 = np.full((8, 8), 1000.0)
    T_obs = np.full((8, 8), 1050.0)
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    pred, b = apply_anchor_offset(T0, T_obs, mask, tau=0.0)
    assert abs(b - 50.0) < 1e-12
    np.testing.assert_allclose(pred, 1050.0)


def test_param_mean_loco_auto_and_no_intercept_penalty():
    from combustion_recon.models.baselines import _PENALTY

    assert _PENALTY[0, 0] == 0.0
    assert np.all(np.diag(_PENALTY)[1:] == 1.0)
    # Synthetic condition means with clear NH3 + phi dependence
    H = W = 16
    fields, ops = [], []
    for nh3 in (0.0, 0.2, 0.4, 0.8):
        for phi in (0.85, 1.0, 1.15, 1.25):
            T = np.full((H, W), 1500.0 + 500.0 * nh3 + 200.0 * (phi - 1.0) + 50.0 * (phi - 1.0) ** 2)
            fields.append(T)
            ops.append((nh3, phi))
    model = ParamConditionedMeanPredictor.fit(fields, ops, ridge_alpha="auto")
    assert model.selected_basis in ("quadratic", "piecewise_phi1")
    assert model.selected_ridge_alpha in (1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    assert np.isfinite(model.loco_rmse_K)
    pred = model.predict(0.4, 1.0)
    assert pred.shape == (H, W)
    # Should recover near the generating mean at a train-like point
    assert abs(float(pred.mean()) - (1500.0 + 200.0)) < 30.0


def test_oracle_pod_projection_beats_sparse_gappy():
    frames = _synthetic_frames(24, H=32, W=32, seed=5)
    pod = GappyPODReconstructor.fit(frames, R=8, max_train_frames=None, seed=0)
    fr = frames[0]
    oracle = pod.project_oracle_T(fr["T"], nh3=fr["nh3"], phi=fr["phi"])
    mask = build_layout_mask("random_5pct", seed=1, height=32, width=32)
    sparse = pod.reconstruct(
        fr["vi"], fr["vj"], fr["vk"], vel_mask=mask, nh3=fr["nh3"], phi=fr["phi"]
    )
    rmse_o = float(np.sqrt(np.mean((oracle - fr["T"]) ** 2)))
    rmse_s = float(np.sqrt(np.mean((sparse - fr["T"]) ** 2)))
    assert rmse_o < rmse_s
    assert np.isfinite(oracle).all()
