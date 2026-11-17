"""Executable isolation / invariance checks for Month-2 baselines.

Mutations are applied to **source frames before** sample construction so that
predictions can change when observed inputs change, and must not change when
only hidden fields change.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from combustion_recon.models.baselines import (
    GappyPODReconstructor,
    GlobalTrainMeanPredictor,
    ParamConditionedMeanPredictor,
)
from combustion_recon.models.month2_baseline_adapters import (
    observed_grids_from_model_inputs,
    predict_gappy,
    predict_global_mean,
    predict_param_mean,
)


def _centered_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    err = pred - true
    return float(np.sqrt(np.mean((err - err.mean()) ** 2)))


def _copy_frame(fr: dict[str, Any]) -> dict[str, Any]:
    out = dict(fr)
    for k in ("T", "vi", "vj", "vk", "umag", "x2d", "y2d"):
        if k in fr and isinstance(fr[k], np.ndarray):
            out[k] = fr[k].copy()
    return out


def run_isolation_battery(
    *,
    frames: list[dict[str, Any]],
    make_sample: Callable[..., dict[str, Any]],
    pods_free: dict[bool, GappyPODReconstructor],
    pods_tblock: dict[bool, GappyPODReconstructor],
    global_mean: GlobalTrainMeanPredictor,
    param_mean: ParamConditionedMeanPredictor,
    mask_seed: int,
    site_fraction: float,
    atol: float = 1e-8,
    max_frames: int = 2,
) -> dict[str, Any]:
    """Return measured invariants with tolerances; raises AssertionError on failure."""
    if not frames:
        raise ValueError("isolation battery requires at least one frame")
    use = frames[:max_frames]
    cases = 0
    records: list[dict[str, Any]] = []

    for fr0 in use:
        for p_on in (False, True):
            cases += 1
            fr = _copy_frame(fr0)
            pod_free = pods_free[p_on]
            pod_tb = pods_tblock[p_on]

            s0 = make_sample(
                fr,
                K=0,
                p_conditioning=p_on,
                replica_id=0,
                mask_seed=mask_seed,
                site_fraction=site_fraction,
            )
            s9 = make_sample(
                fr,
                K=9,
                p_conditioning=p_on,
                replica_id=0,
                mask_seed=mask_seed,
                site_fraction=site_fraction,
            )
            assert np.array_equal(
                s0["model_inputs"]["mask_velocity"], s9["model_inputs"]["mask_velocity"]
            ), "velocity mask must match across K"

            grids0 = observed_grids_from_model_inputs(s0["model_inputs"])
            vel_mask = grids0["vel_mask"]

            p_base = predict_gappy(
                pod_free,
                s0["model_inputs"],
                use_temperature_block=False,
                p_conditioning=p_on,
            )

            # Hidden velocity mutation
            fr_hid = _copy_frame(fr)
            fr_hid["vi"] = fr_hid["vi"].copy()
            fr_hid["vj"] = fr_hid["vj"].copy()
            fr_hid["vi"][~vel_mask] += 50.0
            fr_hid["vj"][~vel_mask] -= 25.0
            s_hid0 = make_sample(
                fr_hid,
                K=0,
                p_conditioning=p_on,
                replica_id=0,
                mask_seed=mask_seed,
                site_fraction=site_fraction,
            )
            p_hid = predict_gappy(
                pod_free,
                s_hid0["model_inputs"],
                use_temperature_block=False,
                p_conditioning=p_on,
            )
            d_hid = float(np.max(np.abs(p_base - p_hid)))
            assert d_hid <= atol, f"hidden velocity changed free pred: {d_hid}"

            # vk source mutation (adapters ignore vk via zero placeholder)
            fr_vk = _copy_frame(fr)
            fr_vk["vk"] = np.zeros_like(fr_vk["vi"]) + 99.0
            s_vk = make_sample(
                fr_vk,
                K=0,
                p_conditioning=p_on,
                replica_id=0,
                mask_seed=mask_seed,
                site_fraction=site_fraction,
            )
            p_vk = predict_gappy(
                pod_free, s_vk["model_inputs"], use_temperature_block=False, p_conditioning=p_on
            )
            d_vk = float(np.max(np.abs(p_base - p_vk)))
            assert d_vk <= atol, f"vk source mutation changed 2C pred: {d_vk}"

            # Free ignores T
            fr_T = _copy_frame(fr)
            fr_T["T"] = fr_T["T"].copy() + 300.0
            s_Tfree = make_sample(
                fr_T,
                K=0,
                p_conditioning=p_on,
                replica_id=0,
                mask_seed=mask_seed,
                site_fraction=site_fraction,
            )
            p_Tfree = predict_gappy(
                pod_free,
                s_Tfree["model_inputs"],
                use_temperature_block=False,
                p_conditioning=p_on,
            )
            d_Tfree = float(np.max(np.abs(p_base - p_Tfree)))
            assert d_Tfree <= atol, f"free changed when T mutated: {d_Tfree}"

            # Assisted: off-anchor T mutation; anchors held fixed
            tmask = np.asarray(s9["model_inputs"]["mask_temperature"], dtype=bool)
            fr_off = _copy_frame(fr)
            fr_off["T"] = fr_off["T"].copy()
            fr_off["T"][~tmask] += 400.0
            fr_off["T"][tmask] = fr["T"][tmask]
            s9_off = make_sample(
                fr_off,
                K=9,
                p_conditioning=p_on,
                replica_id=0,
                mask_seed=mask_seed,
                site_fraction=site_fraction,
            )
            p_tb = predict_gappy(
                pod_tb, s9["model_inputs"], use_temperature_block=True, p_conditioning=p_on
            )
            p_tb_off = predict_gappy(
                pod_tb, s9_off["model_inputs"], use_temperature_block=True, p_conditioning=p_on
            )
            d_off = float(np.max(np.abs(p_tb - p_tb_off)))
            assert d_off <= atol, f"off-anchor T changed T-block: {d_off}"
            assert np.allclose(
                s9["model_inputs"]["anchor_values_physical"],
                s9_off["model_inputs"]["anchor_values_physical"],
            )

            # Positive: observed velocity must move Gappy
            fr_obs = _copy_frame(fr)
            fr_obs["vi"] = fr_obs["vi"].copy()
            fr_obs["vi"][vel_mask] += 3.0
            s_obs = make_sample(
                fr_obs,
                K=0,
                p_conditioning=p_on,
                replica_id=0,
                mask_seed=mask_seed,
                site_fraction=site_fraction,
            )
            p_obs = predict_gappy(
                pod_free, s_obs["model_inputs"], use_temperature_block=False, p_conditioning=p_on
            )
            d_obs = float(np.max(np.abs(p_base - p_obs)))
            assert d_obs > 1e-4, f"observed velocity mutation did not change Gappy: {d_obs}"

            # GlobalMean ignores sensors
            g0 = predict_global_mean(global_mean)
            assert np.allclose(g0, predict_global_mean(global_mean))

            if p_on:
                p_pm = predict_param_mean(param_mean, s0["model_inputs"])
                p_pm_obs = predict_param_mean(param_mean, s_obs["model_inputs"])
                assert np.allclose(p_pm, p_pm_obs), "ParamMean must ignore velocity"
                mi_p = dict(s0["model_inputs"])
                op = dict(mi_p["op_params"])
                op["nh3_fraction"] = float(op["nh3_fraction"]) + 0.15
                mi_p["op_params"] = op
                p_pm2 = predict_param_mean(param_mean, mi_p)
                assert float(np.max(np.abs(p_pm - p_pm2))) > 1e-4
            else:
                mi_fake = dict(s0["model_inputs"])
                mi_fake["op_params"] = {
                    "nh3_fraction": 0.99,
                    "phi": 1.4,
                    "equivalence_ratio": 1.4,
                }
                p_fake = predict_gappy(
                    pod_free,
                    mi_fake,
                    use_temperature_block=False,
                    p_conditioning=False,
                )
                assert float(np.max(np.abs(p_base - p_fake))) <= atol

            # K=0 same model/hyperparams: T-block flag == free
            p_free_k0 = predict_gappy(
                pod_tb, s0["model_inputs"], use_temperature_block=False, p_conditioning=p_on
            )
            p_tb_k0 = predict_gappy(
                pod_tb, s0["model_inputs"], use_temperature_block=True, p_conditioning=p_on
            )
            d_k0 = float(np.max(np.abs(p_free_k0 - p_tb_k0)))
            assert d_k0 <= atol, f"K=0 Tblock flag != free: {d_k0}"

            p_free9 = predict_gappy(
                pod_free,
                s9["model_inputs"],
                use_temperature_block=False,
                anchor_offset=False,
                p_conditioning=p_on,
            )
            p_ao = predict_gappy(
                pod_free,
                s9["model_inputs"],
                use_temperature_block=False,
                anchor_offset=True,
                anchor_tau=1.0,
                p_conditioning=p_on,
            )
            true = fr["T"]
            c0 = _centered_rmse(p_free9, true)
            c1 = _centered_rmse(p_ao, true)
            assert abs(c0 - c1) <= 1e-9, f"AO changed Centered-RMSE: {c0} vs {c1}"

            records.append(
                {
                    "sim_id": fr["sim_id"],
                    "frame": int(fr["frame"]),
                    "p_conditioning": p_on,
                    "hidden_vel_maxabs": d_hid,
                    "vk_maxabs": d_vk,
                    "free_T_mut_maxabs": d_Tfree,
                    "off_anchor_T_maxabs": d_off,
                    "observed_vel_maxabs": d_obs,
                    "K0_tblock_vs_free": d_k0,
                    "AO_centered_rmse_absdiff": abs(c0 - c1),
                }
            )

    return {
        "n_cases": cases,
        "atol": atol,
        "max_frames": max_frames,
        "passed": True,
        "records": records,
        "summary": {
            "max_hidden_vel": max(r["hidden_vel_maxabs"] for r in records),
            "max_off_anchor": max(r["off_anchor_T_maxabs"] for r in records),
            "min_observed_response": min(r["observed_vel_maxabs"] for r in records),
            "max_AO_crmse_diff": max(r["AO_centered_rmse_absdiff"] for r in records),
        },
    }
