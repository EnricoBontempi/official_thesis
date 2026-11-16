"""Adapters that call Month-1 baselines using only Month-2 allowed observations.

At inference, full-grid arrays are reconstructed from observed values + masks.
Unobserved entries are placeholders (zeros) and must never be read as physics;
GappyPOD / AO index only via masks. vk is a zero placeholder in the 2C path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from combustion_recon.models.baselines import (
    GappyPODReconstructor,
    GlobalTrainMeanPredictor,
    ParamConditionedMeanPredictor,
    apply_anchor_offset,
)


def observed_grids_from_model_inputs(model_inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Build inference containers from model_inputs only (no metadata hidden fields)."""
    mask_v = np.asarray(model_inputs["mask_velocity"], dtype=bool)
    H, W = mask_v.shape
    vi = np.zeros((H, W), dtype=np.float64)
    vj = np.zeros((H, W), dtype=np.float64)
    vk = np.zeros((H, W), dtype=np.float64)  # unused in 2C reconstruct
    sites = np.asarray(model_inputs["velocity_site_index"], dtype=np.int64)
    # Prefer physical velocity values for baselines (trained in physical units)
    if "velocity_values_physical" in model_inputs:
        vals = np.asarray(model_inputs["velocity_values_physical"], dtype=np.float64)
    else:
        vals = np.asarray(model_inputs["velocity_values"], dtype=np.float64)
    if sites.size:
        ys = sites // W
        xs = sites % W
        vi[ys, xs] = vals[:, 0]
        vj[ys, xs] = vals[:, 1]

    t_mask = np.asarray(model_inputs["mask_temperature"], dtype=bool)
    T_obs = np.zeros((H, W), dtype=np.float64)
    if int(model_inputs.get("K", 0)) > 0 and t_mask.any():
        a_sites = np.asarray(model_inputs["anchor_site_index"], dtype=np.int64)
        if "anchor_values_physical" in model_inputs:
            a_vals = np.asarray(model_inputs["anchor_values_physical"], dtype=np.float64)
        else:
            a_vals = np.asarray(model_inputs["anchor_values"], dtype=np.float64)
        ay = a_sites // W
        ax = a_sites % W
        T_obs[ay, ax] = a_vals
    return {
        "vi": vi,
        "vj": vj,
        "vk": vk,
        "vel_mask": mask_v,
        "T_obs": T_obs,
        "T_mask": t_mask,
    }


def op_params_from_model_inputs(model_inputs: dict[str, Any]) -> tuple[float | None, float | None]:
    if "op_params" not in model_inputs:
        return None, None
    op = model_inputs["op_params"]
    return float(op["nh3_fraction"]), float(op["phi"])


def predict_global_mean(model: GlobalTrainMeanPredictor) -> np.ndarray:
    return model.predict()


def predict_param_mean(
    model: ParamConditionedMeanPredictor,
    model_inputs: dict[str, Any],
) -> np.ndarray:
    nh3, phi = op_params_from_model_inputs(model_inputs)
    if nh3 is None or phi is None:
        raise ValueError("ParamMean requires p_on model_inputs with op_params")
    return model.predict(nh3, phi)


def predict_param_mean_ao(
    model: ParamConditionedMeanPredictor,
    model_inputs: dict[str, Any],
    *,
    tau: float = 1.0,
) -> np.ndarray:
    nh3, phi = op_params_from_model_inputs(model_inputs)
    if nh3 is None or phi is None:
        raise ValueError("ParamMean+AO requires p_on")
    grids = observed_grids_from_model_inputs(model_inputs)
    T0 = model.predict(nh3, phi)
    T_cal, _ = apply_anchor_offset(T0, grids["T_obs"], grids["T_mask"], tau=tau)
    return T_cal


def predict_gappy(
    model: GappyPODReconstructor,
    model_inputs: dict[str, Any],
    *,
    use_temperature_block: bool = False,
    anchor_offset: bool = False,
    anchor_tau: float = 1.0,
    p_conditioning: bool = False,
) -> np.ndarray:
    grids = observed_grids_from_model_inputs(model_inputs)
    if p_conditioning:
        nh3, phi = op_params_from_model_inputs(model_inputs)
        if nh3 is None or phi is None:
            raise ValueError("GappyPOD p_on requires op_params in model_inputs")
    else:
        # Must not read p from elsewhere
        nh3, phi = 0.0, 1.0
        if model.use_param_mean:
            raise ValueError("Model is conditioned but adapter called with p_conditioning=False")
    return model.reconstruct(
        grids["vi"],
        grids["vj"],
        grids["vk"],
        vel_mask=grids["vel_mask"],
        T_obs=grids["T_obs"] if (use_temperature_block or anchor_offset) else None,
        T_mask=grids["T_mask"] if (use_temperature_block or anchor_offset) else None,
        nh3=nh3,
        phi=phi,
        use_temperature_block=use_temperature_block,
        anchor_offset=anchor_offset,
        anchor_tau=anchor_tau,
    )
