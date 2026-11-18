"""Model baselines and neural reconstructors.

GLU symbols are loaded lazily so U-Net-only paths do not import vendored
``external/GLU_Reconstruction`` (and its ``einops`` dependency) at package import.
"""

from __future__ import annotations

from typing import Any

from combustion_recon.models.baselines import (
    GappyPODReconstructor,
    GlobalTrainMeanPredictor,
    LocalVelocityTemperatureRegressor,
    ParamConditionedMeanPredictor,
    apply_anchor_offset,
)
from combustion_recon.models.unet import (
    DenseUNet,
    SparseUNet,
    build_dense_unet_from_config,
    build_sparse_unet_from_config,
)

_GLU_EXPORTS = {
    "AdaptedDeterministicGLU",
    "build_glu_from_config",
    "load_glu_minimal_config",
    "phys_to_isotropic_q",
    "domain_geometry_from_axes",
    "domain_geometry_from_center_L",
    "load_domain_geometry_npz",
    "topk_distance_then_site_index",
    "make_synthetic_query_grid",
}

__all__ = [
    "GappyPODReconstructor",
    "GlobalTrainMeanPredictor",
    "LocalVelocityTemperatureRegressor",
    "ParamConditionedMeanPredictor",
    "apply_anchor_offset",
    "DenseUNet",
    "SparseUNet",
    "build_dense_unet_from_config",
    "build_sparse_unet_from_config",
    *sorted(_GLU_EXPORTS),
]


def __getattr__(name: str) -> Any:
    if name in _GLU_EXPORTS:
        from combustion_recon.models import glu_reconstructor as _glu

        return getattr(_glu, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
