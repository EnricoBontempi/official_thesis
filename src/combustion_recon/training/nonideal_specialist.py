"""Specialist retraining on ONE degradation regime (N1, N2, G1, G1-eq, N1G1).

* Training observations: the historical TRAIN 5% sites (epoch-dependent TRAIN masks, unchanged), then the
  frozen Block I operators with realizations drawn in a TRAIN namespace: key = (sim_id, frame_index, mask epoch).
  The key does not contain the model seed or the family, so the same sample at the same epoch receives the same
  noise / region in every specialist of the same regime. N1/N2 share the standardized noise (N2 = 3 x N1);
  G1, N1G1 and G1-eq share the region (G1-eq removes the same number of sites, uniformly).
* Selection stays the original clean one (VAL 5%, three replicas). The specialist-scenario VAL metric is only
  recorded, on fixed SELECTION-namespace realizations, and never used for the scheduler, early stop or best.
* T is never used to generate noise, regions or sites (the operators have no T argument).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from combustion_recon.data.nonideal_observations import (
    HISTORICAL_SEEDS,
    CellSpec,
    DegradationSpec,
    VelocityScale,
    file_sha256,
    load_degradation_spec,
    load_velocity_scale,
)
from combustion_recon.evaluation.nonideal_eval import degrade_item
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset
from combustion_recon.training.sparse_scientific_loop import SparseScientificTrainer

REGIMES = ("N1", "N2", "G1", "G1-eq", "N1G1")
EVAL_SEED = 20261125  # Block II evaluation namespace (frozen)
TRAIN_SEED = 20261201
SELECTION_SEED = 20261202
GLU_MICROBATCH = 1


class SpecialistContractError(ValueError):
    pass


def namespace_spec(spec: DegradationSpec, seed: int) -> DegradationSpec:
    if seed in HISTORICAL_SEEDS or seed == EVAL_SEED:
        raise SpecialistContractError(f"seed {seed} collides with a historical or evaluation seed")
    return dataclasses.replace(spec, degradation_seed=int(seed))


def validate_specialist_block(block: dict[str, Any], *, family: str, tc: dict[str, Any], root: Path,
                              allow_identity: bool = False) -> dict[str, Any]:
    """Refuse anything that is not the authorised specialist design. Returns resolved objects.

    ``allow_identity`` (smoke only) accepts regime "R0": the identity cell, used to check that the
    specialist code path reproduces the original training path when nothing is degraded.
    """
    if block.get("regime") not in REGIMES and not (allow_identity and block.get("regime") == "R0"):
        raise SpecialistContractError(f"regime must be one of {REGIMES}")
    if int(block.get("train_degradation_seed", -1)) != TRAIN_SEED or int(block.get("selection_degradation_seed", -1)) != SELECTION_SEED:
        raise SpecialistContractError("train/selection degradation seeds differ from the frozen contract")
    if len({TRAIN_SEED, SELECTION_SEED, EVAL_SEED}) != 3 or {TRAIN_SEED, SELECTION_SEED} & HISTORICAL_SEEDS:
        raise SpecialistContractError("namespace seeds must be distinct from each other and from historical seeds")
    if block.get("clean_fraction") not in (0, 0.0):
        raise SpecialistContractError("no clean samples in a specialist: clean_fraction must be 0")
    if tc.get("p_conditioning") is not True:
        raise SpecialistContractError("specialists are p_on only")
    mb = tc.get("microbatch_size")
    if family == "adapted_glu" and mb != GLU_MICROBATCH:
        raise SpecialistContractError("GLU specialists must use microbatch_size 1 (gradient accumulation to batch 8)")
    if family == "sparse_unet" and mb not in (None, "null"):
        raise SpecialistContractError("U-Net specialists keep the original batch-8 forward (microbatch null)")
    spec_path = root / block["degradation_spec"]
    if file_sha256(spec_path) != block["degradation_spec_sha256"]:
        raise SpecialistContractError("degradation spec hash differs from the frozen one")
    spec = load_degradation_spec(spec_path)
    if spec.degradation_seed != EVAL_SEED:
        raise SpecialistContractError("the base spec must carry the evaluation seed (it is namespaced for TRAIN/SELECTION here)")
    cell = spec.cells[block["regime"]]
    return {"spec": spec, "cell": cell}


class SpecialistTrainDataset(SparseUNetCacheDataset):
    """Training dataset with the regime applied after the historical TRAIN 5% sampling."""

    def __init__(self, *args, cell: CellSpec, spec: DegradationSpec, scale: VelocityScale, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell, self.scale = cell, scale
        self.spec = namespace_spec(spec, TRAIN_SEED)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = super().__getitem__(i)
        if self.mask_mode != "training":
            raise RuntimeError("SpecialistTrainDataset must stay in training mask mode")
        new, deg = degrade_item(self, item, self.cell, self.spec, self.scale, replica_id=int(self.epoch))
        new["metadata"] = {**new["metadata"], "regime": self.cell.cell_id, "n_surviving": deg.meta["n_surviving"],
                           "degradation_namespace": "train"}
        return new


class SpecialistSelectionDataset(SparseUNetCacheDataset):
    """VAL dataset with the regime applied on fixed SELECTION-namespace realizations (replica = mask replica)."""

    def __init__(self, *args, cell: CellSpec, spec: DegradationSpec, scale: VelocityScale, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell, self.scale = cell, scale
        self.spec = namespace_spec(spec, SELECTION_SEED)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = super().__getitem__(i)
        new, deg = degrade_item(self, item, self.cell, self.spec, self.scale, replica_id=int(self.replica_id))
        new["metadata"] = {**new["metadata"], "regime": self.cell.cell_id, "n_surviving": deg.meta["n_surviving"],
                           "degradation_namespace": "selection"}
        return new


class SpecialistTrainer(SparseScientificTrainer):
    """Same trainer; additionally records the specialist-scenario VAL macro (never used for selection)."""

    def __init__(self, *args, specialist_val_ds: SparseUNetCacheDataset, regime: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.specialist_val_ds = specialist_val_ds
        self.regime = regime

    def evaluate_val_replicas(self, val_ds, *, method, variant, model_seed, extra_tags=None) -> dict[str, Any]:
        out = super().evaluate_val_replicas(val_ds, method=method, variant=variant, model_seed=model_seed, extra_tags=extra_tags)
        bs = 1 if self.cfg.family == "adapted_glu" else 8  # ragged N: homogeneous batches for the GLU
        per = []
        for rid in self.cfg.val_replicas:
            r = self.evaluate_dataset(
                self.specialist_val_ds, role="val_eval", replica_id=int(rid), method=method, variant=variant,
                model_seed=model_seed, batch_size=bs,
                extra_tags={**(extra_tags or {}), "selection": False, "specialist_regime": self.regime},
            )
            per.append({"replica_id": int(rid), "mean_condition_rmse_K": r["mean_condition_rmse_K"]})
        vals = [p["mean_condition_rmse_K"] for p in per if p["mean_condition_rmse_K"] is not None]
        out["extra_eval_fields"] = {
            "specialist_regime": self.regime,
            "specialist_val_mean_condition_rmse_K": float(np.mean(vals)) if vals else None,
            "specialist_val_per_replica": per,
            "specialist_val_used_for_selection": False,
        }
        return out
