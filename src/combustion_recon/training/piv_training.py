"""Training / selection / evaluation datasets with virtual-PIV observations (mission 02, Phase 4).

Same pattern as ``nonideal_specialist``: the historical item is built by ``SparseUNetCacheDataset`` machinery, then the
observation packet is replaced by the one of ``PivObservationGenerator`` (frozen spec v1.1).  The model families,
the recipe, the normaliser, the selection protocol and the checkpoint/resume logic are the existing ones.

Namespaces (spec seeds, never the historical or mission-00 ones):
  * TRAIN dataset      : role ``train``,     replica = epoch  (one new error realisation per (frame, epoch));
  * SELECTION / diag.  : role ``selection``, replica = mask replica id (3 replicas), same observation as in training;
  * EVALUATION (Phase 3): role ``eval``,     replica = evaluation replica.
The realisation key has no model seed and no family, so U-Net and GLU receive identical observations.
Temperature is never an input of the generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from combustion_recon.data.nonideal_observations import file_sha256, load_velocity_scale
from combustion_recon.data.piv_observations import (
    ROLE_SEEDS,
    PivObservationGenerator,
    PivSpec,
    load_piv_spec,
)
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset
from combustion_recon.training.sparse_view import SPARSE_CHANNEL_ORDER_TUPLE, build_sparse_unet_input

PIV_SPEC_REL = "configs/observations/piv_observation_spec_v1_1.yaml"
TRAINABLE_CONFIGS = ("IDEAL-A", "PIV-A", "IDEAL-B-in", "PIV-B-in")   # the only observations of the main campaign (T1-T6)
BRIDGE = "HIST-5PCT"                                                   # T7: historical random 5 % sites, ideal values
CELLS = {
    "T1": ("sparse_unet", "IDEAL-A"), "T2": ("sparse_unet", "PIV-A"), "T3": ("sparse_unet", "IDEAL-B-in"), "T4": ("sparse_unet", "PIV-B-in"),
    "T5-ideal": ("adapted_glu", "IDEAL-A"), "T5-piv": ("adapted_glu", "PIV-A"), "T6-ideal": ("adapted_glu", "IDEAL-B-in"),
    "T6-piv": ("adapted_glu", "PIV-B-in"), "T7": ("adapted_glu", BRIDGE),
}


class PivContractError(ValueError):
    pass


def validate_piv_block(block: Mapping[str, Any], *, family: str, tc: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Refuse anything that is not the authorised main-campaign design."""
    cell = block.get("cell")
    if cell not in CELLS:
        raise PivContractError(f"cell must be one of {sorted(CELLS)}")
    fam, obs = CELLS[cell]
    if family != fam or block.get("observation") != obs:
        raise PivContractError(f"cell {cell} is {fam}/{obs}, config says {family}/{block.get('observation')}")
    if tc.get("p_conditioning") is not True:
        raise PivContractError("the PIV campaign is p_on only")
    mb = tc.get("microbatch_size")
    if family == "adapted_glu" and mb != 1:
        raise PivContractError("GLU cells use microbatch_size 1 with accumulation to batch 8 (uniform recipe, R3 matrix)")
    if family == "sparse_unet" and mb not in (None, "null"):
        raise PivContractError("U-Net cells keep the batch-8 forward")
    if block.get("train_role") != "train" or block.get("selection_role") != "selection":
        raise PivContractError("namespaces must be train / selection")
    if {ROLE_SEEDS["train"], ROLE_SEEDS["selection"]} & {20261114, 20261116, 20261118, 20261125, 20261201, 20261202}:
        raise PivContractError("namespace seeds collide with historical seeds")
    spec_path = root / block["spec"]
    if file_sha256(spec_path) != block["spec_sha256"]:
        raise PivContractError("observation spec hash differs from the frozen one")
    out: dict[str, Any] = {"cell": cell, "observation": obs, "spec_path": spec_path}
    if obs != BRIDGE:
        spec = load_piv_spec(spec_path, expected_sha256=block["spec_sha256"])
        if obs not in spec.raw["configs"]:
            raise PivContractError(f"{obs} not in the spec")
        out["spec"] = spec
    return out


def make_generator(spec: PivSpec, stats_path, x_vec_m: np.ndarray, y_vec_m: np.ndarray, expected_stats_sha256: str | None = None) -> PivObservationGenerator:
    scale = load_velocity_scale(stats_path, expected_sha256=expected_stats_sha256)
    return PivObservationGenerator(spec, scale, x_vec_m=x_vec_m, y_vec_m=y_vec_m)


class PivDataset(SparseUNetCacheDataset):
    """Dataset whose observation is a virtual-PIV packet.

    ``obs_role``: 'train' (replica = epoch), 'selection' or 'eval' (replica = ``replica_id``).
    ``config_id``/``cfg_override``/``restrict_by_item``: see ``PivObservationGenerator.observe``.
    """

    def __init__(self, *args, generator: PivObservationGenerator, config_id: str, obs_role: str, cfg_override=None, restrict_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        if obs_role not in ROLE_SEEDS:
            raise PivContractError(f"obs_role {obs_role!r}")
        self.gen, self.config_id, self.obs_role = generator, config_id, obs_role
        self.cfg_override, self.restrict_fn = cfg_override, restrict_fn

    def _replica(self) -> int:
        return int(self.epoch) if self.obs_role == "train" else int(self.replica_id)

    def __getitem__(self, i: int) -> dict[str, Any]:
        idx = self.indices[i]
        meta = dict(self.frames[idx])
        vi_p, vj_p = self._phys_velocities(idx)
        restrict = self.restrict_fn(self, i, meta) if self.restrict_fn is not None else None
        po = self.gen.observe(self.config_id, vi_p, vj_p, sim_id=str(meta["sim_id"]), frame_index=int(meta["frame_index"]), role=self.obs_role,
                              replica=self._replica(), restrict_sites=restrict, cfg_override=self.cfg_override)
        obs = po.obs
        x = build_sparse_unet_input(
            sites=obs.sites, velocity_values_norm=obs.velocity_values_norm, x_scaled=self.x_scaled, y_scaled=self.y_scaled,
            nh3_norm=float(self.p_norm[idx, 0]), phi_norm=float(self.p_norm[idx, 1]), p_conditioning=self.p_conditioning)
        y = np.asarray(self.t_norm[idx], dtype=np.float32)[None, ...]
        model_inputs = {
            "input_tensor": torch.from_numpy(np.array(x, dtype=np.float32, copy=True)), "channel_order": list(SPARSE_CHANNEL_ORDER_TUPLE),
            "p_conditioning": self.p_conditioning, "K": 0, "site_fraction": obs.site_fraction, "task_mode": "temperature_free",
            "mask_mode": "piv", "epoch": self.epoch, "replica_id": self.replica_id, "layout_id": obs.layout_id,
            "velocity_mask_seed": obs.velocity_mask_seed, "mask_hash": obs.mask_hash, "n_sites": obs.n_sites,
        }
        return {
            "model_inputs": model_inputs,
            "targets": {"temperature_normalized": torch.from_numpy(np.array(y, dtype=np.float32, copy=True)),
                        "temperature": torch.from_numpy(np.array(self.t_K[idx], dtype=np.float32, copy=True)[None, ...])},
            "observation": obs, "glu_tokens": obs.to_glu_tokens(), "baseline_physical": obs.to_baseline_physical(),
            "piv": po,
            "metadata": {**meta, "cache_index": idx, "mask_hash": obs.mask_hash, "velocity_mask_seed": obs.velocity_mask_seed,
                         "site_fraction": obs.site_fraction, "replica_id": self._replica(), "layout_id": obs.layout_id,
                         "p_conditioning": self.p_conditioning, "epoch": self.epoch, "mask_mode": "piv",
                         "piv_config": self.config_id, "piv_namespace": self.obs_role, "piv_obs_hash": po.obs_hash,
                         "piv_n_observed": po.meta["n_observed"]},
        }
