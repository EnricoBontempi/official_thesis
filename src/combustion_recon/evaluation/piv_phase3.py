"""Phase-3 helpers (mission 02): evaluation of the EXISTING checkpoints on PIV observations.

``HistoricalSiteGenerator`` adds the pseudo-configurations that need the historical random 5 % sites of mission 00:
  E0  : ideal values at the historical sites                      (replay of the original input; gate against the S7/00 tables)
  E0v : ideal values at historical sites that are valid vectors of PIV-A
  E1  : filtered (A) values at the same nodes as E0v                (filter at historical layout: direct bridge to 00)
  E2  : filtered + correlated error (A) at the same nodes           (error at historical layout)
The sites come from the frozen nested eval-mask payload (same files as the original evaluation); temperature is never used.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from combustion_recon.data.piv_observations import PivObservation, PivObservationGenerator, PivSpec
from combustion_recon.data.sparse_eval_masks import load_nested_sites

PSEUDO = {
    "E0": {"base": "IDEAL-full", "restrict": "historical", "note": "ideal values at the historical 5 % sites (replay)"},
    "E0v": {"base": "IDEAL-A", "restrict": "historical", "note": "historical sites that are valid vectors of A, ideal values"},
    "E1": {"base": "FILT-A", "restrict": "historical", "note": "same nodes as E0v, filtered values (A)"},
    "E2": {"base": "PIV-A", "restrict": "historical", "note": "same nodes as E0v, filtered + error (A)"},
}


def with_pseudo_configs(spec: PivSpec) -> PivSpec:
    raw = copy.deepcopy(dict(spec.raw))
    for name, d in PSEUDO.items():
        c = copy.deepcopy(raw["configs"][d["base"]])
        c.update({"restrict": d["restrict"], "role": "phase3", "note": d["note"]})
        raw["configs"][name] = c
    return PivSpec(raw=raw, sha256=spec.sha256)  # spec hash stays the frozen one: the pseudo-configs add no parameter


class HistoricalSiteGenerator(PivObservationGenerator):
    def __init__(self, spec, scale, *, x_vec_m, y_vec_m, mask_root: str | Path, role_by_sim: Mapping[str, str]):
        super().__init__(spec, scale, x_vec_m=x_vec_m, y_vec_m=y_vec_m)
        self.mask_root, self.role_by_sim = Path(mask_root), dict(role_by_sim)

    def historical_sites(self, sim_id: str, frame_index: int, replica: int) -> np.ndarray:
        role = self.role_by_sim[sim_id]
        out = load_nested_sites(role=role, sim_id=sim_id, frame_index=int(frame_index), replica_id=int(replica), payload_root=self.mask_root)
        return np.asarray(out["sites_5pct"], dtype=np.int64)

    def observe(self, config_id, vi_phys, vj_phys, *, sim_id, frame_index, role, replica=0, **kw) -> PivObservation:
        cfg = self.spec.config(config_id)
        if cfg.get("restrict") == "historical":
            kw["restrict_sites"] = self.historical_sites(sim_id, frame_index, replica)
        return super().observe(config_id, vi_phys, vj_phys, sim_id=sim_id, frame_index=frame_index, role=role, replica=replica, **kw)
