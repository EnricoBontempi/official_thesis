"""Temperature-assisted dataset: the free sparse observation plus K observed temperature anchors."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from combustion_recon.data.t_anchors import H, W, K_VALUES, AnchorManifest, train_sites
from combustion_recon.models.unet import ASSISTED_CHANNEL_ORDER
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset


_PAYLOAD_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _payload_table(root, role: str) -> dict[str, Any]:
    """Persisted nested eval sites (5%) decompressed ONCE per process: the free loader reopens the npz for every frame."""
    key = (str(root), role)
    if key not in _PAYLOAD_CACHE:
        z = np.load(f"{root}/{role}_nested_sites.npz", allow_pickle=True)
        sims, frames, reps = z["sim_id"].tolist(), np.asarray(z["frame_index"]), np.asarray(z["replica_id"])
        _PAYLOAD_CACHE[key] = {
            "sites": np.asarray(z["sites_5pct"], dtype=np.int64), "hash": [str(x) for x in z["hash_5pct"]],
            "seed": np.asarray(z["velocity_mask_seed"], dtype=np.int64),
            "row": {(str(s), int(f), int(r)): i for i, (s, f, r) in enumerate(zip(sims, frames, reps))},
        }
        z.close()
    return _PAYLOAD_CACHE[key]


class AssistedDataset(SparseUNetCacheDataset):
    """Items are ``ds[pos]`` (K = ``eval_K``, eval layout of ``replica_id``) or ``ds[(pos, K)]`` (TRAIN: random anchors).

    Only the OBSERVED temperature values enter the model input; hidden T is never read. T values come from the
    normalized target with the frozen fit-only normalizer (ideal, noise-free).
    """

    def __init__(self, *args: Any, anchors: AnchorManifest, eval_K: int = 0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.anchors = anchors
        self.eval_K = int(eval_K)

    def _resolve_sites(self, idx: int, meta: dict[str, Any]):  # type: ignore[override]
        if self.mask_mode == "evaluation" and abs(float(self.site_fraction) - 0.05) < 1e-12 and self.require_eval_mask_payload:
            tab = _payload_table(self.eval_mask_payload_root, self.role)
            row = tab["row"].get((str(meta["sim_id"]), int(meta["frame_index"]), int(self.replica_id)))
            if row is None:
                raise KeyError(f"required eval mask payload missing for {self.role}/{meta['sim_id']}/{meta['frame_index']}/r{self.replica_id}")
            return tab["sites"][row].copy(), int(tab["seed"][row]), tab["hash"][row]
        return super()._resolve_sites(idx, meta)

    def _anchor_sites(self, meta: dict[str, Any], K: int) -> np.ndarray:
        if K not in K_VALUES:
            raise ValueError(f"K={K} not in {K_VALUES}")
        if self.mask_mode == "training":
            return train_sites(self.epoch, meta["sim_id"], int(meta["frame_index"]), K)
        return self.anchors.sites(self.replica_id, K)

    def __getitem__(self, i):  # type: ignore[override]
        if isinstance(i, tuple):
            pos, K = int(i[0]), int(i[1])
        else:
            pos, K = int(i), int(self.eval_K)
        item = super().__getitem__(pos)
        meta = item["metadata"]
        idx = int(meta["cache_index"])
        sites = self._anchor_sites(meta, K)
        vals = np.asarray(self.t_norm[idx], dtype=np.float32).reshape(-1)[sites] if K else np.zeros((0,), dtype=np.float32)
        x7 = item["model_inputs"]["input_tensor"].numpy()
        tmap = np.zeros((H * W,), dtype=np.float32)
        mt = np.zeros((H * W,), dtype=np.float32)
        if K:
            tmap[sites] = vals
            mt[sites] = 1.0
        x9 = np.concatenate([x7[:5], tmap.reshape(1, H, W), mt.reshape(1, H, W), x7[5:7]], axis=0)
        item["model_inputs"]["input_tensor"] = torch.from_numpy(np.array(x9, dtype=np.float32, copy=True))
        item["model_inputs"]["channel_order"] = list(ASSISTED_CHANNEL_ORDER)
        item["model_inputs"]["K"] = K
        item["model_inputs"]["task_mode"] = "temperature_assisted"
        iy, ix = sites // W, sites % W
        item["glu_tokens"] = dict(item["glu_tokens"])
        item["glu_tokens"]["t_coords_phys"] = np.stack([self.x[ix], self.y[iy]], axis=1).astype(np.float64) if K else np.zeros((0, 2))
        item["glu_tokens"]["t_values_norm"] = vals.reshape(-1, 1).astype(np.float32)
        item["glu_tokens"]["t_site_index"] = sites.astype(np.int64)
        meta["K"] = K
        meta["t_sites"] = sites
        meta["task_mode"] = "temperature_assisted"
        return item
