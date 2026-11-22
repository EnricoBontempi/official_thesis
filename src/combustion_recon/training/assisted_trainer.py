"""Temperature-assisted trainer: variable TRAIN K per optimizer batch, selection on the six-K common-domain VAL macro."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from combustion_recon.data.t_anchors import K_VALUES, domain_metrics, train_K_for_update
from combustion_recon.evaluation.month2_evaluator import denormalize_temperature
from combustion_recon.training.sparse_dataset import collate_sparse_batch
from combustion_recon.training.sparse_scientific_loop import SparseScientificTrainer

DIAG_K = (0, 9, 32)   # TRAIN/seen diagnostics in the loop (cost); VAL selection always uses all six K


class AssistedSampler(Sampler):
    """Yield (index, K) pairs: one K per optimizer batch = f(update number); same K for every example of the batch."""

    def __init__(self, indices, update_at_start: int, batch_size: int):
        self.indices = [int(i) for i in indices]
        self.u0 = int(update_at_start)
        self.bs = int(batch_size)

    def __iter__(self):
        for pos, idx in enumerate(self.indices):
            yield (idx, train_K_for_update(self.u0 + pos // self.bs + 1))

    def __len__(self):
        return len(self.indices)


class AssistedTrainer(SparseScientificTrainer):
    def __init__(self, *args: Any, anchors=None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.anchors = anchors
        self.diag_K = DIAG_K
        self._last_per_K: dict[tuple[str, int], dict[int, float]] = {}

    def evaluate_val_replicas(self, val_ds, **kw):
        out = super().evaluate_val_replicas(val_ds, **kw)
        for item in out["per_replica"]:
            item["per_K_common_rmse_K"] = self._last_per_K.get(("val_eval", int(item["replica_id"])))
        return out

    def _train_sampler(self, epoch_indices, epoch):
        return AssistedSampler(epoch_indices, int(self.state.update), int(self.cfg.batch_size))

    def _forward_norm(self, batch: dict[str, Any], *, decode_chunked: bool | None = None) -> torch.Tensor:
        if self.cfg.family == "sparse_unet":
            return self.model(batch["input_tensor"].to(self.device))
        tokens = batch["glu_tokens"]
        dev = self.device

        def stack(key, dtype):
            return torch.stack([torch.from_numpy(np.asarray(t[key], dtype=dtype)) for t in tokens], dim=0).to(dev)

        Ks = {int(t["t_site_index"].shape[0]) for t in tokens}
        if len(Ks) != 1:
            raise RuntimeError(f"mixed K inside a batch: {Ks}")
        p = batch["p_norm_tensor"].to(dev)
        if not self.cfg.p_conditioning:
            p = torch.zeros_like(p)
        Y = self.query_coords_phys.unsqueeze(0).expand(len(tokens), -1, -1).contiguous()
        use_chunk = self.cfg.decode_chunked_eval if decode_chunked is None else bool(decode_chunked)
        return self.model(
            stack("coords_phys", np.float32), stack("values_norm", np.float32), stack("site_index", np.int64), p, Y,
            t_coords_phys=stack("t_coords_phys", np.float32), t_values_norm=stack("t_values_norm", np.float32),
            t_site_index=stack("t_site_index", np.int64), p_conditioning=self.cfg.p_conditioning,
            decode_chunked=use_chunk, chunk_size=self.cfg.decode_chunk_size, return_flat=False)

    @torch.no_grad()
    def evaluate_dataset(self, ds, *, role: str, replica_id: int, method: str, variant: str, model_seed: int,
                         batch_size: int = 8, extra_tags: dict[str, Any] | None = None) -> dict[str, Any]:
        self.eval_mode()
        ds.replica_id = int(replica_id)
        ds.mask_mode = "evaluation"
        Ks = K_VALUES if role == "val_eval" else self.diag_K
        layout32 = self.anchors.orders[int(replica_id)]
        rows: list[dict[str, Any]] = []
        for K in Ks:
            ds.eval_K = int(K)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_sparse_batch)
            for batch in loader:
                if self.cfg.family == "adapted_glu":
                    batch["p_norm_tensor"] = torch.tensor(
                        [[float(ds.p_norm[int(m["cache_index"]), 0]), float(ds.p_norm[int(m["cache_index"]), 1])] for m in batch["metadata"]],
                        dtype=torch.float32)
                pred_n = self._forward_norm(batch, decode_chunked=True).detach().cpu().numpy()
                true_K = batch["target_T_K"].numpy()
                for i, meta in enumerate(batch["metadata"]):
                    pred_K = denormalize_temperature(pred_n[i, 0], self.normalizer)
                    row = {"sim_id": str(meta["sim_id"]), "frame_index": int(meta["frame_index"]), "role": role, "K": int(K),
                           "mask_replica_id": int(replica_id), "site_fraction": float(self.cfg.site_fraction),
                           "p_conditioning": bool(self.cfg.p_conditioning), "model_seed": int(model_seed),
                           "task_mode": "temperature_assisted", "mask_hash": str(meta.get("mask_hash", "")),
                           **domain_metrics(pred_K, true_K[i, 0], meta["t_sites"], layout32)}
                    if extra_tags:
                        row.update(extra_tags)
                    rows.append(row)
        df = pd.DataFrame(rows)
        per_k = {}
        for K, g in df.groupby("K"):
            per_k[int(K)] = float(g.groupby("sim_id")["common_rmse_K"].mean().mean())
        macro = float(np.mean([per_k[k] for k in Ks]))
        self._last_per_K[(role, int(replica_id))] = per_k
        return {"rows": rows, "aggregation": None, "mean_condition_rmse_K": macro, "replica_id": int(replica_id), "per_K_common_rmse_K": per_k}
