"""Sparse U-Net dataset over the verified dense Month-2 cache + mask contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from combustion_recon.data.sparse_eval_masks import (
    DEFAULT_PAYLOAD_ROOT,
    load_nested_sites,
)
from combustion_recon.training.dense_cache import (
    DEFAULT_CACHE_ROOT,
    DenseCacheError,
    DenseCachePaths,
    FORBIDDEN_ROLES,
    FrameRole,
    validate_dense_cache_manifest,
)
from combustion_recon.training.sparse_view import (
    SPARSE_CHANNEL_ORDER_TUPLE,
    build_canonical_observation,
    build_sparse_unet_input,
    sites_from_eval_seed,
    sites_from_training_mask,
)

MaskMode = Literal["training", "evaluation", "fixed"]
CoverageKey = Literal[0.01, 0.05, 0.20]


class SparseUNetCacheDataset(Dataset):
    """Builds 7-channel sparse inputs from dense-cache fields + site masks.

    ``set_epoch`` must be called by the runner (including after resume) so
    training masks depend on epoch, not worker/order.
    """

    def __init__(
        self,
        role: FrameRole,
        *,
        cache_root: str | Path | None = None,
        p_conditioning: bool = True,
        indices: Sequence[int] | None = None,
        mask_mode: MaskMode = "training",
        site_fraction: float = 0.05,
        train_mask_seed: int = 20261118,
        eval_mask_seed: int = 20261116,
        layout_id: str = "random_sparse_primary",
        replica_id: int = 0,
        epoch: int = 0,
        eval_mask_payload_root: str | Path | None = None,
        fixed_sites_by_cache_index: dict[int, np.ndarray] | None = None,
        require_denorm_phys: bool = True,
        require_eval_mask_payload: bool = False,
        vi_phys: np.ndarray | None = None,
        vj_phys: np.ndarray | None = None,
        normalizer: Any | None = None,
    ):
        if role in FORBIDDEN_ROLES:
            raise DenseCacheError(f"refusing TEST role {role!r}")
        if float(site_fraction) not in (0.01, 0.05, 0.20) and mask_mode == "evaluation":
            raise ValueError("evaluation coverages must be 0.01/0.05/0.20")
        self.role = role
        self.p_conditioning = bool(p_conditioning)
        self.mask_mode = mask_mode
        self.site_fraction = float(site_fraction)
        self.train_mask_seed = int(train_mask_seed)
        self.eval_mask_seed = int(eval_mask_seed)
        self.layout_id = str(layout_id)
        self.replica_id = int(replica_id)
        self.epoch = int(epoch)
        self.eval_mask_payload_root = Path(eval_mask_payload_root) if eval_mask_payload_root else DEFAULT_PAYLOAD_ROOT
        self.fixed_sites_by_cache_index = fixed_sites_by_cache_index or {}
        self.require_denorm_phys = bool(require_denorm_phys)
        self.require_eval_mask_payload = bool(require_eval_mask_payload)
        self.normalizer = normalizer

        self.root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
        self.paths = DenseCachePaths.from_root(self.root)
        if not self.paths.manifest.exists():
            raise FileNotFoundError(f"Missing dense cache manifest at {self.paths.manifest}")
        self.manifest = json.loads(self.paths.manifest.read_text())
        validate_dense_cache_manifest(
            self.manifest,
            cache_root=self.root,
            require_roles=(role,),
        )

        rd = self.root / role
        self.vi = np.load(rd / "vi_norm.npy", mmap_mode="r")
        self.vj = np.load(rd / "vj_norm.npy", mmap_mode="r")
        self.t_norm = np.load(rd / "target_T_norm.npy", mmap_mode="r")
        self.t_K = np.load(rd / "target_T_K.npy", mmap_mode="r")
        self.p_norm = np.load(rd / "p_norm.npy", mmap_mode="r")
        self.frames = json.loads((rd / "frames.json").read_text())
        geom = np.load(self.paths.geometry)
        self.x_scaled = np.asarray(geom["x_scaled"], dtype=np.float32)
        self.y_scaled = np.asarray(geom["y_scaled"], dtype=np.float32)
        self.x = np.asarray(geom["x"], dtype=np.float64)
        self.y = np.asarray(geom["y"], dtype=np.float64)

        # Optional physical velocity arrays (for baseline packet). If absent,
        # denormalize from fit-only stats when normalizer is provided.
        self._vi_phys = vi_phys
        self._vj_phys = vj_phys

        n = int(self.vi.shape[0])
        if indices is None:
            self.indices = list(range(n))
        else:
            self.indices = [int(i) for i in indices]
            if any(i < 0 or i >= n for i in self.indices):
                raise IndexError("cache index out of range")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def get_frame_meta(self, i: int) -> dict[str, Any]:
        return dict(self.frames[self.indices[i]])

    def _phys_velocities(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self._vi_phys is not None and self._vj_phys is not None:
            return np.asarray(self._vi_phys[idx]), np.asarray(self._vj_phys[idx])
        if self.normalizer is None:
            if self.require_denorm_phys:
                raise DenseCacheError(
                    "physical velocities required but normalizer/phys arrays missing"
                )
            # Fall back: treat norm as phys only for isolation tests that opt out.
            return np.asarray(self.vi[idx]), np.asarray(self.vj[idx])
        vi_n = np.asarray(self.vi[idx], dtype=np.float64)
        vj_n = np.asarray(self.vj[idx], dtype=np.float64)
        return (
            self.normalizer.denormalize_field("Velocity_i", vi_n),
            self.normalizer.denormalize_field("Velocity_j", vj_n),
        )

    def _resolve_sites(self, idx: int, meta: dict[str, Any]) -> tuple[np.ndarray, int, str]:
        if self.mask_mode == "fixed":
            if idx not in self.fixed_sites_by_cache_index:
                raise KeyError(f"fixed sites missing for cache_index={idx}")
            sites = np.asarray(self.fixed_sites_by_cache_index[idx], dtype=np.int64)
            return np.sort(sites), -1, "fixed"
        if self.mask_mode == "training":
            return sites_from_training_mask(
                train_mask_seed=self.train_mask_seed,
                epoch=self.epoch,
                sim_id=meta["sim_id"],
                frame_index=int(meta["frame_index"]),
                layout_id=self.layout_id,
                site_fraction=self.site_fraction,
            )
        # evaluation: prefer persisted nested payload at 1/5/20%
        cov = float(self.site_fraction)
        key = {0.01: "sites_1pct", 0.05: "sites_5pct", 0.20: "sites_20pct"}.get(cov)
        if key is not None and self.eval_mask_payload_root.exists():
            try:
                loaded = load_nested_sites(
                    role=self.role,
                    sim_id=meta["sim_id"],
                    frame_index=int(meta["frame_index"]),
                    replica_id=self.replica_id,
                    payload_root=self.eval_mask_payload_root,
                )
                sites = np.asarray(loaded[key], dtype=np.int64)
                hash_key = {
                    "sites_1pct": "hash_1pct",
                    "sites_5pct": "hash_5pct",
                    "sites_20pct": "hash_20pct",
                }[key]
                return sites, int(loaded["velocity_mask_seed"]), str(loaded[hash_key])
            except KeyError as exc:
                if self.require_eval_mask_payload:
                    raise KeyError(
                        "required eval mask payload missing for "
                        f"role={self.role} sim={meta['sim_id']} frame={meta['frame_index']} "
                        f"replica={self.replica_id} coverage={cov}: {exc}"
                    ) from exc
        elif self.require_eval_mask_payload:
            raise FileNotFoundError(
                f"require_eval_mask_payload=True but payload root missing/unusable: "
                f"{self.eval_mask_payload_root}"
            )
        # Fallback: seed regeneration (5% historical path; 1/20% non-nested).
        # Forbidden when require_eval_mask_payload is set (Opus/Astra contract).
        if self.require_eval_mask_payload:
            raise RuntimeError("eval mask payload required; refusing seed regeneration fallback")
        return sites_from_eval_seed(
            eval_mask_seed=self.eval_mask_seed,
            sim_id=meta["sim_id"],
            frame_index=int(meta["frame_index"]),
            replica_id=self.replica_id,
            layout_id=self.layout_id,
            site_fraction=self.site_fraction,
        )

    def __getitem__(self, i: int) -> dict[str, Any]:
        idx = self.indices[i]
        meta = dict(self.frames[idx])
        sites, vel_seed, mhash = self._resolve_sites(idx, meta)
        vi_n = np.asarray(self.vi[idx], dtype=np.float64)
        vj_n = np.asarray(self.vj[idx], dtype=np.float64)
        vi_p, vj_p = self._phys_velocities(idx)

        obs = build_canonical_observation(
            sites=sites,
            vi_phys=vi_p,
            vj_phys=vj_p,
            vi_norm=vi_n,
            vj_norm=vj_n,
            x_coords=self.x,
            y_coords=self.y,
            site_fraction=self.site_fraction,
            layout_id=self.layout_id,
            velocity_mask_seed=vel_seed,
        )
        x = build_sparse_unet_input(
            sites=obs.sites,
            velocity_values_norm=obs.velocity_values_norm,
            x_scaled=self.x_scaled,
            y_scaled=self.y_scaled,
            nh3_norm=float(self.p_norm[idx, 0]),
            phi_norm=float(self.p_norm[idx, 1]),
            p_conditioning=self.p_conditioning,
        )
        y = np.asarray(self.t_norm[idx], dtype=np.float32)[None, ...]
        model_inputs = {
            "input_tensor": torch.from_numpy(np.array(x, dtype=np.float32, copy=True)),
            "channel_order": list(SPARSE_CHANNEL_ORDER_TUPLE),
            "p_conditioning": self.p_conditioning,
            "K": 0,
            "site_fraction": self.site_fraction,
            "task_mode": "temperature_free",
            "mask_mode": self.mask_mode,
            "epoch": self.epoch,
            "replica_id": self.replica_id,
            "layout_id": self.layout_id,
            "velocity_mask_seed": obs.velocity_mask_seed,
            "mask_hash": obs.mask_hash,
            "n_sites": obs.n_sites,
        }
        return {
            "model_inputs": model_inputs,
            "targets": {
                "temperature_normalized": torch.from_numpy(
                    np.array(y, dtype=np.float32, copy=True)
                ),
                "temperature": torch.from_numpy(
                    np.array(self.t_K[idx], dtype=np.float32, copy=True)[None, ...]
                ),
            },
            "observation": obs,
            "glu_tokens": obs.to_glu_tokens(),
            "baseline_physical": obs.to_baseline_physical(),
            "metadata": {
                **meta,
                "cache_index": idx,
                "mask_hash": obs.mask_hash,
                "velocity_mask_seed": obs.velocity_mask_seed,
                "site_fraction": self.site_fraction,
                "replica_id": self.replica_id,
                "layout_id": self.layout_id,
                "p_conditioning": self.p_conditioning,
                "epoch": self.epoch,
                "mask_mode": self.mask_mode,
                "resolved_mask_tag": mhash,
            },
        }


def collate_sparse_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    x = torch.stack([b["model_inputs"]["input_tensor"] for b in batch], dim=0)
    y = torch.stack([b["targets"]["temperature_normalized"] for b in batch], dim=0)
    y_K = torch.stack([b["targets"]["temperature"] for b in batch], dim=0)
    if not batch[0]["model_inputs"]["p_conditioning"]:
        # Enforce p_off zeros for the whole batch.
        if not torch.all(x[:, -2:] == 0):
            raise AssertionError("p_off batch has non-zero parameter channels")
    return {
        "input_tensor": x,
        "target_T_norm": y,
        "target_T_K": y_K,
        "metadata": [b["metadata"] for b in batch],
        "p_conditioning": bool(batch[0]["model_inputs"]["p_conditioning"]),
        "observations": [b["observation"] for b in batch],
        "baseline_physical": [b["baseline_physical"] for b in batch],
        "glu_tokens": [b["glu_tokens"] for b in batch],
    }
