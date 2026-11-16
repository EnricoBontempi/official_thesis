"""Month-2 reconstruction dataset with explicit model_inputs / targets / metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from torch.utils.data import Dataset

from combustion_recon.data.freeze import FreezeGateError, assert_test_access_allowed
from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.data.month2_observations import build_month2_observation_pack
from combustion_recon.data.month2_protocol import (
    TEST_ROLES,
    FrameRole,
    build_index,
    load_frame_protocol,
    load_yaml,
)
from combustion_recon.data.shard_access import ShardStore

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PILOT = ROOT / "configs" / "experiments" / "month2_pilot.yaml"

TaskMode = Literal["temperature_free", "temperature_assisted"]


class Month2ReconDataset(Dataset):
    """Frame-role-aware dataset for the Month-2 pilot contract.

    Initialization does **not** decode Arrow payloads. TEST roles refuse payload
    reads unless a valid selection freeze is provided.
    """

    def __init__(
        self,
        role: FrameRole,
        *,
        task_mode: TaskMode = "temperature_free",
        K: int = 0,
        site_fraction: float | None = None,
        p_conditioning: bool = False,
        layout_id: str = "random_sparse_primary",
        mask_seed: int = 20261116,
        replica_id: int = 0,
        mask_mode: Literal["evaluation", "training"] = "evaluation",
        epoch: int = 0,
        train_mask_seed: int = 20261116,
        raw_dir: str | Path | None = None,
        frame_protocol_path: str | Path | None = None,
        pilot_path: str | Path | None = None,
        splits_path: str | Path | None = None,
        sim_ids: Sequence[str] | None = None,
        normalizer: Month2FitOnlyNormalizer | None = None,
        stats_path: str | Path | None = None,
        store: ShardStore | None = None,
        max_cached_frames: int = 512,
        allow_test_payloads: bool = False,
        freeze_path: str | Path | None = None,
        return_dense_temperature_channels: bool | None = None,
    ):
        self.role: FrameRole = role
        self.task_mode = task_mode
        self.K = int(K)
        if task_mode == "temperature_free" and self.K != 0:
            raise ValueError("temperature_free requires K=0")
        if task_mode == "temperature_assisted" and self.K < 0:
            raise ValueError("K must be >= 0")

        self.protocol = load_frame_protocol(frame_protocol_path)
        self.pilot = load_yaml(pilot_path if pilot_path is not None else DEFAULT_PILOT)
        self.site_fraction = (
            float(site_fraction)
            if site_fraction is not None
            else float(self.pilot["sparsity"]["primary"]["site_fraction"])
        )
        self.p_conditioning = bool(p_conditioning)
        self.layout_id = layout_id
        self.mask_seed = int(mask_seed)
        self.replica_id = int(replica_id)
        self.mask_mode = mask_mode
        self.epoch = int(epoch)
        self.train_mask_seed = int(train_mask_seed)
        self.allow_test_payloads = bool(allow_test_payloads)
        self.freeze_path = freeze_path
        self.return_dense_temperature_channels = (
            bool(return_dense_temperature_channels)
            if return_dense_temperature_channels is not None
            else (task_mode == "temperature_assisted")
        )

        self._index = build_index(
            role,
            protocol=self.protocol,
            splits_path=splits_path,
            sim_ids=sim_ids,
        )
        self.store = store or ShardStore(raw_dir=raw_dir, max_cached_frames=max_cached_frames)

        if normalizer is not None:
            self.normalizer = normalizer
        elif stats_path is not None or (ROOT / "data/manifests/train_norm_stats_month2_fit.json").exists():
            # For fold subsets, callers should pass expected_sim_ids via a pre-built normalizer.
            self.normalizer = Month2FitOnlyNormalizer(stats_path)
        else:
            self.normalizer = None

        # Manifests may describe TEST indices; payload access remains gated.
        self._freeze: dict[str, Any] | None = None
        if role in TEST_ROLES:
            if not self.allow_test_payloads:
                # Index OK; __getitem__ will refuse unless freeze validated later
                pass
            else:
                self._freeze = assert_test_access_allowed(freeze_path=freeze_path)

    def __len__(self) -> int:
        return len(self._index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _ensure_payload_allowed(self) -> None:
        if self.role not in TEST_ROLES:
            return
        if not self.allow_test_payloads:
            raise FreezeGateError(
                f"Payload access for role={self.role!r} is disabled by default "
                "(selection/tuning must not load TEST fields)."
            )
        if self._freeze is None:
            self._freeze = assert_test_access_allowed(freeze_path=self.freeze_path)

    def get_identity(self, idx: int) -> dict[str, Any]:
        return dict(self._index[idx])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        self._ensure_payload_allowed()
        ident = self._index[idx]
        sim_id = ident["sim_id"]
        frame_index = int(ident["frame_index"])
        fr = self.store.get_frame(sim_id, frame_index)

        T = fr.temperature.astype(np.float64)
        vi = fr.velocity_i.astype(np.float64)
        vj = fr.velocity_j.astype(np.float64)

        if self.normalizer is not None:
            vi_n = self.normalizer.normalize_field("Velocity_i", vi)
            vj_n = self.normalizer.normalize_field("Velocity_j", vj)
            t_n = self.normalizer.normalize_field("Temperature", T)
        else:
            vi_n, vj_n, t_n = vi, vj, T

        # Effective K for observation pack
        K_eff = 0 if self.task_mode == "temperature_free" else self.K
        pack = build_month2_observation_pack(
            velocity_i=vi,
            velocity_j=vj,
            temperature=T,
            x_coords=fr.x,
            y_coords=fr.y,
            site_fraction=self.site_fraction,
            K=K_eff,
            layout_id=self.layout_id,
            mask_seed=self.mask_seed,
            sim_id=sim_id,
            frame_index=frame_index,
            replica_id=self.replica_id,
            mode=self.mask_mode,
            epoch=self.epoch,
            train_mask_seed=self.train_mask_seed,
            include_dense_temperature_channels=self.return_dense_temperature_channels,
            vi_norm=vi_n,
            vj_norm=vj_n,
            t_norm=t_n,
        )

        H, W = T.shape
        # Normalized co-located velocity values for model consumption
        if pack.n_sites:
            ys = pack.velocity_site_index // W
            xs = pack.velocity_site_index % W
            vel_values_norm = np.stack([vi_n[ys, xs], vj_n[ys, xs]], axis=1).astype(np.float64)
        else:
            vel_values_norm = np.zeros((0, 2), dtype=np.float64)

        if K_eff > 0:
            ay = pack.anchor_site_index // W
            ax = pack.anchor_site_index % W
            anchor_values_norm = t_n[ay, ax].astype(np.float64)
            # Physical + normalized anchors
            anchor_values_phys = pack.anchor_values.astype(np.float64)
        else:
            anchor_values_norm = np.zeros((0,), dtype=np.float64)
            anchor_values_phys = np.zeros((0,), dtype=np.float64)

        model_inputs: dict[str, Any] = {
            "velocity_coords": pack.velocity_coords,
            "velocity_values": vel_values_norm,
            "velocity_values_physical": pack.velocity_values,
            "velocity_site_index": pack.velocity_site_index,
            "mask_velocity": pack.mask_velocity,
            "scalar_coords": pack.scalar_coords,
            "scalar_var_ids": pack.scalar_var_ids,
            # scalar values: velocity norms + optional T norms at anchors only
            "scalar_values": _scalar_values_normalized(pack, vi_n, vj_n, t_n, K_eff),
            "dense_values": pack.dense_values,
            "dense_channel_names": list(pack.dense_channel_names),
            "query_coords": _query_coords(fr.x, fr.y),
            "K": int(K_eff),
            "site_fraction": float(self.site_fraction),
            "task_mode": self.task_mode,
        }
        if K_eff > 0:
            model_inputs["anchor_coords"] = pack.anchor_coords
            model_inputs["anchor_values"] = anchor_values_norm
            model_inputs["anchor_values_physical"] = anchor_values_phys
            model_inputs["anchor_site_index"] = pack.anchor_site_index
            model_inputs["mask_temperature"] = pack.mask_temperature
        else:
            # Keep keys present for collate stability, but empty / all-false
            model_inputs["anchor_coords"] = pack.anchor_coords
            model_inputs["anchor_values"] = anchor_values_norm
            model_inputs["anchor_values_physical"] = anchor_values_phys
            model_inputs["anchor_site_index"] = pack.anchor_site_index
            model_inputs["mask_temperature"] = pack.mask_temperature

        if self.p_conditioning:
            if self.normalizer is not None:
                p_vec = self.normalizer.normalize_op_params(
                    ident["nh3_fraction"], phi=ident["phi"]
                )
            else:
                p_vec = np.asarray(
                    [ident["nh3_fraction"], ident["phi"]], dtype=np.float64
                )
            model_inputs["op_params"] = {
                "nh3_fraction": float(ident["nh3_fraction"]),
                "phi": float(ident["phi"]),
                "equivalence_ratio": float(ident["equivalence_ratio"]),
                "normalized": p_vec,
            }
        # else: deliberately omit op_params from model_inputs (p_off)

        targets = {
            "temperature": T,  # physical Kelvin
            "temperature_normalized": t_n,
        }

        metadata = {
            "sim_id": sim_id,
            "split": ident["split"],
            "role": self.role,
            "frame_index": frame_index,
            "time": float(fr.t),
            "nh3_fraction": float(ident["nh3_fraction"]),
            "phi": float(ident["phi"]),
            "equivalence_ratio": float(ident["equivalence_ratio"]),
            "nh3_pct": int(ident["nh3_pct"]),
            "layout_id": self.layout_id,
            "mask_mode": self.mask_mode,
            "mask_seed": self.mask_seed,
            "replica_id": self.replica_id,
            "epoch": self.epoch if self.mask_mode == "training" else None,
            "velocity_mask_seed": pack.velocity_mask_seed,
            "anchor_perm_seed": pack.anchor_perm_seed,
            "velocity_mask_hash": pack.velocity_mask_hash,
            "temperature_mask_hash": pack.temperature_mask_hash,
            "n_velocity_sites": pack.n_sites,
            "K": int(K_eff),
            "p_conditioning": self.p_conditioning,
            "task_mode": self.task_mode,
            "grid": [int(H), int(W)],
            "channels": {"temperature": 10, "velocity_i": 11, "velocity_j": 12},
            "x": fr.x.copy(),
            "y": fr.y.copy(),
            # Hidden fields kept ONLY for evaluation/tests — never merge into model_inputs
            "_hidden_velocity_k": fr.velocity_k.astype(np.float64),
            "_hidden_full_velocity_i": vi,
            "_hidden_full_velocity_j": vj,
        }

        return {
            "model_inputs": model_inputs,
            "targets": targets,
            "metadata": metadata,
        }


def _query_coords(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)


def _scalar_values_normalized(
    pack,
    vi_n: np.ndarray,
    vj_n: np.ndarray,
    t_n: np.ndarray,
    K_eff: int,
) -> np.ndarray:
    """Scalar token values with normalized velocities and (optional) T anchors only."""
    H, W = vi_n.shape
    if pack.n_sites == 0 and K_eff == 0:
        return np.zeros((0,), dtype=np.float64)
    ys = pack.velocity_site_index // W
    xs = pack.velocity_site_index % W
    parts = [vi_n[ys, xs], vj_n[ys, xs]]
    if K_eff > 0:
        ay = pack.anchor_site_index // W
        ax = pack.anchor_site_index % W
        parts.append(t_n[ay, ax])
    return np.concatenate([p.astype(np.float64) for p in parts], axis=0)


def extract_model_inputs(sample: dict[str, Any]) -> dict[str, Any]:
    """Return only the model_inputs dict (never the full sample)."""
    return sample["model_inputs"]


class InputIsolationProbe:
    """Tiny deterministic probe that consumes **only** ``model_inputs``.

    Not a scientific model and not a substitute for future U-Net/GLU forward tests.
    """

    def __call__(self, model_inputs: dict[str, Any]) -> np.ndarray:
        # Deterministic fingerprint from allowed tensors only.
        vel = np.asarray(model_inputs["velocity_values"], dtype=np.float64)
        coords = np.asarray(model_inputs["velocity_coords"], dtype=np.float64)
        dense = np.asarray(model_inputs["dense_values"], dtype=np.float64)
        acc = np.array(
            [
                float(vel.sum()) if vel.size else 0.0,
                float(coords.sum()) if coords.size else 0.0,
                float(dense.sum()) if dense.size else 0.0,
                float(model_inputs.get("K", 0)),
                float(model_inputs.get("site_fraction", 0.0)),
            ],
            dtype=np.float64,
        )
        if "op_params" in model_inputs:
            acc = np.concatenate(
                [acc, np.asarray(model_inputs["op_params"]["normalized"], dtype=np.float64)]
            )
        if int(model_inputs.get("K", 0)) > 0:
            acc = np.concatenate(
                [acc, np.asarray(model_inputs["anchor_values"], dtype=np.float64).reshape(-1)]
            )
        # Fixed-size digest
        h = np.zeros(8, dtype=np.float64)
        h[: min(8, acc.size)] = acc[:8]
        if acc.size > 8:
            h[-1] += float(acc[8:].sum())
        return h
