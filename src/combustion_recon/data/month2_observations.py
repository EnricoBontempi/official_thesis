"""Month-2 observation packs: co-located velocity vectors, anchors, dense U-Net views."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from combustion_recon.data.masks import (
    VAR_ID,
    nested_temperature_anchor_permutation,
    random_sparse_mask,
    stable_seed,
    temperature_anchor_mask,
)

LayoutMode = Literal["evaluation", "training"]


def observation_mask_seed(
    *,
    mask_seed: int,
    sim_id: str,
    frame_index: int,
    layout_id: str,
    replica_id: int,
) -> int:
    """Deterministic seed independent of dataset order / worker / model."""
    return stable_seed(int(mask_seed), sim_id, int(frame_index), layout_id, int(replica_id))


def training_mask_seed(
    *,
    train_mask_seed: int,
    epoch: int,
    sim_id: str,
    frame_index: int,
    layout_id: str,
) -> int:
    """Epoch-scoped RNG seed, separate from evaluation mask replicas."""
    return stable_seed("train_mask", int(train_mask_seed), int(epoch), sim_id, int(frame_index), layout_id)


def mask_hash(mask: np.ndarray) -> str:
    raw = np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
    return hashlib.sha256(raw).hexdigest()


@dataclass
class Month2ObservationPack:
    """Observation views derived from one physical sample + one observation plan."""

    # Co-located velocity vector tokens (preferred)
    velocity_coords: np.ndarray  # (N, 2)
    velocity_values: np.ndarray  # (N, 2) columns [vi, vj]
    velocity_site_index: np.ndarray  # (N,) flat indices 0..H*W-1
    mask_velocity: np.ndarray  # (H, W) bool

    # Scalar-token compatibility view (same measurements, duplicated sites per component)
    scalar_coords: np.ndarray  # (M, 2)
    scalar_var_ids: np.ndarray  # (M,)
    scalar_values: np.ndarray  # (M,)

    # Temperature anchors (empty when K=0)
    anchor_coords: np.ndarray  # (K, 2)
    anchor_values: np.ndarray  # (K,)
    anchor_site_index: np.ndarray  # (K,)
    mask_temperature: np.ndarray  # (H, W) bool

    # Dense U-Net pack: normalized values with zero placeholder + explicit mask channels
    # Channels: [vi_norm_or_0, vj_norm_or_0, vel_mask, optional T_norm_or_0, optional T_mask]
    dense_values: np.ndarray  # (C, H, W) float32
    dense_channel_names: list[str]

    # Provenance
    layout_id: str
    site_fraction: float
    n_sites: int
    K: int
    velocity_mask_seed: int
    anchor_perm_seed: int
    velocity_mask_hash: str
    temperature_mask_hash: str


def _mesh_xy(x_coords: np.ndarray, y_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(x_coords, y_coords, indexing="xy")


def build_month2_observation_pack(
    *,
    velocity_i: np.ndarray,
    velocity_j: np.ndarray,
    temperature: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    site_fraction: float,
    K: int,
    layout_id: str,
    mask_seed: int,
    sim_id: str,
    frame_index: int,
    replica_id: int = 0,
    mode: LayoutMode = "evaluation",
    epoch: int = 0,
    train_mask_seed: int = 0,
    include_dense_temperature_channels: bool = True,
    vi_norm: np.ndarray | None = None,
    vj_norm: np.ndarray | None = None,
    t_norm: np.ndarray | None = None,
) -> Month2ObservationPack:
    """Build sparse + dense observation views from the same masks.

    - Velocity sites are co-located: one site → (vi, vj).
    - Scalar tokens duplicate sites per component for legacy-compatible APIs.
    - Dense pack uses zero placeholder for unobserved values + explicit mask channels.
    - Does **not** interpolate through hidden velocity locations.
    - Layout seeds never depend on temperature values.
    """
    if K < 0:
        raise ValueError("K must be >= 0")
    H, W = temperature.shape
    if velocity_i.shape != (H, W) or velocity_j.shape != (H, W):
        raise ValueError("velocity/temperature shape mismatch")

    if mode == "evaluation":
        vel_seed = observation_mask_seed(
            mask_seed=mask_seed,
            sim_id=sim_id,
            frame_index=frame_index,
            layout_id=layout_id,
            replica_id=replica_id,
        )
    elif mode == "training":
        vel_seed = training_mask_seed(
            train_mask_seed=train_mask_seed,
            epoch=epoch,
            sim_id=sim_id,
            frame_index=frame_index,
            layout_id=layout_id,
        )
    else:
        raise ValueError(mode)

    vel_mask = random_sparse_mask(site_fraction, seed=vel_seed, height=H, width=W)
    # Nested anchors from a permutation that does not depend on K or T values.
    # Same velocity mask for K=0 and K>0 (vel_mask computed once above).
    anchor_seed = stable_seed(vel_seed, "temp_anchors")
    if K == 0:
        t_mask = np.zeros((H, W), dtype=bool)
    else:
        t_mask = temperature_anchor_mask(
            K, seed=anchor_seed, height=H, width=W, nested=True, max_k=32
        )

    xx, yy = _mesh_xy(x_coords, y_coords)
    ys, xs = np.where(vel_mask)
    site_index = ys * W + xs
    vel_coords = np.stack([xx[ys, xs], yy[ys, xs]], axis=1).astype(np.float64)
    vel_values = np.stack(
        [velocity_i[ys, xs].astype(np.float64), velocity_j[ys, xs].astype(np.float64)],
        axis=1,
    )

    # Scalar compatibility view
    scalar_coords = np.concatenate([vel_coords, vel_coords], axis=0)
    scalar_var_ids = np.concatenate(
        [
            np.full(ys.size, VAR_ID["velocity_i"], dtype=np.int64),
            np.full(ys.size, VAR_ID["velocity_j"], dtype=np.int64),
        ]
    )
    scalar_values = np.concatenate(
        [velocity_i[ys, xs].astype(np.float64), velocity_j[ys, xs].astype(np.float64)]
    )

    ay, ax = np.where(t_mask)
    anchor_site = ay * W + ax
    if ay.size:
        # Preserve nested order from the shared permutation
        perm = nested_temperature_anchor_permutation(seed=anchor_seed, height=H, width=W, max_k=32)
        order_map = {int(p): i for i, p in enumerate(perm)}
        order = np.argsort([order_map[int(s)] for s in anchor_site])
        ay, ax, anchor_site = ay[order], ax[order], anchor_site[order]
        anchor_coords = np.stack([xx[ay, ax], yy[ay, ax]], axis=1).astype(np.float64)
        anchor_values = temperature[ay, ax].astype(np.float64)
        # append T scalar tokens
        scalar_coords = np.concatenate([scalar_coords, anchor_coords], axis=0)
        scalar_var_ids = np.concatenate(
            [scalar_var_ids, np.full(ay.size, VAR_ID["temperature"], dtype=np.int64)]
        )
        scalar_values = np.concatenate([scalar_values, anchor_values])
    else:
        anchor_coords = np.zeros((0, 2), dtype=np.float64)
        anchor_values = np.zeros((0,), dtype=np.float64)

    # Dense U-Net pack
    vi_src = vi_norm if vi_norm is not None else velocity_i.astype(np.float64)
    vj_src = vj_norm if vj_norm is not None else velocity_j.astype(np.float64)
    t_src = t_norm if t_norm is not None else temperature.astype(np.float64)

    vi_dense = np.zeros((H, W), dtype=np.float32)
    vj_dense = np.zeros((H, W), dtype=np.float32)
    vi_dense[vel_mask] = vi_src[vel_mask].astype(np.float32)
    vj_dense[vel_mask] = vj_src[vel_mask].astype(np.float32)
    vel_mask_f = vel_mask.astype(np.float32)

    channels = [vi_dense, vj_dense, vel_mask_f]
    names = ["velocity_i_filled0", "velocity_j_filled0", "velocity_mask"]
    if include_dense_temperature_channels:
        t_dense = np.zeros((H, W), dtype=np.float32)
        t_dense[t_mask] = t_src[t_mask].astype(np.float32)
        channels.extend([t_dense, t_mask.astype(np.float32)])
        names.extend(["temperature_filled0", "temperature_mask"])

    dense = np.stack(channels, axis=0)

    return Month2ObservationPack(
        velocity_coords=vel_coords,
        velocity_values=vel_values,
        velocity_site_index=site_index.astype(np.int64),
        mask_velocity=vel_mask,
        scalar_coords=scalar_coords,
        scalar_var_ids=scalar_var_ids,
        scalar_values=scalar_values,
        anchor_coords=anchor_coords,
        anchor_values=anchor_values,
        anchor_site_index=anchor_site.astype(np.int64),
        mask_temperature=t_mask,
        dense_values=dense,
        dense_channel_names=names,
        layout_id=layout_id,
        site_fraction=float(site_fraction),
        n_sites=int(vel_mask.sum()),
        K=int(K),
        velocity_mask_seed=int(vel_seed),
        anchor_perm_seed=int(anchor_seed),
        velocity_mask_hash=mask_hash(vel_mask),
        temperature_mask_hash=mask_hash(t_mask),
    )


def observation_recipe_dict(pack: Month2ObservationPack) -> dict[str, Any]:
    return {
        "layout_id": pack.layout_id,
        "site_fraction": pack.site_fraction,
        "n_sites": pack.n_sites,
        "K": pack.K,
        "velocity_mask_seed": pack.velocity_mask_seed,
        "anchor_perm_seed": pack.anchor_perm_seed,
        "velocity_mask_hash": pack.velocity_mask_hash,
        "temperature_mask_hash": pack.temperature_mask_hash,
        "dense_channel_names": list(pack.dense_channel_names),
        "token_scheme": "co_located_velocity_vectors_plus_optional_T_anchors",
    }
