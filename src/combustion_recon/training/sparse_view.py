"""Canonical sparse observation view shared by U-Net, future GLU, and baselines.

Physical m/s and normalized values are kept distinct. Baseline adapters must
consume physical velocities; U-Net/GLU consume normalized velocities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from combustion_recon.data.masks import random_sparse_mask
from combustion_recon.data.month2_observations import (
    mask_hash,
    observation_mask_seed,
    training_mask_seed,
)
from combustion_recon.data.sparse_eval_masks import N_SITES, sites_to_mask
from combustion_recon.models.unet import SPARSE_CHANNEL_ORDER

SPARSE_CHANNEL_ORDER_TUPLE = SPARSE_CHANNEL_ORDER


@dataclass(frozen=True)
class CanonicalSparseObservation:
    """One observation packet derived from selected velocity sites."""

    sites: np.ndarray  # sorted flat indices
    mask: np.ndarray  # (H, W) bool
    velocity_coords: np.ndarray  # (N, 2) physical
    velocity_values_phys: np.ndarray  # (N, 2) m/s
    velocity_values_norm: np.ndarray  # (N, 2) normalized
    n_sites: int
    site_fraction: float
    layout_id: str
    velocity_mask_seed: int
    mask_hash: str
    H: int = 128
    W: int = 128

    def to_unet_grid(
        self,
        *,
        x_scaled: np.ndarray,
        y_scaled: np.ndarray,
        nh3_norm: float,
        phi_norm: float,
        p_conditioning: bool = True,
    ) -> np.ndarray:
        return build_sparse_unet_input(
            sites=self.sites,
            velocity_values_norm=self.velocity_values_norm,
            x_scaled=x_scaled,
            y_scaled=y_scaled,
            nh3_norm=nh3_norm,
            phi_norm=phi_norm,
            p_conditioning=p_conditioning,
            height=self.H,
            width=self.W,
        )

    def to_glu_tokens(self) -> dict[str, np.ndarray]:
        """Vector tokens for future GLU: [x, y, vi_norm, vj_norm] ordered by site index."""
        return {
            "coords_phys": np.asarray(self.velocity_coords, dtype=np.float64),
            "values_norm": np.asarray(self.velocity_values_norm, dtype=np.float64),
            "values_phys": np.asarray(self.velocity_values_phys, dtype=np.float64),
            "site_index": np.asarray(self.sites, dtype=np.int64),
        }

    def to_baseline_physical(self) -> dict[str, np.ndarray]:
        """Physical m/s packet for GappyPOD / linear adapters (never normalized)."""
        return {
            "coords": np.asarray(self.velocity_coords, dtype=np.float64),
            "values_ms": np.asarray(self.velocity_values_phys, dtype=np.float64),
            "site_index": np.asarray(self.sites, dtype=np.int64),
            "mask": np.asarray(self.mask, dtype=bool),
            "units": "m/s",
        }


def build_sparse_unet_input(
    *,
    sites: np.ndarray,
    velocity_values_norm: np.ndarray,
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    nh3_norm: float,
    phi_norm: float,
    p_conditioning: bool = True,
    height: int = 128,
    width: int = 128,
) -> np.ndarray:
    """Stack 7 channels; hidden sites are exact zero in normalized space."""
    H, W = int(height), int(width)
    if x_scaled.shape != (H, W) or y_scaled.shape != (H, W):
        raise ValueError("x_scaled/y_scaled shape mismatch")
    sites = np.asarray(sites, dtype=np.int64).ravel()
    vals = np.asarray(velocity_values_norm, dtype=np.float64)
    if vals.ndim != 2 or vals.shape[1] != 2 or vals.shape[0] != sites.size:
        raise ValueError(f"velocity_values_norm expected (N,2) with N={sites.size}")

    vi = np.zeros((H, W), dtype=np.float32)
    vj = np.zeros((H, W), dtype=np.float32)
    Mv = np.zeros((H, W), dtype=np.float32)
    if sites.size:
        ys = sites // W
        xs = sites % W
        vi[ys, xs] = vals[:, 0].astype(np.float32)
        vj[ys, xs] = vals[:, 1].astype(np.float32)
        Mv[ys, xs] = 1.0

    if p_conditioning:
        nh3_map = np.full((H, W), float(nh3_norm), dtype=np.float32)
        phi_map = np.full((H, W), float(phi_norm), dtype=np.float32)
    else:
        nh3_map = np.zeros((H, W), dtype=np.float32)
        phi_map = np.zeros((H, W), dtype=np.float32)

    out = np.stack(
        [
            vi,
            vj,
            Mv,
            np.asarray(x_scaled, dtype=np.float32),
            np.asarray(y_scaled, dtype=np.float32),
            nh3_map,
            phi_map,
        ],
        axis=0,
    )
    assert out.shape[0] == 7
    assert list(SPARSE_CHANNEL_ORDER_TUPLE) == list(SPARSE_CHANNEL_ORDER)
    return out


def assert_p_off_zeros(x: np.ndarray | Any, *, atol: float = 0.0) -> None:
    """Last two channels must be exactly zero for p_off."""
    arr = np.asarray(x)
    if arr.ndim == 3:
        last = arr[-2:]
    elif arr.ndim == 4:
        last = arr[:, -2:]
    else:
        raise ValueError("expected (C,H,W) or (B,C,H,W)")
    if not np.all(np.abs(last) <= atol):
        raise AssertionError("p_off channels are not zero")


def sites_from_training_mask(
    *,
    train_mask_seed: int,
    epoch: int,
    sim_id: str,
    frame_index: int,
    layout_id: str = "random_sparse_primary",
    site_fraction: float = 0.05,
    height: int = 128,
    width: int = 128,
) -> tuple[np.ndarray, int, str]:
    """Epoch-dependent TRAIN mask (namespace distinct from eval)."""
    vel_seed = training_mask_seed(
        train_mask_seed=train_mask_seed,
        epoch=epoch,
        sim_id=sim_id,
        frame_index=frame_index,
        layout_id=layout_id,
    )
    mask = random_sparse_mask(site_fraction, seed=vel_seed, height=height, width=width)
    sites = np.flatnonzero(mask).astype(np.int64)
    expected = N_SITES.get(float(site_fraction))
    if expected is not None and sites.size != expected:
        # Tolerant for fractions without fixed table entry.
        pass
    return sites, int(vel_seed), mask_hash(mask)


def sites_from_eval_seed(
    *,
    eval_mask_seed: int,
    sim_id: str,
    frame_index: int,
    replica_id: int,
    layout_id: str = "random_sparse_primary",
    site_fraction: float = 0.05,
    height: int = 128,
    width: int = 128,
) -> tuple[np.ndarray, int, str]:
    vel_seed = observation_mask_seed(
        mask_seed=eval_mask_seed,
        sim_id=sim_id,
        frame_index=frame_index,
        layout_id=layout_id,
        replica_id=replica_id,
    )
    mask = random_sparse_mask(site_fraction, seed=vel_seed, height=height, width=width)
    sites = np.flatnonzero(mask).astype(np.int64)
    return sites, int(vel_seed), mask_hash(mask)


def build_canonical_observation(
    *,
    sites: np.ndarray,
    vi_phys: np.ndarray,
    vj_phys: np.ndarray,
    vi_norm: np.ndarray,
    vj_norm: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    site_fraction: float,
    layout_id: str,
    velocity_mask_seed: int,
    height: int = 128,
    width: int = 128,
) -> CanonicalSparseObservation:
    sites = np.sort(np.asarray(sites, dtype=np.int64).ravel())
    mask = sites_to_mask(sites, height=height, width=width)
    ys = sites // width
    xs = sites % width
    xx, yy = np.meshgrid(np.asarray(x_coords, dtype=np.float64), np.asarray(y_coords, dtype=np.float64), indexing="xy")
    coords = np.stack([xx[ys, xs], yy[ys, xs]], axis=1).astype(np.float64)
    phys = np.stack(
        [vi_phys[ys, xs].astype(np.float64), vj_phys[ys, xs].astype(np.float64)],
        axis=1,
    )
    norm = np.stack(
        [vi_norm[ys, xs].astype(np.float64), vj_norm[ys, xs].astype(np.float64)],
        axis=1,
    )
    return CanonicalSparseObservation(
        sites=sites,
        mask=mask,
        velocity_coords=coords,
        velocity_values_phys=phys,
        velocity_values_norm=norm,
        n_sites=int(sites.size),
        site_fraction=float(site_fraction),
        layout_id=layout_id,
        velocity_mask_seed=int(velocity_mask_seed),
        mask_hash=mask_hash(mask),
        H=height,
        W=width,
    )
