"""Degraded velocity observations for the non-ideal sensitivity study (Block I).

Observation model, in this order::

    physical field -> historical sites -> noise in m/s -> invalidation
                   -> frozen normalization -> model input

* N  : i.i.d. Gaussian noise per velocity component, sigma given in **m/s**.
* G1 : union of discs (contiguous gaps) independent of T and of any model error.
* G1-eq : uniform random removal of the same number of sites as G1.

Nothing here reads temperature, and invalidated sites are never filled: the
output packet simply has fewer sites (U-Net: value 0 with Mv=0 through
``to_unet_grid``; GLU: the token is absent).  Spatial averaging, regular
resampling, T-dependent gaps and spurious vectors are NOT implemented on purpose.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from combustion_recon.data.masks import stable_seed
from combustion_recon.data.month2_observations import mask_hash
from combustion_recon.data.sparse_eval_masks import sites_to_mask
from combustion_recon.training.sparse_view import CanonicalSparseObservation

SPEC_FORMAT = "nonideal_degradation_spec_v1"
NAMESPACE = "nonideal_degradation_v1"
HISTORICAL_SEEDS = frozenset({20261114, 20261116, 20261118})
EVAL_ROLES = ("seen_train_eval", "val_eval")
QUALIFICATION_ROLES = ("train_fit",)
FORBIDDEN_ROLES = ("test", "test_eval", "TEST")
GRID_HW = (128, 128)
GLU_TOP_K = 16
VELOCITY_FIELDS = ("Velocity_i", "Velocity_j")


class DegradationError(ValueError):
    """Unsupported or invalid degradation request."""


# --------------------------------------------------------------------------- scale

@dataclass(frozen=True)
class VelocityScale:
    """Frozen z-score parameters (m/s) of (vi, vj). Never refit."""

    mean: tuple[float, float]
    std: tuple[float, float]
    stats_sha256: str | None = None

    def noise_ms_to_norm(self, noise_ms: np.ndarray) -> np.ndarray:
        """Convert an m/s perturbation to normalized units: delta/std (the mean cancels)."""
        return np.asarray(noise_ms, dtype=np.float64) / np.asarray(self.std, dtype=np.float64)


def load_velocity_scale(stats_path: str | Path, *, expected_sha256: str | None = None) -> VelocityScale:
    p = Path(stats_path)
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and sha != expected_sha256:
        raise DegradationError(f"stats sha256 mismatch: {sha} != {expected_sha256}")
    fields = json.loads(raw)["fields"]
    return VelocityScale(
        mean=tuple(float(fields[n]["mean"]) for n in VELOCITY_FIELDS),
        std=tuple(float(fields[n]["std"]) for n in VELOCITY_FIELDS),
        stats_sha256=sha,
    )


# --------------------------------------------------------------------------- spec

@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    noise_sigma_ms: float | None  # None: no noise operator
    gap: str | None  # None | "g1" | "g1_eq"
    role: str = ""


@dataclass(frozen=True)
class G1Params:
    area_fraction: float
    diameter_min_mm: float
    diameter_max_mm: float
    max_attempts: int
    max_discs: int
    min_surviving_sites: int


@dataclass(frozen=True)
class DegradationSpec:
    spec_id: str
    degradation_seed: int
    cells: dict[str, CellSpec]
    g1: G1Params
    noise_reference_sigma_ms: float  # level that defines the standardized realization (sigma=1 m/s)
    raw: dict[str, Any] = field(compare=False, default_factory=dict)


_FORBIDDEN_NOISE_KEYS = ("sigma_norm", "sigma_normalized", "noise_norm", "sigma_z")


def _reject_normalized_noise(obj: Any, path: str = "") -> None:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if str(k) in _FORBIDDEN_NOISE_KEYS:
                raise DegradationError(f"noise must be specified in m/s; forbidden key {path}/{k}")
            _reject_normalized_noise(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _reject_normalized_noise(v, f"{path}[{i}]")


def parse_degradation_spec(raw: Mapping[str, Any]) -> DegradationSpec:
    """Validate a spec mapping. Raises DegradationError on any violation."""
    _reject_normalized_noise(raw)
    if raw.get("format") != SPEC_FORMAT:
        raise DegradationError(f"format must be {SPEC_FORMAT!r}")
    seeds = raw.get("seeds", {})
    dseed = seeds.get("degradation_seed")
    if not isinstance(dseed, int) or isinstance(dseed, bool):
        raise DegradationError("seeds.degradation_seed must be an int")
    hist = set(HISTORICAL_SEEDS) | {int(s) for s in seeds.get("forbidden_historical_seeds", [])}
    if dseed in hist:
        raise DegradationError(f"degradation_seed {dseed} coincides with a historical seed")
    if seeds.get("namespace") != NAMESPACE:
        raise DegradationError(f"seeds.namespace must be {NAMESPACE!r}")

    roles = raw.get("roles", {})
    for r in list(roles.get("evaluation", [])) + list(roles.get("qualification", [])):
        if r in FORBIDDEN_ROLES or "test" in str(r).lower():
            raise DegradationError(f"forbidden role {r!r}")
    if tuple(roles.get("evaluation", [])) != EVAL_ROLES:
        raise DegradationError(f"evaluation roles must be {EVAL_ROLES}")
    if tuple(roles.get("qualification", [])) != QUALIFICATION_ROLES:
        raise DegradationError(f"qualification roles must be {QUALIFICATION_ROLES}")

    ops = raw.get("operators", {})
    noise = ops.get("noise", {})
    if noise.get("unit") != "m/s":
        raise DegradationError("operators.noise.unit must be 'm/s'")
    ref = float(noise.get("standardized_reference_sigma_ms", 1.0))
    if ref != 1.0:
        raise DegradationError("standardized realization must have unit sigma (1 m/s)")

    g = ops.get("g1", {})
    if g.get("family") != "union_of_discs":
        raise DegradationError("g1.family must be 'union_of_discs'")
    if g.get("dependence_on_temperature") is not False:
        raise DegradationError("g1.dependence_on_temperature must be false")
    g1 = G1Params(
        area_fraction=float(g["area_fraction"]),
        diameter_min_mm=float(g["diameter_mm"]["min"]),
        diameter_max_mm=float(g["diameter_mm"]["max"]),
        max_attempts=int(g["rejection"]["max_attempts"]),
        max_discs=int(g["max_discs"]),
        min_surviving_sites=int(g["rejection"]["min_surviving_sites"]),
    )
    if not 0.0 < g1.area_fraction < 1.0:
        raise DegradationError("g1.area_fraction must be in (0,1)")
    if not 0.0 < g1.diameter_min_mm <= g1.diameter_max_mm:
        raise DegradationError("invalid g1 diameter range")
    if g1.min_surviving_sites < GLU_TOP_K:
        raise DegradationError(f"min_surviving_sites must be >= GLU top_k={GLU_TOP_K}")

    cells: dict[str, CellSpec] = {}
    for c in raw.get("cells", []):
        cid = str(c["id"])
        if cid in cells:
            raise DegradationError(f"duplicate cell {cid}")
        sig = c.get("noise_sigma_ms")
        if sig is not None:
            sig = float(sig)
            if not math.isfinite(sig) or sig < 0.0:
                raise DegradationError(f"invalid sigma for {cid}")
        gap = c.get("gap")
        if gap not in (None, "g1", "g1_eq"):
            raise DegradationError(f"unknown gap operator {gap!r} in {cid}")
        if gap == "g1_eq" and sig is not None:
            raise DegradationError("g1_eq is a pure control: no noise")
        cells[cid] = CellSpec(cid, sig, gap, str(c.get("role", "")))
    if not cells:
        raise DegradationError("no cells")
    return DegradationSpec(
        spec_id=str(raw.get("spec_id", "")),
        degradation_seed=int(dseed),
        cells=cells,
        g1=g1,
        noise_reference_sigma_ms=ref,
        raw=dict(raw),
    )


def load_degradation_spec(path: str | Path) -> DegradationSpec:
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise DegradationError("spec must be a mapping")
    return parse_degradation_spec(raw)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- seeds

def degradation_seed_for(
    spec_seed: int, operator: str, sim_id: str, frame_index: int, replica_id: int, *extra: object
) -> int:
    """Namespace-isolated seed.

    The level (sigma) is NOT part of the key: N1 and N2 share one standardized
    realization, and N1 / N1G1 share the same noise at surviving sites.
    The ``NAMESPACE`` string prefix cannot collide with the historical
    ``observation_mask_seed`` / ``training_mask_seed`` payload layouts.
    """
    return stable_seed(NAMESPACE, int(spec_seed), operator, sim_id, int(frame_index), int(replica_id), *extra)


# --------------------------------------------------------------------------- operators

def standardized_noise(
    spec_seed: int, sim_id: str, frame_index: int, replica_id: int, *, height: int = 128, width: int = 128
) -> np.ndarray:
    """(H*W, 2) standard normal draws indexed by flat site index (unit sigma, dimensionless)."""
    rng = np.random.default_rng(degradation_seed_for(spec_seed, "noise", sim_id, frame_index, replica_id))
    return rng.standard_normal((height * width, 2))


@dataclass(frozen=True)
class G1Region:
    mask: np.ndarray  # (H, W) bool, True = invalidated
    n_cells: int
    area_fraction_effective: float
    area_mm2: float
    n_discs: int
    attempt: int
    seed: int


def g1_region(
    params: G1Params,
    *,
    spec_seed: int,
    sim_id: str,
    frame_index: int,
    replica_id: int,
    x_m: np.ndarray,
    y_m: np.ndarray,
    attempt: int = 0,
) -> G1Region:
    """Union of discs until the covered cell fraction reaches ``area_fraction``.

    Disc diameter ~ U[d_min, d_max] mm, centre ~ U over the bounding box of cell
    centres (both in physical mm, Euclidean distance, so anisotropic cells are
    handled).  Discs are clipped by the domain (no wrap).  Overlaps are allowed;
    adding stops at the first disc after which covered cells >= target.
    Uses only seeds and geometry: no velocity, no T, no model output.
    """
    h, w = len(y_m), len(x_m)
    xmm = np.asarray(x_m, dtype=np.float64) * 1e3
    ymm = np.asarray(y_m, dtype=np.float64) * 1e3
    xx, yy = np.meshgrid(xmm, ymm, indexing="xy")  # (H, W): columns follow x, rows follow y
    target = int(math.ceil(params.area_fraction * h * w))
    seed = degradation_seed_for(spec_seed, "g1", sim_id, frame_index, replica_id, int(attempt))
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), dtype=bool)
    n_discs = 0
    while int(mask.sum()) < target:
        if n_discs >= params.max_discs:
            raise DegradationError("g1: max_discs reached before target area")
        d = rng.uniform(params.diameter_min_mm, params.diameter_max_mm)
        cx = rng.uniform(xmm[0], xmm[-1])
        cy = rng.uniform(ymm[0], ymm[-1])
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.5 * d) ** 2
        n_discs += 1
    n_cells = int(mask.sum())
    cell_mm2 = float(np.mean(np.diff(xmm)) * np.mean(np.diff(ymm)))
    return G1Region(
        mask=mask,
        n_cells=n_cells,
        area_fraction_effective=n_cells / (h * w),
        area_mm2=n_cells * cell_mm2,
        n_discs=n_discs,
        attempt=int(attempt),
        seed=int(seed),
    )


def g1_remove(
    params: G1Params,
    sites: np.ndarray,
    *,
    spec_seed: int,
    sim_id: str,
    frame_index: int,
    replica_id: int,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> tuple[G1Region, np.ndarray]:
    """Generate a region with deterministic retries; returns (region, boolean 'removed' per site).

    Rejection (independent of T and of any model): zero removed sites, or fewer than
    ``min_surviving_sites`` survivors.  Attempt k uses its own seed.  Exhaustion raises.
    """
    sites = np.asarray(sites, dtype=np.int64)
    for attempt in range(params.max_attempts):
        reg = g1_region(
            params, spec_seed=spec_seed, sim_id=sim_id, frame_index=frame_index,
            replica_id=replica_id, x_m=x_m, y_m=y_m, attempt=attempt,
        )
        removed = reg.mask.ravel()[sites]
        n_rem = int(removed.sum())
        if n_rem == 0 or sites.size - n_rem < params.min_surviving_sites:
            continue
        return reg, removed
    raise DegradationError(
        f"g1: no acceptable region in {params.max_attempts} attempts ({sim_id}, {frame_index}, r{replica_id})"
    )


def g1_eq_removed(
    sites: np.ndarray, n_remove: int, *, spec_seed: int, sim_id: str, frame_index: int, replica_id: int
) -> np.ndarray:
    """Boolean per site: uniform random removal of exactly ``n_remove`` original sites."""
    sites = np.asarray(sites)
    if not 0 < n_remove < sites.size:
        raise DegradationError(f"g1_eq: invalid n_remove={n_remove} for {sites.size} sites")
    rng = np.random.default_rng(degradation_seed_for(spec_seed, "g1_eq", sim_id, frame_index, replica_id))
    pick = rng.choice(sites.size, size=int(n_remove), replace=False)
    removed = np.zeros(sites.size, dtype=bool)
    removed[pick] = True
    return removed


# --------------------------------------------------------------------------- driver

@dataclass(frozen=True)
class DegradedObservation:
    obs: CanonicalSparseObservation  # surviving sites only, values after noise
    cell_id: str
    original_sites: np.ndarray
    removed_sites: np.ndarray
    region_mask: np.ndarray | None  # G1 region on the dense grid (None for G1-eq / no gap)
    meta: dict[str, Any]


def _subset(obs: CanonicalSparseObservation, keep: np.ndarray, *, values_phys, values_norm, layout_id) -> CanonicalSparseObservation:
    sites = obs.sites[keep]
    mask = sites_to_mask(sites, height=obs.H, width=obs.W)
    return CanonicalSparseObservation(
        sites=sites,
        mask=mask,
        velocity_coords=obs.velocity_coords[keep],
        velocity_values_phys=values_phys[keep],
        velocity_values_norm=values_norm[keep],
        n_sites=int(sites.size),
        site_fraction=obs.site_fraction,
        layout_id=layout_id,
        velocity_mask_seed=obs.velocity_mask_seed,
        mask_hash=mask_hash(mask),
        H=obs.H,
        W=obs.W,
    )


def degrade_observation(
    obs: CanonicalSparseObservation,
    cell: CellSpec,
    spec: DegradationSpec,
    *,
    scale: VelocityScale,
    sim_id: str,
    frame_index: int,
    replica_id: int,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> DegradedObservation:
    """Apply one matrix cell to the original packet.

    The input packet is the historical one (same frames and sites for every cell).
    Signature deliberately has no temperature argument.
    """
    if obs.sites.size and np.any(np.diff(obs.sites) <= 0):
        raise DegradationError("obs.sites must be sorted and unique")
    n0 = int(obs.sites.size)
    vals_phys = np.array(obs.velocity_values_phys, dtype=np.float64, copy=True)
    vals_norm = np.array(obs.velocity_values_norm, dtype=np.float64, copy=True)

    noise_ms = None
    if cell.noise_sigma_ms:  # None or 0.0 -> identity on values (bit-exact)
        z = standardized_noise(spec.degradation_seed, sim_id, frame_index, replica_id, height=obs.H, width=obs.W)
        noise_ms = cell.noise_sigma_ms * z[obs.sites]  # (N0, 2) in m/s, generated before invalidation
        vals_phys = vals_phys + noise_ms
        vals_norm = vals_norm + scale.noise_ms_to_norm(noise_ms)

    region_mask = None
    info: dict[str, Any] = {"n_discs": None, "attempt": None, "region_cells": None}
    if cell.gap == "g1":
        reg, removed = g1_remove(
            spec.g1, obs.sites, spec_seed=spec.degradation_seed, sim_id=sim_id,
            frame_index=frame_index, replica_id=replica_id, x_m=x_m, y_m=y_m,
        )
        region_mask = reg.mask
        info.update(
            n_discs=reg.n_discs, attempt=reg.attempt, region_cells=reg.n_cells,
            area_fraction_effective=reg.area_fraction_effective, area_mm2=reg.area_mm2,
            gap_seed=reg.seed,
        )
    elif cell.gap == "g1_eq":
        # Same count as G1 for this frame/replica: recompute G1 (deterministic) and use only its size.
        _, g1_removed = g1_remove(
            spec.g1, obs.sites, spec_seed=spec.degradation_seed, sim_id=sim_id,
            frame_index=frame_index, replica_id=replica_id, x_m=x_m, y_m=y_m,
        )
        removed = g1_eq_removed(
            obs.sites, int(g1_removed.sum()), spec_seed=spec.degradation_seed,
            sim_id=sim_id, frame_index=frame_index, replica_id=replica_id,
        )
    else:
        removed = np.zeros(n0, dtype=bool)

    keep = ~removed
    n_keep = int(keep.sum())
    if n_keep == 0:
        raise DegradationError("no surviving sites")
    if cell.gap is not None and n_keep < spec.g1.min_surviving_sites:
        raise DegradationError(f"only {n_keep} surviving sites (< {spec.g1.min_surviving_sites})")

    if cell.noise_sigma_ms is None and cell.gap is None:
        new_obs = obs  # identity: the very same historical packet
    else:
        new_obs = _subset(obs, keep, values_phys=vals_phys, values_norm=vals_norm,
                          layout_id=f"{obs.layout_id}|{spec.spec_id}|{cell.cell_id}")
    meta = {
        "cell_id": cell.cell_id,
        "spec_id": spec.spec_id,
        "sim_id": sim_id,
        "frame_index": int(frame_index),
        "replica_id": int(replica_id),
        "n_original": n0,
        "n_surviving": n_keep,
        "n_removed": int(removed.sum()),
        "removed_site_fraction": float(removed.sum()) / n0,
        "noise_sigma_ms": cell.noise_sigma_ms,
        "gap": cell.gap,
        "original_mask_hash": obs.mask_hash,
        "surviving_mask_hash": new_obs.mask_hash,
        **info,
    }
    return DegradedObservation(
        obs=new_obs,
        cell_id=cell.cell_id,
        original_sites=obs.sites,
        removed_sites=obs.sites[removed],
        region_mask=region_mask,
        meta=meta,
    )


def glu_batching_policy(n_sites_per_frame: list[int]) -> list[list[int]]:
    """Homogeneous-N batches in frame order (existing GLU wrapper forbids ragged batches).

    Consecutive frames are grouped up to 8 while N is identical; any change of N starts a
    new batch.  With constant N (R0, N1, N2) this is exactly the historical batch of 8.
    With G1-type cells N varies per frame, hence batches of 1.
    """
    out: list[list[int]] = []
    cur: list[int] = []
    for i, n in enumerate(n_sites_per_frame):
        if cur and (n != n_sites_per_frame[cur[0]] or len(cur) == 8):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out
