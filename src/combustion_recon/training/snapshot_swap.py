"""Deterministic velocity snapshot-swap control (U5).

Coordinates, NH3/phi, and temperature targets stay on the destination frame.
Only planar velocity channels (vi, vj) are taken from the donor.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from combustion_recon.training.dense_cache import CHANNEL_ORDER, DenseUNetCacheDataset


def build_within_condition_swap_pairs(
    frames: Sequence[dict[str, Any]],
    *,
    role: str,
) -> list[dict[str, Any]]:
    """Pair each frame j with (j + floor(n/2)) mod n within the same condition.

    No cross-condition or cross-role pairing. Index distances are not converted
    to physical times (anomalous temporal shards keep index semantics only).
    """
    by_sim: dict[str, list[dict[str, Any]]] = {}
    for row in frames:
        if str(row.get("role", role)) != role:
            raise ValueError("mixed roles in snapshot-swap input")
        by_sim.setdefault(str(row["sim_id"]), []).append(dict(row))

    pairs: list[dict[str, Any]] = []
    for sim_id, rows in sorted(by_sim.items()):
        rows = sorted(rows, key=lambda r: int(r["frame_index"]))
        n = len(rows)
        if n == 0:
            continue
        if n == 1:
            raise ValueError(f"cannot build swap pairs for singleton condition {sim_id}")
        shift = n // 2
        if shift % n == 0:
            raise ValueError(f"degenerate shift for {sim_id} with n={n}")
        for j, dest in enumerate(rows):
            src = rows[(j + shift) % n]
            if int(src["frame_index"]) == int(dest["frame_index"]):
                raise ValueError(f"fixed point in swap pairing for {sim_id}")
            if int(src["cache_index"]) == int(dest["cache_index"]):
                raise ValueError(f"fixed cache_index in swap pairing for {sim_id}")
            pairs.append(
                {
                    "role": role,
                    "sim_id": sim_id,
                    "dest_frame_index": int(dest["frame_index"]),
                    "dest_cache_index": int(dest["cache_index"]),
                    "src_frame_index": int(src["frame_index"]),
                    "src_cache_index": int(src["cache_index"]),
                    "index_shift": int(shift),
                    "n_frames_in_condition": n,
                    "note": "index distance ≠ verified physical Δt on anomalous shards",
                }
            )
    return pairs


def pairs_by_dest_cache_index(pairs: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for p in pairs:
        key = int(p["dest_cache_index"])
        if key in out:
            raise ValueError(f"duplicate dest_cache_index in pairing: {key}")
        out[key] = dict(p)
    return out


def inverse_pairs(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Invert dest←src velocity mapping (for aligned reconstruction checks)."""
    inv: list[dict[str, Any]] = []
    for p in pairs:
        inv.append(
            {
                **{k: p[k] for k in ("role", "sim_id", "index_shift", "n_frames_in_condition", "note") if k in p},
                "dest_frame_index": int(p["src_frame_index"]),
                "dest_cache_index": int(p["src_cache_index"]),
                "src_frame_index": int(p["dest_frame_index"]),
                "src_cache_index": int(p["dest_cache_index"]),
                "inverted_from_forward_pair": True,
            }
        )
    return inv


def apply_velocity_swap_to_input(
    dest_input: np.ndarray,
    src_velocity: np.ndarray,
) -> np.ndarray:
    """Replace vi/vj channels (0,1) of dest_input with src velocity (2,H,W)."""
    out = np.array(dest_input, copy=True, dtype=np.float32)
    if src_velocity.ndim == 3 and src_velocity.shape[0] >= 2:
        out[0] = np.asarray(src_velocity[0], dtype=np.float32)
        out[1] = np.asarray(src_velocity[1], dtype=np.float32)
    elif src_velocity.ndim == 4 and src_velocity.shape[1] >= 2:
        raise ValueError("batch velocity not supported here")
    else:
        raise ValueError("src_velocity must be (2,H,W)")
    return out


def build_swapped_input_from_cache(
    ds: DenseUNetCacheDataset,
    dataset_index: int,
    *,
    pair_lookup: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a sample dict with vi/vj swapped; coords/p/target unchanged.

    Does not mutate mmap cache arrays.
    """
    item = ds[dataset_index]
    dest_cache_index = int(ds.indices[dataset_index])
    if dest_cache_index not in pair_lookup:
        raise KeyError(f"no swap pair for dest_cache_index={dest_cache_index}")
    pair = pair_lookup[dest_cache_index]
    src_idx = int(pair["src_cache_index"])
    src_vel = np.stack(
        [
            np.asarray(ds.vi[src_idx], dtype=np.float32),
            np.asarray(ds.vj[src_idx], dtype=np.float32),
        ],
        axis=0,
    )
    x_aligned = item["model_inputs"]["input_tensor"].detach().cpu().numpy()
    x_swapped = apply_velocity_swap_to_input(x_aligned, src_vel)

    # Non-velocity channels must match destination.
    if not np.allclose(x_swapped[2:], x_aligned[2:], rtol=0.0, atol=0.0):
        raise RuntimeError("swap altered non-velocity channels")
    # Velocity must match donor.
    if not np.allclose(x_swapped[0], ds.vi[src_idx], rtol=0.0, atol=0.0):
        raise RuntimeError("swapped vi does not match donor")
    if not np.allclose(x_swapped[1], ds.vj[src_idx], rtol=0.0, atol=0.0):
        raise RuntimeError("swapped vj does not match donor")

    out = {
        "model_inputs": {
            **item["model_inputs"],
            "input_tensor": torch.from_numpy(np.array(x_swapped, dtype=np.float32, copy=True)),
            "channel_order": list(CHANNEL_ORDER),
            "snapshot_control": "swapped_velocity",
            "swap_pair": dict(pair),
        },
        "targets": item["targets"],
        "metadata": {
            **item["metadata"],
            "snapshot_control": "swapped_velocity",
            "src_frame_index": int(pair["src_frame_index"]),
            "src_cache_index": int(pair["src_cache_index"]),
            "dest_frame_index": int(pair["dest_frame_index"]),
            "dest_cache_index": int(pair["dest_cache_index"]),
        },
    }
    return out


def reconstruct_aligned_input_via_inverse_swap(
    ds: DenseUNetCacheDataset,
    dataset_index: int,
    *,
    forward_pair_lookup: Mapping[int, Mapping[str, Any]],
) -> np.ndarray:
    """Recover aligned input from a swapped view without writing the cache.

    Takes the swapped tensor (dest coords/p + src velocity) and restores
    destination velocity using the inverse pair map.
    """
    swapped = build_swapped_input_from_cache(
        ds, dataset_index, pair_lookup=forward_pair_lookup
    )
    x_swapped = swapped["model_inputs"]["input_tensor"].detach().cpu().numpy()
    dest_cache_index = int(ds.indices[dataset_index])
    # Inverse: put back dest's own velocity.
    dest_vel = np.stack(
        [
            np.asarray(ds.vi[dest_cache_index], dtype=np.float32),
            np.asarray(ds.vj[dest_cache_index], dtype=np.float32),
        ],
        axis=0,
    )
    return apply_velocity_swap_to_input(x_swapped, dest_vel)
