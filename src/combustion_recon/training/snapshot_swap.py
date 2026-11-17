"""Deterministic velocity snapshot-swap control (prepared for U5; not executed here)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def build_within_condition_swap_pairs(
    frames: Sequence[dict[str, Any]],
    *,
    role: str,
) -> list[dict[str, Any]]:
    """Pair each frame j with (j + floor(n/2)) mod n within the same condition.

    Coordinates, p, and targets stay on the destination frame; only velocity
    fields would be taken from the source. No cross-condition or cross-role
    pairing. Index distances are not converted to physical times.
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
        shift = n // 2
        for j, dest in enumerate(rows):
            src = rows[(j + shift) % n]
            if int(src["frame_index"]) == int(dest["frame_index"]) and n > 1:
                raise ValueError(f"fixed point in swap pairing for {sim_id}")
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


def apply_velocity_swap_to_input(
    dest_input: np.ndarray,
    src_velocity: np.ndarray,
) -> np.ndarray:
    """Replace vi/vj channels (0,1) of dest_input with src velocity (2,H,W) or (H,W)×2."""
    out = np.array(dest_input, copy=True)
    if src_velocity.ndim == 3 and src_velocity.shape[0] >= 2:
        out[0] = src_velocity[0]
        out[1] = src_velocity[1]
    elif src_velocity.ndim == 4 and src_velocity.shape[1] >= 2:
        raise ValueError("batch velocity not supported here")
    else:
        raise ValueError("src_velocity must be (2,H,W)")
    return out
