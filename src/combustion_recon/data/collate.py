"""Collate helpers for variable-length Month-2 observation tokens."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _pad_np(arrays: list[np.ndarray], pad_value: float | int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pad a list of (N_i, ...) arrays to (B, N_max, ...). Returns values, validity mask (B, N_max)."""
    if not arrays:
        raise ValueError("empty batch")
    lengths = [int(a.shape[0]) for a in arrays]
    n_max = max(lengths) if lengths else 0
    b = len(arrays)
    if n_max == 0:
        # preserve trailing dims if any from a prototype
        proto = arrays[0]
        out_shape = (b, 0) + proto.shape[1:]
        return np.zeros(out_shape, dtype=proto.dtype), np.zeros((b, 0), dtype=bool)
    trailing = arrays[0].shape[1:]
    out = np.full((b, n_max) + trailing, pad_value, dtype=arrays[0].dtype)
    valid = np.zeros((b, n_max), dtype=bool)
    for i, a in enumerate(arrays):
        n = a.shape[0]
        if n:
            out[i, :n] = a
            valid[i, :n] = True
    return out, valid


def collate_month2_batch(samples: list[dict[str, Any]], *, as_torch: bool = True) -> dict[str, Any]:
    """Collate Month2ReconDataset samples with padding masks for variable-length tokens."""
    model_inputs_list = [s["model_inputs"] for s in samples]
    targets_list = [s["targets"] for s in samples]
    meta_list = [s["metadata"] for s in samples]

    vel_coords, vel_valid = _pad_np([m["velocity_coords"] for m in model_inputs_list])
    vel_vals, _ = _pad_np([m["velocity_values"] for m in model_inputs_list])
    vel_sites, _ = _pad_np([m["velocity_site_index"] for m in model_inputs_list], pad_value=-1)

    sc_coords, sc_valid = _pad_np([m["scalar_coords"] for m in model_inputs_list])
    sc_vals, _ = _pad_np([m["scalar_values"] for m in model_inputs_list])
    sc_ids, _ = _pad_np([m["scalar_var_ids"] for m in model_inputs_list], pad_value=-1)

    anc_coords, anc_valid = _pad_np([m["anchor_coords"] for m in model_inputs_list])
    anc_vals, _ = _pad_np([m["anchor_values"] for m in model_inputs_list])
    anc_sites, _ = _pad_np([m["anchor_site_index"] for m in model_inputs_list], pad_value=-1)

    dense = np.stack([m["dense_values"] for m in model_inputs_list], axis=0)
    query = np.stack([m["query_coords"] for m in model_inputs_list], axis=0)
    mask_vel = np.stack([m["mask_velocity"] for m in model_inputs_list], axis=0)
    mask_t = np.stack([m["mask_temperature"] for m in model_inputs_list], axis=0)

    model_inputs: dict[str, Any] = {
        "velocity_coords": vel_coords,
        "velocity_values": vel_vals,
        "velocity_site_index": vel_sites,
        "velocity_valid": vel_valid,
        "scalar_coords": sc_coords,
        "scalar_values": sc_vals,
        "scalar_var_ids": sc_ids,
        "scalar_valid": sc_valid,
        "anchor_coords": anc_coords,
        "anchor_values": anc_vals,
        "anchor_site_index": anc_sites,
        "anchor_valid": anc_valid,
        "dense_values": dense,
        "dense_channel_names": list(model_inputs_list[0]["dense_channel_names"]),
        "query_coords": query,
        "mask_velocity": mask_vel,
        "mask_temperature": mask_t,
        "K": np.asarray([m["K"] for m in model_inputs_list], dtype=np.int64),
        "site_fraction": np.asarray(
            [m["site_fraction"] for m in model_inputs_list], dtype=np.float64
        ),
        "task_mode": [m["task_mode"] for m in model_inputs_list],
    }

    # op_params only if present for all (p_on); otherwise omit
    if all("op_params" in m for m in model_inputs_list):
        model_inputs["op_params_normalized"] = np.stack(
            [m["op_params"]["normalized"] for m in model_inputs_list], axis=0
        )
        model_inputs["nh3_fraction"] = np.asarray(
            [m["op_params"]["nh3_fraction"] for m in model_inputs_list], dtype=np.float64
        )
        model_inputs["phi"] = np.asarray(
            [m["op_params"]["phi"] for m in model_inputs_list], dtype=np.float64
        )
    elif any("op_params" in m for m in model_inputs_list):
        raise ValueError("Batch mixes p_on and p_off samples; collate refuses silent merge")

    targets = {
        "temperature": np.stack([t["temperature"] for t in targets_list], axis=0),
        "temperature_normalized": np.stack(
            [t["temperature_normalized"] for t in targets_list], axis=0
        ),
    }

    # Metadata: keep list-of-dicts; strip hidden full fields from default batch to limit RAM
    metadata = []
    for m in meta_list:
        md = {k: v for k, v in m.items() if not str(k).startswith("_hidden_")}
        metadata.append(md)

    batch = {"model_inputs": model_inputs, "targets": targets, "metadata": metadata}

    if as_torch:
        batch["model_inputs"] = _to_torch(batch["model_inputs"])
        batch["targets"] = _to_torch(batch["targets"])
    return batch


def _to_torch(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_torch(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        if obj.dtype == bool:
            return torch.from_numpy(obj.copy())
        if np.issubdtype(obj.dtype, np.integer):
            return torch.from_numpy(obj.astype(np.int64))
        return torch.from_numpy(obj.astype(np.float32))
    return obj
