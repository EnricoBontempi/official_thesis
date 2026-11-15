"""Observation mask generators and sparse token extraction.

Missing / unobserved grid locations are never represented as physical zeros.
Use :func:`masked_field_to_tokens` to emit ``(x, y, var_id, value)`` only at
observed sites.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAYOUTS_PATH = ROOT / "configs" / "masks" / "observation_layouts.yaml"


def stable_seed(*parts: object) -> int:
    """Process-independent 32-bit seed from stringifiable parts (zlib.crc32)."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int(zlib.crc32(payload) & 0xFFFFFFFF)

TaskMode = Literal["temperature_free", "temperature_assisted"]

VAR_ID = {
    "velocity_i": 11,
    "velocity_j": 12,
    "velocity_k": 13,
    "temperature": 10,
}


@dataclass(frozen=True)
class ObservationTokens:
    """Sparse observation packet.

    Attributes
    ----------
    coords : (M, 2) float
        Physical ``(x, y)`` for each observation.
    var_ids : (M,) int
        Channel / variable identity.
    values : (M,) float
        Observed scalar values (never padded with zeros for missing sites).
    mask_velocity : (H, W) bool
        True where velocity sensors are present.
    mask_temperature : (H, W) bool
        True where temperature anchors are present (all-False for Task A).
    """

    coords: np.ndarray
    var_ids: np.ndarray
    values: np.ndarray
    mask_velocity: np.ndarray
    mask_temperature: np.ndarray


def load_layouts(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else DEFAULT_LAYOUTS_PATH
    with p.open() as f:
        return yaml.safe_load(f)


def _full_false(h: int = 128, w: int = 128) -> np.ndarray:
    return np.zeros((h, w), dtype=bool)


def random_sparse_mask(
    fraction: float,
    *,
    seed: int,
    height: int = 128,
    width: int = 128,
) -> np.ndarray:
    """Deterministic Bernoulli / subsample mask with approximately ``fraction`` True."""
    n = height * width
    k = max(1, int(round(fraction * n)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    mask = _full_false(height, width)
    mask.ravel()[idx] = True
    return mask


def horizontal_line_mask(row: int, height: int = 128, width: int = 128) -> np.ndarray:
    mask = _full_false(height, width)
    mask[int(row), :] = True
    return mask


def vertical_line_mask(col: int, height: int = 128, width: int = 128) -> np.ndarray:
    mask = _full_false(height, width)
    mask[:, int(col)] = True
    return mask


def box_mask(
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    *,
    height: int = 128,
    width: int = 128,
    missing: bool = True,
) -> np.ndarray:
    """If ``missing=True``, return a mask that is False inside the box (occlusion).

    Observed region = complement of the box.
    """
    observed = np.ones((height, width), dtype=bool)
    observed[row_start:row_end, col_start:col_end] = False
    return observed if missing else ~observed


def build_layout_mask(
    layout_name: str,
    *,
    seed: int | None = None,
    layouts: dict | None = None,
    height: int = 128,
    width: int = 128,
) -> np.ndarray:
    """Build a boolean observation mask (True = observed) for a named layout."""
    cfg = layouts if layouts is not None else load_layouts()
    if layout_name in cfg.get("random_sparse", {}):
        frac = float(cfg["random_sparse"][layout_name]["fraction"])
        if seed is None:
            seed = int(cfg.get("seed_eval_default", 0))
        return random_sparse_mask(frac, seed=seed, height=height, width=width)

    if layout_name in cfg.get("structured_rakes", {}):
        spec = cfg["structured_rakes"][layout_name]
        if spec["type"] == "horizontal_line":
            return horizontal_line_mask(spec["row"], height, width)
        if spec["type"] == "vertical_line":
            return vertical_line_mask(spec["col"], height, width)
        raise ValueError(f"Unknown rake type: {spec['type']}")

    if layout_name in cfg.get("localized_missing", {}):
        spec = cfg["localized_missing"][layout_name]
        return box_mask(
            spec["row_start"],
            spec["row_end"],
            spec["col_start"],
            spec["col_end"],
            height=height,
            width=width,
            missing=True,
        )

    raise KeyError(f"Unknown layout_name={layout_name!r}")


def nested_temperature_anchor_permutation(
    *,
    seed: int,
    height: int = 128,
    width: int = 128,
    max_k: int | None = None,
) -> np.ndarray:
    """Fixed random permutation of grid sites for nested K-anchor sweeps.

    Taking ``perm[:K]`` for increasing ``K`` guarantees
    ``K=1 ⊂ K=4 ⊂ K=9 ⊂ …`` for a fixed ``seed``.
    """
    n = height * width
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    if max_k is not None:
        return perm[: max(int(max_k), 0)]
    return perm


def temperature_anchor_mask(
    k: int,
    *,
    seed: int,
    height: int = 128,
    width: int = 128,
    exclude: np.ndarray | None = None,
    nested: bool = True,
    max_k: int = 32,
) -> np.ndarray:
    """Place ``k`` sparse temperature sensors (True = T observed).

    By default anchors are sampled independently on the full grid (``exclude``
    is ignored unless ``nested=False`` and ``exclude`` is provided for legacy
    callers). With ``nested=True``, sites come from a fixed permutation so
    larger ``K`` strictly contain smaller ``K``.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return _full_false(height, width)
    n = height * width
    mask = _full_false(height, width)
    if nested:
        perm = nested_temperature_anchor_permutation(
            seed=seed, height=height, width=width, max_k=max(max_k, k)
        )
        idx = perm[: min(k, perm.size)]
    else:
        rng = np.random.default_rng(seed)
        candidates = np.arange(n)
        if exclude is not None:
            free = ~exclude.ravel()
            if int(free.sum()) >= k:
                candidates = np.flatnonzero(free)
        idx = rng.choice(candidates, size=min(k, candidates.size), replace=False)
    mask.ravel()[idx] = True
    return mask


def masked_field_to_tokens(
    fields: dict[int, np.ndarray],
    masks: dict[int, np.ndarray],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert masked fields into sparse tokens.

    Parameters
    ----------
    fields :
        Mapping ``var_id -> (H, W)`` array.
    masks :
        Mapping ``var_id -> (H, W) bool`` (True where that variable is observed).
    x_coords, y_coords :
        1D length-W / length-H physical grids, or 2D ``(H, W)`` meshes.

    Returns
    -------
    coords : (M, 2)
    var_ids : (M,)
    values : (M,)
    """
    if x_coords.ndim == 1 and y_coords.ndim == 1:
        xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
    else:
        xx, yy = x_coords, y_coords

    coords_list: list[np.ndarray] = []
    var_list: list[np.ndarray] = []
    val_list: list[np.ndarray] = []
    for var_id, field in fields.items():
        m = masks[var_id]
        if m.shape != field.shape:
            raise ValueError(f"Mask/field shape mismatch for var_id={var_id}")
        ys, xs = np.where(m)
        if ys.size == 0:
            continue
        coords_list.append(np.stack([xx[ys, xs], yy[ys, xs]], axis=1))
        var_list.append(np.full(ys.shape, int(var_id), dtype=np.int64))
        val_list.append(field[ys, xs].astype(np.float64, copy=False))

    if not coords_list:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
        )
    return (
        np.concatenate(coords_list, axis=0),
        np.concatenate(var_list, axis=0),
        np.concatenate(val_list, axis=0),
    )


def build_observation_tokens(
    velocity_i: np.ndarray,
    velocity_j: np.ndarray,
    temperature: np.ndarray,
    *,
    layout_name: str,
    task_mode: TaskMode = "temperature_free",
    n_temperature_anchors: int = 0,
    seed: int | None = None,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    include_velocity_k: np.ndarray | None = None,
    layouts: dict | None = None,
) -> ObservationTokens:
    """Build sparse tokens for Task A or Task B under a named layout."""
    cfg = layouts if layouts is not None else load_layouts()
    if seed is None:
        seed = int(cfg.get("seed_eval_default", 0))

    vel_mask = build_layout_mask(layout_name, seed=seed, layouts=cfg)
    t_mask = _full_false(*vel_mask.shape)

    if task_mode == "temperature_assisted":
        if n_temperature_anchors <= 0:
            raise ValueError("temperature_assisted requires n_temperature_anchors > 0")
        # Nested independent T anchors (not restricted to velocity sites)
        t_mask = temperature_anchor_mask(
            n_temperature_anchors,
            seed=stable_seed(seed, "temp_anchors"),
            nested=True,
            max_k=32,
        )
    elif task_mode != "temperature_free":
        raise ValueError(f"Unknown task_mode={task_mode!r}")

    fields = {
        VAR_ID["velocity_i"]: velocity_i,
        VAR_ID["velocity_j"]: velocity_j,
    }
    masks = {
        VAR_ID["velocity_i"]: vel_mask,
        VAR_ID["velocity_j"]: vel_mask,
    }
    if include_velocity_k is not None:
        fields[VAR_ID["velocity_k"]] = include_velocity_k
        masks[VAR_ID["velocity_k"]] = vel_mask
    if task_mode == "temperature_assisted":
        fields[VAR_ID["temperature"]] = temperature
        masks[VAR_ID["temperature"]] = t_mask

    coords, var_ids, values = masked_field_to_tokens(fields, masks, x_coords, y_coords)
    return ObservationTokens(
        coords=coords,
        var_ids=var_ids,
        values=values,
        mask_velocity=vel_mask,
        mask_temperature=t_mask,
    )


def assert_no_zero_fill_for_missing(
    tokens: ObservationTokens,
    full_field: np.ndarray,
    var_id: int,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    atol: float = 0.0,
) -> None:
    """Guardrail helper: every token with ``var_id`` must match the true field.

    Also asserts that the number of tokens equals the number of True mask sites
    for that variable (no phantom zero observations).
    """
    if x_coords.ndim == 1 and y_coords.ndim == 1:
        xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
    else:
        xx, yy = x_coords, y_coords

    sel = tokens.var_ids == var_id
    n_tok = int(sel.sum())
    if var_id == VAR_ID["temperature"]:
        n_mask = int(tokens.mask_temperature.sum())
    else:
        n_mask = int(tokens.mask_velocity.sum())
    if n_tok != n_mask:
        raise AssertionError(
            f"Token count {n_tok} != mask True count {n_mask} for var_id={var_id}"
        )
    # Round-trip values
    for (x, y), val in zip(tokens.coords[sel], tokens.values[sel]):
        # nearest grid index
        j = int(np.argmin(np.abs(xx[0, :] - x))) if xx.ndim == 2 else int(np.argmin(np.abs(x_coords - x)))
        i = int(np.argmin(np.abs(yy[:, 0] - y))) if yy.ndim == 2 else int(np.argmin(np.abs(y_coords - y)))
        if abs(float(full_field[i, j]) - float(val)) > atol + 1e-6:
            raise AssertionError("Token value does not match field at claimed location")
