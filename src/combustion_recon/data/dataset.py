"""Arrow-backed combustion dataset filtered by locked operating-condition splits."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import torch
import yaml
from torch.utils.data import Dataset

from combustion_recon.data.masks import (
    ObservationTokens,
    build_observation_tokens,
    load_layouts,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLITS = ROOT / "configs" / "splits" / "splits_interp_extrap.yaml"
DEFAULT_RAW = ROOT / "data" / "raw" / "combustion" / "hf_dataset" / "numerical"
DEFAULT_OC = ROOT / "data" / "manifests" / "operating_conditions.yaml"

SplitName = Literal["train", "val", "test_interp", "test_extrap"]


def load_splits(path: str | Path | None = None) -> dict[str, list[str]]:
    p = Path(path) if path is not None else DEFAULT_SPLITS
    with p.open() as f:
        cfg = yaml.safe_load(f)
    return {
        "train": list(cfg["train"]["sim_ids"]),
        "val": list(cfg["val"]["sim_ids"]),
        "test_interp": list(cfg["test_interp"]["sim_ids"]),
        "test_extrap": list(cfg["test_extrap"]["sim_ids"]),
    }


def parse_sim_id(sim_id: str) -> tuple[int, float]:
    m = re.match(r"(\d+)NH3_(\d+\.?\d*)\.h5", sim_id)
    if not m:
        raise ValueError(f"Cannot parse sim_id={sim_id!r}")
    return int(m.group(1)), float(m.group(2))


def _load_arrow_row(path: Path) -> dict[str, Any]:
    with pa.memory_map(str(path), "r") as mm:
        table = pa.ipc.open_stream(mm).read_all()
    if table.num_rows != 1:
        raise ValueError(f"{path.name}: expected 1 row, got {table.num_rows}")
    return {c: table.column(c)[0].as_py() for c in table.column_names}


def build_sim_id_to_shard(raw_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(raw_dir.glob("data-*-of-00030.arrow")):
        row = _load_arrow_row(path)
        mapping[row["sim_id"]] = path
    return mapping


class CombustionReconDataset(Dataset):
    """Full-resolution (128×128) sparse-observation temperature reconstruction samples.

    Returns a dict with:
      - ``tokens``: ObservationTokens (or tensor pack)
      - ``query_coords``: (H*W, 2) physical query locations
      - ``target_T``: (H, W) temperature
      - ``op_params``: ``{nh3_fraction, equivalence_ratio, nh3_pct}``
      - ``metadata``: sim_id, split, frame_index, etc.
    """

    def __init__(
        self,
        split: SplitName,
        *,
        raw_dir: str | Path | None = None,
        splits_path: str | Path | None = None,
        layout_name: str = "random_5pct",
        task_mode: str = "temperature_free",
        n_temperature_anchors: int = 0,
        frame_stride: int = 50,
        max_frames_per_traj: int | None = None,
        seed: int = 20261114,
        include_velocity_k: bool = False,
        return_torch: bool = True,
    ):
        self.split = split
        self.raw_dir = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW
        self.splits = load_splits(splits_path)
        if split not in self.splits:
            raise KeyError(split)
        self.sim_ids = list(self.splits[split])
        self.layout_name = layout_name
        self.task_mode = task_mode
        self.n_temperature_anchors = n_temperature_anchors
        self.frame_stride = int(frame_stride)
        self.max_frames_per_traj = max_frames_per_traj
        self.seed = int(seed)
        self.include_velocity_k = include_velocity_k
        self.return_torch = return_torch
        self.layouts = load_layouts()

        # Map sim_id -> shard (prefer operating_conditions.yaml if present)
        self._shard_of: dict[str, Path] = {}
        oc_path = DEFAULT_OC
        if oc_path.exists():
            with oc_path.open() as f:
                oc = yaml.safe_load(f)
            for cond in oc.get("conditions", []):
                sid = cond["sim_id"]
                shard = cond.get("shard_filename")
                if shard:
                    self._shard_of[sid] = self.raw_dir / shard
        # Fill any missing via scan (expensive; only if needed)
        missing = [s for s in self.sim_ids if s not in self._shard_of or not self._shard_of[s].exists()]
        if missing:
            full_map = build_sim_id_to_shard(self.raw_dir)
            for s in self.sim_ids:
                self._shard_of[s] = full_map[s]

        self._index: list[tuple[str, int]] = []
        self._cache: dict[str, dict[str, Any]] = {}
        for sid in self.sim_ids:
            meta = self._ensure_loaded(sid)
            n_t = int(meta["shape_t"])
            frames = list(range(0, n_t, self.frame_stride))
            if self.max_frames_per_traj is not None:
                frames = frames[: self.max_frames_per_traj]
            for fi in frames:
                self._index.append((sid, fi))

    def _ensure_loaded(self, sim_id: str) -> dict[str, Any]:
        if sim_id in self._cache:
            return self._cache[sim_id]
        path = self._shard_of[sim_id]
        row = _load_arrow_row(path)
        shape = (int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"]))
        nchan = int(row["numerical_channels"])
        numerical = np.frombuffer(row["numerical"], dtype=np.float32).reshape(*shape, nchan)
        x = np.frombuffer(row["x"], dtype=np.float64).reshape(int(row["x_shape"]))
        y = np.frombuffer(row["y"], dtype=np.float64).reshape(int(row["y_shape"]))
        t = np.frombuffer(row["t"], dtype=np.float64).reshape(int(row["t_shape"]))
        nh3_pct, phi = parse_sim_id(sim_id)
        pack = {
            "sim_id": sim_id,
            "numerical": numerical,
            "x": x,
            "y": y,
            "t": t,
            "shape_t": shape[0],
            "shape_h": shape[1],
            "shape_w": shape[2],
            "nh3_pct": nh3_pct,
            "nh3_fraction": nh3_pct / 100.0,
            "equivalence_ratio": phi,
        }
        self._cache[sim_id] = pack
        return pack

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sim_id, frame = self._index[idx]
        traj = self._ensure_loaded(sim_id)
        num = traj["numerical"][frame]  # (H,W,C)
        T = num[:, :, 10]
        vi = num[:, :, 11]
        vj = num[:, :, 12]
        vk = num[:, :, 13]
        u_mag = np.sqrt(vi * vi + vj * vj + vk * vk)

        H, W = T.shape
        xx, yy = np.meshgrid(traj["x"], traj["y"], indexing="xy")
        query_coords = np.stack([xx.ravel(), yy.ravel()], axis=1)

        tokens = build_observation_tokens(
            vi,
            vj,
            T,
            layout_name=self.layout_name,
            task_mode=self.task_mode,  # type: ignore[arg-type]
            n_temperature_anchors=self.n_temperature_anchors,
            seed=self.seed + idx,
            x_coords=traj["x"],
            y_coords=traj["y"],
            include_velocity_k=vk if self.include_velocity_k else None,
            layouts=self.layouts,
        )

        sample = {
            "tokens": tokens,
            "query_coords": query_coords.astype(np.float64),
            "target_T": T.astype(np.float64),
            "velocity_i": vi.astype(np.float64),
            "velocity_j": vj.astype(np.float64),
            "velocity_k": vk.astype(np.float64),
            "velocity_mag": u_mag.astype(np.float64),
            "op_params": {
                "nh3_fraction": float(traj["nh3_fraction"]),
                "equivalence_ratio": float(traj["equivalence_ratio"]),
                "nh3_pct": int(traj["nh3_pct"]),
            },
            "metadata": {
                "sim_id": sim_id,
                "split": self.split,
                "frame_index": int(frame),
                "time": float(traj["t"][frame]),
                "layout_name": self.layout_name,
                "task_mode": self.task_mode,
                "n_temperature_anchors": int(self.n_temperature_anchors),
                "grid": [H, W],
            },
        }
        if self.return_torch:
            sample["query_coords"] = torch.from_numpy(sample["query_coords"]).float()
            sample["target_T"] = torch.from_numpy(sample["target_T"]).float()
            sample["token_coords"] = torch.from_numpy(tokens.coords).float()
            sample["token_var_ids"] = torch.from_numpy(tokens.var_ids).long()
            sample["token_values"] = torch.from_numpy(tokens.values).float()
        return sample
