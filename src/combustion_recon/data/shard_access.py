"""Lazy Arrow shard access with compact frame caching for Month 2.

Full numerical trajectories are ~2 GB float32 each. This module:
- does **not** load payloads at initialization;
- decodes at most one shard at a time when materializing frames;
- retains only requested channels/frames in a compact float32 cache;
- never keeps a durable full-trajectory dict like the Month-1 loader.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa

from combustion_recon.data.month2_protocol import load_shard_map

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW = ROOT / "data" / "raw" / "combustion" / "hf_dataset" / "numerical"

# Channel indices in the packed numerical array
CH_T = 10
CH_VI = 11
CH_VJ = 12
CH_VK = 13


@dataclass(frozen=True)
class CompactFrame:
    """Single-frame physical fields needed by the Month-2 primary protocol."""

    temperature: np.ndarray  # (H, W) float32
    velocity_i: np.ndarray
    velocity_j: np.ndarray
    velocity_k: np.ndarray  # stored for isolation tests; not placed in 2C inputs
    x: np.ndarray  # (W,) float64
    y: np.ndarray  # (H,) float64
    t: float
    sim_id: str
    frame_index: int


class ShardStore:
    """Per-process shard store with compact LRU cache of frames."""

    def __init__(
        self,
        *,
        raw_dir: str | Path | None = None,
        shard_map: dict[str, str] | None = None,
        max_cached_frames: int = 2048,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW
        self.shard_map = shard_map if shard_map is not None else load_shard_map()
        self.max_cached_frames = int(max_cached_frames)
        self._frame_cache: OrderedDict[tuple[str, int], CompactFrame] = OrderedDict()
        self._coords_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._lock = threading.RLock()
        self.stats: dict[str, Any] = {
            "shard_decodes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "frames_materialized": 0,
        }

    def shard_path(self, sim_id: str) -> Path:
        if sim_id not in self.shard_map:
            raise KeyError(f"Unknown sim_id={sim_id!r}")
        path = self.raw_dir / self.shard_map[sim_id]
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _read_arrow_row(self, path: Path) -> dict[str, Any]:
        with pa.memory_map(str(path), "r") as mm:
            table = pa.ipc.open_stream(mm).read_all()
        if table.num_rows != 1:
            raise ValueError(f"{path.name}: expected 1 row, got {table.num_rows}")
        return {c: table.column(c)[0].as_py() for c in table.column_names}

    def _materialize_frames(
        self, sim_id: str, frame_indices: Sequence[int]
    ) -> dict[int, CompactFrame]:
        """Decode one shard, extract requested frames, discard the full array."""
        path = self.shard_path(sim_id)
        row = self._read_arrow_row(path)
        shape = (int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"]))
        nchan = int(row["numerical_channels"])
        numerical = np.frombuffer(row["numerical"], dtype=np.float32).reshape(*shape, nchan)
        x = np.frombuffer(row["x"], dtype=np.float64).reshape(int(row["x_shape"])).copy()
        y = np.frombuffer(row["y"], dtype=np.float64).reshape(int(row["y_shape"])).copy()
        t = np.frombuffer(row["t"], dtype=np.float64).reshape(int(row["t_shape"])).copy()
        self._coords_cache[sim_id] = (x, y, t)
        self.stats["shard_decodes"] += 1

        out: dict[int, CompactFrame] = {}
        for fi in frame_indices:
            fi = int(fi)
            if fi < 0 or fi >= shape[0]:
                raise IndexError(f"{sim_id}: frame {fi} out of range [0, {shape[0]})")
            frame = numerical[fi]
            compact = CompactFrame(
                temperature=np.array(frame[:, :, CH_T], dtype=np.float32, copy=True),
                velocity_i=np.array(frame[:, :, CH_VI], dtype=np.float32, copy=True),
                velocity_j=np.array(frame[:, :, CH_VJ], dtype=np.float32, copy=True),
                velocity_k=np.array(frame[:, :, CH_VK], dtype=np.float32, copy=True),
                x=x,
                y=y,
                t=float(t[fi]),
                sim_id=sim_id,
                frame_index=fi,
            )
            out[fi] = compact
            self.stats["frames_materialized"] += 1
        # Drop full payload references before returning
        del numerical, row
        return out

    def _cache_put(self, key: tuple[str, int], frame: CompactFrame) -> None:
        self._frame_cache[key] = frame
        self._frame_cache.move_to_end(key)
        while len(self._frame_cache) > self.max_cached_frames:
            self._frame_cache.popitem(last=False)

    def get_frame(self, sim_id: str, frame_index: int) -> CompactFrame:
        key = (sim_id, int(frame_index))
        with self._lock:
            if key in self._frame_cache:
                self.stats["cache_hits"] += 1
                self._frame_cache.move_to_end(key)
                return self._frame_cache[key]
            self.stats["cache_misses"] += 1
            materialized = self._materialize_frames(sim_id, [frame_index])
            frame = materialized[int(frame_index)]
            self._cache_put(key, frame)
            return frame

    def prefetch(self, sim_id: str, frame_indices: Iterable[int]) -> None:
        """Decode a shard once and cache many frames (used for fit-stats / compact warm-up)."""
        frames = sorted({int(f) for f in frame_indices})
        with self._lock:
            missing = [f for f in frames if (sim_id, f) not in self._frame_cache]
            if not missing:
                self.stats["cache_hits"] += len(frames)
                return
            materialized = self._materialize_frames(sim_id, missing)
            for fi, frame in materialized.items():
                self._cache_put((sim_id, fi), frame)
            self.stats["cache_misses"] += len(missing)

    def clear_cache(self) -> None:
        with self._lock:
            self._frame_cache.clear()

    @property
    def cached_frame_count(self) -> int:
        return len(self._frame_cache)

    def estimate_compact_cache_bytes(self) -> int:
        """Approximate resident compact-cache bytes (4 fields × H×W float32 + coords shared)."""
        n = self.cached_frame_count
        if n == 0:
            return 0
        # 4 channels * 128 * 128 * 4 bytes
        return n * 4 * 128 * 128 * 4
