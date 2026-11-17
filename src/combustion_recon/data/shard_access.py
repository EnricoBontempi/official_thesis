"""Lazy Arrow shard access with compact frame caching for Month 2.

Full numerical trajectories are ~2 GB float32 each. This module:
- does **not** load payloads at initialization;
- decodes at most one shard at a time when materializing frames;
- retains only requested channels/frames in a compact float32 cache;
- never keeps a durable full-trajectory dict like the Month-1 loader.

I/O contract for runners (Step 2.3+)
------------------------------------
``get_frame(sim_id, frame)`` on a cache miss decodes the **entire** Arrow
numerical blob once for that call. Repeated ``get_frame`` for many frames of
the same trajectory without warming the cache therefore costs one decode per
miss (not per unique shard overall).

**Supported efficient path:** call ``prefetch(sim_id, frame_indices)`` so all
requested frames of that trajectory are extracted in **one** decode, then use
``get_frame`` (cache hits). Size ``max_cached_frames`` to hold at least one
trajectory's needed frames (e.g. 38 for ``train_fit``) if you want subsequent
random access without re-decoding. Counters: ``stats['shard_decodes']``,
``cache_hits``, ``cache_misses``, ``frames_materialized``.

Temporal policy
---------------
Valid timelines (finite, non-decreasing, length matching ``shape_t``) are
accepted as-is. Generic extrapolation / padding of short ``t`` arrays is
**refused**. One documented packaging anomaly
(``40NH3_1.2.h5`` / ``data-00022-of-00030.arrow``) may be read in
**single-frame-by-index** mode with ``time_validated=False`` and **no**
invented timestamps when the file SHA/size match the remote inventory.
"""

from __future__ import annotations

import hashlib
import threading
import warnings
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

# Narrow, identity-bound exception (mission M1). Do not generalize.
KNOWN_T_LEN_MISMATCH: dict[str, dict[str, Any]] = {
    "40NH3_1.2.h5": {
        "shard_filename": "data-00022-of-00030.arrow",
        "expected_sha256": "3d908c0dc2913dc658fd227c5750d99a7f4c4293ae29a634cd96d02129e4dbdb",
        "expected_size_bytes": 2098221056,
        "shape_t": 2001,
        "t_len": 2000,
        "audit_t_start": 1.5,
        "audit_t_end": 2.0,
        "note": (
            "numerical frame 2000 is an exact duplicate of frame 1999; "
            "t has 2000 validated samples spanning audit [1.5, 2.0]. "
            "Index 2000 has no original timestamp — single-frame fields only."
        ),
    }
}


class TimelineError(ValueError):
    """Raised when temporal coordinates are invalid or unhandled."""


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
    time_validated: bool = True


def validate_timeline(t: np.ndarray, *, sim_id: str) -> None:
    """Refuse non-finite or decreasing timelines."""
    if t.size == 0:
        raise TimelineError(f"{sim_id}: empty t coordinate array")
    if not np.isfinite(t).all():
        raise TimelineError(f"{sim_id}: non-finite values in t")
    if t.size >= 2 and np.any(np.diff(t) < 0):
        raise TimelineError(f"{sim_id}: t is not non-decreasing")


def sha256_file(path: Path, *, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class ShardStore:
    """Per-process shard store with compact LRU cache of frames."""

    def __init__(
        self,
        *,
        raw_dir: str | Path | None = None,
        shard_map: dict[str, str] | None = None,
        max_cached_frames: int = 2048,
        verify_known_mismatch_sha: bool = True,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW
        self.shard_map = shard_map if shard_map is not None else load_shard_map()
        self.max_cached_frames = int(max_cached_frames)
        self.verify_known_mismatch_sha = bool(verify_known_mismatch_sha)
        self._frame_cache: OrderedDict[tuple[str, int], CompactFrame] = OrderedDict()
        self._coords_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._sha_cache: dict[str, str] = {}
        self._lock = threading.RLock()
        self.stats: dict[str, Any] = {
            "shard_decodes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "frames_materialized": 0,
            "time_unvalidated_frames": 0,
            "known_t_mismatch_hits": 0,
        }

    def shard_path(self, sim_id: str) -> Path:
        if sim_id not in self.shard_map:
            raise KeyError(f"Unknown sim_id={sim_id!r}")
        path = self.raw_dir / self.shard_map[sim_id]
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def file_sha256(self, sim_id: str) -> str:
        with self._lock:
            if sim_id in self._sha_cache:
                return self._sha_cache[sim_id]
        digest = sha256_file(self.shard_path(sim_id))
        with self._lock:
            self._sha_cache[sim_id] = digest
        return digest

    def _read_arrow_row(self, path: Path) -> dict[str, Any]:
        with pa.memory_map(str(path), "r") as mm:
            table = pa.ipc.open_stream(mm).read_all()
        if table.num_rows != 1:
            raise ValueError(f"{path.name}: expected 1 row, got {table.num_rows}")
        return {c: table.column(c)[0].as_py() for c in table.column_names}

    def _resolve_time(
        self,
        *,
        sim_id: str,
        path: Path,
        fi: int,
        t: np.ndarray,
        shape_t: int,
        numerical: np.ndarray,
    ) -> tuple[float, bool]:
        """Return (t_value, time_validated) for frame index ``fi``."""
        validate_timeline(t, sim_id=sim_id)
        if t.size == shape_t:
            return float(t[fi]), True

        if t.size > shape_t:
            raise TimelineError(
                f"{sim_id}: t_len={t.size} exceeds shape_t={shape_t}; refusing read"
            )

        # t shorter than numerical — only the documented packaging anomaly is allowed.
        spec = KNOWN_T_LEN_MISMATCH.get(sim_id)
        if spec is None:
            raise TimelineError(
                f"{sim_id}: t_len={t.size} < shape_t={shape_t} without a known exception; "
                "refusing generic timestamp padding"
            )
        if int(spec["shape_t"]) != shape_t or int(spec["t_len"]) != int(t.size):
            raise TimelineError(
                f"{sim_id}: t/shape mismatch does not match documented exception "
                f"(got shape_t={shape_t}, t_len={t.size})"
            )
        if path.name != spec["shard_filename"]:
            raise TimelineError(
                f"{sim_id}: shard filename {path.name!r} != documented {spec['shard_filename']!r}"
            )
        size_bytes = path.stat().st_size
        if size_bytes != int(spec["expected_size_bytes"]):
            raise TimelineError(
                f"{sim_id}: size {size_bytes} != expected {spec['expected_size_bytes']}"
            )
        if self.verify_known_mismatch_sha:
            digest = self.file_sha256(sim_id)
            if digest != spec["expected_sha256"]:
                raise TimelineError(
                    f"{sim_id}: sha256 {digest} != expected {spec['expected_sha256']}"
                )

        self.stats["known_t_mismatch_hits"] = int(self.stats.get("known_t_mismatch_hits", 0)) + 1
        if fi < t.size:
            return float(t[fi]), True

        # Index beyond validated t: allow single-frame fields only if they are
        # an exact duplicate of the last indexed frame (observed for this shard).
        last = int(t.size - 1)
        if not np.array_equal(numerical[fi], numerical[last]):
            raise TimelineError(
                f"{sim_id}: frame {fi} is beyond t_len={t.size} and is not an exact "
                f"duplicate of frame {last}; refusing unvalidated time association"
            )
        warnings.warn(
            f"{sim_id}: serving frame {fi} with time_validated=False "
            f"(no original timestamp; fields duplicate frame {last}). {spec['note']}",
            stacklevel=2,
        )
        self.stats["time_unvalidated_frames"] = int(self.stats.get("time_unvalidated_frames", 0)) + 1
        return float("nan"), False

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
        # Cache only the original t (never extrapolated).
        self._coords_cache[sim_id] = (x, y, t)
        self.stats["shard_decodes"] += 1

        out: dict[int, CompactFrame] = {}
        for fi in frame_indices:
            fi = int(fi)
            if fi < 0 or fi >= shape[0]:
                raise IndexError(f"{sim_id}: frame {fi} out of range [0, {shape[0]})")
            t_val, time_ok = self._resolve_time(
                sim_id=sim_id,
                path=path,
                fi=fi,
                t=t,
                shape_t=shape[0],
                numerical=numerical,
            )
            frame = numerical[fi]
            compact = CompactFrame(
                temperature=np.array(frame[:, :, CH_T], dtype=np.float32, copy=True),
                velocity_i=np.array(frame[:, :, CH_VI], dtype=np.float32, copy=True),
                velocity_j=np.array(frame[:, :, CH_VJ], dtype=np.float32, copy=True),
                velocity_k=np.array(frame[:, :, CH_VK], dtype=np.float32, copy=True),
                x=x,
                y=y,
                t=t_val,
                sim_id=sim_id,
                frame_index=fi,
                time_validated=time_ok,
            )
            out[fi] = compact
            self.stats["frames_materialized"] += 1
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
        """Decode a shard **once** and cache many frames (supported multi-frame path)."""
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
        return n * 4 * 128 * 128 * 4
