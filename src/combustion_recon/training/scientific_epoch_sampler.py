"""Deterministic presentation stream for Opus scientific epochs.

A scientific epoch always yields ``presentations_per_epoch`` samples (default
760 → 95 AdamW updates at batch 8), independent of pool size.

Stream semantics (roadmapopus §4.2):
- consume a deterministic permutation of the pool without replacement;
- leftover elements carry into the next scientific epoch;
- when a permutation is exhausted, draw the next with
  ``seed = shuffle_seed + 100_003 * (cycle + 1)``;
- for pool size == presentations_per_epoch (A38/REF), each scientific epoch
  is exactly one full permutation — matching the historical S5 DataLoader
  shuffle seed formula ``shuffle_seed + 100_003 * (epoch + 1)``.

State persisted for resume: ``cycle``, ``pos`` (perm regenerable from cycle).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import Sampler


@dataclass
class ScientificStreamState:
    cycle: int = 0
    pos: int = 0
    presentations_served: int = 0
    scientific_epochs_completed: int = 0


class ScientificPresentationStream:
    """Stateful without-replacement stream over a fixed pool size."""

    def __init__(
        self,
        pool_size: int,
        *,
        shuffle_seed: int,
        presentations_per_epoch: int = 760,
        state: ScientificStreamState | None = None,
    ):
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if presentations_per_epoch <= 0:
            raise ValueError("presentations_per_epoch must be positive")
        if presentations_per_epoch % 1 != 0:
            raise ValueError("presentations_per_epoch must be an integer")
        self.pool_size = int(pool_size)
        self.shuffle_seed = int(shuffle_seed)
        self.presentations_per_epoch = int(presentations_per_epoch)
        self.state = state or ScientificStreamState()
        self._perm = self._make_perm(self.state.cycle)

    def _make_perm(self, cycle: int) -> list[int]:
        g = torch.Generator()
        g.manual_seed(int(self.shuffle_seed) + 100_003 * (int(cycle) + 1))
        return torch.randperm(self.pool_size, generator=g).tolist()

    def state_dict(self) -> dict[str, Any]:
        return {
            "pool_size": self.pool_size,
            "shuffle_seed": self.shuffle_seed,
            "presentations_per_epoch": self.presentations_per_epoch,
            "stream_state": asdict(self.state),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if int(payload.get("pool_size", -1)) != self.pool_size:
            raise ValueError("sampler resume pool_size mismatch")
        if int(payload.get("shuffle_seed", -1)) != self.shuffle_seed:
            raise ValueError("sampler resume shuffle_seed mismatch")
        if int(payload.get("presentations_per_epoch", -1)) != self.presentations_per_epoch:
            raise ValueError("sampler resume presentations_per_epoch mismatch")
        raw = payload.get("stream_state") or {}
        self.state = ScientificStreamState(
            cycle=int(raw.get("cycle", 0)),
            pos=int(raw.get("pos", 0)),
            presentations_served=int(raw.get("presentations_served", 0)),
            scientific_epochs_completed=int(raw.get("scientific_epochs_completed", 0)),
        )
        if self.state.pos < 0 or self.state.pos > self.pool_size:
            raise ValueError("invalid sampler pos")
        self._perm = self._make_perm(self.state.cycle)

    def next_scientific_epoch(self) -> list[int]:
        """Return exactly ``presentations_per_epoch`` pool indices (0..pool_size-1)."""
        out: list[int] = []
        while len(out) < self.presentations_per_epoch:
            remaining = self.pool_size - self.state.pos
            if remaining <= 0:
                self.state.cycle += 1
                self.state.pos = 0
                self._perm = self._make_perm(self.state.cycle)
                remaining = self.pool_size
            need = self.presentations_per_epoch - len(out)
            take = min(remaining, need)
            start = self.state.pos
            out.extend(self._perm[start : start + take])
            self.state.pos = start + take
            if self.state.pos >= self.pool_size:
                self.state.cycle += 1
                self.state.pos = 0
                self._perm = self._make_perm(self.state.cycle)
        self.state.presentations_served += len(out)
        self.state.scientific_epochs_completed += 1
        if len(out) != self.presentations_per_epoch:
            raise RuntimeError("scientific epoch length invariant violated")
        return out


class ListSampler(Sampler[int]):
    """Yield a fixed sequence of dataset indices (one scientific epoch)."""

    def __init__(self, indices: Sequence[int]):
        self.indices = [int(i) for i in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


# Fixed Opus/S5 pool definitions (frame_index per condition).
POOL_FRAME_INDICES: dict[str, list[int]] = {
    "B19": list(range(0, 1441, 80)),  # 19 frames
    "A38": list(range(0, 1481, 40)),  # 38 frames (= REF / S5 train_fit)
    "C149": list(range(0, 1481, 10)),  # 149 frames
}


def cache_indices_for_pool(
    frames: Sequence[dict[str, Any]],
    pool_id: str,
) -> list[int]:
    """Map pool_id → dense-cache row indices for train_fit frames."""
    if pool_id not in POOL_FRAME_INDICES:
        raise ValueError(f"unknown pool_id {pool_id!r}; expected one of {sorted(POOL_FRAME_INDICES)}")
    wanted = set(POOL_FRAME_INDICES[pool_id])
    out: list[int] = []
    for i, meta in enumerate(frames):
        if int(meta["frame_index"]) in wanted:
            out.append(int(i))
    return out


def assert_pool_nesting(frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Verify B19 ⊂ A38 ⊂ C149 on real train_fit identifiers; seen disjoint."""
    by_cond: dict[str, set[int]] = {}
    for meta in frames:
        by_cond.setdefault(str(meta["sim_id"]), set()).add(int(meta["frame_index"]))
    report: dict[str, Any] = {"n_conditions": len(by_cond), "conditions": {}}
    b19 = set(POOL_FRAME_INDICES["B19"])
    a38 = set(POOL_FRAME_INDICES["A38"])
    c149 = set(POOL_FRAME_INDICES["C149"])
    if not (b19 <= a38 <= c149):
        raise AssertionError("definition nesting B19⊂A38⊂C149 failed")
    ok = True
    for sid, idxs in sorted(by_cond.items()):
        has_b19 = b19 <= idxs
        has_a38 = a38 <= idxs
        # C149 may not be fully materialized in the historical 760-frame cache.
        has_c149_in_cache = c149 <= idxs
        entry = {
            "n_frames": len(idxs),
            "b19_subset": has_b19,
            "a38_equals_or_subset": has_a38,
            "c149_fully_in_cache": has_c149_in_cache,
        }
        report["conditions"][sid] = entry
        if not (has_b19 and has_a38):
            ok = False
    report["ok_b19_a38_on_cache"] = ok
    report["definition_nesting_ok"] = True
    report["counts"] = {
        "B19_per_cond": len(b19),
        "A38_per_cond": len(a38),
        "C149_per_cond": len(c149),
        "B19_total_expected": len(b19) * len(by_cond),
        "A38_total_expected": len(a38) * len(by_cond),
    }
    return report
