#!/usr/bin/env python3
"""Paths one O5 task actually reads, taken from the matrix, for the launcher preflight.

Operational only. It names the files the evaluator consumes for this run's pool:
a B19/A38 task never lists the C149 cache, and a C149 task never lists the
historical cache. The launcher stats these paths under a timeout; the hash and
integrity gates stay in scripts/26_opus_o5_eval.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

CHECKPOINT_NAMES = ("best", "best_seen_diagnostic", "last")


def required_paths(matrix: dict[str, Any], run_id: str) -> list[Path]:
    shared = matrix["shared_inputs"]
    matches = [row for row in matrix["runs"] if row["run_id"] == run_id]
    if len(matches) != 1:
        raise SystemExit(f"matrix does not contain exactly one {run_id}")
    entry = matches[0]
    if str(entry["pool_id"]) == "C149":
        cache_root = Path(shared["c149_cache_root"])
        hash_file = Path(shared["c149_hash_file"])
    else:
        cache_root = Path(shared["historical_cache_root"])
        hash_file = Path(shared["historical_hash_file"])
    mask_root = Path(shared["mask_root"])
    training_dir = Path(entry["training_dir"])
    paths = [
        cache_root / "manifest.json",
        hash_file,
        Path(shared["stats"]),
        mask_root / "manifest.json",
        training_dir / "run_identity.json",
    ]
    paths += [training_dir / "checkpoints" / f"{name}.pt" for name in CHECKPOINT_NAMES]
    return paths


def check_access(paths: list[Path], stat: Callable[[Path], object]) -> list[Path]:
    """Return the paths that raised; ``stat`` is injected so tests can record calls."""
    failed = []
    for path in paths:
        try:
            stat(path)
        except OSError:
            failed.append(path)
    return failed


def main() -> int:
    matrix = json.loads(Path(sys.argv[1]).read_text())
    for path in required_paths(matrix, sys.argv[2]):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
