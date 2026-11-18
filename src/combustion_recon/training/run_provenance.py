"""Run-manifest provenance helpers (git SHA, tree state, source hashes)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Sequence

from combustion_recon.data.month2_protocol import sha256_file

ROOT = Path(__file__).resolve().parents[3]

# Scientific sources whose hashes must be recorded for U4/U6-class runs.
DEFAULT_SCIENTIFIC_SOURCES: tuple[str, ...] = (
    "src/combustion_recon/training/dense_cache.py",
    "src/combustion_recon/training/epoch_loop.py",
    "src/combustion_recon/training/trainer.py",
    "src/combustion_recon/training/snapshot_swap.py",
    "src/combustion_recon/training/run_provenance.py",
    "src/combustion_recon/models/unet.py",
    "src/combustion_recon/data/month2_normalizer.py",
    "src/combustion_recon/evaluation/month2_evaluator.py",
    "scripts/07_train_reconstructor.py",
    "configs/experiments/month2_dense_unet.yaml",
    "configs/models/unet_small.yaml",
)


def _git(args: Sequence[str], *, cwd: Path = ROOT) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def collect_run_provenance(
    *,
    root: Path | None = None,
    extra_sources: Sequence[str | Path] = (),
    scientific_sources: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Collect SHA, working-tree status, and hashes of scientific sources/configs."""
    base = Path(root) if root is not None else ROOT
    sources = list(scientific_sources if scientific_sources is not None else DEFAULT_SCIENTIFIC_SOURCES)
    sources.extend(str(p) for p in extra_sources)

    head = _git(["rev-parse", "HEAD"], cwd=base)
    status_porcelain = _git(["status", "--porcelain"], cwd=base) or ""
    status_lines = [ln for ln in status_porcelain.splitlines() if ln.strip()]
    untracked = [ln[3:] for ln in status_lines if ln.startswith("??")]
    dirty = [ln for ln in status_lines if not ln.startswith("??")]

    source_hashes: dict[str, str | None] = {}
    missing: list[str] = []
    for rel in sources:
        path = base / rel if not Path(rel).is_absolute() else Path(rel)
        key = str(path.relative_to(base)) if path.is_relative_to(base) else str(path)
        if not path.is_file():
            source_hashes[key] = None
            missing.append(key)
            continue
        source_hashes[key] = sha256_file(path)

    return {
        "git_commit": head,
        "working_tree": {
            "clean": len(status_lines) == 0,
            "dirty_entries": dirty,
            "untracked_entries": untracked,
            "status_porcelain": status_lines,
            "note": "Outputs generated after start should remain distinguishable from pre-existing dirty/untracked files.",
        },
        "scientific_source_sha256": source_hashes,
        "missing_scientific_sources": missing,
    }
