#!/usr/bin/env python3
"""Aggregate a finished Opus O5 directory. Does not rank runs or launch jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.evaluation.opus_o5_plan import (  # noqa: E402
    CHECKPOINTS,
    COVERAGES,
    REPLICAS,
    REQUIRED_RUN_IDS,
    ROLES,
    aggregate_frames,
)


def _coverage_label(value: float) -> str:
    return f"{float(value):.2f}"


def collect(out_root: Path) -> pd.DataFrame:
    frames = []
    for run_id in REQUIRED_RUN_IDS:
        run_dir = out_root / run_id
        marker = run_dir / "COMPLETE.json"
        if not marker.exists():
            raise SystemExit(f"missing COMPLETE.json for {run_id}")
        payload = json.loads(marker.read_text())
        if payload.get("ok") is not True:
            raise SystemExit(f"{run_id} is not complete")
        for name in CHECKPOINTS:
            for coverage in COVERAGES:
                for role in ROLES:
                    for replica in REPLICAS:
                        path = (
                            run_dir
                            / "per_frame"
                            / name
                            / f"cov{_coverage_label(coverage)}"
                            / role
                            / f"rep{replica}.csv"
                        )
                        if not path.exists():
                            raise SystemExit(f"missing {path}")
                        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    if args.dest.exists():
        raise SystemExit(f"refusing to overwrite {args.dest}")
    tables = aggregate_frames(collect(args.out_root))
    args.dest.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(args.dest / f"{name}.csv", index=False)
    note = {
        "source": "opus_o5_per_frame",
        "ranking": "none",
        "tables": list(tables),
        "coverage_note": "1% and 20% generalize the TRAIN density of 5%.",
    }
    (args.dest / "aggregation_note.json").write_text(json.dumps(note, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
