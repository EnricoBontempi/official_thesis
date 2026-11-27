#!/usr/bin/env python3
"""Graphics for the 30 p_on specialists. CPU only. No training and no inference.

Reads the frozen evaluation the specialist campaign already ran: freeze.json, the best-checkpoint
per-frame CSVs, fields_val_frame1000_best.npz, the numeric package and the Block II baseline CSV.
Publishes a package only when every expected run is present and the arrays agree with the metrics.
A missing or inconsistent input writes GRAPHICS_BLOCK_REPORT.md and no COMPLETE.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.evaluation.specialist_graphics import (  # noqa: E402
    BASELINE_EVAL_ROOT_DEFAULT,
    BASELINE_PER_FRAME_DEFAULT,
    PRODUCTION,
    GraphicsBlocked,
    run_graphics,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--fields-index", type=Path, required=True)
    ap.add_argument("--baseline-per-frame", type=Path, default=Path(BASELINE_PER_FRAME_DEFAULT))
    ap.add_argument("--baseline-eval-root", type=Path, default=Path(BASELINE_EVAL_ROOT_DEFAULT))
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--git-sha", default="")
    a = ap.parse_args(argv)
    # The production contract is fixed: 30 runs, frame 1000, 6174 rows. A smaller run is a refusal.
    try:
        run_graphics(
            freeze_path=a.freeze,
            eval_root=a.eval_root,
            package_dir=a.package,
            fields_index_path=a.fields_index,
            baseline_per_frame=a.baseline_per_frame,
            baseline_eval_root=a.baseline_eval_root,
            dest=a.dest,
            spec=PRODUCTION,
            git_sha=a.git_sha,
        )
    except GraphicsBlocked as exc:
        print("GRAPHICS_BLOCKED\n" + str(exc), file=sys.stderr)
        return 3
    print(f"GRAPHICS_OK {a.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
