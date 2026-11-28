#!/usr/bin/env python3
"""Pre-evaluation CPU audit & freeze helper for Mission 02 (Virtual PIV).

Reads a ``piv_postproc_manifest_v1`` manifest, classifies every run's training status
into ``plateau``, ``budget_exhausted``, ``incomplete``, ``failed``, ``missing`` (or
``frozen_external``), writes ``run_status_audit.csv``, ``run_status_summary.json`` and
``STATUS_REPORT.md``, and optionally prints colon-separated valid run IDs for Slurm
evaluation array submission.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.evaluation.piv_eval import atomic_write_csv  # noqa: E402
from combustion_recon.evaluation.piv_manifest import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    audit_manifest,
    load_manifest,
    render_status_report_md,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--print-valid-train-runs",
        action="store_true",
        help="Print colon-separated run_ids whose training is valid (plateau / budget_exhausted / frozen_external)",
    )
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest, allow_test_with_r5_auth=False)
    audit = audit_manifest(manifest, smoke=args.smoke)

    args.dest.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(audit["table"], args.dest / "run_status_audit.csv")
    atomic_write_json(args.dest / "run_status_summary.json", audit["summary"])
    atomic_write_text(args.dest / "STATUS_REPORT.md", render_status_report_md(audit))

    if args.print_valid_train_runs:
        df = audit["table"]
        valid_ids = list(dict.fromkeys(df[df["training_valid"]]["run_id"].tolist()))
        print(":".join(valid_ids))
    else:
        s = audit["summary"]
        print(
            f"audit ok: runs={s['n_runs_total']} training={s['training_status_counts']} "
            f"eval={s['eval_status_counts']} strict_complete={s['strict_complete']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
