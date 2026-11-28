#!/usr/bin/env python3
"""Safe Tier-2 scratch field pruner enforcing the Mission 02 storage & I/O policy.

Deletes temporary full-VAL prediction scratch archives (``scratch_val_rep0_*.npz``)
ONLY AFTER verifying that:
1. All full-field metrics (isotherm 1800/2100 K, distance bins, in/out-field) have been
   computed and saved in ``per_frame/`` CSVs;
2. Compact spatial error accumulators (``spatial/*_rep0.npz``) exist;
3. Selected representative & best/worst field bundles (``fields_selected_*.npz`` +
   ``fields_manifest.json`` with valid SHA-256) are present and intact;
4. The post-processing package ``package_manifest.json`` (and ``COMPLETE.json`` unless
   ``--allow-partial``) is verified.

Runs in ``--dry-run`` mode by default; pass ``--execute`` to actually remove scratch files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.evaluation.piv_manifest import load_manifest, sha256_file  # noqa: E402

REQUIRED_FULL_FIELD_COLS = (
    "iso1800_sym_dist_mm",
    "iso2100_sym_dist_mm",
    "in_field_rmse_K",
    "dist_d0_rmse_K",
)


def verify_ready_to_prune(eval_kind_dir: Path) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not (eval_kind_dir / "COMPLETE.json").is_file():
        reasons.append("COMPLETE.json missing")
    csvs = sorted((eval_kind_dir / "per_frame").rglob("rep*.csv"))
    if not csvs:
        reasons.append("no per_frame CSVs found")
    else:
        hdr = set(pd.read_csv(csvs[0], nrows=1).columns)
        missing_cols = [c for c in REQUIRED_FULL_FIELD_COLS if c not in hdr]
        if missing_cols:
            reasons.append(f"full-field metrics missing from per_frame CSV: {missing_cols}")
    if not list((eval_kind_dir / "spatial").rglob("*_rep0.npz")):
        reasons.append("spatial accumulator .npz files missing")
    fman = eval_kind_dir / "fields_manifest.json"
    if not fman.is_file():
        reasons.append("fields_manifest.json missing")
    else:
        m = json.loads(fman.read_text(encoding="utf-8"))
        npz_p = eval_kind_dir / m.get("file", "")
        if not npz_p.is_file() or sha256_file(npz_p) != m.get("file_sha256"):
            reasons.append("fields_selected archive missing or sha256 mismatch")
    return (len(reasons) == 0), reasons


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--package-dest", type=Path, required=True)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--execute", action="store_true", help="Actually delete verified scratch files (default is dry-run)")
    args = ap.parse_args(argv)

    if not (args.package_dest / "package_manifest.json").is_file():
        raise SystemExit(f"REFUSE: {args.package_dest / 'package_manifest.json'} not found")
    if not args.allow_partial and not (args.package_dest / "COMPLETE.json").is_file():
        raise SystemExit("REFUSE: COMPLETE.json not found (pass --allow-partial only for diagnostic cleanup)")

    manifest = load_manifest(args.manifest, allow_test_with_r5_auth=False)
    total_bytes = 0
    total_files = 0
    for run in manifest["runs"]:
        edir = run.get("eval_dir")
        if not edir or not Path(edir).is_dir():
            continue
        for kind in manifest.get("expected_checkpoints", ["best", "last"]):
            kdir = Path(edir) / kind
            if not kdir.is_dir():
                continue
            scratches = sorted(kdir.rglob("scratch_val_rep0_*.npz"))
            if not scratches:
                continue
            ok, reasons = verify_ready_to_prune(kdir)
            if not ok:
                print(f"SKIP {run['run_id']}/{kind}: not verified ({'; '.join(reasons)})")
                continue
            for sp in scratches:
                sz = sp.stat().st_size
                total_bytes += sz
                total_files += 1
                if args.execute:
                    sp.unlink()
                    print(f"DELETED {sp} ({sz / 1e6:.2f} MB)")
                else:
                    print(f"DRY-RUN would delete {sp} ({sz / 1e6:.2f} MB)")

    mode = "DELETED" if args.execute else "DRY-RUN"
    print(f"{mode}: {total_files} scratch files ({total_bytes / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
