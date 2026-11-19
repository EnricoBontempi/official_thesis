#!/usr/bin/env python3
"""Second-pass median9 / mean9 diagnostic on saved Astra fields.

This does not run a forward and does not alter the primary free/oracle/AO9
tables. A CSV that stores only ``anchor_mean_resid_K`` is refused: the median
of nine residuals is not a function of their mean.

``from-npz`` is for the saved frame-1000 fields only. It is not the seen/VAL
population. Before writing median9 it checks regenerated free/oracle/AO9
against the primary per-frame CSV. A miss does not update that reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from combustion_recon.evaluation.astra_level import (
    AO_TAU,
    saved_columns_have_anchor_residuals,
    t9_site_indices,
    treat_frame_with_level_diagnostics,
)

PRIMARY_ATOL_K = 0.05
PRIMARY_TREATMENTS = ("free", "oracle", "ao9")
COMPARE_KEYS = (
    "delta_applied_K",
    "delta_oracle_K",
    "delta_ao9_K",
    "all_rmse_K",
    "all_mae_K",
    "all_bias_K",
    "all_centered_rmse_K",
    "off_rmse_K",
    "off_bias_K",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_csv(path: Path) -> dict:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    has_resid = saved_columns_have_anchor_residuals(header)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "n_columns": len(header),
        "has_anchor_mean_resid_K": "anchor_mean_resid_K" in header,
        "has_nine_residuals": has_resid,
        "median_recoverable": has_resid,
        "reason": (
            None
            if has_resid
            else "anchor_mean_resid_K does not determine the median of the nine residuals"
        ),
    }


def _rows_from_fields(pred: np.ndarray, true: np.ndarray, sites: np.ndarray, identity: dict) -> tuple[list[dict], list[dict]]:
    rows, detail = treat_frame_with_level_diagnostics(pred, true, sites, tau=AO_TAU)
    metric_rows = []
    for row in rows:
        metric_rows.append({**identity, **row})
    residual_rows = []
    for i in range(detail["resid_K"].size):
        residual_rows.append(
            {
                **identity,
                "anchor_id": i,
                "flat_index": int(detail["anchor_flat_index"][i]),
                "row": int(detail["anchor_row"][i]),
                "col": int(detail["anchor_col"][i]),
                "resid_K": float(detail["resid_K"][i]),
                "ao9_contribution_K": float(detail["ao9_contribution_K"][i]),
                "delta_mean9_K": float(detail["delta_mean9_K"]),
                "delta_median9_K": float(detail["delta_median9_K"]),
                "delta_ao9_K": float(detail["delta_ao9_K"]),
                "delta_oracle_K": float(detail["delta_oracle_K"]),
            }
        )
    return metric_rows, residual_rows


def _primary_mismatches(regenerated: pd.DataFrame, reference: pd.DataFrame) -> list[dict]:
    misses = []
    keys = ["run_id", "sim_id", "frame_index", "replica_id", "treatment"]
    ref = reference[reference["treatment"].isin(PRIMARY_TREATMENTS)]
    for _, row in regenerated[regenerated["treatment"].isin(PRIMARY_TREATMENTS)].iterrows():
        hit = ref
        for key in keys:
            hit = hit[hit[key] == row[key]]
        if len(hit) != 1:
            misses.append({"identity": {k: row[k] for k in keys}, "error": f"reference rows {len(hit)}"})
            continue
        src = hit.iloc[0]
        for col in COMPARE_KEYS:
            delta = float(row[col]) - float(src[col])
            if abs(delta) > PRIMARY_ATOL_K:
                misses.append(
                    {
                        "identity": {k: row[k] for k in keys},
                        "column": col,
                        "regenerated": float(row[col]),
                        "reference": float(src[col]),
                        "delta_K": delta,
                    }
                )
    return misses


def from_npz(fields_root: Path, reference_root: Path, out_dir: Path) -> int:
    sites = t9_site_indices()
    metric_rows: list[dict] = []
    residual_rows: list[dict] = []
    provenance = []
    for run_dir in sorted(p for p in fields_root.iterdir() if p.is_dir()):
        npz_path = run_dir / "val_frame1000_rep0_fields.npz"
        if not npz_path.is_file():
            continue
        payload = np.load(npz_path)
        if not np.array_equal(payload["t9_sites"], sites):
            raise SystemExit(f"REFUSE: t9_sites mismatch in {npz_path}")
        sims = sorted({key.split("__")[0] for key in payload.files if key.endswith("__pred_K")})
        for sim in sims:
            pred = np.asarray(payload[f"{sim}__pred_K"], np.float64)
            true = np.asarray(payload[f"{sim}__true_K"], np.float64)
            identity = {
                "run_id": run_dir.name,
                "role": "val_eval",
                "replica_id": 0,
                "sim_id": sim,
                "frame_index": 1000,
                "scope": "example_frame1000_rep0_only",
            }
            metrics, residuals = _rows_from_fields(pred, true, sites, identity)
            metric_rows.extend(metrics)
            residual_rows.extend(residuals)
        provenance.append({"run_id": run_dir.name, "npz": str(npz_path), "sha256": _sha256(npz_path)})
    if not metric_rows:
        raise SystemExit(f"REFUSE: no val_frame1000_rep0_fields.npz under {fields_root}")
    regenerated = pd.DataFrame(metric_rows)
    ref_frames = []
    for item in provenance:
        csv_path = reference_root / item["run_id"] / "per_frame_val_eval.csv.gz"
        if not csv_path.is_file():
            raise SystemExit(f"REFUSE: missing primary CSV {csv_path}")
        frame = pd.read_csv(csv_path)
        frame = frame[(frame["frame_index"] == 1000) & (frame["replica_id"] == 0)]
        ref_frames.append(frame)
        item["reference_csv"] = str(csv_path)
        item["reference_sha256"] = _sha256(csv_path)
    reference = pd.concat(ref_frames, ignore_index=True)
    misses = _primary_mismatches(regenerated, reference)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "example_frame1000_rep0_only",
        "not_the_seen_or_val_population": True,
        "primary_atol_K": PRIMARY_ATOL_K,
        "primary_match": not misses,
        "n_mismatches": len(misses),
        "mismatches": misses[:30],
        "provenance": provenance,
        "note": (
            "AO9 uses sum(resid)/(9+tau). mean9 is sum(resid)/9. "
            "median9 is the fifth ordered residual. "
            "A median-versus-AO9 difference is not only a robustness difference."
        ),
    }
    (out_dir / "primary_reproduction.json").write_text(json.dumps(report, indent=2))
    if misses:
        print(f"REFUSE: primary free/oracle/AO9 mismatch, {len(misses)} cells > {PRIMARY_ATOL_K} K")
        return 3
    regenerated.to_csv(out_dir / "example_frame1000_rep0_metrics.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(out_dir / "example_frame1000_rep0_residuals.csv", index=False)
    _write_paired(regenerated, out_dir / "example_frame1000_rep0_paired.csv")
    print(f"wrote example tables to {out_dir}")
    return 0


def _write_paired(frame: pd.DataFrame, path: Path) -> None:
    """Paired treatment differences on the example frames. No winner is chosen."""
    index = ["run_id", "sim_id", "frame_index", "replica_id"]
    metrics = ["all_rmse_K", "all_bias_K", "all_centered_rmse_K", "off_rmse_K", "delta_applied_K"]
    pairs = (
        ("median9", "ao9"),
        ("median9", "mean9"),
        ("mean9", "ao9"),
        ("median9", "oracle"),
        ("mean9", "oracle"),
        ("ao9", "oracle"),
        ("median9", "free"),
        ("ao9", "free"),
    )
    rows = []
    for keys, group in frame.groupby(index, sort=False):
        ident = dict(zip(index, keys if isinstance(keys, tuple) else (keys,)))
        by = {str(rec.treatment): rec for rec in group.itertuples(index=False)}
        for left, right in pairs:
            for metric in metrics:
                delta = float(getattr(by[left], metric)) - float(getattr(by[right], metric))
                rows.append({**ident, "contrast": f"{left}_minus_{right}", "metric": metric, "delta": delta})
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    inspect = sub.add_parser("inspect-csv")
    inspect.add_argument("csv", type=Path)
    fields = sub.add_parser("from-npz")
    fields.add_argument("--fields-root", type=Path, required=True)
    fields.add_argument("--reference-root", type=Path, required=True)
    fields.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "inspect-csv":
        info = inspect_csv(args.csv)
        json.dump(info, sys.stdout, indent=2)
        print()
        return 0 if info["median_recoverable"] else 2
    return from_npz(args.fields_root, args.reference_root, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
