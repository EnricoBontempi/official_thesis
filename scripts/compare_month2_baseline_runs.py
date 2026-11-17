#!/usr/bin/env python3
"""Compare a consolidation replica against the historical 822ed52 tables.

Tolerances must match the preflight comparison_rule (default atol=1e-6 K, rtol=1e-8).
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST_DIR = ROOT / "results" / "tables"
DEFAULT_ATOL = 1.0e-6
DEFAULT_RTOL = 1.0e-8

PAIR_KEYS = [
    "method",
    "role",
    "p_conditioning",
    "K",
    "site_fraction",
    "variant",
    "mask_replica_id",
    "sim_id",
    "frame_index",
    "R",
    "ridge_lambda",
    "w_T",
    "observation_hash",
]

METRIC_CANDIDATES = [
    "rmse_K",
    "mae_K",
    "abs_bias_K",
    "centered_rmse_K",
    "nrmse_train_sigma",
    "mean_signed_error_K",
    "anchor_rmse_K",
    "off_anchor_rmse_K",
]


def _read_table(path: Path) -> pd.DataFrame:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f)
    return pd.read_csv(path)


def _norm_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in PAIR_KEYS:
        if c not in out.columns:
            out[c] = np.nan
    # normalize bool-like
    if out["p_conditioning"].dtype == object:
        out["p_conditioning"] = out["p_conditioning"].map(
            lambda x: True if str(x).lower() == "true" else False if str(x).lower() == "false" else x
        )
    return out


def _key_series(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in PAIR_KEYS if c in df.columns]

    def row_key(r):
        parts = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                parts.append(f"{c}=NA")
            elif isinstance(v, float):
                parts.append(f"{c}={v:.12g}")
            else:
                parts.append(f"{c}={v}")
        return "|".join(parts)

    return df.apply(row_key, axis=1)


def compare_frames(
    hist: pd.DataFrame,
    new: pd.DataFrame,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    hist = _norm_key_frame(hist)
    new = _norm_key_frame(new)
    hist = hist.assign(_key=_key_series(hist)).drop_duplicates("_key", keep="first")
    new = new.assign(_key=_key_series(new)).drop_duplicates("_key", keep="first")
    h_keys = set(hist["_key"])
    n_keys = set(new["_key"])
    missing_in_new = sorted(h_keys - n_keys)
    extra_in_new = sorted(n_keys - h_keys)
    common = sorted(h_keys & n_keys)

    metrics = [m for m in METRIC_CANDIDATES if m in hist.columns and m in new.columns]
    h_idx = hist.set_index("_key")
    n_idx = new.set_index("_key")
    max_abs = {}
    n_exceed = {}
    worst = []
    for m in metrics:
        diffs = []
        for k in common:
            a = h_idx.at[k, m]
            b = n_idx.at[k, m]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) or pd.isna(b):
                diffs.append((k, m, float("nan"), "nan_mismatch"))
                continue
            a = float(a)
            b = float(b)
            absd = abs(a - b)
            tol = atol + rtol * abs(a)
            if absd > tol:
                diffs.append((k, m, absd, "exceed"))
            else:
                diffs.append((k, m, absd, "ok"))
        abs_vals = [d[2] for d in diffs if np.isfinite(d[2])]
        max_abs[m] = float(max(abs_vals)) if abs_vals else 0.0
        n_exceed[m] = int(sum(1 for d in diffs if d[3] == "exceed" or d[3] == "nan_mismatch"))
        for d in diffs:
            if d[3] != "ok":
                worst.append({"key": d[0], "metric": d[1], "abs_diff": d[2], "status": d[3]})

    worst = sorted(
        worst,
        key=lambda x: (-1 if not np.isfinite(x["abs_diff"]) else -x["abs_diff"]),
    )[:50]

    classification = "identical_or_within_tolerance"
    if missing_in_new or extra_in_new:
        classification = "difference_unexplained_keys"
    elif any(v > 0 for v in n_exceed.values()):
        classification = "difference_unexplained_metrics"

    return {
        "n_hist": int(len(hist)),
        "n_new": int(len(new)),
        "n_common": int(len(common)),
        "n_missing_in_new": len(missing_in_new),
        "n_extra_in_new": len(extra_in_new),
        "missing_in_new_sample": missing_in_new[:20],
        "extra_in_new_sample": extra_in_new[:20],
        "metrics_compared": metrics,
        "max_abs_diff": max_abs,
        "n_exceed": n_exceed,
        "worst_rows": worst,
        "classification": classification,
        "atol": atol,
        "rtol": rtol,
    }


def compare_search(hist: pd.DataFrame, new: pd.DataFrame) -> dict[str, Any]:
    cols = [c for c in ["task", "p_conditioning", "K", "ridge_lambda", "w_T", "R"] if c in hist.columns]
    h = hist.copy()
    n = new.copy()
    for c in cols:
        if c not in n.columns:
            return {"ok": False, "reason": f"missing column {c} in new search"}
    # sort and compare scores
    merge = h.merge(
        n,
        on=cols,
        how="outer",
        suffixes=("_hist", "_new"),
        indicator=True,
    )
    score_h = "score_mean_condition_rmse_K_hist"
    score_n = "score_mean_condition_rmse_K_new"
    if score_h not in merge.columns:
        # pandas may keep name if only one side - handle
        if "score_mean_condition_rmse_K" in h.columns and "score_mean_condition_rmse_K" in n.columns:
            merge = h.merge(n, on=cols, how="outer", suffixes=("_hist", "_new"), indicator=True)
            score_h = "score_mean_condition_rmse_K_hist"
            score_n = "score_mean_condition_rmse_K_new"
    diffs = []
    for _, r in merge.iterrows():
        if r["_merge"] != "both":
            diffs.append({"row": {c: r[c] for c in cols}, "status": str(r["_merge"])})
            continue
        a, b = float(r[score_h]), float(r[score_n])
        absd = abs(a - b)
        diffs.append({"row": {c: r[c] for c in cols}, "abs_diff": absd, "hist": a, "new": b})
    max_diff = max((d.get("abs_diff", float("nan")) for d in diffs if "abs_diff" in d), default=float("nan"))
    return {
        "n_rows_hist": len(h),
        "n_rows_new": len(n),
        "max_abs_score_diff": float(max_diff) if np.isfinite(max_diff) else None,
        "details_sample": diffs[:30],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", type=Path, required=True)
    ap.add_argument("--hist-dir", type=Path, default=HIST_DIR)
    ap.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    ap.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    args = ap.parse_args()

    hist_frame = args.hist_dir / "month2_baseline_per_frame.csv"
    new_frame = args.new_dir / "month2_baseline_per_frame.csv.gz"
    if not new_frame.exists():
        new_frame = args.new_dir / "month2_baseline_per_frame.csv"

    hist_pf = _read_table(hist_frame)
    new_pf = _read_table(new_frame)
    frame_cmp = compare_frames(hist_pf, new_pf, atol=args.atol, rtol=args.rtol)

    hist_sum = _read_table(args.hist_dir / "month2_baseline_summary.csv")
    new_sum = _read_table(args.new_dir / "month2_baseline_summary.csv")
    # summary pairing on coarse keys
    sum_keys = [
        c
        for c in ["method", "role", "p_conditioning", "K", "variant", "R", "ridge_lambda", "w_T"]
        if c in hist_sum.columns and c in new_sum.columns
    ]
    sum_merge = hist_sum.merge(new_sum, on=sum_keys, how="outer", suffixes=("_hist", "_new"), indicator=True)
    sum_metrics = []
    for m in ["rmse_K", "abs_bias_K", "centered_rmse_K"]:
        if f"{m}_hist" in sum_merge.columns and f"{m}_new" in sum_merge.columns:
            both = sum_merge[sum_merge["_merge"] == "both"]
            d = (both[f"{m}_hist"] - both[f"{m}_new"]).abs()
            sum_metrics.append(
                {
                    "metric": m,
                    "max_abs": float(d.max()) if len(d) else None,
                    "n_exceed": int((d > (args.atol + args.rtol * both[f"{m}_hist"].abs())).sum())
                    if len(d)
                    else 0,
                }
            )

    hist_search = _read_table(args.hist_dir / "month2_baseline_search.csv")
    new_search = _read_table(args.new_dir / "month2_baseline_search.csv")
    search_cmp = compare_search(hist_search, new_search)

    # selected winners from manifests if present
    hist_manifest = ROOT / "results" / "meta" / "month2_baseline_run_manifest.json"
    new_manifest = args.new_dir / "run_manifest.json"
    selected = {}
    if hist_manifest.exists() and new_manifest.exists():
        hs = json.loads(hist_manifest.read_text())["selected"]
        ns = json.loads(new_manifest.read_text())["selected"]
        selected = {"historical": hs, "new": ns, "equal": hs == ns}

    out = {
        "historical_dir": str(args.hist_dir),
        "new_dir": str(args.new_dir),
        "per_frame": frame_cmp,
        "summary_metrics": sum_metrics,
        "search": search_cmp,
        "selected": selected,
        "overall_classification": frame_cmp["classification"],
    }
    out_path = args.new_dir / "comparison.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    # compact markdown table
    lines = [
        "# Baseline consolidation comparison",
        "",
        f"- overall: **{out['overall_classification']}**",
        f"- common per-frame keys: {frame_cmp['n_common']}",
        f"- missing in new: {frame_cmp['n_missing_in_new']}",
        f"- extra in new: {frame_cmp['n_extra_in_new']}",
        "",
        "## Max abs metric diffs (per-frame)",
        "",
        "| metric | max_abs | n_exceed |",
        "|---|---:|---:|",
    ]
    for m, v in frame_cmp["max_abs_diff"].items():
        lines.append(f"| {m} | {v:.6g} | {frame_cmp['n_exceed'][m]} |")
    lines += ["", f"selected_equal: {selected.get('equal')}", ""]
    (args.new_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"wrote": str(out_path), "classification": out["overall_classification"]}, indent=2))


if __name__ == "__main__":
    main()
