"""Slurm compute accounting and run-ledger parser for Mission 02 (Virtual PIV).

Reuses the exact accounting method of Mission 00 (``scripts/73_nonideal_compute_accounting.py``
and ``roadmap_tecnica_piv.md`` §8.6):
* Reads ``sacct -D -P`` pipe-separated exports (``-D`` preserves every requeued/preempted
  attempt as its own record so nothing is double-counted or dropped).
* Computes ``gpu_s = ElapsedRaw * n_gpu`` (from ``AllocTRES``) and ``cpu_s = ElapsedRaw * AllocCPUS``.
* Joins optional per-run JSONL ledgers (attempts, preemptions/signals, extensions) and
  per-run ``setting_costs.csv`` wall times (explicitly labelled as end-to-end wall time
  including observation building, I/O, forward, and metrics — NOT inference latency).
* Handles absent Slurm logs gracefully so post-processing can still run on partial data.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _parse_n_gpu(tres: str) -> int:
    m = re.search(r"gres/gpu(?:[:=][^,=]+)?=(\d+)", tres or "")
    if m:
        return int(m.group(1))
    m2 = re.search(r"gres/gpu=(\d+)", tres or "")
    return int(m2.group(1)) if m2 else 0


def _parse_n_cpu(tres: str, alloc_cpus_col: str = "") -> int:
    m = re.search(r"cpu=(\d+)", tres or "")
    if m:
        return int(m.group(1))
    try:
        return int(alloc_cpus_col)
    except (TypeError, ValueError):
        return 0


def _infer_phase(job_name: str, phase_map: Mapping[str, str] | None = None) -> str:
    if phase_map and job_name in phase_map:
        return str(phase_map[job_name])
    jn = (job_name or "").lower()
    if "prof" in jn or "smoke" in jn:
        return "F2.7 / smoke (GPU)"
    if "train" in jn:
        return "Phase 4 training (GPU)"
    if "ext" in jn:
        return "Phase 4 continuation 1000->1200 (GPU)"
    if "eval" in jn:
        return "Evaluation (GPU)"
    if "freeze" in jn or "post" in jn or "agg" in jn or "atlas" in jn:
        return "Post-processing (CPU)"
    return "other"


def parse_sacct_psv(
    sacct_path: str | Path,
    *,
    job_run_map: Mapping[str, str] | None = None,
    array_run_map: Mapping[str, list[str]] | None = None,
    phase_map: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a ``sacct -D -P`` pipe-separated file into job-attempt and phase tables."""
    p = Path(sacct_path)
    if not p.is_file():
        return pd.DataFrame(), pd.DataFrame()

    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="|"))

    jobs = [r for r in rows if "." not in str(r.get("JobID", ""))]
    steps = {str(r.get("JobID", "")): r for r in rows if str(r.get("JobID", "")).endswith(".batch")}

    out_jobs: list[dict[str, Any]] = []
    for r in jobs:
        base = str(r.get("JobID", ""))
        arr, _, task = base.partition("_")
        st = steps.get(base + ".batch", {})
        el = int(r.get("ElapsedRaw") or 0)
        tres = str(r.get("AllocTRES") or "")
        ng = _parse_n_gpu(tres)
        ncpu = _parse_n_cpu(tres, str(r.get("AllocCPUS") or ""))

        run_id = ""
        if job_run_map and base in job_run_map:
            run_id = str(job_run_map[base])
        elif array_run_map and arr in array_run_map and task.isdigit():
            lst = array_run_map[arr]
            if int(task) < len(lst):
                run_id = str(lst[int(task)])
        elif array_run_map and str(r.get("JobName", "")) in array_run_map and task.isdigit():
            lst = array_run_map[str(r.get("JobName", ""))]
            if int(task) < len(lst):
                run_id = str(lst[int(task)])

        out_jobs.append(
            {
                "job_id": base,
                "array_id": arr,
                "task": task,
                "job_name": str(r.get("JobName", "")),
                "phase": _infer_phase(str(r.get("JobName", "")), phase_map=phase_map),
                "run_id": run_id,
                "partition": str(r.get("Partition", "")),
                "qos": str(r.get("QOS", "")),
                "account": str(r.get("Account", "")),
                "state": str(r.get("State", "")),
                "exit_code": str(r.get("ExitCode", "")),
                "submit": str(r.get("Submit", "")),
                "start": str(r.get("Start", "")),
                "end": str(r.get("End", "")),
                "elapsed_s": el,
                "n_gpu": ng,
                "alloc_cpus": ncpu,
                "gpu_s": el * ng,
                "gpu_h": (el * ng) / 3600.0,
                "cpu_s": el * ncpu,
                "cpu_h": (el * ncpu) / 3600.0,
                "node": str(r.get("NodeList", "")),
                "timelimit": str(r.get("Timelimit", "")),
                "batch_max_rss": str(st.get("MaxRSS", "")),
                "batch_total_cpu": str(st.get("TotalCPU", "")),
            }
        )

    jobs_df = pd.DataFrame(out_jobs)
    if jobs_df.empty:
        return jobs_df, pd.DataFrame()
    jobs_df = jobs_df.sort_values(["start", "job_id"]).reset_index(drop=True)

    ph = defaultdict(
        lambda: {
            "records": 0,
            "completed": 0,
            "preempted": 0,
            "failed_or_cancelled": 0,
            "other_states": 0,
            "gpu_s": 0,
            "cpu_s": 0,
            "elapsed_s": 0,
        }
    )
    for _, j in jobs_df.iterrows():
        d = ph[str(j["phase"])]
        d["records"] += 1
        st = str(j["state"]).split()[0]
        if st == "COMPLETED":
            d["completed"] += 1
        elif st == "PREEMPTED":
            d["preempted"] += 1
        elif st in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"):
            d["failed_or_cancelled"] += 1
        else:
            d["other_states"] += 1
        d["gpu_s"] += int(j["gpu_s"])
        d["cpu_s"] += int(j["cpu_s"])
        d["elapsed_s"] += int(j["elapsed_s"])

    phase_rows = []
    for phase_name, d in sorted(ph.items()):
        phase_rows.append(
            {
                "phase": phase_name,
                **d,
                "gpu_h": d["gpu_s"] / 3600.0,
                "cpu_h": d["cpu_s"] / 3600.0,
            }
        )
    phases_df = pd.DataFrame(phase_rows)
    return jobs_df, phases_df


def collect_compute_accounting(
    manifest: Mapping[str, Any],
    audit_table: pd.DataFrame,
) -> dict[str, Any]:
    """Build job-, phase-, run-, and eval-config-level accounting tables."""
    acc_cfg = manifest.get("slurm_accounting") or {}
    sacct_path = acc_cfg.get("sacct_psv")
    jobs_df = pd.DataFrame()
    phases_df = pd.DataFrame()
    if sacct_path and Path(sacct_path).is_file():
        jobs_df, phases_df = parse_sacct_psv(
            sacct_path,
            job_run_map=acc_cfg.get("job_run_map"),
            array_run_map=acc_cfg.get("array_run_map"),
            phase_map=acc_cfg.get("phase_map"),
        )

    # Collect per-run eval wall times from setting_costs.csv
    eval_cost_parts: list[pd.DataFrame] = []
    seen_dirs: set[str] = set()
    for _, row in audit_table.iterrows():
        sdir = str(row.get("eval_source_dir") or "")
        if not sdir or sdir in seen_dirs:
            continue
        seen_dirs.add(sdir)
        cp = Path(sdir) / "setting_costs.csv"
        if cp.is_file():
            cdf = pd.read_csv(cp)
            if not cdf.empty:
                cdf["family"] = str(row["family"])
                cdf["train_cell"] = str(row["train_cell"])
                eval_cost_parts.append(cdf)

    eval_costs_df = pd.concat(eval_cost_parts, ignore_index=True) if eval_cost_parts else pd.DataFrame()

    # Build per-run accounting table combining audit, training summary elapsed_s, sacct, and eval costs
    run_rows: list[dict[str, Any]] = []
    unique_runs = audit_table.drop_duplicates(subset=["run_id"]) if not audit_table.empty else pd.DataFrame()
    for _, r in unique_runs.iterrows():
        rid = str(r["run_id"])
        t_elapsed_s = float("nan")
        tdir = Path(str(r["training_dir"])) if r.get("training_dir") else None
        if tdir and (tdir / "summary.json").is_file():
            try:
                t_elapsed_s = float(json.loads((tdir / "summary.json").read_text(encoding="utf-8")).get("elapsed_s", float("nan")))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass

        sub_j = jobs_df[jobs_df["run_id"] == rid] if (not jobs_df.empty and "run_id" in jobs_df.columns) else pd.DataFrame()
        sacct_gpu_h = float(sub_j["gpu_h"].sum()) if not sub_j.empty else float("nan")
        sacct_attempts = int(len(sub_j)) if not sub_j.empty else None
        sacct_preempts = int((sub_j["state"] == "PREEMPTED").sum()) if not sub_j.empty else None

        sub_e = eval_costs_df[eval_costs_df["run_id"] == rid] if (not eval_costs_df.empty and "run_id" in eval_costs_df.columns) else pd.DataFrame()
        eval_wall_s = float(sub_e["wall_s"].sum()) if not sub_e.empty else float("nan")

        run_rows.append(
            {
                "run_id": rid,
                "group": str(r["group"]),
                "family": str(r["family"]),
                "train_cell": str(r["train_cell"]),
                "model_seed": int(r["model_seed"]),
                "training_status": str(r["training_status"]),
                "budget_exhausted": bool(r["budget_exhausted"]),
                "epochs_completed": r["epochs_completed"],
                "best_val_epoch": r["best_val_epoch"],
                "train_summary_elapsed_s": t_elapsed_s,
                "sacct_gpu_h": sacct_gpu_h,
                "sacct_attempts": sacct_attempts,
                "sacct_preemptions": sacct_preempts,
                "ledger_attempts": r["ledger_attempts"],
                "ledger_extension": r["ledger_extension"],
                "ledger_signals": r["ledger_signals"],
                "eval_wall_s_end_to_end": eval_wall_s,
            }
        )

    by_run_df = pd.DataFrame(run_rows)
    eval_by_cfg = (
        eval_costs_df.groupby(["family", "eval_config", "role"], as_index=False)
        .agg(
            n_units=("wall_s", "count"),
            mean_wall_s=("wall_s", "mean"),
            total_wall_s=("wall_s", "sum"),
            total_frames=("n_frames", "sum"),
        )
        if not eval_costs_df.empty
        else pd.DataFrame()
    )

    return {
        "jobs": jobs_df,
        "phases": phases_df,
        "by_run": by_run_df,
        "eval_costs_by_config": eval_by_cfg,
        "has_sacct": not jobs_df.empty,
    }


__all__ = [
    "collect_compute_accounting",
    "parse_sacct_psv",
]
