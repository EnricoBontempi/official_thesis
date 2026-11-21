#!/usr/bin/env python3
"""Effective S7 costs from Slurm accounting and the evaluation outputs: compute vs queue wait.

Training: every attempt of every run (requeues kept as separate attempts, ``sacct -D``).
Evaluation: every task of the full-evaluation arrays. Wait = Start - Submit of that attempt
(for a requeued attempt Submit is the requeue time); compute = End - Start. Reads only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

TRAIN_ARRAYS = ("29799680",)
TRAIN_SINGLES = ("29966737", "29645053", "29645054")


def _sacct(job_ids: list[str]) -> pd.DataFrame:
    out = subprocess.run(
        ["sacct", "-D", "-X", "-P", "-j", ",".join(job_ids),
         "-o", "JobID,JobIDRaw,State,Submit,Start,End,Elapsed,NodeList,Partition,Account,QOS"],
        check=True, capture_output=True, text=True,
    ).stdout
    rows = [line.split("|") for line in out.strip().splitlines()]
    return pd.DataFrame(rows[1:], columns=rows[0])


def _t(s: str):
    return datetime.fromisoformat(s) if s and s not in ("Unknown", "None") else None


def _seconds(a, b):
    return (b - a).total_seconds() if a and b else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args()
    if args.dest.exists():
        raise SystemExit(f"refusing to overwrite {args.dest}")
    matrix = json.loads(args.matrix.read_text())
    raw_to_run = {str(j): e["run_id"] for e in matrix["runs"] for j in e["training_job_ids"]}
    tr = _sacct(list(TRAIN_ARRAYS + TRAIN_SINGLES))
    tr = tr[tr["JobIDRaw"].isin(raw_to_run)].copy()
    tr["run_id"] = tr["JobIDRaw"].map(raw_to_run)
    for c in ("Submit", "Start", "End"):
        tr[c + "_t"] = tr[c].map(_t)
    tr["compute_s"] = [_seconds(a, b) for a, b in zip(tr["Start_t"], tr["End_t"])]
    tr["queue_wait_s"] = [_seconds(a, b) for a, b in zip(tr["Submit_t"], tr["Start_t"])]
    tr = tr.sort_values(["run_id", "Start_t"])
    tr["attempt"] = tr.groupby("run_id").cumcount() + 1
    cols = ["run_id", "attempt", "JobID", "JobIDRaw", "State", "Submit", "Start", "End", "NodeList",
            "partition", "account", "qos", "compute_s", "queue_wait_s"]
    tr = tr.rename(columns={"Partition": "partition", "Account": "account", "QOS": "qos"})
    by_attempt = tr[cols]
    by_run = by_attempt.groupby("run_id").agg(
        n_attempts=("attempt", "max"), compute_h=("compute_s", lambda s: s.sum() / 3600),
        queue_wait_h=("queue_wait_s", lambda s: s.sum() / 3600)).reset_index()
    missing = sorted(set(raw_to_run.values()) - set(by_run["run_id"]))
    if missing:
        raise SystemExit(f"no accounting rows for {missing}")
    eval_rows = []
    for sub in sorted(args.eval_root.glob("submission_*.json")):
        meta = json.loads(sub.read_text())
        if meta.get("mode") != "full":
            continue
        ids = meta["runs"].split()
        df = _sacct([str(meta["array_job_id"])])
        for r in df.itertuples():
            task = int(r.JobID.split("_")[1]) if "_" in r.JobID and r.JobID.split("_")[1].isdigit() else None
            if task is None:
                continue
            run = ids[task]
            sc = args.eval_root / run / "setting_costs.csv"
            fwd = float(pd.read_csv(sc)["wall_s"].sum()) if sc.exists() else float("nan")
            prov = args.eval_root / run / "provenance.json"
            mem = json.loads(prov.read_text()).get("max_memory_bytes") if prov.exists() else None
            a, b, c = _t(r.Submit), _t(r.Start), _t(r.End)
            eval_rows.append({"run_id": run, "JobID": r.JobID, "State": r.State, "Submit": r.Submit, "Start": r.Start,
                              "End": r.End, "NodeList": r.NodeList, "compute_s": _seconds(b, c),
                              "queue_wait_s": _seconds(a, b), "forward_wall_s_sum": fwd, "max_gpu_memory_bytes": mem})
    args.dest.mkdir(parents=True, exist_ok=False)
    by_attempt.to_csv(args.dest / "training_costs_by_attempt.csv", index=False)
    by_run.to_csv(args.dest / "training_costs_by_run.csv", index=False)
    pd.DataFrame(eval_rows).to_csv(args.dest / "eval_costs_by_task.csv", index=False)
    print(by_run.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
