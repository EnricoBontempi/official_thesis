#!/usr/bin/env python3
"""Compute accounting for Blocks II-III of the non-ideal study, from Slurm accounting and the run ledgers. No new measurement.

Inputs (all published under results/nonideal/specialists/accounting/):
  raw/sacct_*.psv   `sacct -D -u <user> -S 2026-11-23 -E 2026-11-28 -P` with the columns in its header (-D keeps every
                    requeued/preempted attempt of an array task as its own record, so nothing is counted twice)
  raw/runs_meta.tgz per-run ledger (_ledger/*.jsonl), progress.json, summary.json, slurm_context*.txt
  eval_costs_by_run.csv / eval_costs_by_cell.csv  (copied from the evaluation provenance, measured by the evaluation code)
  submission.json, chain_ids.json  (array task index <-> run id)
GPU-seconds = ElapsedRaw x number of GPUs in AllocTRES (1 for every job here); CPU-seconds = ElapsedRaw x AllocCPUS.
Nothing is a latency: a wall time measured around an evaluation includes observation building, I/O, forward and metrics.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tarfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PHASE = {
    "ni_eval_smoke": "B2 smoke (GPU)", "ni_eval_full": "B2 evaluation of 12 original models (GPU)",
    "ni_smoke": "B3 smoke (GPU)", "ni_smoke_noise": "B3 smoke (GPU)", "ni_smoke_resume": "B3 smoke (GPU)", "ni_smoke_eval": "B3 smoke (GPU)",
    "ni_train_normal": "B3 training, mit_normal_gpu", "ni_train_preempt": "B3 training, mit_preemptable", "ni_ext_preempt": "B3 continuation 1000->1200, mit_preemptable",
    "ni_eval_spec": "B3 evaluation of 30 specialists (GPU)",
    "ni_verify": "CPU verify", "ni_aggregate": "CPU aggregate", "ni_aggregate_spec": "CPU aggregate", "ni_stage_b": "CPU stage B/C",
    "ni_stage_c": "CPU stage B/C", "ni_graphics_spec": "CPU graphics",
}


def t(s: str):
    return datetime.fromisoformat(s) if s and s not in ("Unknown", "None") else None


def n_gpu(tres: str) -> int:
    m = re.search(r"gres/gpu=(\d+)", tres or "")
    return int(m.group(1)) if m else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", type=Path, required=True)
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--freeze", type=Path, required=True)
    a = ap.parse_args()
    raw = a.acc / "raw"
    rows = list(csv.DictReader(open(next(raw.glob("sacct_*.psv"))), delimiter="|"))
    jobs = [r for r in rows if "." not in r["JobID"]]
    steps = {r["JobID"]: r for r in rows if r["JobID"].endswith(".batch")}
    sub = json.loads(a.submission.read_text())
    normal = sub["normal_runs"].split()
    preempt = sub["preempt_runs"].split()
    out_jobs = []
    for r in jobs:
        base = r["JobID"]
        arr, _, task = base.partition("_")
        st = steps.get(base + ".batch", {})
        el = int(r["ElapsedRaw"] or 0)
        ng = n_gpu(r["AllocTRES"])
        cpus = int(re.search(r"cpu=(\d+)", r["AllocTRES"]).group(1)) if "cpu=" in r["AllocTRES"] else 0
        run_id = ""
        if r["JobName"] == "ni_train_normal" and task != "":
            run_id = normal[int(task)]
        elif r["JobName"] == "ni_train_preempt" and task != "":
            run_id = preempt[int(task)]
        elif r["JobName"] == "ni_ext_preempt":
            run_id = "NI-glu-on-N1-s2"
        out_jobs.append({"job_id": base, "array_id": arr, "task": task, "name": r["JobName"], "phase": PHASE.get(r["JobName"], "other"), "run_id": run_id,
                         "partition": r["Partition"], "qos": r["QOS"], "account": r["Account"], "state": r["State"], "exit": r["ExitCode"],
                         "submit": r["Submit"], "start": r["Start"], "end": r["End"], "elapsed_s": el, "n_gpu": ng, "alloc_cpus": cpus,
                         "gpu_s": el * ng, "cpu_s": el * cpus, "alloc_tres": r["AllocTRES"], "node": r["NodeList"], "timelimit": r["Timelimit"],
                         "batch_max_rss": st.get("MaxRSS", ""), "batch_total_cpu": st.get("TotalCPU", "")})
    out_jobs.sort(key=lambda x: (x["start"], x["job_id"]))
    with open(a.acc / "jobs_all_attempts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_jobs[0])); w.writeheader(); w.writerows(out_jobs)

    # phases
    ph = defaultdict(lambda: {"records": 0, "completed": 0, "preempted": 0, "other_states": 0, "gpu_s": 0, "cpu_s": 0, "elapsed_s": 0, "first_start": "9", "last_end": ""})
    for j in out_jobs:
        p = ph[j["phase"]]
        p["records"] += 1
        p["completed"] += j["state"] == "COMPLETED"
        p["preempted"] += j["state"] == "PREEMPTED"
        p["other_states"] += j["state"] not in ("COMPLETED", "PREEMPTED")
        p["gpu_s"] += j["gpu_s"]; p["cpu_s"] += j["cpu_s"]; p["elapsed_s"] += j["elapsed_s"]
        p["first_start"] = min(p["first_start"], j["start"]); p["last_end"] = max(p["last_end"], j["end"])
    with open(a.acc / "phases.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "records", "completed", "preempted", "other_states", "elapsed_job_hours", "gpu_hours", "cpu_hours_allocated", "first_start", "last_end"])
        for k in sorted(ph):
            p = ph[k]
            w.writerow([k, p["records"], p["completed"], p["preempted"], p["other_states"], f"{p['elapsed_s']/3600:.3f}", f"{p['gpu_s']/3600:.3f}", f"{p['cpu_s']/3600:.3f}", p["first_start"], p["last_end"]])

    # per training run
    freeze = json.loads(a.freeze.read_text())["runs"]
    meta = tarfile.open(raw / "runs_meta.tgz")
    def rd(name):
        try:
            return meta.extractfile(name).read().decode()
        except KeyError:
            return None
    tr = []
    for rid, fz in sorted(freeze.items()):
        recs = [j for j in out_jobs if j["run_id"] == rid and j["phase"].startswith(("B3 training", "B3 continuation"))]
        recs.sort(key=lambda x: x["start"])
        prog = json.loads(rd(f"./{rid}/progress.json") or rd(f"{rid}/progress.json"))
        summ = json.loads(rd(f"./{rid}/summary.json") or rd(f"{rid}/summary.json"))
        led = [json.loads(l) for l in (rd(f"./_ledger/{rid}.jsonl") or rd(f"_ledger/{rid}.jsonl")).splitlines() if l.strip()]
        gpu_s = sum(x["gpu_s"] for x in recs)
        main_recs = [x for x in recs if x["phase"].startswith("B3 training")]
        ext_recs = [x for x in recs if x["phase"].startswith("B3 continuation")]
        first_sub, last_end = recs[0]["submit"], max(x["end"] for x in recs)
        span = (t(last_end) - t(recs[0]["start"])).total_seconds()
        queue0 = (t(recs[0]["start"]) - t(first_sub)).total_seconds()
        gaps = sum(max(0.0, (t(b["start"]) - t(a_["end"])).total_seconds()) for a_, b in zip(recs, recs[1:]))
        epochs = fz["epochs_completed"]
        tr.append({"run_id": rid, "family": fz["family"], "regime": fz["regime"], "seed": fz["seed"], "partition": recs[0]["partition"],
                   "gpu": next((e.get("gpu") for e in led if e.get("event") == "attempt_start"), ""),
                   "epochs_completed": epochs, "updates": prog["update"], "presentations_samples_seen": summ["samples_seen"],
                   "best_epoch": fz["best_val_epoch"], "n_sacct_records": len(recs), "n_preempted_records": sum(x["state"] == "PREEMPTED" for x in recs),
                   "n_ledger_attempts": sum(e.get("event") == "attempt_start" for e in led), "n_signals": sum(e.get("event") == "signal" for e in led),
                   "gpu_hours_all_attempts": round(gpu_s / 3600, 3), "gpu_hours_main": round(sum(x["gpu_s"] for x in main_recs) / 3600, 3),
                   "gpu_hours_continuation": round(sum(x["gpu_s"] for x in ext_recs) / 3600, 3),
                   "gpu_seconds_per_completed_epoch": round(gpu_s / epochs, 2),
                   "gpu_hours_in_preempted_records": round(sum(x["gpu_s"] for x in recs if x["state"] == "PREEMPTED") / 3600, 3),
                   "calendar_first_start_to_last_end_h": round(span / 3600, 2), "queue_wait_first_attempt_h": round(queue0 / 3600, 2),
                   "gaps_between_attempts_h": round(gaps / 3600, 2), "first_submit": first_sub, "first_start": recs[0]["start"], "last_end": last_end,
                   
                   "progress_elapsed_s_last_attempt": round(prog["elapsed_s"], 1), "n_eval_history": len(json.loads(rd(f"./{rid}/eval_history.json") or "[]")) if rd(f"./{rid}/eval_history.json") else ""})
    with open(a.acc / "training_by_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tr[0])); w.writeheader(); w.writerows(tr)
    print("jobs", len(out_jobs), "runs", len(tr))


if __name__ == "__main__":
    main()
