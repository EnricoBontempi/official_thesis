#!/usr/bin/env python3
"""Helper for the S7 launcher: matrix rows, preflight paths, attempt plan, ledger.

Subcommands (all print to stdout, machine readable):
  row    <matrix.json> <array_task_id|run_id>        -> shell KEY=VALUE lines
  paths  <matrix.json> <checkout> <run_id>           -> one path per line (inputs read by the task)
  plan   <out_root> <run_id> <job_id> <restart>      -> LaunchPlan JSON
  plan-extend <out_root> <run_id> <job_id> <restart> <new_n_epochs> -> LaunchPlan JSON (extend|refuse)
  preserve <out_root> <run_id> <tag> key=value ...   -> copy the finished tranche + MANIFEST.json
  ledger <out_root> <run_id> key=value ...           -> append one ledger event
  last   <run_dir>                                   -> progress summary (epoch/status) for logs
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.training.s7_launch import (  # noqa: E402
    ledger_append,
    plan_attempt,
    plan_extension,
    preserve_tranche,
)


def _matrix(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _find(matrix: dict, key: str) -> dict:
    for r in matrix["new_runs"]:
        if str(r["array_index"]) == key or r["run_id"] == key:
            return r
    raise SystemExit(f"matrix has no new run {key!r}")


def main(argv: list[str]) -> int:
    cmd = argv[1]
    if cmd == "row":
        m = _matrix(argv[2])
        r = _find(m, argv[3])
        s = m["shared_inputs"]
        out = {
            "RUN_ID": r["run_id"],
            "FAMILY": r["family"],
            "P_ON": str(r["p_conditioning"]).lower(),
            "SEED": r["seed"],
            "CONFIG": r["config"],
            "MATRIX_OUT_ROOT": m["out_root"],
            "CACHE": s["cache_root"],
            "CACHE_HASHES": s["cache_hash_file"],
            "EXPECT_CACHE_AGG": s["expect_cache_aggregate_sha256"],
            "MASKS": s["mask_root"],
            "EXPECT_MASK_FILE": s["mask_manifest_file_sha256"],
            "EXPECT_MASK_AGG": s["mask_aggregate_payload_sha256"],
            "STATS_REL": s["stats_relpath"],
            "EXPECT_STATS": s["stats_sha256"],
        }
        for k, v in out.items():
            print(f"{k}={shlex.quote(str(v))}")
        return 0
    if cmd == "paths":
        m = _matrix(argv[2])
        s = m["shared_inputs"]
        r = _find(m, argv[4])
        cache = Path(s["cache_root"])
        mask = Path(s["mask_root"])
        paths = [
            cache / "manifest.json",
            cache / "geometry.npz",
            Path(s["cache_hash_file"]),
            Path(argv[3]) / s["stats_relpath"],
            Path(argv[3]) / r["config"],
            mask / "manifest.json",
        ]
        paths += [cache / role / n for role in ("train_fit", "seen_train_eval", "val_eval")
                  for n in ("frames.json", "p_norm.npy", "vi_norm.npy", "vj_norm.npy",
                            "target_T_norm.npy", "target_T_K.npy")]
        paths += [mask / f"{role}_nested_sites.npz" for role in ("train_fit", "seen_train_eval", "val_eval")]
        for p in paths:
            print(p)
        return 0
    if cmd == "plan":
        plan = plan_attempt(Path(argv[2]), argv[3], slurm_job_id=argv[4], restart_count=int(argv[5]))
        print(plan.to_json())
        return 0
    if cmd == "plan-extend":
        plan = plan_extension(
            Path(argv[2]), argv[3], slurm_job_id=argv[4], restart_count=int(argv[5]), new_n_epochs=int(argv[6])
        )
        print(plan.to_json())
        return 0
    if cmd == "preserve":
        prov = dict(kv.split("=", 1) for kv in argv[5:])
        man = preserve_tranche(Path(argv[2]) / argv[3], argv[4], prov)
        print(json.dumps({"tag": man["tag"], "n_files": len(man["files"])}))
        return 0
    if cmd == "ledger":
        ev = dict(kv.split("=", 1) for kv in argv[4:])
        ledger_append(Path(argv[2]), argv[3], ev)
        return 0
    if cmd == "last":
        d = Path(argv[2])
        prog = d / "progress.json"
        summ = d / "summary.json"
        out = {}
        if prog.is_file():
            try:
                p = json.loads(prog.read_text())
                out.update({k: p.get(k) for k in ("epoch", "update", "status", "best_val_metric", "best_val_epoch", "lr")})
            except json.JSONDecodeError:
                out["progress"] = "unreadable"
        if summ.is_file():
            try:
                out["summary_status"] = json.loads(summ.read_text()).get("status")
            except json.JSONDecodeError:
                out["summary_status"] = "unreadable"
        print(json.dumps(out, sort_keys=True))
        return 0
    raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
