#!/usr/bin/env python3
"""Freeze the evaluation matrix of the temperature-assisted trainings (checkpoints, hashes, epochs, shared forwards, loop references).

Run by the chained verification job after the trainings finished. Refuses a run that is not in a terminal state, whose summary
disagrees with its checkpoints, whose identity is not the assisted study's, or whose loop record is missing. Writes ONE file that
is made read-only; the evaluation and the aggregation consume exactly this file (its sha256 is checked downstream).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.evaluation.opus_o5_plan import CHECKPOINTS, CRITERIA, reuse_plan  # noqa: E402

TERMINAL = {"completed", "stopped_early_plateau"}
EXPERIMENT_ID = "month2_t_assisted_eval_v1"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def weights_sha256(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(str(k).encode())
        h.update(np.ascontiguousarray(sd[k].detach().contiguous().cpu().numpy()).tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-matrix", type=Path, default=ROOT / "configs/experiments/month2_t_assisted_matrix.json")
    ap.add_argument("--free-eval-matrix", type=Path, default=ROOT / "configs/experiments/month2_s7_c149_eval_matrix.json")
    ap.add_argument("--protocol", type=Path, default=ROOT / "configs/experiments/month2_temperature_study_protocol.yaml")
    ap.add_argument("--anchor-manifest", type=Path, default=ROOT / "configs/observations/month2_temperature_anchors_nested_v1.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", default="")
    ap.add_argument("--allow-non-terminal", action="store_true", help="smoke matrices only")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    tm = json.loads(args.train_matrix.read_text())
    free = json.loads(args.free_eval_matrix.read_text())
    out_root = Path(tm["out_root"])
    wanted = [r for r in args.runs.split(",") if r]
    items = [r for r in tm["new_runs"] if not wanted or r["run_id"] in wanted]
    if not wanted and len(items) != 12:
        raise SystemExit("expected 12 runs")
    prot_sha, man_sha = sha256_file(args.protocol), sha256_file(args.anchor_manifest)
    entries = []
    for it in items:
        d = Path(it["out_dir"])
        ident = json.loads((d / "run_identity.json").read_text())
        if (d / "summary.json").exists():
            summary = json.loads((d / "summary.json").read_text())
        elif args.allow_non_terminal:          # smoke only: a running run has no summary yet
            pr = json.loads((d / "progress.json").read_text())
            summary = {"status": pr["status"], "stop_reason": None, "epochs_completed": pr["epoch"], "updates": pr["update"],
                       "best_val_epoch": pr["best_val_epoch"], "best_seen_diagnostic_epoch": ident.get("_best_seen_epoch")}
        else:
            raise SystemExit(f"{it['run_id']}: summary.json missing")
        hist = json.loads((d / "eval_history.json").read_text())
        if not args.allow_non_terminal and summary["status"] not in TERMINAL:
            raise SystemExit(f"{it['run_id']}: status {summary['status']!r} is not terminal")
        a = ident.get("assisted") or {}
        if a.get("protocol_sha256") != prot_sha or a.get("anchor_manifest_sha256") != man_sha:
            raise SystemExit(f"{it['run_id']}: identity protocol/manifest hashes differ from the frozen contract")
        if ident.get("family") != it["family"] or bool(ident.get("p_conditioning")) != bool(it["p_conditioning"]):
            raise SystemExit(f"{it['run_id']}: identity differs from the matrix")
        ledger = [json.loads(x) for x in (out_root / "_ledger" / f"{it['run_id']}.jsonl").read_text().splitlines() if x.strip()]
        starts = [r for r in ledger if r.get("event") == "attempt_start"]
        by_epoch = {int(h["epoch"]): h for h in hist}
        entry = {"run_id": it["run_id"], "family": it["family"], "p_conditioning": bool(it["p_conditioning"]), "model_seed": int(it["seed"]),
                 "training_dir": str(d), "status": summary["status"], "stop_reason": summary.get("stop_reason"),
                 "epochs_completed": int(summary["epochs_completed"]), "updates": int(summary["updates"]),
                 "training_git_shas": sorted({r["git_sha"] for r in starts if r.get("git_sha")} | {ident["git_sha"]}),
                 "training_job_ids": sorted({str(r["slurm_job_id"]) for r in starts}),
                 "n_attempts": len(starts), "checkpoints": {}}
        wsha = {}
        for name in CHECKPOINTS:
            path = d / "checkpoints" / f"{name}.pt"
            pl = torch.load(path, map_location="cpu", weights_only=False)
            if pl.get("checkpoint_kind") != "sparse_scientific_epoch_complete_v1":
                raise SystemExit(f"{it['run_id']}/{name}: not a complete epoch checkpoint")
            epoch = int(pl["epoch_state"]["epoch"])
            idn = (pl.get("extra") or {}).get("run_identity") or {}
            if idn.get("run_id") != it["run_id"] or int(pl["epoch_config"]["seed"]) != int(it["seed"]):
                raise SystemExit(f"{it['run_id']}/{name}: identity/seed mismatch")
            if epoch not in by_epoch and not args.allow_non_terminal:
                raise SystemExit(f"{it['run_id']}/{name}: no loop eval at epoch {epoch}")
            rec = by_epoch.get(epoch, {})      # smoke only: a running run's last.pt need not sit on an eval epoch
            if name == "best" and epoch != int(summary["best_val_epoch"]):
                raise SystemExit(f"{it['run_id']}: best.pt epoch != summary best_val_epoch")
            if name == "best_seen_diagnostic" and summary.get("best_seen_diagnostic_epoch") is not None and epoch != int(summary["best_seen_diagnostic_epoch"]):
                raise SystemExit(f"{it['run_id']}: best_seen epoch != summary")
            if name == "last" and epoch != int(summary["epochs_completed"]) and not args.allow_non_terminal:
                raise SystemExit(f"{it['run_id']}: last.pt epoch != epochs_completed")
            wsha[name] = weights_sha256(pl["model_state_dict"])
            entry["checkpoints"][name] = {
                "epoch": epoch, "file_sha256": sha256_file(path), "weights_sha256": wsha[name], "bytes": path.stat().st_size,
                "criterion": CRITERIA[name], "checkpoint_kind": pl["checkpoint_kind"],
                "loop_reference": {"train_mean_condition_rmse_K": rec.get("train_mean_condition_rmse_K"),
                                   "seen_mean_condition_rmse_K": rec.get("seen_mean_condition_rmse_K"),
                                   "val_mean_condition_rmse_K": rec.get("val_mean_condition_rmse_K"),
                                   "val_per_replica": rec.get("val_per_replica"), "source": "training_loop_not_final_eval"}}
            del pl
        for name, src in reuse_plan(wsha).items():
            entry["checkpoints"][name]["reuse_forward_from"] = src
        vals = [(int(h["epoch"]), float(h["val_mean_condition_rmse_K"])) for h in hist]
        if min(vals, key=lambda t: (t[1], t[0]))[0] != entry["checkpoints"]["best"]["epoch"] and not args.allow_non_terminal:
            raise SystemExit(f"{it['run_id']}: best.pt is not the first absolute VAL minimum")
        entries.append(entry)
    sh = free["shared_inputs"]
    payload = {"experiment_id": EXPERIMENT_ID, "status": "frozen_for_evaluation", "no_test_split": True, "protocol_sha256": prot_sha,
               "anchor_manifest_sha256": man_sha, "K_values": [0, 1, 4, 9, 16, 32], "velocity_coverage": 0.05, "replicas": [0, 1, 2],
               "gate": {"tolerance_K": 0.05, "rule": "macro common-domain RMSE within 0.05 K of the training loop per run/checkpoint/role/replica (and per K on VAL); missing reference = failure"},
               "shared_inputs": sh, "runs": entries}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(args.out, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(f"wrote {args.out} runs={len(entries)} sha256={sha256_file(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
