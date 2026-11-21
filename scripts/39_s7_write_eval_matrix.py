#!/usr/bin/env python3
"""Write the S7 evaluation matrix (12 runs) from the finished training directories.

Reads the training matrix, summaries, ledgers and the three checkpoints of every run, one
checkpoint at a time on CPU (metadata and weight hashes only; no forward). Refuses a run that is
not in a terminal state, whose summary disagrees with its checkpoints, or whose loop record
cannot be matched. Nothing is trained, selected or ranked.

    39_s7_write_eval_matrix.py --out configs/experiments/month2_s7_c149_eval_matrix.json [--runs A,B]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.evaluation.opus_o5_plan import CHECKPOINTS, CRITERIA, reuse_plan  # noqa: E402

EXPERIMENT_ID = "month2_s7_c149_eval_v1"
TRAIN_MATRIX = ROOT / "configs" / "experiments" / "month2_s7_c149_matrix.json"
TERMINAL = {"completed", "stopped_early_plateau"}
STATS_PATH = "/home/bontempi/projects/official_thesis_s5/data/manifests/train_norm_stats_month2_fit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weights_sha256(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        array = state_dict[key].detach().contiguous().cpu().numpy()
        digest.update(str(key).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def read_ledger(out_root: Path, run_id: str) -> list[dict[str, Any]]:
    path = out_root / "_ledger" / f"{run_id}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_list(train: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in train["new_runs"]:
        rows.append(
            {
                "run_id": item["run_id"],
                "family": item["family"],
                "p_conditioning": bool(item["p_conditioning"]),
                "model_seed": int(item["seed"]),
                "shuffle_seed": int(item["shuffle_seed"]),
                "training_dir": item["out_dir"],
                "reused": False,
            }
        )
    for item in train["reused_runs"]:
        rows.append(
            {
                "run_id": item["run_id"],
                "family": item["family"],
                "p_conditioning": bool(item["p_conditioning"]),
                "model_seed": int(item["seed"]),
                "shuffle_seed": int(item["seed"]),
                "training_dir": item["training_dir"],
                "reused": True,
                "training_job": int(item["training_job"]),
                "training_sha": item["training_sha"],
            }
        )
    return rows


def build_entry(item: dict[str, Any], out_root: Path, *, require_terminal: bool) -> dict[str, Any]:
    d = Path(item["training_dir"])
    summary = json.loads((d / "summary.json").read_text())
    identity = json.loads((d / "run_identity.json").read_text())
    history = json.loads((d / "eval_history.json").read_text())
    if require_terminal and summary["status"] not in TERMINAL:
        raise SystemExit(f"{item['run_id']}: status {summary['status']!r} is not terminal")
    ledger = read_ledger(out_root, item["run_id"])
    starts = [r for r in ledger if r.get("event") == "attempt_start"]
    shas = sorted({identity["git_sha"], *[r["git_sha"] for r in starts if r.get("git_sha")]})
    jobs = [str(r["slurm_job_id"]) for r in starts] if starts else [str(item.get("training_job", ""))]
    epochs = {int(h["epoch"]): h for h in history}
    entry: dict[str, Any] = {
        "run_id": item["run_id"],
        "family": item["family"],
        "pool_id": "C149",
        "pool_size": 2980,
        "p_conditioning": item["p_conditioning"],
        "model_seed": item["model_seed"],
        "shuffle_seed": item["shuffle_seed"],
        "reused_not_relaunched": item["reused"],
        "training_git_sha": identity["git_sha"],
        "training_git_shas": shas,
        "training_job_ids": sorted(set(jobs)),
        "training_dir": str(d),
        "status": summary["status"],
        "stop_reason": summary.get("stop_reason"),
        "completed": bool(summary.get("completed")),
        "epochs_completed": int(summary["epochs_completed"]),
        "updates": int(summary["updates"]),
        "extended_past_initial_tranche": any(r.get("event") == "extension_start" for r in ledger),
        "checkpoints": {},
    }
    weight_shas: dict[str, str] = {}
    for name in CHECKPOINTS:
        path = d / "checkpoints" / f"{name}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("checkpoint_kind") != "sparse_scientific_epoch_complete_v1":
            raise SystemExit(f"{item['run_id']}/{name}: not a complete epoch checkpoint")
        state = payload["epoch_state"]
        epoch = int(state["epoch"])
        ident = (payload.get("extra") or {}).get("run_identity") or {}
        if ident.get("run_id") != item["run_id"] or ident.get("family") != item["family"]:
            raise SystemExit(f"{item['run_id']}/{name}: identity mismatch")
        if bool(ident.get("p_conditioning")) != item["p_conditioning"]:
            raise SystemExit(f"{item['run_id']}/{name}: p_conditioning mismatch")
        if int(payload["epoch_config"]["seed"]) != item["model_seed"]:
            raise SystemExit(f"{item['run_id']}/{name}: seed mismatch")
        if epoch not in epochs:
            raise SystemExit(f"{item['run_id']}/{name}: no loop eval at epoch {epoch}")
        rec = epochs[epoch]
        if name == "best":
            if epoch != int(summary["best_val_epoch"]):
                raise SystemExit(f"{item['run_id']}: best.pt epoch {epoch} != summary best_val_epoch")
            loop_value, key = float(rec["val_mean_condition_rmse_K"]), "val_mean_condition_rmse_K"
        elif name == "best_seen_diagnostic":
            if epoch != int(summary["best_seen_diagnostic_epoch"]):
                raise SystemExit(f"{item['run_id']}: best_seen epoch {epoch} != summary")
            loop_value, key = float(rec["seen_mean_condition_rmse_K"]), "seen_mean_condition_rmse_K"
        else:
            if epoch != int(summary["epochs_completed"]):
                raise SystemExit(f"{item['run_id']}: last.pt epoch {epoch} != epochs_completed")
            loop_value, key = float(rec["val_mean_condition_rmse_K"]), "val_mean_condition_rmse_K"
        digest = weights_sha256(payload["model_state_dict"])
        weight_shas[name] = digest
        entry["checkpoints"][name] = {
            "epoch": epoch,
            "file_sha256": sha256_file(path),
            "weights_sha256": digest,
            "bytes": path.stat().st_size,
            "loop_metric_K": loop_value,
            "loop_metric_key": key,
            "loop_metric_source": "training_loop_not_s7_eval",
            "criterion": CRITERIA[name],
            "checkpoint_kind": payload["checkpoint_kind"],
        }
        del payload
    for name, source in reuse_plan(weight_shas).items():
        entry["checkpoints"][name]["reuse_forward_from"] = source
    # best must be the first absolute minimum of the recorded VAL history
    vals = [(int(h["epoch"]), float(h["val_mean_condition_rmse_K"])) for h in history]
    first_min = min(vals, key=lambda t: (t[1], t[0]))
    if first_min[0] != entry["checkpoints"]["best"]["epoch"]:
        raise SystemExit(f"{item['run_id']}: best.pt is not the first absolute VAL minimum")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", default="", help="comma separated subset (smoke matrices)")
    parser.add_argument("--allow-non-terminal", action="store_true", help="smoke matrices only")
    parser.add_argument("--stats-path", default=STATS_PATH)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    train = json.loads(TRAIN_MATRIX.read_text())
    shared = train["shared_inputs"]
    out_root = Path(train["out_root"])
    stats_sha = sha256_file(Path(args.stats_path))
    if stats_sha != shared["stats_sha256"]:
        raise SystemExit("stats sha does not match the training matrix")
    items = run_list(train)
    wanted = [r for r in args.runs.split(",") if r]
    if wanted:
        known = {i["run_id"] for i in items}
        if set(wanted) - known:
            raise SystemExit(f"unknown runs {sorted(set(wanted) - known)}")
        items = [i for i in items if i["run_id"] in wanted]
    elif len(items) != 12:
        raise SystemExit(f"expected 12 runs, found {len(items)}")
    entries = []
    for item in items:
        print(f"[matrix] {item['run_id']}", flush=True)
        entries.append(build_entry(item, out_root, require_terminal=not args.allow_non_terminal))
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "prepared_not_submitted",
        "scope": "free S7 C149: sparse U-Net and adapted GLU, p_on/p_off, seeds 0/1/2",
        "loop_metrics_are_not_s7_eval": True,
        "no_test_split": True,
        "no_refit": True,
        "no_new_selection": True,
        "gate": {"coverage": 0.05, "tolerance_K": 0.05,
                 "rule": "macro RMSE within 0.05 K of the training loop per run/checkpoint/role/replica; "
                         "a missing reference is a failure (roadmapsparse.md S7)"},
        "roles": {"train_fit": "reference760", "seen_train_eval": "seen220", "val_eval": "VAL123"},
        "coverages": [0.01, 0.05, 0.2],
        "replicas": [0, 1, 2],
        "shared_inputs": {
            "mask_root": shared["mask_root"],
            "eval_mask_manifest_file_sha256": shared["mask_manifest_file_sha256"],
            "eval_mask_aggregate_payload_sha256": shared["mask_aggregate_payload_sha256"],
            "stats": args.stats_path,
            "stats_sha256": stats_sha,
            "eval_mask_seed": 20261116,
            "layout_id": "random_sparse_primary",
            "c149_cache_id": shared["cache_id"],
            "c149_cache_root": shared["cache_root"],
            "c149_hash_file": shared["cache_hash_file"],
            "c149_cache_aggregate_sha256": shared["expect_cache_aggregate_sha256"],
        },
        "runs": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} with {len(entries)} runs sha256={sha256_file(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
