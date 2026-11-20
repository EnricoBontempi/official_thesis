#!/usr/bin/env python3
"""Verify the two reused U-Net C149 seed-0 trainings from files on disk (no forward pass).

Reads run_identity/summary/init manifests and hashes the checkpoints. Writes a JSON report.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    matrix = json.loads(Path(sys.argv[1]).read_text())
    out = {"verified_at": subprocess.run(["date", "-Is"], capture_output=True, text=True).stdout.strip(), "runs": []}
    for r in matrix["reused_runs"]:
        d = Path(r["training_dir"])
        ident = json.loads((d / "run_identity.json").read_text())
        summ = json.loads((d / "summary.json").read_text())
        init = json.loads((d / "init_manifest.json").read_text())
        eff = json.loads((d / "effective_config.json").read_text())
        ck = {n: {"sha256": sha(d / "checkpoints" / f"{n}.pt"), "bytes": (d / "checkpoints" / f"{n}.pt").stat().st_size}
              for n in ("best", "last", "best_seen_diagnostic")}
        sacct = subprocess.run(
            ["sacct", "-j", str(r["training_job"]), "-X", "-n", "-P", "-o", "JobID,State,ExitCode,Elapsed,Partition,Account,QOS,NodeList"],
            capture_output=True, text=True).stdout.strip()
        tc = eff["candidate"]["training_candidate"]
        out["runs"].append({
            "run_id": r["run_id"], "training_job": r["training_job"], "sacct": sacct,
            "training_dir": str(d),
            "git_sha_in_identity": ident.get("git_sha"),
            "expected_training_sha": r["training_sha"],
            "family": ident.get("family"), "p_conditioning": ident.get("p_conditioning"),
            "model_seed": init.get("model_seed"), "shuffle_seed": init.get("shuffle_seed"),
            "pool_id": ident.get("pool_id"), "pool_size": ident.get("pool_size"),
            "presentations_per_scientific_epoch": ident.get("presentations_per_scientific_epoch"),
            "warm_start": init.get("warm_start"), "init_state_dict_sha256": init.get("init_state_dict_sha256"),
            "cache_payload_aggregate_sha256": (ident.get("cache_payload_sha256") or {}).get("aggregate_sha256"),
            "stats_sha256": ident.get("stats_sha256"),
            "mask_payload_sha256_legacy_key_is_manifest_file_hash": ident.get("mask_payload_sha256"),
            "config_sha256": ident.get("config_sha256"),
            "train_mask_seed": tc.get("train_mask_seed"), "eval_mask_seed": tc.get("eval_mask_seed"),
            "batch_size": tc.get("batch_size"), "warmup_updates": tc.get("warmup_updates"),
            "summary": {k: summ.get(k) for k in ("status", "stop_reason", "completed", "epochs_completed", "updates",
                                                  "best_val_metric", "best_val_epoch", "best_seen_diagnostic_metric",
                                                  "best_seen_diagnostic_epoch", "elapsed_s", "samples_seen")},
            "checkpoints": ck,
            "partial_interrupt_present": (d / "checkpoints" / "partial_interrupt.pt").exists(),
        })
    Path(sys.argv[2]).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
