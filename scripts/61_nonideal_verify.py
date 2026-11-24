#!/usr/bin/env python3
"""Verify the twelve Block II run directories before aggregation (CPU). Writes VERIFY_OK.json or BLOCK_REPORT.md."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from combustion_recon.data.nonideal_observations import file_sha256, load_degradation_spec  # noqa: E402
from combustion_recon.evaluation import nonideal_eval as ne  # noqa: E402

SAME = ("spec_sha256", "base_spec_sha256", "eval_git_sha", "matrix_sha256", "s7_reference_sha256")


def verify(spec_path: Path, eval_root: Path) -> dict:
    spec = load_degradation_spec(spec_path)
    runs = [r["run_id"] for r in spec.raw["models"]["runs"]]
    problems: list[str] = []
    provs = {}
    if (eval_root / "CAMPAIGN_HALT.json").exists():
        problems.append("CAMPAIGN_HALT.json present: " + (eval_root / "CAMPAIGN_HALT.json").read_text().replace("\n", " "))
    extra = sorted(p.name for p in eval_root.iterdir() if p.is_dir() and p.name not in runs and p.name != "slurm")
    if extra:
        problems.append(f"unexpected run directories: {extra}")
    for rid in runs:
        d = eval_root / rid
        if not d.is_dir():
            problems.append(f"{rid}: missing directory")
            continue
        for marker in ("GATE_FAILED.json", "HALTED.json", "SMOKE_FAILED.json"):
            if (d / marker).exists():
                problems.append(f"{rid}: {marker}")
        if not (d / "COMPLETE.json").exists():
            problems.append(f"{rid}: no COMPLETE.json")
            continue
        gate = json.loads((d / "gate.json").read_text())
        if not gate.get("all_pass"):
            problems.append(f"{rid}: gate not passed")
        provs[rid] = json.loads((d / "provenance.json").read_text())
        srun = next(r for r in spec.raw["models"]["runs"] if r["run_id"] == rid)
        if provs[rid]["checkpoint_file_sha256"] != srun["best_file_sha256"] or provs[rid]["checkpoint_weights_sha256"] != srun["best_weights_sha256"]:
            problems.append(f"{rid}: checkpoint hash differs from spec")
        if provs[rid]["spec_sha256"] != file_sha256(spec_path):
            problems.append(f"{rid}: spec hash differs")
    for key in SAME:
        vals = {p.get(key) for p in provs.values()}
        if len(vals) > 1:
            problems.append(f"{key} not identical across runs: {sorted(map(str, vals))}")
    cons = {json.dumps(p.get("consumed"), sort_keys=True) for p in provs.values()}
    if len(cons) > 1:
        problems.append("consumed inputs (cache/stats/masks) differ across runs")
    return {"ok": not problems, "problems": problems, "n_runs_expected": len(runs), "n_runs_complete": len(provs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)
    res = verify(a.spec, a.eval_root)
    if res["ok"]:
        (a.dest / "VERIFY_OK.json").write_text(json.dumps(res, indent=2) + "\n")
        print("VERIFY_OK")
        return 0
    lines = ["# BLOCK_REPORT — Block II verification failed", "", "Outputs are kept as diagnostic. No retry, no threshold change, no retraining.", ""]
    lines += [f"- {p}" for p in res["problems"]]
    (a.dest / "BLOCK_REPORT.md").write_text("\n".join(lines) + "\n")
    print("BLOCK_REPORT written")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
