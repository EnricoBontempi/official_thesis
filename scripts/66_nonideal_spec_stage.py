#!/usr/bin/env python3
"""Chain helper for the specialist campaign (CPU).

  status    --matrix M            print the classification of every run (JSON) and write stageB_status.json
  extension --matrix M            print one line per pool: "<pool>:<id1>:<id2>..." for runs that need the single continuation
  freeze    --matrix M --dest D --git-sha S   validate every run, write D/freeze.json and BLOCK_REPORT_TRAINING.md if anything is invalid
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from combustion_recon.training import nonideal_campaign as nc  # noqa: E402


def load(matrix: Path):
    m = json.loads(matrix.read_text())
    return m, Path(m["out_root"]), m["new_runs"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("status", "extension", "freeze"))
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--dest", type=Path)
    ap.add_argument("--git-sha", default="", help="frozen SHA of the TRAINING checkout (run_identity.git_sha must equal it)")
    ap.add_argument("--post-git-sha", default="", help="SHA of the verification/aggregation checkout (recorded in freeze.json)")
    ap.add_argument("--smoke-root", type=Path, default=None, help="chain test on non-scientific smoke runs only (output must be named *smoke*)")
    ap.add_argument("--smoke-runs", default="")
    a = ap.parse_args()
    m, root, runs = load(a.matrix)
    if a.smoke_root is not None:
        assert "smoke" in str(a.smoke_root) and a.dest is not None and "smoke" in str(a.dest)
        root = a.smoke_root
        runs = [r for r in runs if r["run_id"] in a.smoke_runs.split(",")]
    if a.cmd in ("status", "extension"):
        cls = {r["run_id"]: nc.classify_run(root, r["run_id"]) for r in runs}
        if a.cmd == "status":
            (root / "_ledger").mkdir(parents=True, exist_ok=True)
            (root / "_ledger" / "stageB_status.json").write_text(json.dumps(cls, indent=1, sort_keys=True) + "\n")
            print(json.dumps({k: v["state"] for k, v in cls.items()}, indent=1, sort_keys=True))
            return 0
        assign = json.loads((root / "_ledger" / "assignment.json").read_text())
        for pool in ("normal", "preempt"):
            ids = [r for r in assign.get(pool, []) if cls.get(r, {}).get("state") == nc.NEEDS_EXTENSION]
            if ids:
                print(pool + ":" + ":".join(ids))
        return 0
    # freeze
    assert a.dest is not None
    a.dest.mkdir(parents=True, exist_ok=True)
    spec_sha = m["degradation_spec_sha256"]
    frozen, problems = {}, []
    for r in runs:
        res = nc.validate_and_freeze_run(root, r, expect_git_sha=a.git_sha, expect_spec_sha=spec_sha, smoke=a.smoke_root is not None)
        res["ledger"] = nc.ledger_summary(root, r["run_id"])
        frozen[r["run_id"]] = res
        if not res["valid"]:
            problems.append(f"{r['run_id']}: {'; '.join(res['problems'])}")
    # init identity across regimes of the same family/seed
    by = {}
    for r in runs:
        res = frozen[r["run_id"]]
        if res.get("init_state_dict_sha256"):
            by.setdefault((r["family"], r["seed"]), set()).add(res["init_state_dict_sha256"])
    for k, v in by.items():
        if len(v) != 1:
            problems.append(f"init differs across regimes for {k}")
    n_budget = sum(1 for v in frozen.values() if v.get("valid") and v.get("budget_exhausted"))
    out = {"n_budget_exhausted_valid": n_budget, "n_plateau_valid": sum(1 for v in frozen.values() if v.get("valid") and not v.get("budget_exhausted")),
           "n_runs": len(runs), "n_valid": sum(bool(v["valid"]) for v in frozen.values()), "git_sha": a.git_sha, "training_git_sha": a.git_sha, "post_processing_git_sha": a.post_git_sha,
           "degradation_spec_sha256": spec_sha, "problems": problems, "runs": frozen}
    (a.dest / "freeze.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    if problems:
        lines = ["# BLOCK_REPORT — specialist training validation", "",
                 "Valid runs are still evaluated (diagnostic); the package cannot be declared complete.", ""] + [f"- {p}" for p in problems]
        (a.dest / "BLOCK_REPORT_TRAINING.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"n_valid": out["n_valid"], "n_problems": len(problems)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
