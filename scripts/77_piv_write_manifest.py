#!/usr/bin/env python3
"""Write the real post-processing manifest (piv_postproc_manifest_v1) of the main campaign from the run matrix.

Spec v1.1 names, all evaluation configurations (including the mandatory SENS-RHO0 and the R2 additions SENS-TAIL, sub5), real paths.
No placeholder is kept. T8-T11 are not in the manifest (not authorised)."""
import argparse, json, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ap = argparse.ArgumentParser()
ap.add_argument("--matrix", type=Path, default=ROOT / "configs/experiments/piv_campaign_matrix.json")
ap.add_argument("--train-root"); ap.add_argument("--eval-root", required=True); ap.add_argument("--training-git-sha", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--only-runs", nargs="*", default=None, help="tests only")
ap.add_argument("--eval-configs", nargs="*", default=None, help="tests only")
ap.add_argument("--sacct-psv", default=None)
a = ap.parse_args()
m = json.loads(a.matrix.read_text()); ex = json.loads((ROOT / "configs/piv/piv_postproc_manifest_example.json").read_text())
spec = yaml.safe_load((ROOT / m["observation_spec"]).read_bytes())
train_root = a.train_root or m["out_root"]
CELL = {"IDEAL-A": "IDEAL-A", "PIV-A": "PIV-A", "IDEAL-B-in": "IDEAL-B", "PIV-B-in": "PIV-B", "HIST-5PCT": "BRIDGE-5pct-IDEAL"}
CFGID = {"HIST-5PCT": "E0"}
runs = []
for r in m["new_runs"]:
    if a.only_runs and r["run_id"] not in a.only_runs: continue
    obs = r["observation"]
    runs.append({"run_id": r["run_id"], "group": r["group"], "family": r["family"], "variant": "p_on", "train_cell": CELL[obs], "train_config_id": CFGID.get(obs, obs),
                 "b_position": "B-in" if "B-in" in obs else None, "model_seed": r["seed"], "conditional": False,
                 "training_dir": f"{train_root}/{r['run_id']}", "eval_dir": f"{a.eval_root}/{r['run_id']}", "ledger_path": f"{train_root}/_ledger/{r['run_id']}.jsonl"})
man = {k: v for k, v in ex.items() if k not in ("note",)}
man.update({"campaign_phase": "phase4_campaign", "observation_spec_path": m["observation_spec"], "observation_spec_sha256": m["observation_spec_sha256"],
            "training_git_sha": a.training_git_sha, "eval_configs": a.eval_configs or list(spec["configs"].keys()), "runs": runs})
man.pop("mission00_baseline_csv", None)
if a.sacct_psv: man["slurm_accounting"] = {"sacct_psv": a.sacct_psv}
else: man.pop("slurm_accounting", None)
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(man, indent=1) + "\n")
print("manifest", a.out, "runs", len(runs), "eval_configs", len(man["eval_configs"]))
