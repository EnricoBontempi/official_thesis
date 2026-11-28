#!/bin/bash
# CPU launcher: submits the whole PIV campaign (piv_submit_campaign.sh) ONLY IF Phase 3 completed for all 12 original models with the
# E0 replay gate passed. Runs via --dependency=afterany on the Phase-3 arrays. Writes the outcome to the ledger; never retries silently.
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -uo pipefail
cd "${PIV_TIP:?}" || exit 2
LED=/home/bontempi/official_thesis_piv_runs/_ledger; mkdir -p "$LED"
P3=/home/bontempi/official_thesis_piv_phase3
OUT="$LED/launcher_$(date +%Y%m%dT%H%M%S).json"
PY="$HOME/envs/official_thesis/bin/python"
V="$($PY - "$P3" <<'PYV'
import json, sys, glob, os
m = json.load(open("configs/experiments/month2_s7_c149_eval_matrix.json"))
bad = []
for r in m["runs"]:
    d = os.path.join(sys.argv[1], r["run_id"], "best")
    try:
        g = json.load(open(os.path.join(d, "gate_replay_E0.json"))); c = json.load(open(os.path.join(d, "COMPLETE.json")))
        if not (g["all_pass"] and c["ok"]): bad.append((r["run_id"], "gate/complete not ok"))
    except Exception as e:
        bad.append((r["run_id"], "missing: " + type(e).__name__))
print(json.dumps({"n_models": len(m["runs"]), "bad": bad}))
PYV
)"
echo "[$(date -Is)] phase3 verification: $V"
if [[ "$V" != *'"bad": []'* ]]; then
  echo "{\"launched\": false, \"reason\": \"Phase 3 not complete / gate failed\", \"phase3\": $V}" > "$OUT"; echo "REFUSE: campaign NOT submitted"; exit 6
fi
echo "[$(date -Is)] launching campaign"
bash scripts/engaging/piv_submit_campaign.sh --tip "$PIV_TIP" --sha "$EXPECT_SHA" 2>&1 | tee -a "$LED/launcher_stdout.log"
rc=${PIPESTATUS[0]}
echo "{\"launched\": $([ $rc -eq 0 ] && echo true || echo false), \"rc\": $rc, \"phase3\": $V}" > "$OUT"; exit $rc
