#!/bin/bash
#SBATCH --job-name=k-aggregate
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
: "${EXPECT_SHA:?}"
: "${K_TIP:?}"
: "${K_NEURAL:?}"
: "${K_BASE:?}"
: "${K_DEST:?}"
: "${K_FORWARD_SHA:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
cd "$K_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD" >&2; exit 5; }
mapfile -t RUNS < <("$PY" -c 'import json; print("\n".join(r["run_id"] for r in json.load(open("configs/experiments/month2_s7_c149_eval_matrix.json"))["runs"]))')
for run in "${RUNS[@]}"; do
  [[ -f "$K_NEURAL/$run/COMPLETE.json" ]] || { echo "REFUSE: missing COMPLETE for $run" >&2; exit 3; }
done
[[ -f "$K_BASE/COMPLETE.json" ]] || { echo "REFUSE: baseline sweep has no COMPLETE marker" >&2; exit 3; }
export PYTHONPATH="$K_TIP/src"
"$PY" scripts/52_temperature_k_aggregate.py --neural-root "$K_NEURAL" --baseline-root "$K_BASE" --dest "$K_DEST" --git-sha "$EXPECT_SHA" --forward-sha "$K_FORWARD_SHA"
