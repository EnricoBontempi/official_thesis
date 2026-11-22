#!/bin/bash
# One frozen S7 run, best and last, scalar K corrections. CUDA, batch 8, no TF32.
#SBATCH --job-name=k-neural
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --open-mode=append
set -euo pipefail

: "${EXPECT_SHA:?EXPECT_SHA required}"
: "${K_TIP:?K_TIP required}"
: "${K_OUT:?K_OUT required}"
: "${K_MODE:?K_MODE required}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
case "$K_MODE" in
  full) [[ "$K_OUT" != *smoke* ]] || { echo "REFUSE: full output path contains smoke" >&2; exit 4; } ;;
  smoke) [[ "$K_OUT" == *smoke* ]] || { echo "REFUSE: smoke output path must contain smoke" >&2; exit 4; } ;;
  *) echo "REFUSE: mode $K_MODE" >&2; exit 4 ;;
esac
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$K_TIP/src"
cd "$K_TIP"
SHA_HEAD="$(git rev-parse HEAD)"
[[ "$SHA_HEAD" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD $SHA_HEAD != $EXPECT_SHA" >&2; exit 5; }
DIRTY="$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)"
[[ -z "$DIRTY" ]] || { echo "REFUSE: dirty checkout: $DIRTY" >&2; exit 5; }

RUN_ID="$("$PY" - <<PYIDX
import json, os
m = json.load(open("configs/experiments/month2_s7_c149_eval_matrix.json"))
ids = [r["run_id"] for r in m["runs"]]
sel = [x for x in os.environ.get("K_RUN_IDS", "").split("|") if x] or ids
print(sel[int(os.environ["SLURM_ARRAY_TASK_ID"])])
PYIDX
)"
echo "[$(date -Is)] k-neural host=$(hostname) job=${SLURM_JOB_ID:-na} task=${SLURM_ARRAY_TASK_ID:-na} run=$RUN_ID mode=$K_MODE sha=$SHA_HEAD" >&2
[[ ! -e "$K_OUT/$RUN_ID" ]] || { echo "REFUSE: $K_OUT/$RUN_ID already exists" >&2; exit 4; }
ARGS=(scripts/49_temperature_k_neural.py
  --matrix configs/experiments/month2_s7_c149_eval_matrix.json
  --run-id "$RUN_ID"
  --s7-eval-root "${K_S7_EVAL_ROOT:-$HOME/official_thesis_s7_c149_eval}"
  --out-root "$K_OUT"
  --eval-git-sha "$SHA_HEAD"
  --device cuda
  --batch-size 8
  --mode "$K_MODE")
if [[ "$K_MODE" == "smoke" ]]; then
  ARGS+=(--smoke-frames "${K_SMOKE_FRAMES:-4}")
fi
"$PY" "${ARGS[@]}"
