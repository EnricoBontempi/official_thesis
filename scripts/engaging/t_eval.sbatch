#!/bin/bash
# Assisted final evaluation: one array task per run of the FROZEN eval matrix. Resources are passed explicitly by the submitter.
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --open-mode=append
set -euo pipefail
: "${EXPECT_SHA:?}" "${T_TIP:?}" "${T_EVAL_MATRIX:?}" "${T_EVAL_OUT:?}"
MODE="${T_EVAL_MODE:-full}"
case "$MODE" in
  full) [[ "$T_EVAL_OUT" != *smoke* ]] || { echo "REFUSE: full with a smoke root" >&2; exit 4; } ;;
  smoke) [[ "$T_EVAL_OUT" == *smoke* ]] || { echo "REFUSE: smoke needs a smoke root" >&2; exit 4; } ;;
  *) echo "REFUSE: mode" >&2; exit 4 ;;
esac
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$T_TIP/src"
cd "$T_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
RUN_ID="$(T_EVAL_MATRIX="$T_EVAL_MATRIX" "$PY" - <<'PYI'
import json, os
m = json.load(open(os.environ["T_EVAL_MATRIX"]))
ids = [r["run_id"] for r in m["runs"]]
sel = [x for x in os.environ.get("T_EVAL_RUNS", "").split(":") if x] or ids
print(sel[int(os.environ["SLURM_ARRAY_TASK_ID"])])
PYI
)"
echo "[$(date -Is)] t_eval launcher host=$(hostname) job=${SLURM_JOB_ID:-na} run=$RUN_ID mode=$MODE sha=$EXPECT_SHA" >&2
[[ ! -e "$T_EVAL_OUT/$RUN_ID" ]] || { echo "REFUSE: output exists: $T_EVAL_OUT/$RUN_ID" >&2; exit 4; }
mkdir -p "$T_EVAL_OUT"
ARGS=(scripts/55_t_eval.py --matrix "$T_EVAL_MATRIX" --run-id "$RUN_ID" --out-root "$T_EVAL_OUT" --eval-git-sha "$EXPECT_SHA" --mode "$MODE")
[[ -n "${T_EVAL_SMOKE_FRAMES:-}" ]] && ARGS+=(--smoke-frames "$T_EVAL_SMOKE_FRAMES")
"$PY" "${ARGS[@]}"
