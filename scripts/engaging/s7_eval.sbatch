#!/bin/bash
# S7 final evaluation (and its smoke/parity checks): one array task per matrix row.
# Partition/account/QoS/time/array/out are passed explicitly by s7_eval_submit.sh; nothing is
# chosen here. The task refuses unless the checkout is the declared SHA and clean.
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --open-mode=append
set -euo pipefail

: "${EXPECT_SHA:?EXPECT_SHA required}"
: "${S7_EVAL_TIP:?S7_EVAL_TIP required}"
: "${S7_EVAL_MATRIX:?S7_EVAL_MATRIX (path relative to the checkout) required}"
: "${S7_EVAL_OUT_ROOT:?S7_EVAL_OUT_ROOT required}"
MODE="${S7_EVAL_MODE:-full}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
case "$MODE" in
  full) [[ "$S7_EVAL_OUT_ROOT" != *smoke* ]] || { echo "REFUSE: full mode with a smoke out-root" >&2; exit 4; } ;;
  smoke|parity) [[ "$S7_EVAL_OUT_ROOT" == *smoke* ]] || { echo "REFUSE: $MODE needs a smoke out-root" >&2; exit 4; } ;;
  *) echo "REFUSE: unknown mode $MODE" >&2; exit 4 ;;
esac
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$S7_EVAL_TIP/src"
cd "$S7_EVAL_TIP"

SHA_HEAD="$(git rev-parse HEAD)"
[[ "$SHA_HEAD" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD $SHA_HEAD != EXPECT_SHA $EXPECT_SHA" >&2; exit 5; }
DIRTY="$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)"
[[ -z "$DIRTY" ]] || { echo "REFUSE: dirty checkout: $DIRTY" >&2; exit 5; }

RUN_ID="$("$PY" - <<PYIDX
import json, os
m = json.load(open(os.environ["S7_EVAL_MATRIX"]))
ids = [r["run_id"] for r in m["runs"]]
sel = [x for x in os.environ.get("S7_EVAL_RUN_IDS", "").split(",") if x] or ids
print(sel[int(os.environ["SLURM_ARRAY_TASK_ID"])])
PYIDX
)"
echo "[$(date -Is)] launcher host=$(hostname) job=${SLURM_JOB_ID:-na} task=${SLURM_ARRAY_TASK_ID:-na} run=$RUN_ID mode=$MODE sha=$SHA_HEAD" >&2

NEEDED="$("$PY" scripts/engaging/o5_preflight_paths.py "$S7_EVAL_MATRIX" "$RUN_ID")"
[[ -n "$NEEDED" ]] || { echo "REFUSE: empty input list for $RUN_ID" >&2; exit 6; }
while read -r P; do
  timeout 120 stat -c '%s' "$P" > /dev/null 2>&1 || { echo "REFUSE: input not reachable from $(hostname): $P" >&2; exit 6; }
done <<< "$NEEDED"
echo "[$(date -Is)] input access ok" >&2

mkdir -p "$S7_EVAL_OUT_ROOT"
[[ ! -e "$S7_EVAL_OUT_ROOT/$RUN_ID" ]] || { echo "REFUSE: output already exists: $S7_EVAL_OUT_ROOT/$RUN_ID" >&2; exit 4; }

ARGS=(scripts/38_s7_eval.py --matrix "$S7_EVAL_MATRIX" --run-id "$RUN_ID" --out-root "$S7_EVAL_OUT_ROOT"
      --eval-git-sha "$SHA_HEAD" --device cuda --mode "$MODE")
[[ -n "${S7_EVAL_SMOKE_FRAMES:-}" ]] && ARGS+=(--smoke-frames "$S7_EVAL_SMOKE_FRAMES")
"$PY" "${ARGS[@]}"
