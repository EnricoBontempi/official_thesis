#!/bin/bash
# Submit S7 evaluation tasks (full, or the smoke/parity checks). Explicit resources, 4-GPU ceiling.
#   s7_eval_submit.sh --tip DIR --sha SHA --matrix REL --out-root DIR --mode full|smoke|parity
#       --partition P --account A --qos Q --time T --max-gpus N [--runs A,B,...] [--dry-run]
# Refuses when: HEAD != SHA or dirty; the out-root already holds a run of the selection; the
# user's GPU jobs plus this array's concurrency would exceed --max-gpus.
set -uo pipefail
TIP=""; SHA=""; MAT=""; OUT=""; MODE=""; PART=""; ACCT=""; QOS=""; TLIM=""; MAXG=""; RUNS=""; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tip) TIP="$2"; shift 2;; --sha) SHA="$2"; shift 2;; --matrix) MAT="$2"; shift 2;; --out-root) OUT="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;; --partition) PART="$2"; shift 2;; --account) ACCT="$2"; shift 2;; --qos) QOS="$2"; shift 2;;
    --time) TLIM="$2"; shift 2;; --max-gpus) MAXG="$2"; shift 2;; --runs) RUNS="$2"; shift 2;; --dry-run) DRY=1; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
for v in TIP SHA MAT OUT MODE PART ACCT QOS TLIM MAXG; do [[ -n "${!v}" ]] || { echo "missing --${v,,}" >&2; exit 2; }; done
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
cd "$TIP" || exit 2
[[ "$(git rev-parse HEAD)" == "$SHA" ]] || { echo "REFUSE: HEAD != $SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
IDS="$(RUNS="$RUNS" MAT="$MAT" "$PY" - <<'PYIDS'
import json, os
m = json.load(open(os.environ["MAT"]))
ids = [r["run_id"] for r in m["runs"]]
sel = [x for x in os.environ["RUNS"].split(",") if x] or ids
bad = set(sel) - set(ids)
assert not bad, f"unknown runs {bad}"
print(" ".join(sel))
PYIDS
)" || { echo "REFUSE: bad run selection" >&2; exit 2; }
N=$(echo "$IDS" | wc -w)
for rid in $IDS; do [[ ! -e "$OUT/$rid" ]] || { echo "REFUSE: $OUT/$rid already exists" >&2; exit 4; }; done
BUSY="$(squeue -u "$USER" -r -h -t RUNNING,PENDING -o '%i|%b' | grep -ci 'gpu' || true)"
if (( BUSY + MAXG > 4 )); then
  echo "REFUSE: $BUSY GPU jobs already queued/running; this submission can use up to $MAXG (ceiling 4)" >&2; exit 7
fi
mkdir -p "$OUT/slurm"
CONC=$(( MAXG < N ? MAXG : N ))
ARGS=(--parsable --account "$ACCT" --partition "$PART" --qos "$QOS" --time "$TLIM" --array="0-$((N-1))%$CONC"
      --job-name="s7_eval_$MODE" --output="$OUT/slurm/slurm-%A_%a.out" --error="$OUT/slurm/slurm-%A_%a.err"
      --export="ALL,S7_EVAL_TIP=$TIP,EXPECT_SHA=$SHA,S7_EVAL_MATRIX=$MAT,S7_EVAL_OUT_ROOT=$OUT,S7_EVAL_MODE=$MODE,S7_EVAL_RUN_IDS=${IDS// /:}${S7_EVAL_SMOKE_FRAMES:+,S7_EVAL_SMOKE_FRAMES=$S7_EVAL_SMOKE_FRAMES}")  # ids joined by ":" (sbatch --export splits on commas)
echo "[$(date -Is)] test-only: sbatch ${ARGS[*]} $TIP/scripts/engaging/s7_eval.sbatch"
sbatch --test-only "${ARGS[@]}" "$TIP/scripts/engaging/s7_eval.sbatch" 2>&1 | grep -v '^$' || true
[[ "$DRY" == "1" ]] && { echo "dry run: not submitted"; exit 0; }
JID="$(sbatch "${ARGS[@]}" "$TIP/scripts/engaging/s7_eval.sbatch")" || { echo "sbatch failed" >&2; exit 3; }
cat > "$OUT/submission_${JID}.json" <<JSON
{"array_job_id": "$JID", "submitted_at": "$(date -Is)", "git_sha": "$SHA", "tip": "$TIP", "mode": "$MODE",
 "matrix": "$MAT", "partition": "$PART", "account": "$ACCT", "qos": "$QOS", "time": "$TLIM",
 "runs": "$(echo $IDS)", "array": "0-$((N-1))%$CONC", "gres": "gpu:l40s:1", "cpus": 8, "mem": "64G"}
JSON
echo "submitted array job $JID ($N tasks, concurrency $CONC)"
