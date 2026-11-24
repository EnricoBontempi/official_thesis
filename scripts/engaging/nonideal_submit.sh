#!/bin/bash
# Submit the Block II chain: eval array (12 tasks) -> verify (afterany) -> aggregate (afterany). Refuses on SHA mismatch,
# dirty checkout, existing outputs, or if the user's GPU jobs + this submission would exceed 4.
#   nonideal_submit.sh --tip DIR --sha SHA --spec REL --out-root DIR --dest DIR --mode smoke|full --partition P --account A --qos Q
#                      --time T --max-gpus N [--runs A,B] [--smoke-frames N] [--dry-run]
set -uo pipefail
TIP=""; SHA=""; SPEC=""; OUT=""; DEST=""; MODE=""; PART=""; ACCT=""; QOS=""; TLIM=""; MAXG=""; RUNS=""; DRY=0; SF=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tip) TIP="$2"; shift 2;; --sha) SHA="$2"; shift 2;; --spec) SPEC="$2"; shift 2;; --out-root) OUT="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;; --mode) MODE="$2"; shift 2;; --partition) PART="$2"; shift 2;; --account) ACCT="$2"; shift 2;;
    --qos) QOS="$2"; shift 2;; --time) TLIM="$2"; shift 2;; --max-gpus) MAXG="$2"; shift 2;; --runs) RUNS="$2"; shift 2;;
    --smoke-frames) SF="$2"; shift 2;; --dry-run) DRY=1; shift;; *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
for v in TIP SHA SPEC OUT DEST MODE PART ACCT QOS TLIM MAXG; do [[ -n "${!v}" ]] || { echo "missing --${v,,}" >&2; exit 2; }; done
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
cd "$TIP" || exit 2
[[ "$(git rev-parse HEAD)" == "$SHA" ]] || { echo "REFUSE: HEAD != $SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
[[ ! -e "$OUT" || -z "$(ls -A "$OUT" 2>/dev/null)" ]] || { echo "REFUSE: out-root $OUT not empty" >&2; exit 4; }
[[ ! -e "$DEST" ]] || { echo "REFUSE: dest $DEST exists" >&2; exit 4; }
IDS="$(RUNS="$RUNS" SPEC="$SPEC" PYTHONPATH="$TIP/src" "$PY" - <<'PYIDS'
import os
from combustion_recon.data.nonideal_observations import load_degradation_spec
ids = [r["run_id"] for r in load_degradation_spec(os.environ["SPEC"]).raw["models"]["runs"]]
sel = [x for x in os.environ["RUNS"].split(",") if x] or ids
assert not set(sel) - set(ids), "unknown run"
print(" ".join(sel))
PYIDS
)" || { echo "REFUSE: bad run selection" >&2; exit 2; }
N=$(echo "$IDS" | wc -w)
BUSY="$(squeue -u "$USER" -r -h -t RUNNING,PENDING -o '%i|%b' | grep -ci 'gpu' || true)"
(( BUSY + MAXG <= 4 )) || { echo "REFUSE: $BUSY GPU jobs queued/running; +$MAXG would exceed 4" >&2; exit 7; }
mkdir -p "$OUT/slurm" "$DEST/slurm"
CONC=$(( MAXG < N ? MAXG : N ))
EXPORT="ALL,EXPECT_SHA=$SHA,NI_TIP=$TIP,NI_SPEC=$SPEC,NI_OUT_ROOT=$OUT,NI_DEST=$DEST,NI_MODE=$MODE,NI_RUN_IDS=${IDS// /:}${SF:+,NI_SMOKE_FRAMES=$SF}"
ARGS=(--parsable --account "$ACCT" --partition "$PART" --qos "$QOS" --time "$TLIM" --array="0-$((N-1))%$CONC"
      --job-name="ni_eval_$MODE" --output="$OUT/slurm/eval-%A_%a.out" --error="$OUT/slurm/eval-%A_%a.err" --export="$EXPORT")
echo "[$(date -Is)] test-only: sbatch ${ARGS[*]} $TIP/scripts/engaging/nonideal_eval.sbatch"
sbatch --test-only "${ARGS[@]}" "$TIP/scripts/engaging/nonideal_eval.sbatch" 2>&1 | grep -v '^$' || true
[[ "$DRY" == "1" ]] && { echo "dry run: not submitted"; exit 0; }
JA="$(sbatch "${ARGS[@]}" "$TIP/scripts/engaging/nonideal_eval.sbatch")" || { echo "sbatch eval failed" >&2; exit 3; }
JV="$(sbatch --parsable --dependency=afterany:"$JA" --output="$DEST/slurm/verify-%j.out" --error="$DEST/slurm/verify-%j.err" --export="$EXPORT" "$TIP/scripts/engaging/nonideal_verify.sbatch")" || { echo "sbatch verify failed" >&2; exit 3; }
JG="$(sbatch --parsable --dependency=afterany:"$JV" --output="$DEST/slurm/aggregate-%j.out" --error="$DEST/slurm/aggregate-%j.err" --export="$EXPORT" "$TIP/scripts/engaging/nonideal_aggregate.sbatch")" || { echo "sbatch aggregate failed" >&2; exit 3; }
cat > "$DEST/submission.json" <<JSON
{"eval_array": "$JA", "verify": "$JV", "aggregate": "$JG", "submitted_at": "$(date -Is)", "git_sha": "$SHA", "tip": "$TIP",
 "mode": "$MODE", "spec": "$SPEC", "out_root": "$OUT", "dest": "$DEST", "partition": "$PART", "account": "$ACCT", "qos": "$QOS",
 "time": "$TLIM", "array": "0-$((N-1))%$CONC", "runs": "$(echo $IDS)", "gres": "gpu:l40s:1", "cpus": 8, "mem": "64G"}
JSON
echo "submitted: eval=$JA verify=$JV aggregate=$JG ($N tasks, concurrency $CONC)"
