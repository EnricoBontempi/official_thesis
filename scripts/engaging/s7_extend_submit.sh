#!/bin/bash
# Submit the authorised continuation of one S7 run (single job, 1 GPU). Explicit resources only.
#   s7_extend_submit.sh --tip DIR --sha SHA --run-id ID --n-epochs N --partition P --account A --qos Q --time T --wall-hours H [--dry-run]
set -uo pipefail
TIP=""; SHA=""; RID=""; NEP=""; PART=""; ACCT=""; QOS=""; TLIM=""; WALL=""; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tip) TIP="$2"; shift 2;; --sha) SHA="$2"; shift 2;; --run-id) RID="$2"; shift 2;; --n-epochs) NEP="$2"; shift 2;;
    --partition) PART="$2"; shift 2;; --account) ACCT="$2"; shift 2;; --qos) QOS="$2"; shift 2;;
    --time) TLIM="$2"; shift 2;; --wall-hours) WALL="$2"; shift 2;; --dry-run) DRY=1; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
for v in TIP SHA RID NEP PART ACCT QOS TLIM WALL; do [[ -n "${!v}" ]] || { echo "missing --${v,,}" >&2; exit 2; }; done
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
cd "$TIP" || exit 2
[[ "$(git rev-parse HEAD)" == "$SHA" ]] || { echo "REFUSE: HEAD != $SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
OUT_ROOT="$("$PY" -c 'import json;print(json.load(open("configs/experiments/month2_s7_c149_matrix.json"))["out_root"])')"
# dry planner check against the real run directory (read-only): must say "extend"
PLAN="$(PYTHONPATH="$TIP/src" "$PY" scripts/engaging/s7_plan.py plan-extend "$OUT_ROOT" "$RID" "0" 0 "$NEP")" || { echo "planner failed" >&2; exit 8; }
echo "planner: $PLAN"
[[ "$PLAN" == *'"action": "extend"'* ]] || { echo "REFUSE: run is not extendable" >&2; exit 4; }
OTHER="$(squeue -u "$USER" -h -o '%i|%j|%T|%b' | grep -i 'gpu' || true)"
GPUS="$(echo -n "$OTHER" | grep -c . || true)"
if (( GPUS >= 4 )); then echo "REFUSE: already $GPUS GPU jobs" >&2; echo "$OTHER" >&2; exit 7; fi
ARGS=(--parsable --account "$ACCT" --partition "$PART" --qos "$QOS" --time "$TLIM" --requeue --job-name=s7_c149_ext
      --export="ALL,S7_TIP=$TIP,EXPECT_SHA=$SHA,S7_WALL_HOURS=$WALL,S7_EXPECT_PARTITION=$PART,S7_EXPECT_ACCOUNT=$ACCT,S7_EXPECT_QOS=$QOS,S7_EXTEND_RUN_ID=$RID,S7_EXTEND_N_EPOCHS=$NEP")
sbatch --test-only "${ARGS[@]}" "$TIP/scripts/engaging/s7_extend.sbatch" 2>&1 | grep -v '^$' || true
[[ "$DRY" == "1" ]] && { echo "dry run: not submitted"; exit 0; }
JID="$(sbatch "${ARGS[@]}" "$TIP/scripts/engaging/s7_extend.sbatch")" || { echo "sbatch failed" >&2; exit 3; }
cat > "$OUT_ROOT/_ledger/submission_extension_${RID}_${JID}.json" <<JSON
{"job_id": "$JID", "run_id": "$RID", "submitted_at": "$(date -Is)", "git_sha": "$SHA", "tip": "$TIP",
 "new_n_epochs": $NEP, "partition": "$PART", "account": "$ACCT", "qos": "$QOS", "time": "$TLIM",
 "wall_hours_per_attempt": $WALL, "gres": "gpu:l40s:1", "cpus": 8, "mem": "64G"}
JSON
echo "submitted extension job $JID"
