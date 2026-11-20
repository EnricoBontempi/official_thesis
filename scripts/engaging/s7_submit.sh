#!/bin/bash
# Submit the ten S7 C149 trainings as ONE array (index 0-9, throttle %4).
# Resources are explicit arguments: nothing is chosen silently.
#
#   s7_submit.sh --tip DIR --sha SHA --partition P --account A --qos Q --time D-HH:MM:SS --wall-hours H [--dry-run]
#
# Refuses when: the checkout is not at --sha or not clean; the combination fails `sbatch --test-only`;
# the user already has any other queued/running GPU job (the four-GPU limit is global, so a second
# array or a smoke job would silently exceed it); a run directory of the matrix already exists.
set -uo pipefail
TIP=""; SHA=""; PART=""; ACCT=""; QOS=""; TLIM=""; WALL=""; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tip) TIP="$2"; shift 2;; --sha) SHA="$2"; shift 2;; --partition) PART="$2"; shift 2;;
    --account) ACCT="$2"; shift 2;; --qos) QOS="$2"; shift 2;; --time) TLIM="$2"; shift 2;;
    --wall-hours) WALL="$2"; shift 2;; --dry-run) DRY=1; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
for v in TIP SHA PART ACCT QOS TLIM WALL; do [[ -n "${!v}" ]] || { echo "missing --${v,,}" >&2; exit 2; }; done
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
cd "$TIP" || exit 2
[[ "$(git rev-parse HEAD)" == "$SHA" ]] || { echo "REFUSE: HEAD != $SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
OUT_ROOT="$("$PY" -c 'import json;print(json.load(open("configs/experiments/month2_s7_c149_matrix.json"))["out_root"])')"
for rid in $("$PY" -c 'import json;print(" ".join(r["run_id"] for r in json.load(open("configs/experiments/month2_s7_c149_matrix.json"))["new_runs"]))'); do
  [[ ! -e "$OUT_ROOT/$rid" ]] || { echo "REFUSE: $OUT_ROOT/$rid already exists" >&2; exit 4; }
done
OTHER="$(squeue -u "$USER" -h -o '%i|%j|%T|%b' | grep -i 'gpu' || true)"
if [[ -n "$OTHER" ]]; then
  echo "REFUSE: existing GPU jobs would share the 4-GPU limit:" >&2; echo "$OTHER" >&2; exit 7
fi
mkdir -p "$OUT_ROOT/slurm"
ARGS=(--parsable --account "$ACCT" --partition "$PART" --qos "$QOS" --time "$TLIM"
      --array=0-9%4 --requeue --job-name=s7_c149
      --export="ALL,S7_TIP=$TIP,EXPECT_SHA=$SHA,S7_WALL_HOURS=$WALL,S7_EXPECT_PARTITION=$PART,S7_EXPECT_ACCOUNT=$ACCT,S7_EXPECT_QOS=$QOS")
echo "[$(date -Is)] test-only: sbatch ${ARGS[*]} $TIP/scripts/engaging/s7_train.sbatch"
sbatch --test-only "${ARGS[@]}" "$TIP/scripts/engaging/s7_train.sbatch" 2>&1 | grep -v '^$' || true
if [[ "$DRY" == "1" ]]; then echo "dry run: not submitted"; exit 0; fi
JID="$(sbatch "${ARGS[@]}" "$TIP/scripts/engaging/s7_train.sbatch")" || { echo "sbatch failed" >&2; exit 3; }
mkdir -p "$OUT_ROOT/_ledger"
cat > "$OUT_ROOT/_ledger/submission_${JID}.json" <<JSON
{"array_job_id": "$JID", "submitted_at": "$(date -Is)", "git_sha": "$SHA", "tip": "$TIP",
 "partition": "$PART", "account": "$ACCT", "qos": "$QOS", "time": "$TLIM", "wall_hours_per_attempt": $WALL,
 "array": "0-9%4", "gres": "gpu:l40s:1", "cpus": 8, "mem": "64G"}
JSON
echo "submitted array job $JID"
squeue -u "$USER" -r -o '%i %j %T %M %l %r %b' | head -20
