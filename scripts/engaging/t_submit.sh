#!/bin/bash
# Submit a subset of the 12 temperature-assisted trainings as ONE array. Explicit resources, global GPU ceilings.
#   t_submit.sh --tip DIR --sha SHA --pool normal|preempt --array 0,1,2 --concurrency N --wall-hours H [--dry-run]
# normal : mit_normal_gpu, account/QoS mit_amf_advanced_gpu, time 06:00:00  (my share: <= 3 concurrent, user total <= 4)
# preempt: mit_preemptable, account mit_general, QoS normal, time 2-00:00:00 (<= 4 concurrent; QoS limit gpu=4)
# Refuses: dirty/wrong checkout; a run dir that exists; a run already queued/running in ANY of my t_assist jobs (one active submission
# per run); concurrency above the pool share; the rejected combination preemptable + Advanced.
set -uo pipefail
TIP=""; SHA=""; POOL=""; ARR=""; CONC=""; WALL=""; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tip) TIP="$2"; shift 2;; --sha) SHA="$2"; shift 2;; --pool) POOL="$2"; shift 2;; --array) ARR="$2"; shift 2;;
    --concurrency) CONC="$2"; shift 2;; --wall-hours) WALL="$2"; shift 2;; --dry-run) DRY=1; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
for v in TIP SHA POOL ARR CONC WALL; do [[ -n "${!v}" ]] || { echo "missing --${v,,}" >&2; exit 2; }; done
case "$POOL" in
  normal)  PART=mit_normal_gpu; ACCT=mit_amf_advanced_gpu; QOS=mit_amf_advanced_gpu; TLIM=06:00:00; MAXC=3 ;;
  preempt) PART=mit_preemptable; ACCT=mit_general; QOS=normal; TLIM=2-00:00:00; MAXC=4 ;;
  *) echo "pool must be normal|preempt" >&2; exit 2 ;;
esac
(( CONC <= MAXC )) || { echo "REFUSE: concurrency $CONC > pool share $MAXC" >&2; exit 7; }
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
cd "$TIP" || exit 2
[[ "$(git rev-parse HEAD)" == "$SHA" ]] || { echo "REFUSE: HEAD != $SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
M=configs/experiments/month2_t_assisted_matrix.json
OUT_ROOT="$("$PY" -c "import json;print(json.load(open('$M'))['out_root'])")"
mapfile -t IDS < <("$PY" - <<PYI
import json
m = json.load(open("$M"))
want = {int(x) for x in "$ARR".split(",")}
for r in m["new_runs"]:
    if r["array_index"] in want:
        print(r["run_id"])
PYI
)
(( ${#IDS[@]} > 0 )) || { echo "REFUSE: empty selection" >&2; exit 2; }
for rid in "${IDS[@]}"; do [[ ! -e "$OUT_ROOT/$rid" ]] || { echo "REFUSE: $OUT_ROOT/$rid already exists (one submission per run)" >&2; exit 4; }; done
# one ACTIVE submission per run: no pending/running t_assist job may already carry these array indices
ACTIVE="$(squeue -u "$USER" -h -r -o '%i|%j|%T' | grep 't_assist' || true)"
if [[ -n "$ACTIVE" ]]; then
  for i in ${ARR//,/ }; do
    grep -E "_${i}\|" <<<"$ACTIVE" >/dev/null && { echo "REFUSE: array index $i already has an active t_assist job" >&2; echo "$ACTIVE" >&2; exit 7; }
  done
fi
NORMAL_BUSY="$(squeue -u "$USER" -h -r -t RUNNING -p mit_normal_gpu -o '%b' | grep -ci gpu || true)"
if [[ "$POOL" == normal ]] && (( NORMAL_BUSY + CONC > 4 )); then echo "REFUSE: $NORMAL_BUSY normal GPUs running + $CONC > 4" >&2; exit 7; fi
mkdir -p "$OUT_ROOT/slurm" "$OUT_ROOT/_ledger"
ARGS=(--parsable --account "$ACCT" --partition "$PART" --qos "$QOS" --time "$TLIM" --requeue --job-name="t_assist_$POOL"
      --array="${ARR}%${CONC}" --output="$OUT_ROOT/slurm/slurm-%A_%a.out" --error="$OUT_ROOT/slurm/slurm-%A_%a.err"
      --export="ALL,S7_TIP=$TIP,EXPECT_SHA=$SHA,S7_WALL_HOURS=$WALL,S7_EXPECT_PARTITION=$PART,S7_EXPECT_ACCOUNT=$ACCT,S7_EXPECT_QOS=$QOS,S7_MATRIX=$TIP/$M,S7_EXTRA_TRAIN_ARGS=--assisted")
echo "[$(date -Is)] test-only: sbatch ${ARGS[*]} scripts/engaging/t_train.sbatch"
sbatch --test-only "${ARGS[@]}" "$TIP/scripts/engaging/t_train.sbatch" 2>&1 | grep -v '^$' || true
[[ "$DRY" == "1" ]] && { echo "dry run: not submitted"; exit 0; }
JID="$(sbatch "${ARGS[@]}" "$TIP/scripts/engaging/t_train.sbatch")" || { echo "sbatch failed" >&2; exit 3; }
cat > "$OUT_ROOT/_ledger/submission_${JID}.json" <<JSON
{"array_job_id": "$JID", "submitted_at": "$(date -Is)", "git_sha": "$SHA", "tip": "$TIP", "pool": "$POOL", "partition": "$PART",
 "account": "$ACCT", "qos": "$QOS", "time": "$TLIM", "wall_hours_per_attempt": $WALL, "array": "${ARR}%${CONC}", "runs": "${IDS[*]}",
 "gres": "gpu:l40s:1", "cpus": 8, "mem": "64G"}
JSON
echo "submitted $POOL array $JID: ${IDS[*]}"
