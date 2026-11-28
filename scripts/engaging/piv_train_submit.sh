#!/bin/bash
# Submit a pool's share of the 35 PIV campaign trainings as ONE array. Explicit resources, global GPU ceilings.
#   nonideal_train_submit.sh --tip DIR --sha SHA --pool normal|preempt --runs ID1,ID2,... --concurrency N --wall-hours H [--time T] [--dry-run]
# normal : mit_normal_gpu, account/QoS mit_amf_advanced_gpu (<= 4 GPUs for me on this pool)
# preempt: mit_preemptable, account mit_general, QoS normal, 2-00:00:00 (<= 4 GPUs; never Advanced)
# Refuses: dirty/wrong checkout; an existing run dir; a run already active in any piv_train job (one submission per run);
# concurrency above the pool share; a run listed twice.
set -uo pipefail
TIP=""; SHA=""; POOL=""; RUNS=""; CONC=""; WALL=""; DRY=0; TLIM_OVR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tip) TIP="$2"; shift 2;; --sha) SHA="$2"; shift 2;; --pool) POOL="$2"; shift 2;; --runs) RUNS="$2"; shift 2;;
    --concurrency) CONC="$2"; shift 2;; --wall-hours) WALL="$2"; shift 2;; --time) TLIM_OVR="$2"; shift 2;; --dry-run) DRY=1; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
for v in TIP SHA POOL RUNS CONC WALL; do [[ -n "${!v}" ]] || { echo "missing --${v,,}" >&2; exit 2; }; done
case "$POOL" in
  normal)  PART=mit_normal_gpu; ACCT=mit_amf_advanced_gpu; QOS=mit_amf_advanced_gpu; TLIM=06:00:00; MAXC=4 ;;
  preempt) PART=mit_preemptable; ACCT=mit_general; QOS=normal; TLIM=2-00:00:00; MAXC=4 ;;
  *) echo "pool must be normal|preempt" >&2; exit 2 ;;
esac
[[ -n "$TLIM_OVR" ]] && TLIM="$TLIM_OVR"
(( CONC <= MAXC )) || { echo "REFUSE: concurrency $CONC > pool share $MAXC" >&2; exit 7; }
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
cd "$TIP" || exit 2
[[ "$(git rev-parse HEAD)" == "$SHA" ]] || { echo "REFUSE: HEAD != $SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
M=configs/experiments/piv_campaign_matrix.json
OUT_ROOT="$("$PY" -c "import json;print(json.load(open('$M'))['out_root'])")"
IFS=',' read -r -a IDS <<< "$RUNS"
"$PY" - "$M" "${IDS[@]}" <<'PYV' || { echo "REFUSE: bad run selection" >&2; exit 2; }
import json, sys
m = json.load(open(sys.argv[1])); ids = sys.argv[2:]
known = {r["run_id"] for r in m["new_runs"]}
assert len(set(ids)) == len(ids), "duplicate run in selection"
assert not set(ids) - known, f"unknown runs {set(ids)-known}"
PYV
for rid in "${IDS[@]}"; do [[ ! -e "$OUT_ROOT/$rid" ]] || { echo "REFUSE: $OUT_ROOT/$rid already exists (one submission per run)" >&2; exit 4; }; done
ACTIVE_NAMES="$(squeue -u "$USER" -h -r -o '%j' | grep -c '^piv_train' || true)"
if [[ -f "$OUT_ROOT/_ledger/assignment.json" ]]; then
  "$PY" - "$OUT_ROOT/_ledger/assignment.json" "${IDS[@]}" <<'PYA' || { echo "REFUSE: a run is already assigned to a pool" >&2; exit 7; }
import json, sys
a = json.load(open(sys.argv[1]))
assigned = {r for v in a.values() for r in v}
dup = assigned & set(sys.argv[2:])
assert not dup, f"already assigned: {dup}"
PYA
fi
NORMAL_BUSY="$(squeue -u "$USER" -h -r -t RUNNING,PENDING -p mit_normal_gpu -o '%b' | grep -ci gpu || true)"
if [[ "$POOL" == normal ]] && (( NORMAL_BUSY + CONC > 4 )); then echo "REFUSE: $NORMAL_BUSY normal GPU jobs + $CONC > 4" >&2; exit 7; fi
mkdir -p "$OUT_ROOT/slurm" "$OUT_ROOT/_ledger"
N=${#IDS[@]}
LIST="$(IFS=:; echo "${IDS[*]}")"
ARGS=(--parsable --account "$ACCT" --partition "$PART" --qos "$QOS" --time "$TLIM" --requeue --job-name="piv_train_$POOL"
      --array="0-$((N-1))%${CONC}" --output="$OUT_ROOT/slurm/slurm-%A_%a.out" --error="$OUT_ROOT/slurm/slurm-%A_%a.err"
      --export="ALL,S7_TIP=$TIP,EXPECT_SHA=$SHA,S7_WALL_HOURS=$WALL,S7_EXPECT_PARTITION=$PART,S7_EXPECT_ACCOUNT=$ACCT,S7_EXPECT_QOS=$QOS,S7_MATRIX=$TIP/$M,PIV_RUN_IDS=$LIST")
echo "[$(date -Is)] test-only: sbatch ${ARGS[*]} scripts/engaging/piv_train.sbatch"
sbatch --test-only "${ARGS[@]}" "$TIP/scripts/engaging/piv_train.sbatch" 2>&1 | grep -v '^$' || true
[[ "$DRY" == "1" ]] && { echo "dry run: not submitted"; exit 0; }
JID="$(sbatch "${ARGS[@]}" "$TIP/scripts/engaging/piv_train.sbatch")" || { echo "sbatch failed" >&2; exit 3; }
"$PY" - "$OUT_ROOT/_ledger/assignment.json" "$POOL" "$JID" "$LIST" "$PART" "$ACCT" "$QOS" "$TLIM" "$WALL" <<'PYW'
import json, os, sys
p, pool, jid, lst, part, acct, qos, tlim, wall = sys.argv[1:]
a = json.load(open(p)) if os.path.exists(p) else {}
a[pool] = list(a.get(pool, [])) + lst.split(":")
a.setdefault("_arrays", {})[f"{pool}:{jid}"] = {"runs": lst.split(":"), "partition": part, "account": acct, "qos": qos, "time": tlim, "wall_hours": float(wall)}
json.dump(a, open(p, "w"), indent=1, sort_keys=True)
PYW
cat > "$OUT_ROOT/_ledger/submission_${JID}.json" <<JSON
{"array_job_id": "$JID", "submitted_at": "$(date -Is)", "git_sha": "$SHA", "tip": "$TIP", "pool": "$POOL", "partition": "$PART",
 "account": "$ACCT", "qos": "$QOS", "time": "$TLIM", "wall_hours_per_attempt": $WALL, "array": "0-$((N-1))%${CONC}", "runs": "${IDS[*]}",
 "gres": "gpu:l40s:1", "cpus": 8, "mem": "64G"}
JSON
echo "submitted $POOL array $JID: ${IDS[*]}"
