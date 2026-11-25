#!/bin/bash
# Submit the whole specialist campaign: normal array + preemptable array (disjoint run lists) + stage B behind both.
# Stage B (CPU) then submits the single continuation, stage C (freeze + evaluation) and the aggregation, all on their own.
#   nonideal_launch_campaign.sh --tip DIR --sha SHA --dest DIR --normal ID,... --preempt ID,... [--normal-time T --normal-wall H --preempt-wall H] [--dry-run]
set -uo pipefail
TIP=""; SHA=""; DEST=""; NRM=""; PRE=""; NT="12:00:00"; NW="11.5"; PW="40"; DRY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tip) TIP="$2"; shift 2;; --sha) SHA="$2"; shift 2;; --dest) DEST="$2"; shift 2;; --normal) NRM="$2"; shift 2;; --preempt) PRE="$2"; shift 2;;
    --normal-time) NT="$2"; shift 2;; --normal-wall) NW="$2"; shift 2;; --preempt-wall) PW="$2"; shift 2;; --dry-run) DRY="--dry-run"; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
for v in TIP SHA DEST NRM PRE; do [[ -n "${!v}" ]] || { echo "missing --${v,,}" >&2; exit 2; }; done
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
cd "$TIP" || exit 2
M=configs/experiments/nonideal_specialist_matrix.json
"$PY" - "$M" "$NRM" "$PRE" <<'PYC' || { echo "REFUSE: the two pools must partition the 30 runs exactly" >&2; exit 2; }
import json, sys
m = json.load(open(sys.argv[1])); ids = {r["run_id"] for r in m["new_runs"]}
a, b = sys.argv[2].split(","), sys.argv[3].split(",")
assert len(a) + len(b) == 30 and set(a) | set(b) == ids and not set(a) & set(b), "pools do not partition the 30 runs"
PYC
[[ ! -e "$DEST" || -n "$DRY" ]] || { echo "REFUSE: dest exists" >&2; exit 4; }
OUT_ROOT="$("$PY" -c "import json;print(json.load(open('$M'))['out_root'])")"
echo "[$(date -Is)] normal pool"
OUTN="$(scripts/engaging/nonideal_train_submit.sh --tip "$TIP" --sha "$SHA" --pool normal --runs "$NRM" --concurrency 4 --wall-hours "$NW" --time "$NT" $DRY 2>&1)"; echo "$OUTN" | tail -3
[[ -n "$DRY" ]] && { echo "dry run: stopping before the preemptable pool"; exit 0; }
JN="$(grep -o 'submitted normal array [0-9]*' <<<"$OUTN" | grep -o '[0-9]*$')"; [[ -n "$JN" ]] || { echo "normal submission failed" >&2; exit 3; }
echo "[$(date -Is)] preemptable pool"
OUTP="$(scripts/engaging/nonideal_train_submit.sh --tip "$TIP" --sha "$SHA" --pool preempt --runs "$PRE" --concurrency 4 --wall-hours "$PW" 2>&1)"; echo "$OUTP" | tail -3
JP="$(grep -o 'submitted preempt array [0-9]*' <<<"$OUTP" | grep -o '[0-9]*$')"; [[ -n "$JP" ]] || { echo "preempt submission failed (normal array $JN stays queued)" >&2; exit 3; }
mkdir -p "$DEST/slurm"
JB="$(sbatch --parsable --dependency=afterany:"$JN":"$JP" --output="$DEST/slurm/stage_b-%j.out" --error="$DEST/slurm/stage_b-%j.err" \
  --export="ALL,EXPECT_SHA=$SHA,NI_TIP=$TIP,NI_DEST=$DEST" "$TIP/scripts/engaging/nonideal_stage_b.sbatch")" || { echo "stage B submission failed" >&2; exit 3; }
cat > "$DEST/submission.json" <<JSON
{"git_sha": "$SHA", "tip": "$TIP", "submitted_at": "$(date -Is)", "normal_array": "$JN", "preempt_array": "$JP", "stage_b": "$JB", "dest": "$DEST",
 "train_out_root": "$OUT_ROOT", "normal_runs": "${NRM//,/ }", "preempt_runs": "${PRE//,/ }", "normal_time": "$NT", "normal_wall_hours": $NW, "preempt_wall_hours": $PW}
JSON
echo "campaign submitted: normal=$JN preempt=$JP stage_b=$JB"
