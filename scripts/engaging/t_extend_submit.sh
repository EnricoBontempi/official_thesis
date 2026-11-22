#!/bin/bash
# Submit the AUTHORISED continuation of selected assisted runs that exhausted their initial tranche (no plateau) as ONE normal-pool array.
#   t_extend_submit.sh --tip DIR --sha SHA --array 7,10,11 --n-epochs 1100 --wall-hours H [--matrix PATH] [--out-root DIR] [--prev-matrix-sha HEX] [--dry-run]
# The planner (s7_plan.py plan-extend) must say "extend" for every run: only completed/reached_max_epochs runs with a complete last.pt of
# that very epoch and no earlier extension qualify; nothing is edited by hand. Refuses an index that already has an active t_assist job.
set -uo pipefail
TIP=""; SHA=""; ARR=""; NEP=""; WALL=""; MATRIX=""; OUTROOT=""; PREV=""; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tip) TIP="$2"; shift 2;; --sha) SHA="$2"; shift 2;; --array) ARR="$2"; shift 2;; --n-epochs) NEP="$2"; shift 2;;
    --wall-hours) WALL="$2"; shift 2;; --matrix) MATRIX="$2"; shift 2;; --out-root) OUTROOT="$2"; shift 2;; --prev-matrix-sha) PREV="$2"; shift 2;;
    --dry-run) DRY=1; shift;; *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
for v in TIP SHA ARR NEP WALL; do [[ -n "${!v}" ]] || { echo "missing --${v,,}" >&2; exit 2; }; done
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
cd "$TIP" || exit 2
[[ "$(git rev-parse HEAD)" == "$SHA" ]] || { echo "REFUSE: HEAD != $SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
M="${MATRIX:-$TIP/configs/experiments/month2_t_assisted_matrix.json}"
OUT_ROOT="${OUTROOT:-$("$PY" -c "import json;print(json.load(open('$M'))['out_root'])")}"
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
ACTIVE="$(squeue -u "$USER" -h -r -o '%i|%j|%T' | grep 't_assist' || true)"
for i in ${ARR//,/ }; do grep -E "_${i}\|" <<<"$ACTIVE" >/dev/null && { echo "REFUSE: index $i already has an active t_assist job" >&2; exit 7; }; done
for rid in "${IDS[@]}"; do
  PLAN="$(PYTHONPATH="$TIP/src" "$PY" scripts/engaging/s7_plan.py plan-extend "$OUT_ROOT" "$rid" "0" 0 "$NEP")" || { echo "planner failed for $rid" >&2; exit 8; }
  echo "planner $rid: $PLAN" | cut -c1-260
  [[ "$PLAN" == *'"action": "extend"'* ]] || { echo "REFUSE: $rid is not extendable" >&2; exit 4; }
done
RUNNING_NORMAL="$(squeue -u "$USER" -h -r -t RUNNING -p mit_normal_gpu -o '%b' | grep -ci gpu || true)"
CONC=$(( 4 - RUNNING_NORMAL )); (( CONC > 3 )) && CONC=3; (( CONC < 1 )) && CONC=1
N=${#IDS[@]}; (( CONC > N )) && CONC=$N
mkdir -p "$OUT_ROOT/slurm" "$OUT_ROOT/_ledger"
ARGS=(--parsable --account mit_amf_advanced_gpu --partition mit_normal_gpu --qos mit_amf_advanced_gpu --time 03:00:00 --requeue --job-name=t_assist_ext
      --array="${ARR}%${CONC}" --output="$OUT_ROOT/slurm/slurm-ext-%A_%a.out" --error="$OUT_ROOT/slurm/slurm-ext-%A_%a.err"
      --export="ALL,S7_TIP=$TIP,EXPECT_SHA=$SHA,S7_WALL_HOURS=$WALL,S7_EXPECT_PARTITION=mit_normal_gpu,S7_EXPECT_ACCOUNT=mit_amf_advanced_gpu,S7_EXPECT_QOS=mit_amf_advanced_gpu,S7_MATRIX=$M,S7_EXTRA_TRAIN_ARGS=--assisted,S7_EXTEND_N_EPOCHS=$NEP,S7_PREV_EVAL_MATRIX_SHA=$PREV,S7_OUT_ROOT=$OUT_ROOT,S7_EXTEND_SCOPE=assisted runs ${IDS[*]// /+} only; n_epochs up to $NEP; state/scheduler/early-stop/K stream kept")
sbatch --test-only "${ARGS[@]}" "$TIP/scripts/engaging/t_train.sbatch" 2>&1 | grep -v '^$' || true
[[ "$DRY" == "1" ]] && { echo "dry run: not submitted"; exit 0; }
JID="$(sbatch "${ARGS[@]}" "$TIP/scripts/engaging/t_train.sbatch")" || { echo "sbatch failed" >&2; exit 3; }
echo "{\"job_id\": \"$JID\", \"submitted_at\": \"$(date -Is)\", \"git_sha\": \"$SHA\", \"runs\": \"${IDS[*]}\", \"array\": \"${ARR}%${CONC}\", \"new_n_epochs\": $NEP, \"previous_eval_matrix_sha256\": \"$PREV\"}" > "$OUT_ROOT/_ledger/submission_extension_$JID.json"
echo "submitted extension array $JID: ${IDS[*]} (concurrency $CONC)"
