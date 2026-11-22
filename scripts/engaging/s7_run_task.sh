#!/bin/bash
# S7 task runner: one run of the S7 C149 matrix, fresh or resumed after preemption.
# Called by s7_train.sbatch (campaign) and by the smoke wrapper. Not an sbatch script.
#
# Contract (see src/combustion_recon/training/s7_launch.py):
#   * a new run never overwrites an existing directory;
#   * a requeued attempt of the SAME Slurm job resumes only from the run's own
#     checkpoints/last.pt (complete epoch boundary); never from a partial;
#   * if no complete checkpoint exists the restart is recorded as a fresh restart;
#   * every phase is logged with a timestamp; access errors fail explicitly.
set -uo pipefail

ts() { date -Is; }
log() { echo "[$(ts)] S7 $*"; }
die() { local code=$1; shift; log "REFUSE($code): $*"; exit "$code"; }

: "${S7_TIP:?S7_TIP (frozen checkout) required}"
: "${EXPECT_SHA:?EXPECT_SHA required}"
: "${S7_TASK:?S7_TASK (array index or run id) required}"
MODE="${S7_MODE:-campaign}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$S7_TIP/src"
cd "$S7_TIP" || die 2 "cannot cd to $S7_TIP"

JOB_ID="${SLURM_JOB_ID:-nojob}"
RESTART="${SLURM_RESTART_COUNT:-0}"
[[ -z "$RESTART" ]] && RESTART=0
HELPER="$S7_TIP/scripts/engaging/s7_plan.py"
MATRIX_JSON="${S7_MATRIX:-$S7_TIP/configs/experiments/month2_s7_c149_matrix.json}"

log "task start host=$(hostname) job=$JOB_ID array=${SLURM_ARRAY_JOB_ID:-na}_${SLURM_ARRAY_TASK_ID:-na} restart=$RESTART mode=$MODE"

# ---- identity of the executed code -------------------------------------------------
SHA_HEAD="$(git rev-parse HEAD)"
[[ "$SHA_HEAD" == "$EXPECT_SHA" ]] || die 5 "HEAD $SHA_HEAD != EXPECT_SHA $EXPECT_SHA"
if [[ -f .git_commit ]]; then
  [[ "$(tr -d '[:space:]' < .git_commit)" == "$SHA_HEAD" ]] || die 5 ".git_commit disagrees with HEAD"
fi
DIRTY="$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)"
[[ -z "$DIRTY" ]] || die 5 "checkout is not clean: $DIRTY"
log "checkout ok sha=$SHA_HEAD"

# ---- declared resources must be the ones actually granted ---------------------------
if [[ "$MODE" == "campaign" ]]; then
  for pair in "PARTITION:${SLURM_JOB_PARTITION:-}" "ACCOUNT:${SLURM_JOB_ACCOUNT:-}" "QOS:${SLURM_JOB_QOS:-}"; do
    key="${pair%%:*}"; got="${pair#*:}"; want_var="S7_EXPECT_${key}"
    [[ -n "${!want_var:-}" ]] || die 6 "$want_var not set"
    [[ "$got" == "${!want_var}" ]] || die 6 "granted $key=$got but submission declared ${!want_var}"
  done
  : "${S7_WALL_HOURS:?S7_WALL_HOURS (per-attempt internal cap) required}"
fi

# ---- row, paths, access preflight (only what this task reads) ----------------------
eval "$("$PY" "$HELPER" row "$MATRIX_JSON" "$S7_TASK")" || die 3 "matrix row for $S7_TASK not found"
OUT_ROOT="${S7_OUT_ROOT:-$MATRIX_OUT_ROOT}"
if [[ "$MODE" == "smoke" && "$OUT_ROOT" != *smoke* ]]; then die 4 "smoke OUT_ROOT must contain 'smoke': $OUT_ROOT"; fi
OUT="$OUT_ROOT/$RUN_ID"
log "run=$RUN_ID family=$FAMILY p_on=$P_ON seed=$SEED out=$OUT"

while read -r P; do
  if ! timeout 60 stat -c '%s' "$P" > /dev/null 2>&1; then
    # Fail fast and loudly: a hung automount must not silently burn GPU hours.
    die 6 "input not reachable within 60 s from $(hostname): $P"
  fi
done < <("$PY" "$HELPER" paths "$MATRIX_JSON" "$S7_TIP" "$RUN_ID")
log "input access ok"

"$PY" - <<'PYGPU' || die 7 "no usable CUDA device"
import torch, sys
ok = torch.cuda.is_available()
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", ok,
      torch.cuda.get_device_name(0) if ok else "no-gpu", flush=True)
sys.exit(0 if ok else 1)
PYGPU
GPU_NAME="$("$PY" -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null | tr ' ' '_')"

# ---- attempt plan ------------------------------------------------------------------
mkdir -p "$OUT_ROOT"
EXTEND_N="${S7_EXTEND_N_EPOCHS:-}"
if [[ -n "$EXTEND_N" && "$RESTART" == "0" ]]; then
  # Authorised continuation of a run that reached its initial tranche (never a plateau).
  PLAN="$("$PY" "$HELPER" plan-extend "$OUT_ROOT" "$RUN_ID" "$JOB_ID" "$RESTART" "$EXTEND_N")" || die 8 "planner crashed"
else
  PLAN="$("$PY" "$HELPER" plan "$OUT_ROOT" "$RUN_ID" "$JOB_ID" "$RESTART")" || die 8 "planner crashed"
fi
ACTION="$("$PY" -c 'import json,sys;print(json.loads(sys.argv[1])["action"])' "$PLAN")"
REASON="$("$PY" -c 'import json,sys;print(json.loads(sys.argv[1])["reason"])' "$PLAN")"
RESUME_FROM="$("$PY" -c 'import json,sys;print(json.loads(sys.argv[1])["resume_from"] or "")' "$PLAN")"
log "plan action=$ACTION reason=\"$REASON\" resume_from=${RESUME_FROM:-none}"
log "plan json $PLAN"
case "$ACTION" in
  refuse)
    "$PY" "$HELPER" ledger "$OUT_ROOT" "$RUN_ID" event=refused "slurm_job_id=$JOB_ID" "restart_count=$RESTART" "reason=$REASON" || true
    exit 4 ;;
  already_terminal)
    "$PY" "$HELPER" ledger "$OUT_ROOT" "$RUN_ID" event=already_terminal "slurm_job_id=$JOB_ID" "restart_count=$RESTART" || true
    log "run already terminal; nothing to do"
    exit 0 ;;
  fresh|fresh_restart_no_checkpoint|resume|extend) ;;
  *) die 8 "unknown plan action $ACTION" ;;
esac

if [[ "$ACTION" == "resume" || "$ACTION" == "extend" ]]; then
  [[ -d "$OUT" ]] || die 4 "$ACTION planned but $OUT missing"
else
  [[ ! -e "$OUT" ]] || die 4 "fresh start but $OUT exists"
  mkdir -p "$OUT"
fi
CTX="$OUT/slurm_context.txt"
[[ "$ACTION" == "resume" ]] && CTX="$OUT/slurm_context_attempt_restart${RESTART}.txt"
[[ "$ACTION" == "extend" ]] && CTX="$OUT/slurm_context_attempt_extension_job${JOB_ID}.txt"
{
  echo "SLURM_JOB_ID=$JOB_ID"; echo "SLURM_RESTART_COUNT=$RESTART"
  echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"; echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
  echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"; echo "SLURM_JOB_ACCOUNT=${SLURM_JOB_ACCOUNT:-}"
  echo "SLURM_JOB_QOS=${SLURM_JOB_QOS:-}"; echo "NODE=$(hostname)"; echo "GPU=$GPU_NAME"
  echo "GIT_SHA=$SHA_HEAD"; echo "CONFIG=$CONFIG"; echo "RUN_ID=$RUN_ID"; echo "PLAN_ACTION=$ACTION"
  echo "WALL_HOURS_PER_ATTEMPT=${S7_WALL_HOURS:-smoke}"; echo "MODE=$MODE"; date -Is
} | tee "$CTX"

if [[ "$ACTION" == "extend" ]]; then
  PREV_SHA="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("git_sha",""))' "$OUT/run_identity.json")"
  PREV_EP="$("$PY" -c 'import json,sys;print(json.loads(sys.argv[1])["details"]["prev_n_epochs"])' "$PLAN")"
  "$PY" "$HELPER" preserve "$OUT_ROOT" "$RUN_ID" "tranche1_epoch${PREV_EP}" \
    "reason=authorised continuation past the initial n_epochs tranche (reached_max_epochs, not a plateau)" \
    "authorised_by=Enrico Bontempi" "authorised_scope=${S7_EXTEND_SCOPE:-this run only, n_epochs up to $EXTEND_N, state/scheduler/early-stop kept}" "previous_eval_matrix_sha256=${S7_PREV_EVAL_MATRIX_SHA:-}" \
    "previous_git_sha=$PREV_SHA" "extension_git_sha=$SHA_HEAD" "extension_job_id=$JOB_ID" "new_n_epochs=$EXTEND_N" \
    || die 8 "could not preserve the first tranche"
  "$PY" "$HELPER" ledger "$OUT_ROOT" "$RUN_ID" event=extension_start "slurm_job_id=$JOB_ID" "new_n_epochs=$EXTEND_N" \
    "previous_git_sha=$PREV_SHA" "extension_git_sha=$SHA_HEAD" || die 8 "ledger write failed"
fi

"$PY" "$HELPER" ledger "$OUT_ROOT" "$RUN_ID" event=attempt_start "slurm_job_id=$JOB_ID" "restart_count=$RESTART" \
  "action=$ACTION" "node=$(hostname)" "gpu=$GPU_NAME" "git_sha=$SHA_HEAD" "mode=$MODE" \
  "partition=${SLURM_JOB_PARTITION:-}" "account=${SLURM_JOB_ACCOUNT:-}" "qos=${SLURM_JOB_QOS:-}" "reason=$REASON" || die 8 "ledger write failed"

# ---- training command ----------------------------------------------------------------
CMD=("$PY" scripts/16_train_sparse_scientific.py
  --config "$CONFIG" --authorize-scientific-s7
  --cache-root "$CACHE" --expect-cache-hashes "$CACHE_HASHES"
  --mask-root "$MASKS"
  --expect-mask-manifest-sha256 "$EXPECT_MASK_FILE" --expect-mask-aggregate-sha256 "$EXPECT_MASK_AGG"
  --expect-stats-sha256 "$EXPECT_STATS" --expect-cache-aggregate-sha256 "$EXPECT_CACHE_AGG"
  --out-root "$OUT_ROOT" --run-id "$RUN_ID" --device cuda --git-sha "$SHA_HEAD")
if [[ "$MODE" == "smoke" ]]; then
  # shellcheck disable=SC2206
  CMD+=(--smoke ${S7_SMOKE_ARGS:-})
  [[ -n "${S7_SMOKE_FAULT_AT:-}" ]] && CMD+=(--smoke-fault-at "$S7_SMOKE_FAULT_AT")
  [[ -n "${S7_SMOKE_WALL_HOURS:-}" ]] && CMD+=(--wall-hours "$S7_SMOKE_WALL_HOURS")
else
  CMD+=(--wall-hours "$S7_WALL_HOURS")
fi
[[ -n "${S7_EXTRA_TRAIN_ARGS:-}" ]] && CMD+=(${S7_EXTRA_TRAIN_ARGS})
[[ "$ACTION" == "resume" || "$ACTION" == "extend" ]] && CMD+=(--resume-from "$RESUME_FROM")
# Every attempt of an extension job (including wall-time requeues) must carry the same n_epochs.
[[ -n "$EXTEND_N" ]] && CMD+=(--allow-extend-max-epochs --extend-n-epochs "$EXTEND_N")
log "exec: ${CMD[*]}"

child=0
on_signal() {
  local sig="$1"
  local state; state="$("$PY" "$HELPER" last "$OUT" 2>/dev/null || echo '{}')"
  log "SIGNAL $sig received (preemption, requeue or time limit). Last recorded state: $state"
  "$PY" "$HELPER" ledger "$OUT_ROOT" "$RUN_ID" event=signal "signal=$sig" "slurm_job_id=$JOB_ID" "state=$state" || true
  if (( child > 0 )); then kill -TERM "$child" 2>/dev/null || true; fi
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal USR1' USR1

"${CMD[@]}" > >(tee -a "$OUT/stdout.log") 2>&1 &
child=$!
rc=0
while true; do
  wait "$child"; rc=$?
  kill -0 "$child" 2>/dev/null || break
done
sleep 2   # let tee drain
STATE="$("$PY" "$HELPER" last "$OUT" 2>/dev/null || echo '{}')"
log "training process exit rc=$rc state=$STATE"
"$PY" "$HELPER" ledger "$OUT_ROOT" "$RUN_ID" event=attempt_end "slurm_job_id=$JOB_ID" "restart_count=$RESTART" "rc=$rc" "state=$STATE" || true

if [[ "$rc" == "0" && "$STATE" == *'"summary_status": "stopped_wall_time"'* ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" && ( "$MODE" == "campaign" || "${S7_SMOKE_REAL_REQUEUE:-0}" == "1" ) ]]; then
    log "internal wall-time cap reached; scheduling a registered continuation (scontrol requeue); not a plateau"
    "$PY" "$HELPER" ledger "$OUT_ROOT" "$RUN_ID" event=continuation_requeue "slurm_job_id=$JOB_ID" || true
    REQ_ID="$SLURM_JOB_ID"
    [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]] && REQ_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    log "scontrol requeue $REQ_ID"
    scontrol requeue "$REQ_ID" && exit 0
    log "scontrol requeue failed"; exit 9
  fi
fi
log "task end rc=$rc"
exit "$rc"
