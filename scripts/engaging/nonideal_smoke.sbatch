#!/bin/bash
# Specialist SMOKE (non-scientific), ONE job per family (FAMILY=unet|glu); shared across all five regimes.
# real C149 cache, masks, entrypoint and launcher logic; tiny epochs; output under *smoke* only.
#   ORIG  original clean S7 run (same seed)                  -> reference for the identity check
#   R0S   specialist code path with nothing degraded         -> must reproduce ORIG (numerically)
#   N1 N2 G1 G1-eq N1G1  the five regimes                    -> finite, cardinalities, p_on, eval records
#   KILL  N1G1 killed inside epoch 2, then requeue-resume    -> resume from last complete epoch
#   T1/T2 N1G1 real size: 1 epoch + 1 full eval, and 2 epochs + 1 eval -> time per epoch and per eval
#SBATCH --job-name=ni_smoke
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
set -uo pipefail
: "${NI_TIP:?}"; : "${EXPECT_SHA:?}"; : "${FAMILY:?unet|glu}"; : "${NI_SMOKE_ROOT:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
JOB="${SLURM_JOB_ID:-local}"
SM="$NI_SMOKE_ROOT/job${JOB}_${FAMILY}"
mkdir -p "$SM"
if [[ "$FAMILY" == "unet" ]]; then SEED=1; else SEED=0; fi
echo "[$(date -Is)] NI-SMOKE start family=$FAMILY seed=$SEED job=$JOB root=$SM host=$(hostname) sha=$EXPECT_SHA"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv -l 5 > "$SM/gpu_mem.csv" 2>/dev/null &
SMI=$!
export S7_TIP="$NI_TIP" EXPECT_SHA PY S7_MODE=smoke
PROD="$NI_TIP/configs/experiments/nonideal_specialist_matrix.json"
SMK="$NI_TIP/configs/experiments/nonideal_specialist_smoke_matrix.json"
TINY="--smoke-presentations 48 --smoke-epochs 4 --smoke-eval-every 2 --smoke-eval-frames 6"
run() { # name task matrix auth smoke_args [fault] [restart]
  local name=$1 task=$2 matrix=$3 auth=$4 sargs=$5 fault=${6:-} restart=${7:-0}
  S7_OUT_ROOT="$SM/$name" S7_TASK="$task" S7_MATRIX="$matrix" S7_AUTH_FLAG="$auth" S7_SMOKE_ARGS="$sargs" S7_SMOKE_FAULT_AT="$fault" \
    SLURM_JOB_ID="smoke_${JOB}_${name}" SLURM_RESTART_COUNT="$restart" bash "$NI_TIP/scripts/engaging/s7_run_task.sh"
}
NIA=--authorize-scientific-nonideal
set +e
declare -A RC
ORIGID="S7-C149-${FAMILY/unet/unet}-on-s${SEED}"
run ORIG "$ORIGID" "$NI_TIP/configs/experiments/month2_s7_c149_matrix.json" --authorize-scientific-s7 "$TINY"; RC[ORIG]=$?
run R0S "NI-${FAMILY}-on-R0smoke-s${SEED}" "$SMK" "$NIA" "$TINY"; RC[R0S]=$?
for REG in N1 N2 G1 G1eq N1G1; do
  run "$REG" "NI-${FAMILY}-on-${REG}-s${SEED}" "$PROD" "$NIA" "$TINY"; RC[$REG]=$?
done
echo "[$(date -Is)] NI-SMOKE KILL: N1G1 killed inside epoch index 2 after 1 batch"
run KILL "NI-${FAMILY}-on-N1G1-s${SEED}" "$PROD" "$NIA" "$TINY" "2:1" 0; RC[KILL1]=$?
echo "[$(date -Is)] NI-SMOKE KILL: requeue attempt must resume from last.pt"
run KILL "NI-${FAMILY}-on-N1G1-s${SEED}" "$PROD" "$NIA" "$TINY" "" 1; RC[KILL2]=$?
echo "[$(date -Is)] NI-SMOKE TIMING: real-size epoch + full eval"
run T1 "NI-${FAMILY}-on-N1G1-s${SEED}" "$PROD" "$NIA" "--smoke-presentations 760 --smoke-epochs 1 --smoke-eval-every 1 --smoke-eval-frames 100000"; RC[T1]=$?
run T2 "NI-${FAMILY}-on-N1G1-s${SEED}" "$PROD" "$NIA" "--smoke-presentations 760 --smoke-epochs 2 --smoke-eval-every 2 --smoke-eval-frames 100000"; RC[T2]=$?
kill $SMI 2>/dev/null
echo "[$(date -Is)] NI-SMOKE rc: $(for k in "${!RC[@]}"; do echo -n "$k=${RC[$k]} "; done)"
echo "[$(date -Is)] NI-SMOKE compare continuous vs killed+resumed (N1G1)"
RID="NI-${FAMILY}-on-N1G1-s${SEED}"
"$PY" "$NI_TIP/scripts/36_s7_smoke_compare.py" --continuous "$SM/N1G1/$RID" --resumed "$SM/KILL/$RID" --out "$SM/compare_resume.json"
echo "[$(date -Is)] NI-SMOKE checks"
PYTHONPATH="$NI_TIP/src" "$PY" "$NI_TIP/scripts/65_nonideal_smoke_checks.py" --smoke-dir "$SM" --family "$FAMILY" --seed "$SEED" --out "$SM/smoke_checks.json"
echo "[$(date -Is)] NI-SMOKE done rc_checks=$?"
