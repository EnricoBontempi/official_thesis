#!/bin/bash
# Smoke follow-up: is a killed+resumed specialist run as close to the continuous one as two identical continuous runs are?
# Two independent kill+resume repetitions (K1, K2) are compared with each other and with the continuous run. Non-scientific.
#SBATCH --job-name=ni_smoke_resume
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --open-mode=append
set -uo pipefail
: "${NI_TIP:?}"; : "${EXPECT_SHA:?}"; : "${FAMILY:?}"; : "${NI_SMOKE_ROOT:?}"; : "${PREV_SMOKE_DIR:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
SM="$NI_SMOKE_ROOT/resume_job${SLURM_JOB_ID:-local}_${FAMILY}"; mkdir -p "$SM"
if [[ "$FAMILY" == "unet" ]]; then SEED=1; else SEED=0; fi
export S7_TIP="$NI_TIP" EXPECT_SHA PY S7_MODE=smoke
PROD="$NI_TIP/configs/experiments/nonideal_specialist_matrix.json"
TINY="--smoke-presentations 48 --smoke-epochs 4 --smoke-eval-every 2 --smoke-eval-frames 6"
RID="NI-${FAMILY}-on-N1G1-s${SEED}"
run() { S7_OUT_ROOT="$SM/$1" S7_TASK="$RID" S7_MATRIX="$PROD" S7_AUTH_FLAG=--authorize-scientific-nonideal S7_SMOKE_ARGS="$TINY" S7_SMOKE_FAULT_AT="$2" SLURM_JOB_ID="smoke_resume_${SLURM_JOB_ID:-x}_$1" SLURM_RESTART_COUNT="$3" bash "$NI_TIP/scripts/engaging/s7_run_task.sh"; }
set +e
for K in K1 K2; do run $K "2:1" 0; run $K "" 1; done
CONT="$PREV_SMOKE_DIR/N1G1/$RID"
"$PY" "$NI_TIP/scripts/36_s7_smoke_compare.py" --continuous "$SM/K1/$RID" --resumed "$SM/K2/$RID" --out "$SM/resumed_K1_vs_K2.json"
"$PY" "$NI_TIP/scripts/36_s7_smoke_compare.py" --continuous "$CONT" --resumed "$SM/K1/$RID" --out "$SM/continuous_vs_K1.json"
"$PY" "$NI_TIP/scripts/36_s7_smoke_compare.py" --continuous "$CONT" --resumed "$SM/K2/$RID" --out "$SM/continuous_vs_K2.json"
echo "RESUME DONE $SM"
