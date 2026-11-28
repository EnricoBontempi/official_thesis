#!/bin/bash
# Smoke follow-up: GPU run-to-run noise floor of the specialist path (two identical uninterrupted runs) and of the original path,
# to judge identity / resume differences against measured noise instead of an arbitrary tolerance. Non-scientific.
#SBATCH --job-name=ni_smoke_noise
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --open-mode=append
set -uo pipefail
: "${NI_TIP:?}"; : "${EXPECT_SHA:?}"; : "${FAMILY:?}"; : "${NI_SMOKE_ROOT:?}"; : "${PREV_SMOKE_DIR:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
SM="$NI_SMOKE_ROOT/noise_job${SLURM_JOB_ID:-local}_${FAMILY}"; mkdir -p "$SM"
if [[ "$FAMILY" == "unet" ]]; then SEED=1; else SEED=0; fi
export S7_TIP="$NI_TIP" EXPECT_SHA PY S7_MODE=smoke
PROD="$NI_TIP/configs/experiments/nonideal_specialist_matrix.json"
TINY="--smoke-presentations 48 --smoke-epochs 4 --smoke-eval-every 2 --smoke-eval-frames 6"
run() { S7_OUT_ROOT="$SM/$1" S7_TASK="$2" S7_MATRIX="$3" S7_AUTH_FLAG="$4" S7_SMOKE_ARGS="$TINY" S7_SMOKE_FAULT_AT="" SLURM_JOB_ID="smoke_noise_${SLURM_JOB_ID:-x}_$1" SLURM_RESTART_COUNT=0 bash "$NI_TIP/scripts/engaging/s7_run_task.sh"; }
set +e
run B "NI-${FAMILY}-on-N1G1-s${SEED}" "$PROD" --authorize-scientific-nonideal
run ORIG2 "S7-C149-${FAMILY}-on-s${SEED}" "$NI_TIP/configs/experiments/month2_s7_c149_matrix.json" --authorize-scientific-s7
"$PY" "$NI_TIP/scripts/36_s7_smoke_compare.py" --continuous "$PREV_SMOKE_DIR/N1G1/NI-${FAMILY}-on-N1G1-s${SEED}" --resumed "$SM/B/NI-${FAMILY}-on-N1G1-s${SEED}" --out "$SM/noise_specialist_A_vs_B.json"
"$PY" "$NI_TIP/scripts/36_s7_smoke_compare.py" --continuous "$PREV_SMOKE_DIR/ORIG/S7-C149-${FAMILY}-on-s${SEED}" --resumed "$SM/ORIG2/S7-C149-${FAMILY}-on-s${SEED}" --out "$SM/noise_original_A_vs_B.json"
"$PY" "$NI_TIP/scripts/36_s7_smoke_compare.py" --continuous "$PREV_SMOKE_DIR/ORIG/S7-C149-${FAMILY}-on-s${SEED}" --resumed "$PREV_SMOKE_DIR/R0S/NI-${FAMILY}-on-R0smoke-s${SEED}" --out "$SM/identity_ORIG_vs_R0S.json"
"$PY" "$NI_TIP/scripts/36_s7_smoke_compare.py" --continuous "$SM/ORIG2/S7-C149-${FAMILY}-on-s${SEED}" --resumed "$PREV_SMOKE_DIR/R0S/NI-${FAMILY}-on-R0smoke-s${SEED}" --out "$SM/identity_ORIG2_vs_R0S.json"
echo "NOISE DONE $SM"
