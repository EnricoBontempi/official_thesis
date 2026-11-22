#!/bin/bash
# T-ASSISTED SMOKE (non-scientific): real C149 cache, real masks, real entrypoint and launcher logic,
# tiny epochs. One family per job (FAMILY=unet|glu). Output under *smoke* only; never feeds the campaign.
# Resources: explicit on the sbatch command line (the approved Advanced normal L40S combination),
# because the account/partition for the campaign is decided separately.
#SBATCH --job-name=t_smoke
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH --output=/home/bontempi/official_thesis_t_smoke_runs/slurm/slurm-%j.out
#SBATCH --error=/home/bontempi/official_thesis_t_smoke_runs/slurm/slurm-%j.err
set -uo pipefail
: "${S7_TIP:?}"; : "${EXPECT_SHA:?}"; : "${FAMILY:?unet|glu}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
REAL_JOB="${SLURM_JOB_ID:-local}"
SM="${SMOKE_ROOT:-$HOME/official_thesis_t_smoke_runs}/job${REAL_JOB}_${FAMILY}"
mkdir -p "$SM"
echo "[$(date -Is)] S7-SMOKE start family=$FAMILY job=$REAL_JOB root=$SM host=$(hostname)"
export S7_TIP EXPECT_SHA PY S7_MODE=smoke
export S7_MATRIX="$S7_TIP/configs/experiments/month2_t_assisted_matrix.json" S7_EXTRA_TRAIN_ARGS="--assisted"
export S7_SMOKE_ARGS="--smoke-presentations 48 --smoke-epochs 4 --smoke-eval-every 2 --smoke-eval-frames 6"
ON="TA-${FAMILY}-on-s1"; OFF="TA-${FAMILY}-off-s1"
run() { # name run_id [fault] [restart]
  local name=$1 rid=$2 fault=${3:-} restart=${4:-0}
  S7_OUT_ROOT="$SM/$name" S7_TASK="$rid" S7_SMOKE_FAULT_AT="$fault" \
    SLURM_JOB_ID="smoke_${REAL_JOB}_${name}" SLURM_RESTART_COUNT="$restart" \
    bash "$S7_TIP/scripts/engaging/s7_run_task.sh"
}
set +e
RESTART="${SLURM_RESTART_COUNT:-0}"; [[ -z "$RESTART" ]] && RESTART=0
if [[ "$RESTART" == "0" ]]; then
echo "[$(date -Is)] S7-SMOKE step A: uninterrupted p_on"
run A "$ON"; rcA=$?
echo "[$(date -Is)] S7-SMOKE step A2: second uninterrupted p_on (nondeterminism control)"
run A2 "$ON"; rcA2=$?
echo "[$(date -Is)] S7-SMOKE step B1: p_on killed (os._exit 137) inside epoch index 2 after 1 batch"
run B "$ON" "2:1" 0; rcB1=$?
echo "[$(date -Is)] S7-SMOKE step B2: requeue attempt (SLURM_RESTART_COUNT=1) must resume from last.pt"
run B "$ON" "" 1; rcB2=$?
echo "[$(date -Is)] S7-SMOKE step B3: a further requeue after completion must be a no-op"
run B "$ON" "" 2; rcB3=$?
echo "[$(date -Is)] S7-SMOKE step B4: same run dir without requeue must be refused"
run B "$ON" "" 0; rcB4=$?
echo "[$(date -Is)] S7-SMOKE step C: uninterrupted p_off"
run C "$OFF"; rcC=$?
echo "[$(date -Is)] S7-SMOKE rc A=$rcA A2=$rcA2 B1=$rcB1 B2=$rcB2 B3=$rcB3 B4=$rcB4 C=$rcC"
# D: internal wall-time cap -> registered continuation. Long smoke (400 epochs) stopped by a 14 s cap.
export S7_SMOKE_ARGS="--smoke-presentations 48 --smoke-epochs 400 --smoke-eval-every 10 --smoke-eval-frames 6"
export S7_SMOKE_WALL_HOURS=0.004
if [[ "${REAL_REQUEUE_TEST:-0}" == "1" ]]; then
  echo "[$(date -Is)] S7-SMOKE step D1 (REAL): wall-time cap, then the runner issues scontrol requeue on this job"
  S7_OUT_ROOT="$SM/D" S7_TASK="$ON" S7_SMOKE_REAL_REQUEUE=1 bash "$S7_TIP/scripts/engaging/s7_run_task.sh"; rcD1=$?
  echo "[$(date -Is)] S7-SMOKE D1 returned rc=$rcD1; the job should now be requeued by Slurm"
  sleep 300
  echo "[$(date -Is)] S7-SMOKE ERROR: still running 300 s after requeue request" ; exit 1
else
  echo "[$(date -Is)] S7-SMOKE step D1 (emulated restart count): wall-time cap"
  run D "$ON" "" 0; rcD1=$?
  echo "[$(date -Is)] S7-SMOKE step D2: continuation attempt must resume from last.pt"
  run D "$ON" "" 1; rcD2=$?
  echo "[$(date -Is)] S7-SMOKE D rc=$rcD1/$rcD2"
fi
else
echo "[$(date -Is)] S7-SMOKE RESTARTED by Slurm (SLURM_RESTART_COUNT=$RESTART): step D2 real continuation"
export S7_SMOKE_ARGS="--smoke-presentations 48 --smoke-epochs 400 --smoke-eval-every 10 --smoke-eval-frames 6"
export S7_SMOKE_WALL_HOURS=0.004
S7_OUT_ROOT="$SM/D" S7_TASK="$ON" bash "$S7_TIP/scripts/engaging/s7_run_task.sh"; rcD2=$?
echo "[$(date -Is)] S7-SMOKE D2 (real requeue) rc=$rcD2"
fi
echo "[$(date -Is)] S7-SMOKE continuation report"
"$PY" "$S7_TIP/scripts/37_s7_smoke_continuation.py" --out-root "$SM/D" --run-id "$ON" \
  --real-requeue "${REAL_REQUEUE_TEST:-0}" --out "$SM/continuation_report.json"; rcCont=$?
echo "[$(date -Is)] S7-SMOKE compare continuous vs killed+resumed"
"$PY" "$S7_TIP/scripts/36_s7_smoke_compare.py" --continuous "$SM/A/$ON" --continuous2 "$SM/A2/$ON" \
  --resumed "$SM/B/$ON" --out "$SM/compare_resume.json"; rcCmp=$?
echo "[$(date -Is)] T-SMOKE assisted checks (isolation, p, tokens, learned T use, overfit, six-K eval)"
PYTHONPATH="$S7_TIP/src" "$PY" "$S7_TIP/scripts/52_t_smoke_checks.py" --family "$FAMILY" \
  --cache-root /home/bontempi/official_thesis_data/month2_dense_unet_cache_c149 \
  --mask-root /home/bontempi/projects/official_thesis_s5/data/processed/month2_sparse_eval_masks \
  --out "$SM/assisted_checks.json"; rcChk=$?
echo "[$(date -Is)] T-SMOKE end continuation_rc=$rcCont compare_rc=$rcCmp checks_rc=$rcChk"
