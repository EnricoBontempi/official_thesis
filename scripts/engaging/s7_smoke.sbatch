#!/bin/bash
# S7 SMOKE (non-scientific): real C149 cache, real masks, real entrypoint and launcher logic,
# tiny epochs. One family per job (FAMILY=unet|glu). Output under *smoke* only; never feeds the campaign.
# Resources: explicit on the sbatch command line (the approved Advanced normal L40S combination),
# because the account/partition for the campaign is decided separately.
#SBATCH --job-name=s7_smoke
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH --output=/home/bontempi/official_thesis_s7_smoke_runs/slurm/slurm-%j.out
#SBATCH --error=/home/bontempi/official_thesis_s7_smoke_runs/slurm/slurm-%j.err
set -uo pipefail
: "${S7_TIP:?}"; : "${EXPECT_SHA:?}"; : "${FAMILY:?unet|glu}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
REAL_JOB="${SLURM_JOB_ID:-local}"
SM="${SMOKE_ROOT:-$HOME/official_thesis_s7_smoke_runs}/job${REAL_JOB}_${FAMILY}"
mkdir -p "$SM"
echo "[$(date -Is)] S7-SMOKE start family=$FAMILY job=$REAL_JOB root=$SM host=$(hostname)"
export S7_TIP EXPECT_SHA PY S7_MODE=smoke
export S7_SMOKE_ARGS="--smoke-presentations 48 --smoke-epochs 4 --smoke-eval-every 2 --smoke-eval-frames 6"
ON="S7-C149-${FAMILY}-on-s1"; OFF="S7-C149-${FAMILY}-off-s1"
run() { # name run_id [fault] [restart]
  local name=$1 rid=$2 fault=${3:-} restart=${4:-0}
  S7_OUT_ROOT="$SM/$name" S7_TASK="$rid" S7_SMOKE_FAULT_AT="$fault" \
    SLURM_JOB_ID="smoke_${REAL_JOB}_${name}" SLURM_RESTART_COUNT="$restart" \
    bash "$S7_TIP/scripts/engaging/s7_run_task.sh"
}
set +e
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

echo "[$(date -Is)] S7-SMOKE compare continuous vs killed+resumed"
"$PY" "$S7_TIP/scripts/36_s7_smoke_compare.py" --continuous "$SM/A/$ON" --continuous2 "$SM/A2/$ON" \
  --resumed "$SM/B/$ON" --out "$SM/compare_resume.json"; rcCmp=$?
echo "[$(date -Is)] S7-SMOKE checks (isolation, p behaviour, 1/5/20% evaluator, memory)"
PYTHONPATH="$S7_TIP/src" "$PY" "$S7_TIP/scripts/35_s7_smoke_checks.py" \
  --run-on "$SM/A/$ON" --run-off "$SM/C/$OFF" \
  --config-on "$S7_TIP/configs/experiments/s7/$ON.yaml" --config-off "$S7_TIP/configs/experiments/s7/$OFF.yaml" \
  --cache-root /orcd/pool/005/bontempi/official_thesis_data/month2_dense_unet_cache_c149 \
  --mask-root /home/bontempi/projects/official_thesis_s5/data/processed/month2_sparse_eval_masks \
  --out "$SM/smoke_checks.json"; rcChk=$?
echo "[$(date -Is)] S7-SMOKE end compare_rc=$rcCmp checks_rc=$rcChk"
