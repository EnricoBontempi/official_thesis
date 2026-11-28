#!/bin/bash
# S7 C149 free campaign: ten new trainings as one array. Partition, account, QoS, time and the
# 4-GPU throttle are NOT hard-coded here: scripts/engaging/s7_submit.sh passes them explicitly
# and exports the same values as S7_EXPECT_*, which the task runner checks against what Slurm
# actually granted.
#SBATCH --job-name=s7_c149
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/bontempi/official_thesis_s7_c149_runs/slurm/slurm-%A_%a.out
#SBATCH --error=/home/bontempi/official_thesis_s7_c149_runs/slurm/slurm-%A_%a.err
set -uo pipefail
export S7_TASK="${SLURM_ARRAY_TASK_ID:?array task id required}"
export S7_MODE=campaign
exec bash "${S7_TIP:?S7_TIP required}/scripts/engaging/s7_run_task.sh"
