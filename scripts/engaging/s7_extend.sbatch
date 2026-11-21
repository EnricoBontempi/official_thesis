#!/bin/bash
# S7: the one authorised continuation of S7-C149-glu-on-s1 past its initial 1000-epoch tranche.
# Submitted by scripts/engaging/s7_extend_submit.sh, which passes partition/account/QoS/time
# explicitly and exports the S7_EXPECT_* values that the task runner checks.
#SBATCH --job-name=s7_c149_ext
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/bontempi/official_thesis_s7_c149_runs/slurm/slurm-ext-%j.out
#SBATCH --error=/home/bontempi/official_thesis_s7_c149_runs/slurm/slurm-ext-%j.err
set -uo pipefail
export S7_TASK="${S7_EXTEND_RUN_ID:?S7_EXTEND_RUN_ID required}"
export S7_MODE=campaign
: "${S7_EXTEND_N_EPOCHS:?S7_EXTEND_N_EPOCHS required}"
exec bash "${S7_TIP:?S7_TIP required}/scripts/engaging/s7_run_task.sh"
