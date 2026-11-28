#!/bin/bash
# Specialist training array: partition/account/QoS/time/array/output are passed explicitly by nonideal_train_submit.sh;
# the task runner checks them against what Slurm granted. Resume/continuation: s7_run_task.sh (own last.pt only).
# SLURM_ARRAY_TASK_ID indexes the pool's run list (NI_RUN_IDS, colon separated), never two pools at once.
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --open-mode=append
set -uo pipefail
IDX="${SLURM_ARRAY_TASK_ID:?array task id required}"
: "${NI_RUN_IDS:?NI_RUN_IDS required}"
IFS=':' read -r -a LIST <<< "$NI_RUN_IDS"
export S7_TASK="${LIST[$IDX]:?no run for array index $IDX}"
export S7_MODE=campaign S7_AUTH_FLAG=--authorize-scientific-nonideal
exec bash "${S7_TIP:?S7_TIP required}/scripts/engaging/s7_run_task.sh"
