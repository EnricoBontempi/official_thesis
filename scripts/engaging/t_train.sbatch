#!/bin/bash
# Temperature-assisted training array (12-run matrix). Partition/account/QoS/time/array/output are passed explicitly
# by t_submit.sh; the task runner checks them against what Slurm granted. Resume/continuation: s7_run_task.sh.
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --open-mode=append
set -uo pipefail
export S7_TASK="${SLURM_ARRAY_TASK_ID:?array task id required}"
export S7_MODE=campaign
exec bash "${S7_TIP:?S7_TIP required}/scripts/engaging/s7_run_task.sh"
