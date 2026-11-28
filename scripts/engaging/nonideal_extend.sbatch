#!/bin/bash
# The single authorised continuation (initial 1000 -> 1200 epochs) of specialists that reached max epochs without plateau.
# Same task runner, own last.pt only, state/scheduler/early-stop rules unchanged. Index -> run via NI_RUN_IDS.
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --open-mode=append
set -uo pipefail
IDX="${SLURM_ARRAY_TASK_ID:?array task id required}"
: "${NI_RUN_IDS:?}" "${S7_EXTEND_N_EPOCHS:?}"
IFS=':' read -r -a LIST <<< "$NI_RUN_IDS"
export S7_TASK="${LIST[$IDX]:?no run for array index $IDX}"
export S7_MODE=campaign S7_AUTH_FLAG=--authorize-scientific-nonideal
exec bash "${S7_TIP:?S7_TIP required}/scripts/engaging/s7_run_task.sh"
