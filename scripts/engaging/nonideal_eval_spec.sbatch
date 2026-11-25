#!/bin/bash
# Specialist evaluation task: one array task = one specialist run (best and last, six cells). Index -> run via NI_RUN_IDS.
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --open-mode=append
set -euo pipefail
: "${EXPECT_SHA:?}" "${NI_TIP:?}" "${NI_EVAL_ROOT:?}" "${NI_FREEZE:?}" "${NI_RUN_IDS:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH; export PYTHONPATH="$NI_TIP/src"; cd "$NI_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
IFS=':' read -r -a LIST <<< "$NI_RUN_IDS"
RUN_ID="${LIST[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date -Is)] spec-eval host=$(hostname) job=${SLURM_JOB_ID:-na} task=${SLURM_ARRAY_TASK_ID} run=$RUN_ID sha=$EXPECT_SHA" >&2
[[ ! -e "$NI_EVAL_ROOT/$RUN_ID" ]] || { echo "REFUSE: output exists" >&2; exit 4; }
mkdir -p "$NI_EVAL_ROOT"
"$PY" scripts/68_nonideal_specialist_eval.py --run-id "$RUN_ID" --out-root "$NI_EVAL_ROOT" --freeze "$NI_FREEZE" --eval-git-sha "$EXPECT_SHA" --device cuda --mode full
