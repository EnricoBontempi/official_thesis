#!/bin/bash
# PIV campaign: evaluation of best then last of one run per array task (resumable; last == best -> no second inference).
# Dependencies/partition/time passed on the sbatch command line. Env: PIV_TIP EXPECT_SHA PIV_RUN_IDS PIV_EVAL_ROOT PIV_MATRIX
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --open-mode=append
set -uo pipefail
cd "${PIV_TIP:?}" || exit 2
[[ "$(git rev-parse HEAD)" == "${EXPECT_SHA:?}" ]] || { echo "REFUSE sha"; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE dirty"; exit 5; }
IFS=':' read -r -a LIST <<< "${PIV_RUN_IDS:?}"
RUN="${LIST[${SLURM_ARRAY_TASK_ID:?}]:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONPATH="$PIV_TIP/src" PYTHONUNBUFFERED=1
TR="$($PY -c "import json;print(json.load(open('${PIV_MATRIX:?}'))['out_root'])")"
if [[ ! -f "$TR/$RUN/checkpoints/best.pt" ]]; then echo "[$(date -Is)] REFUSE: no best.pt for $RUN (training did not produce a checkpoint)"; exit 6; fi
echo "[$(date -Is)] piv_eval run=$RUN job=${SLURM_JOB_ID:-na}_${SLURM_ARRAY_TASK_ID:-na} restart=${SLURM_RESTART_COUNT:-0} host=$(hostname)"
rc=0
for KIND in best last; do
  $PY scripts/76_piv_eval.py --matrix "$PIV_MATRIX" --run-id "$RUN" --kind $KIND --eval-root "${PIV_EVAL_ROOT:?}" --eval-git-sha "$EXPECT_SHA" --eval-set "${PIV_EVAL_SET:-full}" --device cuda
  r=$?; echo "[$(date -Is)] piv_eval $RUN $KIND rc=$r"; [[ $r -ne 0 ]] && rc=$r
done
exit $rc
