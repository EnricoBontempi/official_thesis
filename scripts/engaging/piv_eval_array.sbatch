#!/bin/bash
# Mission 02 (Virtual PIV) — Stage 2 (GPU array): evaluate one run per array task (resumable).
# Dependencies: --dependency=afterok:<FREEZE_JOB_ID> (or afterany:<TRAIN_ARRAY_IDS>)
# Required env vars (supplied by Sonnet's launcher):
#   PIV_TIP         Clean repository checkout path
#   EXPECT_SHA      Expected git HEAD SHA of PIV_TIP
#   PIV_MANIFEST    Path to piv_postproc_manifest_v1 JSON/YAML
#   PIV_RUN_IDS     Colon-separated list of run_ids mapped by SLURM_ARRAY_TASK_ID
#   PIV_EVAL_CMD    Evaluation script relative path (e.g. Sonnet's GPU eval wrapper calling piv_eval.py)
# Optional:
#   PIV_EVAL_EXTRA_ARGS  Additional CLI flags (default: "--resume --device cuda")
#SBATCH --job-name=piv_eval_arr
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
set -euo pipefail

: "${PIV_TIP:?}" "${EXPECT_SHA:?}" "${PIV_MANIFEST:?}" "${PIV_RUN_IDS:?}" "${PIV_EVAL_CMD:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$PIV_TIP/src"
cd "$PIV_TIP"

[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }

IFS=':' read -r -a RUN_LIST <<< "$PIV_RUN_IDS"
RUN_ID="${RUN_LIST[$SLURM_ARRAY_TASK_ID]}"
EXTRA="${PIV_EVAL_EXTRA_ARGS:---resume --device cuda}"

echo "[$(date -Is)] piv_eval host=$(hostname) job=${SLURM_JOB_ID:-na} task=${SLURM_ARRAY_TASK_ID} run=$RUN_ID sha=$EXPECT_SHA" >&2
# shellcheck disable=SC2086
"$PY" "$PIV_EVAL_CMD" \
  --manifest "$PIV_MANIFEST" \
  --run-id "$RUN_ID" \
  --eval-git-sha "$EXPECT_SHA" \
  $EXTRA
