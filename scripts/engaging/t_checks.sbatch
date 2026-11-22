#!/bin/bash
# Assisted smoke checks only (isolation, p, tokens, overfit, learned T use, six-K eval). One family per job. Non-scientific.
#SBATCH --job-name=t_checks
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
set -uo pipefail
: "${EXPECT_SHA:?}" "${T_TIP:?}" "${FAMILY:?}" "${T_CHECK_OUT:?}"
cd "$T_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
PYTHONPATH="$T_TIP/src" PYTHONDONTWRITEBYTECODE=1 ~/envs/official_thesis/bin/python scripts/52_t_smoke_checks.py --family "$FAMILY" \
  --cache-root /home/bontempi/official_thesis_data/month2_dense_unet_cache_c149 \
  --mask-root /home/bontempi/projects/official_thesis_s5/data/processed/month2_sparse_eval_masks --out "$T_CHECK_OUT"
