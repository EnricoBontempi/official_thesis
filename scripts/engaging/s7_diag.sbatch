#!/bin/bash
# S7 small GPU diagnostic (batch dependence of the best checkpoints). Not a scientific output.
#SBATCH --job-name=s7_batch_diag
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
set -euo pipefail
: "${EXPECT_SHA:?}" "${S7_TIP:?}" "${S7_DIAG_OUT:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
export PYTHONPATH="$S7_TIP/src"
cd "$S7_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
[[ ! -e "$S7_DIAG_OUT" ]] || { echo "REFUSE: output exists" >&2; exit 4; }
"$PY" scripts/45_s7_batch_diag.py --matrix configs/experiments/month2_s7_c149_eval_matrix.json --out-root "$S7_DIAG_OUT" --eval-git-sha "$EXPECT_SHA"
