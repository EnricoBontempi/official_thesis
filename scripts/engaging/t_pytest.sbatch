#!/bin/bash
# Temperature study: run pytest files on one L40S (tests are device-aware). Not a scientific job.
#SBATCH --job-name=t_pytest
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
set -uo pipefail
: "${EXPECT_SHA:?}" "${T_TIP:?}" "${T_TESTS:?}"
cd "$T_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
export PYTHONPATH="$T_TIP/src" PYTHONDONTWRITEBYTECODE=1
nvidia-smi -L
~/envs/official_thesis/bin/python -m pytest -q -p no:cacheprovider $T_TESTS
