#!/bin/bash
# Assisted aggregation (CPU only). Refuses an incomplete or inconsistent evaluation itself.
#SBATCH --job-name=t_aggregate
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
: "${EXPECT_SHA:?}" "${T_TIP:?}" "${T_EVAL_MATRIX:?}" "${T_EVAL_OUT:?}" "${T_PACKAGE_DEST:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
unset PYTHONPATH
export PYTHONPATH="$T_TIP/src"
cd "$T_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
ARGS=(scripts/56_t_aggregate.py --matrix "$T_EVAL_MATRIX" --eval-root "$T_EVAL_OUT" --dest "$T_PACKAGE_DEST")
[[ -n "${T_TRAIN_JOBS:-}" ]] && ARGS+=(--train-jobs "${T_TRAIN_JOBS//:/,}")
[[ -n "${T_EVAL_JOBS:-}" ]] && ARGS+=(--eval-jobs "$T_EVAL_JOBS")
[[ "${T_AGG_SMOKE:-0}" == "1" ]] && ARGS+=(--smoke)
"$PY" "${ARGS[@]}"
