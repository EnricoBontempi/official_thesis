#!/bin/bash
# S7 aggregation (CPU only): costs from Slurm accounting, then tables/figures/index.
# Submitted with --dependency=afterok on the full-evaluation arrays; refuses an incomplete matrix itself.
#SBATCH --job-name=s7_aggregate
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
: "${EXPECT_SHA:?}" "${S7_AGG_TIP:?}" "${S7_EVAL_ROOT:?}" "${S7_PACKAGE_DEST:?}" "${S7_EVAL_SHA:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$S7_AGG_TIP/src"
cd "$S7_AGG_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
M=configs/experiments/month2_s7_c149_eval_matrix.json
COSTS="${S7_PACKAGE_DEST}_costs"
"$PY" scripts/41_s7_costs.py --matrix "$M" --eval-root "$S7_EVAL_ROOT" --dest "$COSTS"
"$PY" scripts/40_s7_aggregate.py --matrix "$M" --eval-root "$S7_EVAL_ROOT" --dest "$S7_PACKAGE_DEST" \
  --eval-sha "$S7_EVAL_SHA" --costs-dir "$COSTS"
