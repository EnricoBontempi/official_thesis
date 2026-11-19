#!/bin/bash
# GappyPOD nested 1/20%: CPU-only on HOME tip. No GPU. No pool. No S6 touch.
#SBATCH --job-name=gappy_amf_cpu
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
# No GPU / no --gres (CPU partition).
set -euo pipefail

TIP="${TIP:-$HOME/projects/official_thesis_s5}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
OUT_ROOT="${OUT_ROOT:-$HOME/official_thesis_gappypod_nested_1_20_amf_v2}"
HIST="${HIST:-$TIP/results/tables/month2_baseline_summary.csv}"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$TIP/src"

cd "$TIP"
mkdir -p "$OUT_ROOT/artifacts" "$OUT_ROOT/slurm_${SLURM_JOB_ID}"
LOGDIR="$OUT_ROOT/slurm_${SLURM_JOB_ID}"
GIT_SHA="$(cat .git_commit 2>/dev/null || echo unknown)"

{
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION"
  echo "SLURM_JOB_ACCOUNT=${SLURM_JOB_ACCOUNT:-unset}"
  echo "SLURM_JOB_QOS=${SLURM_JOB_QOS:-unset}"
  echo "TIP=$TIP"
  echo "OUT_ROOT=$OUT_ROOT"
  echo "GIT_SHA=$GIT_SHA"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "MEM_REQUEST=128G"
  date -Is
  "$PY" -c "import numpy,pandas,yaml; print('numpy', numpy.__version__, 'pandas', pandas.__version__)"
  free -h | head -2
} | tee "$LOGDIR/slurm_context.txt"

# Refuse GPU accidentally
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NOTE: nvidia-smi present on node; job requests zero GPU and must stay CPU-only" | tee -a "$LOGDIR/slurm_context.txt"
fi

"$PY" -u scripts/15_gappypod_nested_coverage_eval.py \
  --authorize-refit \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --gate-atol-rmse-K 0.05 \
  --equiv-sample 8 \
  --git-sha "$GIT_SHA" \
  --out-dir "$OUT_ROOT" \
  --historical-summary "$HIST" \
  --cache-root "$TIP/data/processed/month2_dense_unet_cache" \
  --mask-root "$TIP/data/processed/month2_sparse_eval_masks" \
  --stats "$TIP/data/manifests/train_norm_stats_month2_fit.json" \
  --config "$TIP/configs/experiments/month2_sparse_gappypod_nested.yaml" \
  2>&1 | tee "$LOGDIR/stdout.log"

echo "GAPPY_DONE out=$OUT_ROOT log=$LOGDIR/stdout.log"
