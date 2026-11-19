#!/bin/bash
# S6 offline eval + snapshot-swap (no training). HOME tip only; do not use ORCD pool.
#SBATCH --job-name=s6_sparse_eval
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail

TIP="${TIP:-$HOME/projects/official_thesis_s5}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/official_thesis_s5_runs}"
# One Slurm job per S5 run_id (set ONLY_RUN_ID). Separate OUT_ROOT avoids races.
if [[ -z "${ONLY_RUN_ID:-}" ]]; then
  echo "Refusing: set ONLY_RUN_ID to one S5 run directory name" >&2
  exit 78
fi
OUT_ROOT="${OUT_ROOT:-$HOME/official_thesis_s6_results/$ONLY_RUN_ID}"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$TIP/src"

cd "$TIP"
mkdir -p "$OUT_ROOT"
JOB_OUT="$OUT_ROOT/slurm_${SLURM_JOB_ID}"
mkdir -p "$JOB_OUT"

{
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "TIP=$TIP"
  echo "RUNS_ROOT=$RUNS_ROOT"
  echo "OUT_ROOT=$OUT_ROOT"
  echo "ONLY_RUN_ID=$ONLY_RUN_ID"
  echo "GIT_COMMIT_FILE=$(cat .git_commit 2>/dev/null || true)"
  date -Is
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
} | tee "$JOB_OUT/slurm_context.txt"

# Optional: CHECKPOINTS="last" to skip re-eval of completed best coverages.
# Swap always uses best.pt unless SKIP_SWAP=1.
EXTRA_ARGS=()
if [[ -n "${CHECKPOINTS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=(--checkpoints ${CHECKPOINTS})
fi
if [[ "${SKIP_SWAP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-swap)
fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-existing)
fi
if [[ "${SKIP_FIGURES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-figures)
fi

"$PY" scripts/17_s6_sparse_eval_and_swap.py \
  --tip "$TIP" \
  --runs-root "$RUNS_ROOT" \
  --out-root "$OUT_ROOT" \
  --cache-root "$TIP/data/processed/month2_dense_unet_cache" \
  --mask-root "$TIP/data/processed/month2_sparse_eval_masks" \
  --stats "$TIP/data/manifests/train_norm_stats_month2_fit.json" \
  --baseline-summary "$TIP/results/tables/month2_baseline_summary.csv" \
  --u5-swap-pairs "$TIP/results/month2_dense_unet/u5_pon_seed0_snapshot_control_v1/swap_pairs.json" \
  --device cuda \
  --batch-size 8 \
  --expect-git-sha "38ac7fe40178982ac1bd33c5d5602a5e7d928d3b" \
  --only-run-id "$ONLY_RUN_ID" \
  "${EXTRA_ARGS[@]}" \
  | tee "$JOB_OUT/stdout.log"

echo "S6_DONE out=$OUT_ROOT job_log=$JOB_OUT"
