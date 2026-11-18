#!/bin/bash
# S4 diagnostic profiling: sparse U-Net (AMF normal GPU, prefer L40S).
# No automatic requeue. No S5 training.
#SBATCH --job-name=s4_prof_unet
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
# No automatic requeue/restart on this diagnostic job.
set -euo pipefail

TIP="${TIP:-$HOME/orcd/pool/official_thesis}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
CACHE="${CACHE:-$HOME/orcd/pool/official_thesis_data/month2_dense_unet_cache}"
HASHES="${HASHES:-$HOME/orcd/pool/official_thesis_data/cache_payload_hashes.json}"
MASKS="${MASKS:-$HOME/orcd/pool/official_thesis_data/month2_sparse_eval_masks}"
OUT_ROOT="${OUT_ROOT:-$HOME/orcd/pool/official_thesis_runs}"
OUT="${OUT:-$OUT_ROOT/s4_profile_sparse_unet_${SLURM_JOB_ID}}"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$TIP/src"

mkdir -p "$OUT"
cd "$TIP"
SHA="$(cat .git_commit 2>/dev/null || git rev-parse HEAD)"

{
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "SLURM_JOB_NAME=$SLURM_JOB_NAME"
  echo "SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION"
  echo "SLURM_JOB_ACCOUNT=$SLURM_JOB_ACCOUNT"
  echo "SLURM_JOB_QOS=$SLURM_JOB_QOS"
  echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "GIT_SHA=$SHA"
  date -Is
} | tee "$OUT/slurm_context.txt"

echo "host=$(hostname) job=$SLURM_JOB_ID"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"

"$PY" scripts/engaging/s4_sparse_profile.py \
  --family sparse_unet \
  --cache-root "$CACHE" \
  --expect-cache-hashes "$HASHES" \
  --mask-root "$MASKS" \
  --out-dir "$OUT" \
  --git-sha "$SHA" \
  | tee "$OUT/stdout.log"

echo "S4_PROFILE_UNET_DONE out=$OUT"
