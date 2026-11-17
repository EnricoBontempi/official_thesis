#!/bin/bash
# Single authorized U4 scientific run: p_on seed 0, max 1000 epochs.
# No automatic requeue/restart on error or timeout. No mid-epoch resume.
set -euo pipefail

TIP="${TIP:-$HOME/orcd/pool/official_thesis}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
CACHE="${CACHE:-$HOME/orcd/pool/official_thesis_data/month2_dense_unet_cache}"
HASHES="${HASHES:-$HOME/orcd/pool/official_thesis_data/cache_payload_hashes.json}"
OUT_ROOT="${OUT_ROOT:-$HOME/orcd/pool/official_thesis_runs}"
RUN_ID="${RUN_ID:-u4_pon_seed0_engaging_v1}"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$TIP/src"

mkdir -p "$OUT_ROOT"
cd "$TIP"

echo "host=$(hostname) job=$SLURM_JOB_ID cuda_visible=${CUDA_VISIBLE_DEVICES:-}"
echo "TIP=$TIP PY=$PY CACHE=$CACHE RUN_ID=$RUN_ID"
"$PY" -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu'); print('git', open('.git_commit').read().strip() if __import__('pathlib').Path('.git_commit').exists() else 'n/a')"

# Record scheduler context
{
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "SLURM_JOB_NAME=$SLURM_JOB_NAME"
  echo "SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION"
  echo "SLURM_JOB_ACCOUNT=$SLURM_JOB_ACCOUNT"
  echo "SLURM_JOB_QOS=$SLURM_JOB_QOS"
  echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  date -Is
} | tee "$OUT_ROOT/${RUN_ID}_slurm_context.txt"

"$PY" scripts/07_train_reconstructor.py train-u4 \
  --authorize-scientific-u4 \
  --config configs/experiments/month2_dense_unet_u4_candidate.yaml \
  --dense-config configs/experiments/month2_dense_unet.yaml \
  --cache-root "$CACHE" \
  --expect-cache-hashes "$HASHES" \
  --out-root "$OUT_ROOT" \
  --run-id "$RUN_ID" \
  --device cuda \
  --wall-hours 6

echo "U4_TRAIN_DONE run=$OUT_ROOT/$RUN_ID"
