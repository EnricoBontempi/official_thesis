#!/bin/bash
# S5 screening launcher — Enrico-authorized neural freeze (four runs only).
# No afterok. No reuse of S4 profiling weights. No automatic requeue.
#SBATCH --job-name=s5_unet_pon
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
# Frozen ops from S4 profiling (still requires AUTHORIZE_S5_SUBMIT=1 at submit time):
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail

if [[ "${AUTHORIZE_S5_SUBMIT:-0}" != "1" ]]; then
  echo "Refusing: set AUTHORIZE_S5_SUBMIT=1 only after Enrico authorizes S5 (no auto-launch)." >&2
  exit 78
fi

source "$(dirname "$0")/_s5_train_common.inc.sh"
CFG="configs/experiments/s5_sparse_unet_pon_seed0.yaml"
RUN_ID="s5_sparse_unet_pon_seed0"
OUT="$OUT_ROOT/$RUN_ID"
mkdir -p "$OUT"

{
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "SLURM_JOB_NAME=$SLURM_JOB_NAME"
  echo "SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION"
  echo "SLURM_JOB_ACCOUNT=$SLURM_JOB_ACCOUNT"
  echo "SLURM_JOB_QOS=$SLURM_JOB_QOS"
  echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "GIT_SHA=$SHA"
  echo "CONFIG=$CFG"
  echo "RUN_ID=$RUN_ID"
  echo "NOTE=neural protocol frozen; S5 authorized; wall_cap_internal=5.5h under Slurm 6h"
  date -Is
} | tee "$OUT/slurm_context.txt"

"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"

# Requires YAML scientific_training_authorized:true AND CLI flag.
# microbatch null (frozen). Internal wall guard 5.5h < Slurm 6h.
"$PY" scripts/16_train_sparse_scientific.py \
  --config "$CFG" \
  --authorize-scientific-s5 \
  --cache-root "$CACHE" \
  --expect-cache-hashes "$HASHES" \
  --mask-root "$MASKS" \
  --out-root "$OUT_ROOT" \
  --run-id "$RUN_ID" \
  --device cuda \
  --git-sha "$SHA" \
  | tee "$OUT/stdout.log"

echo "S5_DONE run=$OUT"
