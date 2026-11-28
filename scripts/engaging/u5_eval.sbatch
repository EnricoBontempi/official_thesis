#!/bin/bash
# U5 snapshot-swap evaluation on frozen U4 best.pt (no training).
set -euo pipefail

TIP="${TIP:-$HOME/orcd/pool/official_thesis}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
CACHE="${CACHE:-$HOME/orcd/pool/official_thesis_data/month2_dense_unet_cache}"
HASHES="${HASHES:-$HOME/orcd/pool/official_thesis_data/cache_payload_hashes.json}"
CKPT="${CKPT:-$HOME/orcd/pool/official_thesis_runs/u4_pon_seed0_engaging_v1/checkpoints/best.pt}"
U4_DIR="${U4_DIR:-$TIP/results/month2_dense_unet/u4_pon_seed0_engaging_v1}"
OUT="${OUT:-$HOME/orcd/pool/official_thesis_runs/u5_pon_seed0_snapshot_control_v1}"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$TIP/src"

cd "$TIP"
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "TIP=$TIP CKPT=$CKPT OUT=$OUT"
"$PY" -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"

# Prefer local results path inside checkout for U4 tables if present; else Engaging run dir copy
if [ ! -f "$U4_DIR/eval_history.json" ]; then
  U4_DIR="$HOME/orcd/pool/official_thesis_runs/u4_pon_seed0_engaging_v1"
fi

"$PY" scripts/09_dense_snapshot_control.py \
  --cache-root "$CACHE" \
  --expect-cache-hashes "$HASHES" \
  --checkpoint "$CKPT" \
  --expect-ckpt-sha256 6d456abcb5653c16a7a90df40eda377caf42ffb94fd829a65f4c06be58a03634 \
  --u4-run-dir "$U4_DIR" \
  --out-dir "$OUT" \
  --device cuda \
  --batch-size 8

echo "U5_DONE out=$OUT"
