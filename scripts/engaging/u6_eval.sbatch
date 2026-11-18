#!/bin/bash
# U6 evaluation: p_off best vs p_on + snapshot-swap using U5 pairings (no training).
set -euo pipefail

TIP="${TIP:-$HOME/orcd/pool/official_thesis}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
CACHE="${CACHE:-$HOME/orcd/pool/official_thesis_data/month2_dense_unet_cache}"
HASHES="${HASHES:-$HOME/orcd/pool/official_thesis_data/cache_payload_hashes.json}"
U6_DIR="${U6_DIR:-$HOME/orcd/pool/official_thesis_runs/u6_poff_seed0_engaging_v1}"
U4_DIR="${U4_DIR:-$HOME/orcd/pool/official_thesis_runs/u4_pon_seed0_engaging_v1}"
U5_DIR="${U5_DIR:-$TIP/results/month2_dense_unet/u5_pon_seed0_snapshot_control_v1}"
CKPT_U6="${CKPT_U6:-$U6_DIR/checkpoints/best.pt}"
CKPT_U4="${CKPT_U4:-$U4_DIR/checkpoints/best.pt}"
EXPECT_U6_SHA="${EXPECT_U6_SHA:?set EXPECT_U6_SHA to best.pt sha256}"
OUT="${OUT:-$HOME/orcd/pool/official_thesis_runs/u6_poff_seed0_snapshot_control_v1}"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$TIP/src"

cd "$TIP"
echo "host=$(hostname) job=$SLURM_JOB_ID"
echo "TIP=$TIP CKPT_U6=$CKPT_U6 OUT=$OUT"

# Prefer tip-local result mirrors when present
if [ ! -f "$U5_DIR/swap_pairs.json" ]; then
  U5_DIR="$HOME/orcd/pool/official_thesis_runs/u5_pon_seed0_snapshot_control_v1"
fi
if [ ! -f "$TIP/results/month2_dense_unet/u4_pon_seed0_engaging_v1/selection.json" ]; then
  :
else
  U4_TIP="$TIP/results/month2_dense_unet/u4_pon_seed0_engaging_v1"
  # Keep Engaging checkpoint path; use tip tables for comparison metadata when available
  U4_META="$U4_TIP"
fi
U4_META="${U4_META:-$U4_DIR}"

"$PY" -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"

"$PY" scripts/10_u6_poff_eval_and_swap.py \
  --cache-root "$CACHE" \
  --expect-cache-hashes "$HASHES" \
  --u6-checkpoint "$CKPT_U6" \
  --expect-u6-ckpt-sha256 "$EXPECT_U6_SHA" \
  --u6-run-dir "$U6_DIR" \
  --u4-run-dir "$U4_META" \
  --u5-dir "$U5_DIR" \
  --u4-checkpoint "$CKPT_U4" \
  --out-dir "$OUT" \
  --device cuda \
  --batch-size 8

echo "U6_EVAL_DONE out=$OUT"
