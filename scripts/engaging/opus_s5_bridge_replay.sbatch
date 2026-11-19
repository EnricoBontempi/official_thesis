#!/bin/bash
# Inference-only replay of historical S5 U-Net best.pt at 5%.
# Does not train and does not write into the S5 run directories.
#SBATCH --job-name=opus_s5_bridge_replay
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/bontempi/official_thesis_opus_bridge_runs/slurm-%j.out
#SBATCH --error=/home/bontempi/official_thesis_opus_bridge_runs/slurm-%j.err
set -euo pipefail

if [[ "${AUTHORIZE_OPUS_S5_BRIDGE_REPLAY:-0}" != "1" ]]; then
  echo "Refusing: set AUTHORIZE_OPUS_S5_BRIDGE_REPLAY=1" >&2
  exit 78
fi

TIP="${TIP:-$HOME/projects/official_thesis_opus_bridge}"
S5_TIP="${S5_TIP:-$HOME/projects/official_thesis_s5}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
RUN_ROOT="${RUN_ROOT:-$HOME/official_thesis_s5_runs}"
CACHE="${CACHE:-$S5_TIP/data/processed/month2_dense_unet_cache}"
HASHES="${HASHES:-$S5_TIP/data/manifests/month2_dense_unet_cache_payload_hashes_s5_homecopy.json}"
MASKS="${MASKS:-$S5_TIP/data/processed/month2_sparse_eval_masks}"
STATS="${STATS:-$S5_TIP/data/manifests/train_norm_stats_month2_fit.json}"
OUT_ROOT="${OUT_ROOT:-$HOME/official_thesis_opus_bridge_runs}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$TIP/src"
cd "$TIP"

SHA="$(git rev-parse HEAD)"
echo "GIT_SHA=$SHA"
echo "TIP=$TIP"
date -Is
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"

"$PY" scripts/19_opus_s5_bridge_replay.py \
  --expected "$TIP/results/month2/opus_data_study/bridge_s5/expected_unet_best_5pct.json" \
  --run-root "$RUN_ROOT" \
  --cache-root "$CACHE" \
  --cache-hashes "$HASHES" \
  --mask-root "$MASKS" \
  --stats "$STATS" \
  --out "$OUT_ROOT/replay_unet_best_5pct.json" \
  --device cuda

echo "OPUS_S5_BRIDGE_REPLAY_DONE"
