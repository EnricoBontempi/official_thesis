#!/bin/bash
# Opus wave-1 launcher — Enrico-authorized overnight matrix (four runs).
# No afterok. No S5 warm start. No automatic requeue / preemptable fallback.
#SBATCH --job-name=opus_b19_on_s0
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail

if [[ "${AUTHORIZE_OPUS_SUBMIT:-0}" != "1" ]]; then
  echo "Refusing: set AUTHORIZE_OPUS_SUBMIT=1 only for the authorized Opus wave-1 matrix." >&2
  exit 78
fi

# Common env inlined: under sbatch $0 is the Slurm spool copy, so a relative source fails.
TIP="${TIP:-$HOME/projects/official_thesis_opus}"
S5_TIP="${S5_TIP:-$HOME/projects/official_thesis_s5}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
CACHE="${CACHE:-$S5_TIP/data/processed/month2_dense_unet_cache}"
HASHES="${HASHES:-$S5_TIP/data/manifests/month2_dense_unet_cache_payload_hashes_s5_homecopy.json}"
MASKS="${MASKS:-$S5_TIP/data/processed/month2_sparse_eval_masks}"
OUT_ROOT="${OUT_ROOT:-$HOME/official_thesis_opus_runs}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$TIP/src"
cd "$TIP"
SHA="$(cat .git_commit 2>/dev/null || git rev-parse HEAD)"
CFG="configs/experiments/opus_O-B19-on-s0.yaml"
RUN_ID="O-B19-on-s0"
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
  echo "CACHE=$CACHE"
  echo "HASHES=$HASHES"
  echo "MASKS=$MASKS"
  echo "TIP=$TIP"
  echo "NOTE=opus wave-1; scientific epoch 760/95; wall_cap_internal=5.5h under Slurm 6h"
  date -Is
} | tee "$OUT/slurm_context.txt"

"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"

"$PY" scripts/16_train_sparse_scientific.py \
  --config "$CFG" \
  --authorize-scientific-opus \
  --cache-root "$CACHE" \
  --expect-cache-hashes "$HASHES" \
  --mask-root "$MASKS" \
  --out-root "$OUT_ROOT" \
  --run-id "$RUN_ID" \
  --device cuda \
  --git-sha "$SHA" \
  | tee "$OUT/stdout.log"

echo "OPUS_DONE run=$OUT"
