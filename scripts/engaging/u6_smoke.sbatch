#!/bin/bash
# U6 GPU smoke + short profiling (diagnostic weights only; p_off path).
# Does NOT start the scientific U6 run. Account/partition/QoS on sbatch CLI.
set -euo pipefail

TIP="${TIP:-$HOME/orcd/pool/official_thesis}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
CACHE="${CACHE:-$HOME/orcd/pool/official_thesis_data/month2_dense_unet_cache}"
HASHES="${HASHES:-$HOME/orcd/pool/official_thesis_data/cache_payload_hashes.json}"
OUT="${OUT:-$HOME/orcd/pool/official_thesis_runs/u6_smoke_profile_$SLURM_JOB_ID}"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$TIP/src"

mkdir -p "$OUT"
cd "$TIP"

echo "host=$(hostname) job=$SLURM_JOB_ID cuda_visible=${CUDA_VISIBLE_DEVICES:-}"
echo "TIP=$TIP PY=$PY CACHE=$CACHE OUT=$OUT"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"

"$PY" scripts/engaging/u6_smoke_profile.py \
  --cache-root "$CACHE" \
  --expect-cache-hashes "$HASHES" \
  --out-dir "$OUT" \
  --profile-epochs 2 \
  --max-compute-minutes 10

echo "U6_SMOKE_PROFILE_DONE out=$OUT"
