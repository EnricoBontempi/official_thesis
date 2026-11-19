#!/bin/bash
# Opus wave-2 C149 launcher — separate tip from wave-1; no warm start.
#SBATCH --job-name=opus_c149_off_s0
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail

if [[ "${AUTHORIZE_OPUS_C149_SUBMIT:-0}" != "1" ]]; then
  echo "Refusing: set AUTHORIZE_OPUS_C149_SUBMIT=1 for authorized C149 wave-2 only." >&2
  exit 78
fi

# Inlined env: sbatch $0 is a spool copy.
TIP="${TIP:-$HOME/projects/official_thesis_opus_c149}"
S5_TIP="${S5_TIP:-$HOME/projects/official_thesis_s5}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
CACHE="${CACHE:-/orcd/pool/005/bontempi/official_thesis_data/month2_dense_unet_cache_c149}"
HASHES="${HASHES:-$CACHE/payload_hashes.json}"
MASKS="${MASKS:-$S5_TIP/data/processed/month2_sparse_eval_masks}"
VERIFY_REPORT="${VERIFY_REPORT:-$CACHE/c149_build_verify_report.json}"
OUT_ROOT="${OUT_ROOT:-$HOME/official_thesis_opus_c149_runs}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$TIP/src"
cd "$TIP"
SHA="$(cat .git_commit 2>/dev/null || git rev-parse HEAD)"
CFG="configs/experiments/opus_O-C149-off-s0.yaml"
RUN_ID="O-C149-off-s0"
OUT="$OUT_ROOT/$RUN_ID"
mkdir -p "$OUT"

if [[ ! -f "$VERIFY_REPORT" ]]; then
  echo "REFUSE: missing C149 verify report $VERIFY_REPORT" >&2
  exit 3
fi
"$PY" - <<PY
import json
from pathlib import Path
rep=json.loads(Path("$VERIFY_REPORT").read_text())
man=json.loads(Path("$CACHE/manifest.json").read_text())
assert rep.get("status")=="ok", rep
assert man.get("cache_id")=="month2_dense_unet_cache_opus_c149_v1", man.get("cache_id")
assert man.get("counts",{}).get("train_fit")==2980, man.get("counts")
print("C149_CACHE_GATE_OK", man["counts"])
PY

{
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "SLURM_JOB_NAME=$SLURM_JOB_NAME"
  echo "SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION"
  echo "SLURM_JOB_ACCOUNT=$SLURM_JOB_ACCOUNT"
  echo "SLURM_JOB_QOS=$SLURM_JOB_QOS"
  echo "GIT_SHA=$SHA"
  echo "CONFIG=$CFG"
  echo "RUN_ID=$RUN_ID"
  echo "CACHE=$CACHE"
  echo "HASHES=$HASHES"
  echo "MASKS=$MASKS"
  echo "TIP=$TIP"
  echo "NOTE=opus C149; scientific epoch 760/95; wall_cap_internal=5.5h under Slurm 6h"
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

echo "OPUS_C149_DONE run=$OUT"
