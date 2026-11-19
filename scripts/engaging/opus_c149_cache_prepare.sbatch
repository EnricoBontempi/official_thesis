#!/bin/bash
# Opus C149 cache verify on Advanced CPU. Fails closed on any gate failure.
# Rebuild from RAW only if RAW is set; otherwise verify-only on staged cache.
#SBATCH --job-name=opus_c149_cache
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail

if [[ "${AUTHORIZE_OPUS_C149_SUBMIT:-0}" != "1" ]]; then
  echo "Refusing: AUTHORIZE_OPUS_C149_SUBMIT=1 required" >&2
  exit 78
fi

TIP="${TIP:-$HOME/projects/official_thesis_opus_c149}"
S5_TIP="${S5_TIP:-$HOME/projects/official_thesis_s5}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
CACHE="${CACHE:-/orcd/pool/005/bontempi/official_thesis_data/month2_dense_unet_cache_c149}"
HIST="${HIST:-$S5_TIP/data/processed/month2_dense_unet_cache}"
RAW="${RAW:-}"
OUT_LOG="${OUT_LOG:-$HOME/official_thesis_opus_c149_runs/c149_cache_prepare}"
export PYTHONUNBUFFERED=1
unset PYTHONPATH
export PYTHONPATH="$TIP/src"
cd "$TIP"
mkdir -p "$OUT_LOG" "$CACHE"

{
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "TIP=$TIP"
  echo "CACHE=$CACHE"
  echo "HIST=$HIST"
  echo "RAW=${RAW:-VERIFY_ONLY}"
  date -Is
} | tee "$OUT_LOG/slurm_context.txt"

if [[ -n "${RAW}" ]]; then
  "$PY" scripts/18_build_opus_c149_cache.py \
    --out-root "$CACHE" \
    --raw-dir "$RAW" \
    --historical-cache "$HIST" \
    --overwrite \
    --report "$CACHE/c149_build_verify_report.json" \
    | tee "$OUT_LOG/stdout.log"
else
  "$PY" scripts/18_build_opus_c149_cache.py \
    --out-root "$CACHE" \
    --historical-cache "$HIST" \
    --verify-only \
    --report "$CACHE/c149_build_verify_report.json" \
    | tee "$OUT_LOG/stdout.log"
fi

export CACHE
"$PY" - <<PY
from pathlib import Path
import json
from combustion_recon.training.dense_cache import compute_cache_payload_hashes
cache=Path("$CACHE")
h=compute_cache_payload_hashes(cache)
(cache/"payload_hashes.json").write_text(json.dumps(h, indent=2)+"\n")
print("WROTE", cache/"payload_hashes.json")
PY

echo "C149_CACHE_PREPARE_DONE"
