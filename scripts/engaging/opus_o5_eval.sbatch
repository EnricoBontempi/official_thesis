#!/bin/bash
# Opus O5 final evaluation. Prepared, not submitted by this file.
# One array task per scientific-stream run. Historical S5 is not a task.
#SBATCH --job-name=opus_o5
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/home/bontempi/official_thesis_opus_o5_runs/slurm-%A_%a.out
#SBATCH --error=/home/bontempi/official_thesis_opus_o5_runs/slurm-%A_%a.err
set -euo pipefail

if [[ "${AUTHORIZE_OPUS_O5:-0}" != "1" ]]; then
  echo "Refusing: set AUTHORIZE_OPUS_O5=1 only when Enrico authorizes submission." >&2
  exit 78
fi
if [[ -z "${EXPECT_SHA:-}" ]]; then
  echo "Refusing: EXPECT_SHA is required." >&2
  exit 78
fi

TIP="${TIP:-$HOME/projects/official_thesis_opus_o5_v2}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
OUT_ROOT="${OUT_ROOT:-$HOME/official_thesis_opus_o5_runs}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$TIP/src"
cd "$TIP"

SHA_HEAD="$(git rev-parse HEAD)"
if [[ "$SHA_HEAD" != "$EXPECT_SHA" ]]; then
  echo "REFUSE: HEAD $SHA_HEAD != EXPECT_SHA $EXPECT_SHA" >&2
  exit 5
fi
if [[ -f .git_commit ]]; then
  SHA_FILE="$(tr -d '[:space:]' < .git_commit)"
  if [[ "$SHA_FILE" != "$SHA_HEAD" ]]; then
    echo "REFUSE: .git_commit $SHA_FILE != HEAD $SHA_HEAD" >&2
    exit 5
  fi
fi
DIRTY="$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)"
if [[ -n "$DIRTY" ]]; then
  echo "REFUSE: dirty O5 checkout" >&2
  printf '%s\n' "$DIRTY" >&2
  exit 5
fi

mapfile -t RUNS < <("$PY" - << 'PY'
import json
from pathlib import Path
matrix = json.loads(Path("configs/experiments/opus_o5_matrix.json").read_text())
for row in matrix["runs"]:
    print(row["run_id"])
PY
)
RUN_ID="${RUNS[$SLURM_ARRAY_TASK_ID]}"
OUT="$OUT_ROOT/$RUN_ID"
mkdir -p "$OUT_ROOT"
if [[ -e "$OUT" ]]; then
  echo "REFUSE: output already exists: $OUT" >&2
  exit 4
fi

"$PY" scripts/26_opus_o5_eval.py \
  --matrix configs/experiments/opus_o5_matrix.json \
  --run-id "$RUN_ID" \
  --out-root "$OUT_ROOT" \
  --eval-git-sha "$SHA_HEAD" \
  --device cuda
