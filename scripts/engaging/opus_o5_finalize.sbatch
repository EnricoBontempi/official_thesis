#!/bin/bash
# Final O5 gate on a CPU node. Submitted with --dependency=afterany on the O5 jobs.
# Builds the package only if all eight runs are valid; otherwise writes BLOCK_REPORT.md.
#SBATCH --job-name=opus_o5_final
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/bontempi/official_thesis_opus_o5_final/slurm-%j.out
#SBATCH --error=/home/bontempi/official_thesis_opus_o5_final/slurm-%j.err
set -euo pipefail
TIP="${TIP:?TIP required}"
EXPECT_SHA="${EXPECT_SHA:?EXPECT_SHA required}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
OUT_ROOT="${OUT_ROOT:-$HOME/official_thesis_opus_o5_attempt2_runs}"
DEST_BASE="${DEST_BASE:-$HOME/official_thesis_opus_o5_final}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
unset PYTHONPATH
export PYTHONPATH="$TIP/src"
cd "$TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != EXPECT_SHA" >&2; exit 5; }
DIRTY="$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)"
[[ -z "$DIRTY" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
DEST="$DEST_BASE/package_${SLURM_JOB_ID}"
"$PY" scripts/30_opus_o5_finalize.py \
  --out-root "$OUT_ROOT" \
  --dest "$DEST" \
  --eval-sha 918286809d3cbb33d155b70a6e03a9b898bcab9e \
  --eval-sha 29b94345f0cac3f725662ad687a15ff841c7c730 \
  --jobs "${O5_JOBS:-29728465,29732630}" \
  --finalize-sha "$EXPECT_SHA"
