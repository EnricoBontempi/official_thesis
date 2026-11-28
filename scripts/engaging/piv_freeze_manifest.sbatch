#!/bin/bash
# Mission 02 (Virtual PIV) — Stage 1 (CPU): audit training runs and emit status/freeze report.
# Dependencies: --dependency=afterany:<TRAIN_ARRAY_IDS>
# Required env vars (supplied by Sonnet's launcher):
#   PIV_TIP         Clean repository checkout path
#   EXPECT_SHA      Expected git HEAD SHA of PIV_TIP
#   PIV_MANIFEST    Path to piv_postproc_manifest_v1 JSON/YAML
#   PIV_DEST        Output directory for status report & post-processing package
#SBATCH --job-name=piv_freeze_audit
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
set -euo pipefail

: "${PIV_TIP:?}" "${EXPECT_SHA:?}" "${PIV_MANIFEST:?}" "${PIV_DEST:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
export PYTHONPATH="$PIV_TIP/src"
cd "$PIV_TIP"

[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }

mkdir -p "$PIV_DEST/slurm"
echo "[$(date -Is)] piv_freeze_audit host=$(hostname) job=${SLURM_JOB_ID:-na} sha=$EXPECT_SHA manifest=$PIV_MANIFEST" >&2
"$PY" scripts/piv_postproc/audit_and_freeze.py --manifest "$PIV_MANIFEST" --dest "$PIV_DEST"
