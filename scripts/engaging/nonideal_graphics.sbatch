#!/bin/bash
# CPU graphics for the 30 p_on specialists.
# Submitted by the aggregate job's post_hook, after that job has written the package or a block report.
# It does not train, infer, or wait on the training arrays. A partial matrix is refused.
#SBATCH --job-name=ni_graphics_spec
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
: "${NI_DEST:?}" "${NI_EVAL_ROOT:?}" "${NI_PACKAGE:?}" "${NI_FIELDS_INDEX:?}" "${NI_FREEZE:?}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
unset PYTHONPATH
export PYTHONPATH="$ROOT/src"
cd "$ROOT"
SHA="$(git rev-parse HEAD)"
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty graphics checkout" >&2; exit 5; }
echo "[$(date -Is)] graphics host=$(hostname) job=${SLURM_JOB_ID:-na} sha=$SHA dest=$NI_DEST/graphics" >&2
# This file is the graphics checkout. The training campaign's revision is a different repository and is not compared here.
"$PY" scripts/70_nonideal_specialist_graphics.py \
  --freeze "$NI_FREEZE" \
  --eval-root "$NI_EVAL_ROOT" \
  --package "$NI_PACKAGE" \
  --fields-index "$NI_FIELDS_INDEX" \
  --baseline-per-frame "${NI_BASELINE_PER_FRAME:-/home/bontempi/official_thesis_nonideal_b2_pkg/package/tables/per_frame_all.csv.gz}" \
  --baseline-eval-root "${NI_BASELINE_EVAL:-/home/bontempi/official_thesis_nonideal_b2_eval}" \
  --dest "$NI_DEST/graphics" \
  --git-sha "$SHA"
