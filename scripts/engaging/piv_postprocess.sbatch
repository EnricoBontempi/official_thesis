#!/bin/bash
# Mission 02 (Virtual PIV) — Stage 3 (CPU): resumable aggregation, accounting, main figures & atlas.
# Dependencies: --dependency=afterany:<EVAL_ARRAY_JOB_IDS>
# Required env vars (supplied by Sonnet's launcher):
#   PIV_TIP         Clean repository checkout path
#   EXPECT_SHA      Expected git HEAD SHA of PIV_TIP
#   PIV_MANIFEST    Path to piv_postproc_manifest_v1 JSON/YAML
#   PIV_DEST        Output package directory
# Optional:
#   PIV_SACCT_START Start date for sacct -D export (YYYY-MM-DD); if set, exports fresh sacct snapshot
#   PIV_POST_FLAGS  Extra CLI flags (e.g. "--allow-partial" for monitoring; empty for strict final package)
#SBATCH --job-name=piv_postproc
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
set -euo pipefail

: "${PIV_TIP:?}" "${EXPECT_SHA:?}" "${PIV_MANIFEST:?}" "${PIV_DEST:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$PIV_TIP/src"
cd "$PIV_TIP"

[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }

mkdir -p "$PIV_DEST/slurm"
echo "[$(date -Is)] piv_postproc host=$(hostname) job=${SLURM_JOB_ID:-na} sha=$EXPECT_SHA dest=$PIV_DEST" >&2

# Optional: capture reproducible sacct -D snapshot (same method as Mission 00)
if [[ -n "${PIV_SACCT_START:-}" ]] && command -v sacct >/dev/null 2>&1; then
  mkdir -p "$PIV_DEST/accounting_raw"
  SACCT_OUT="$PIV_DEST/accounting_raw/sacct_${PIV_SACCT_START}_$(date +%Y-%m-%d).psv"
  sacct -D -u "${USER}" -S "$PIV_SACCT_START" -P \
    --format=JobID,JobName,Partition,QOS,Account,State,ExitCode,Submit,Start,End,ElapsedRaw,AllocCPUS,AllocTRES,NodeList,Timelimit,MaxRSS,TotalCPU \
    > "$SACCT_OUT" || echo "WARN: sacct export failed; continuing without fresh sacct" >&2
fi

POST_FLAGS="${PIV_POST_FLAGS:-}"
# shellcheck disable=SC2086
"$PY" scripts/piv_postproc/run_postprocessing.py \
  --manifest "$PIV_MANIFEST" \
  --dest "$PIV_DEST" \
  $POST_FLAGS
