#!/bin/bash
# CPU: aggregation + figures + summary, only if VERIFY_OK exists (afterany of the verify job); otherwise BLOCK_REPORT stays.
#SBATCH --job-name=ni_aggregate
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -uo pipefail
: "${EXPECT_SHA:?}" "${NI_TIP:?}" "${NI_SPEC:?}" "${NI_OUT_ROOT:?}" "${NI_DEST:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
unset PYTHONPATH; export PYTHONPATH="$NI_TIP/src"; cd "$NI_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
if [[ ! -f "$NI_DEST/verify/VERIFY_OK.json" ]]; then
  echo "No VERIFY_OK: see $NI_DEST/verify/BLOCK_REPORT.md. Outputs stay diagnostic." >&2
  cp "$NI_DEST/verify/BLOCK_REPORT.md" "$NI_DEST/BLOCK_REPORT.md" 2>/dev/null || echo "verification job produced no report" > "$NI_DEST/BLOCK_REPORT.md"
  exit 0
fi
"$PY" scripts/62_nonideal_aggregate.py --spec "$NI_SPEC" --eval-root "$NI_OUT_ROOT" --dest "$NI_DEST/package" --verify-dir "$NI_DEST/verify" \
  && cp "$NI_DEST/package/FINAL_SUMMARY.md" "$NI_DEST/FINAL_SUMMARY.md" \
  || { echo "aggregation failed" > "$NI_DEST/BLOCK_REPORT.md"; exit 1; }
