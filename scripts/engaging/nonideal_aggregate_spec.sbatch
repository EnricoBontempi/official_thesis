#!/bin/bash
# CPU: aggregate the specialist evaluation (refuses incomplete matrices) -> package + FINAL_SUMMARY.md, else BLOCK_REPORT.md.
#SBATCH --job-name=ni_aggregate_spec
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
set -uo pipefail
: "${EXPECT_SHA:?}" "${NI_TIP:?}" "${NI_DEST:?}" "${NI_EVAL_ROOT:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg; unset PYTHONPATH; export PYTHONPATH="$NI_TIP/src"; cd "$NI_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
"$PY" scripts/69_nonideal_specialist_aggregate.py --freeze "$NI_DEST/freeze.json" --eval-root "$NI_EVAL_ROOT" --dest "$NI_DEST/package" \
  && cp "$NI_DEST/package/FINAL_SUMMARY.md" "$NI_DEST/FINAL_SUMMARY.md" \
  || { [[ -f "$NI_DEST/BLOCK_REPORT.md" ]] || cp "$NI_DEST/package_BLOCK_REPORT.md" "$NI_DEST/BLOCK_REPORT.md" 2>/dev/null || echo "aggregation refused or failed; see logs in $NI_DEST/slurm" > "$NI_DEST/BLOCK_REPORT.md"; exit 1; }
