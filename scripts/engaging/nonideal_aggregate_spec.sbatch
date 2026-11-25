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
  || { [[ -f "$NI_DEST/BLOCK_REPORT.md" ]] || echo "aggregation refused or failed; see logs in $NI_DEST/slurm" > "$NI_DEST/BLOCK_REPORT.md"; AGG_RC=1; }

# Optional post-processing hook (graphics): if $NI_DEST/post_hook.path names an sbatch script, it is submitted once the aggregation
# has finished (success or block). No duplicated chain: the hook only reads the frozen outputs.
if [[ -f "$NI_DEST/post_hook.path" ]]; then
  HOOK="$(cat "$NI_DEST/post_hook.path")"
  sbatch --parsable --export="ALL,NI_DEST=$NI_DEST,NI_EVAL_ROOT=$NI_EVAL_ROOT,NI_PACKAGE=$NI_DEST/package,NI_FIELDS_INDEX=$NI_DEST/package/fields_index.json,NI_FREEZE=$NI_DEST/freeze.json" "$HOOK" \
    | tee -a "$NI_DEST/post_hook_submitted.txt"
fi
exit "${AGG_RC:-0}"
