#!/bin/bash
# Submit the single CPU graphics job for the 30 specialists, with every path explicit and verified.
# Usage: nonideal_graphics_submit.sh <graphics_checkout> <expected_graphics_sha> <campaign_dest>
# Reads only frozen artefacts under <campaign_dest>. Does not train, infer or touch the numeric package.
set -euo pipefail
ROOT="$(cd "${1:?graphics checkout}" && pwd -P)"
EXPECT="${2:?expected graphics sha}"
DEST="$(cd "${3:?campaign dest}" && pwd -P)"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECT" ]] || { echo "REFUSE: $ROOT HEAD != $EXPECT" >&2; exit 5; }
[[ -z "$(git -C "$ROOT" status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout $ROOT" >&2; exit 5; }
for f in freeze.json package/COMPLETE.json package/fields_index.json package/tables/per_run_scenario.csv; do
  [[ -f "$DEST/$f" ]] || { echo "REFUSE: missing $DEST/$f" >&2; exit 5; }
done
[[ -d "$DEST/eval" ]] || { echo "REFUSE: missing $DEST/eval" >&2; exit 5; }
mkdir -p "$DEST/slurm"
sbatch --parsable --chdir="$ROOT" \
  --output="$DEST/slurm/graphics-%j.out" --error="$DEST/slurm/graphics-%j.err" \
  --export="ALL,NI_GRAPHICS_ROOT=$ROOT,EXPECT_GRAPHICS_SHA=$EXPECT,NI_DEST=$DEST,NI_EVAL_ROOT=$DEST/eval,NI_PACKAGE=$DEST/package,NI_FIELDS_INDEX=$DEST/package/fields_index.json,NI_FREEZE=$DEST/freeze.json" \
  "$ROOT/scripts/engaging/nonideal_graphics.sbatch"
