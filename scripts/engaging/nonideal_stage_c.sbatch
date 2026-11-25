#!/bin/bash
# Stage C (CPU): validate and freeze the 30 trainings (declared scientific states only), then submit the specialist evaluation
# array on the valid runs and, behind it, the aggregation. Invalid or missing runs go into BLOCK_REPORT_TRAINING.md; valid ones are
# still evaluated (diagnostic) and the package is never declared complete.
#SBATCH --job-name=ni_stage_c
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -uo pipefail
: "${EXPECT_SHA:?}" "${NI_TIP:?}" "${NI_DEST:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; unset PYTHONPATH; export PYTHONPATH="$NI_TIP/src"; cd "$NI_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
M=configs/experiments/nonideal_specialist_matrix.json
EVAL_ROOT="$NI_DEST/eval"
mkdir -p "$NI_DEST/slurm"
"$PY" scripts/66_nonideal_spec_stage.py freeze --matrix "$M" --dest "$NI_DEST" --git-sha "$EXPECT_SHA" || { echo "freeze failed" > "$NI_DEST/BLOCK_REPORT.md"; exit 1; }
LIST="$("$PY" - "$NI_DEST/freeze.json" <<'PYL'
import json, sys
f = json.load(open(sys.argv[1]))
print(":".join(k for k, v in f["runs"].items() if v.get("valid")))
PYL
)"
if [[ -z "$LIST" ]]; then echo "no valid run to evaluate" > "$NI_DEST/BLOCK_REPORT.md"; exit 1; fi
N=$(awk -F: '{print NF}' <<< "$LIST")
mkdir -p "$EVAL_ROOT/slurm"
JE="$(sbatch --parsable --account mit_amf_advanced_gpu --partition mit_normal_gpu --qos mit_amf_advanced_gpu --time 02:30:00 \
  --array="0-$((N-1))%4" --job-name=ni_eval_spec --output="$EVAL_ROOT/slurm/eval-%A_%a.out" --error="$EVAL_ROOT/slurm/eval-%A_%a.err" \
  --export="ALL,EXPECT_SHA=$EXPECT_SHA,NI_TIP=$NI_TIP,NI_EVAL_ROOT=$EVAL_ROOT,NI_FREEZE=$NI_DEST/freeze.json,NI_RUN_IDS=$LIST" \
  "$NI_TIP/scripts/engaging/nonideal_eval_spec.sbatch")" || { echo "eval submit failed" > "$NI_DEST/BLOCK_REPORT.md"; exit 1; }
JG="$(sbatch --parsable --dependency=afterany:"$JE" --output="$NI_DEST/slurm/aggregate-%j.out" --error="$NI_DEST/slurm/aggregate-%j.err" \
  --export="ALL,EXPECT_SHA=$EXPECT_SHA,NI_TIP=$NI_TIP,NI_DEST=$NI_DEST,NI_EVAL_ROOT=$EVAL_ROOT" "$NI_TIP/scripts/engaging/nonideal_aggregate_spec.sbatch")" \
  || { echo "aggregate submit failed" > "$NI_DEST/BLOCK_REPORT.md"; exit 1; }
echo "eval=$JE aggregate=$JG runs=$N" | tee -a "$NI_DEST/stageC.log"
"$PY" - "$NI_DEST/chain_ids.json" "$SLURM_JOB_ID" "$JE" "$JG" <<'PYI'
import json, os, sys
p, c, e, g = sys.argv[1:]
d = json.load(open(p)) if os.path.exists(p) else {}
d.update({"stage_c": c, "specialist_eval_array": e, "aggregate": g})
json.dump(d, open(p, "w"), indent=1)
PYI
