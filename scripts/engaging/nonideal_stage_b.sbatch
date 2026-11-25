#!/bin/bash
# Stage B (CPU, afterany of both training arrays): classify the 30 runs; submit the single authorised continuation (1000 -> 1200 epochs)
# for the runs that reached 1000 epochs without plateau, on the pool each run was assigned to; then submit stage C behind it.
#SBATCH --job-name=ni_stage_b
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -uo pipefail
: "${EXPECT_SHA:?}" "${NI_TIP:?}" "${NI_DEST:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; unset PYTHONPATH; export PYTHONPATH="$NI_TIP/src"; cd "$NI_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
M=configs/experiments/nonideal_specialist_matrix.json
OUT_ROOT="$("$PY" -c "import json;print(json.load(open('$M'))['out_root'])")"
mkdir -p "$NI_DEST/slurm"
"$PY" scripts/66_nonideal_spec_stage.py status --matrix "$M" > "$NI_DEST/stageB_status_print.json"
EXT_JOBS=()
while read -r LINE; do
  [[ -n "$LINE" ]] || continue
  POOL="${LINE%%:*}"; LIST="${LINE#*:}"
  mapfile -t P < <("$PY" - "$OUT_ROOT/_ledger/assignment.json" "$POOL" <<'PYP'
import json, sys
a = json.load(open(sys.argv[1])); arr = [v for k, v in a["_arrays"].items() if k.startswith(sys.argv[2] + ":")][0]
print(arr["partition"]); print(arr["account"]); print(arr["qos"]); print(arr["time"]); print(arr["wall_hours"])
PYP
)
  N=$(awk -F: '{print NF}' <<< "$LIST")
  JID="$(sbatch --parsable --account "${P[1]}" --partition "${P[0]}" --qos "${P[2]}" --time "${P[3]}" --requeue --job-name="ni_ext_$POOL" \
    --array="0-$((N-1))%4" --output="$OUT_ROOT/slurm/slurm-ext-%A_%a.out" --error="$OUT_ROOT/slurm/slurm-ext-%A_%a.err" \
    --export="ALL,S7_TIP=$NI_TIP,EXPECT_SHA=$EXPECT_SHA,S7_WALL_HOURS=${P[4]},S7_EXPECT_PARTITION=${P[0]},S7_EXPECT_ACCOUNT=${P[1]},S7_EXPECT_QOS=${P[2]},S7_MATRIX=$NI_TIP/$M,NI_RUN_IDS=$LIST,S7_EXTEND_N_EPOCHS=1200,S7_EXTEND_SCOPE=single authorised continuation 1000->1200 (Enrico 2026-11-24)" \
    "$NI_TIP/scripts/engaging/nonideal_extend.sbatch")" || { echo "extension submit failed for $POOL" >&2; continue; }
  echo "extension $POOL job=$JID runs=$LIST" | tee -a "$NI_DEST/stageB.log"
  EXT_JOBS+=("$JID")
done < <("$PY" scripts/66_nonideal_spec_stage.py extension --matrix "$M")
DEP=""
if (( ${#EXT_JOBS[@]} > 0 )); then DEP="--dependency=afterany:$(IFS=:; echo "${EXT_JOBS[*]}")"; fi
JC="$(sbatch --parsable $DEP --output="$NI_DEST/slurm/stage_c-%j.out" --error="$NI_DEST/slurm/stage_c-%j.err" \
  --export=ALL "$NI_TIP/scripts/engaging/nonideal_stage_c.sbatch")" || { echo "stage C submit failed" > "$NI_DEST/BLOCK_REPORT.md"; exit 1; }
echo "stage C job=$JC dep=${DEP:-none}" | tee -a "$NI_DEST/stageB.log"
"$PY" - "$NI_DEST/chain_ids.json" "$SLURM_JOB_ID" "$JC" "${EXT_JOBS[*]:-}" <<'PYI'
import json, os, sys
p, b, c, ext = sys.argv[1:]
d = json.load(open(p)) if os.path.exists(p) else {}
d.update({"stage_b": b, "stage_c_submitted": c, "extension_arrays": ext.split()})
json.dump(d, open(p, "w"), indent=1)
PYI
