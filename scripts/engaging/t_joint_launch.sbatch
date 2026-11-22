#!/bin/bash
# Runs after chain v2's verify job (afterany). Reads the aggregation job id that chain v2 recorded and submits the
# joint post-processing behind it (afterany, so a failed aggregation yields a BLOCCO file, not a hang). If chain v2
# stopped without submitting aggregation, writes the BLOCCO and stops. Submits at most once.
#SBATCH --job-name=t_joint_launch
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -uo pipefail
: "${EXPECT_SHA:?}" "${T_TIP:?}" "${T_JOINT_DEST:?}" "${T_V2_ROOT:?}"
mkdir -p "$T_JOINT_DEST/slurm"
if [[ -e "$T_JOINT_DEST/joint_submitted.json" ]]; then echo "already submitted; refusing a duplicate" >&2; exit 0; fi
block() {
  printf '# Joint package NOT complete\n\nThe joint package is not declared complete. Nothing was relaunched.\n\n- %s\n' "$1" > "$T_JOINT_DEST/JOINT_BLOCCO.md"
  printf '{"ok": false, "complete": false, "reason": "%s"}\n' "$1" > "$T_JOINT_DEST/JOINT_STATUS.json"; exit 0; }
CH="$T_V2_ROOT/chain_v2_submitted.json"
[[ -f "$T_V2_ROOT/INCOMPLETE_v2.json" ]] && block "assisted chain v2 stopped (INCOMPLETE_v2.json in $T_V2_ROOT)"
[[ -f "$CH" ]] || block "chain_v2_submitted.json missing in $T_V2_ROOT: the v2 evaluation/aggregation was not submitted"
AG=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["aggregate"])' "$CH") || block "cannot read the aggregation job id"
JJ=$(sbatch --parsable --dependency=afterany:"$AG" --output="$T_JOINT_DEST/slurm/joint-%j.out" --error="$T_JOINT_DEST/slurm/joint-%j.err" \
  --export=ALL,EXPECT_SHA="$EXPECT_SHA",T_TIP="$T_TIP",T_JOINT_DEST="$T_JOINT_DEST",T_CURSOR_PACKAGE="$T_CURSOR_PACKAGE",T_CURSOR_NEURAL="$T_CURSOR_NEURAL",T_CURSOR_BASELINE="$T_CURSOR_BASELINE",T_V2_ROOT="$T_V2_ROOT" \
  "$T_TIP/scripts/engaging/t_joint.sbatch") || block "sbatch of the joint job failed"
printf '{"ok": true, "stage": "joint_submitted", "at": "%s", "joint_job": "%s", "after_aggregate": "%s", "git_sha": "%s"}\n' "$(date -Is)" "$JJ" "$AG" "$EXPECT_SHA" > "$T_JOINT_DEST/joint_submitted.json"
echo "joint job $JJ after aggregate $AG"
