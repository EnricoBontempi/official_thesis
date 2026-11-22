#!/bin/bash
# Chain stage 1 (CPU, runs by itself after the training arrays have ended, whatever their outcome):
# verify that the 12 trainings concluded -> freeze the checkpoint matrix -> submit the evaluation array -> submit the aggregation.
# NO blind recovery: if anything is not terminal/clean, it writes INCOMPLETE.json with the causes and stops. It never resubmits a training.
#SBATCH --job-name=t_chain_verify
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -uo pipefail
: "${EXPECT_SHA:?}" "${T_TIP:?}" "${T_TRAIN_JOBS:?}" "${T_EVAL_ROOT:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
export PYTHONPATH="$T_TIP/src"
cd "$T_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
mkdir -p "$T_EVAL_ROOT"
OUT="$T_EVAL_ROOT"
M=configs/experiments/month2_t_assisted_matrix.json
REASONS="$("$PY" - <<PYV
import json, subprocess
m = json.load(open("$M"))
root = m["out_root"]
reasons = []
for j in "$T_TRAIN_JOBS".split(","):
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", j, "-o", "JobID,State,ExitCode"], capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        jid, st, ec = line.split("|")
        if st != "COMPLETED" or ec != "0:0":
            reasons.append(f"slurm {jid}: {st} {ec}")
for r in m["new_runs"]:
    d = f"{root}/{r['run_id']}"
    try:
        s = json.load(open(f"{d}/summary.json"))
    except Exception as e:
        reasons.append(f"{r['run_id']}: summary.json unreadable ({e})"); continue
    if s.get("status") not in ("completed", "stopped_early_plateau"):
        reasons.append(f"{r['run_id']}: status {s.get('status')} / {s.get('stop_reason')}")
    try:
        led = [json.loads(x) for x in open(f"{root}/_ledger/{r['run_id']}.jsonl").read().splitlines() if x.strip()]
        for ev in led:
            if ev.get("event") == "refused" or "fresh_restart_no_checkpoint" in json.dumps(ev):
                reasons.append(f"{r['run_id']}: ledger event {ev.get('event')} {ev.get('reason','')}")
    except Exception as e:
        reasons.append(f"{r['run_id']}: ledger unreadable ({e})")
print("\n".join(reasons))
PYV
)"
if [[ -n "$REASONS" ]]; then
  printf '{"ok": false, "stage": "verify_training", "checked_at": "%s", "train_jobs": "%s", "reasons": %s}\n' "$(date -Is)" "$T_TRAIN_JOBS" "$("$PY" -c 'import json,sys;print(json.dumps(sys.stdin.read().strip().splitlines()))' <<<"$REASONS")" > "$OUT/INCOMPLETE.json"
  echo "INCOMPLETE: $REASONS" >&2
  exit 1
fi
"$PY" scripts/54_t_write_eval_matrix.py --out "$OUT/matrix_frozen.json" || { echo '{"ok": false, "stage": "freeze_matrix"}' > "$OUT/INCOMPLETE.json"; exit 1; }
FROZEN="$OUT/matrix_frozen.json"
RUNNING_NORMAL="$(squeue -u "$USER" -h -r -t RUNNING -p mit_normal_gpu -o '%b' | grep -ci gpu || true)"
CONC=$(( 4 - RUNNING_NORMAL )); (( CONC > 3 )) && CONC=3; (( CONC < 1 )) && CONC=1
mkdir -p "$OUT/slurm"
EV=$(sbatch --parsable --account mit_amf_advanced_gpu --partition mit_normal_gpu --qos mit_amf_advanced_gpu --time 03:00:00 --array=0-11%$CONC \
  --job-name=t_eval_full --output="$OUT/slurm/eval-%A_%a.out" --error="$OUT/slurm/eval-%A_%a.err" \
  --export=ALL,T_TIP="$T_TIP",EXPECT_SHA="$EXPECT_SHA",T_EVAL_MATRIX="$FROZEN",T_EVAL_OUT="$OUT/runs",T_EVAL_MODE=full "$T_TIP/scripts/engaging/t_eval.sbatch") || { echo '{"ok": false, "stage": "submit_eval"}' > "$OUT/INCOMPLETE.json"; exit 1; }
AG=$(sbatch --parsable --dependency=afterok:$EV --kill-on-invalid-dep=yes --output="$OUT/slurm/aggregate-%j.out" --error="$OUT/slurm/aggregate-%j.err" \
  --export=ALL,T_TIP="$T_TIP",EXPECT_SHA="$EXPECT_SHA",T_EVAL_MATRIX="$FROZEN",T_EVAL_OUT="$OUT/runs",T_PACKAGE_DEST="$OUT/package",T_TRAIN_JOBS="$T_TRAIN_JOBS",T_EVAL_JOBS="$EV" "$T_TIP/scripts/engaging/t_aggregate.sbatch") || { echo '{"ok": false, "stage": "submit_aggregate"}' > "$OUT/INCOMPLETE.json"; exit 1; }
printf '{"ok": true, "stage": "chain_submitted", "at": "%s", "eval_array": "%s", "aggregate": "%s", "concurrency": %s, "matrix_sha256": "%s", "git_sha": "%s"}\n' \
  "$(date -Is)" "$EV" "$AG" "$CONC" "$(sha256sum "$FROZEN" | cut -d' ' -f1)" "$EXPECT_SHA" > "$OUT/chain_submitted.json"
echo "chain submitted: eval array $EV, aggregation $AG (concurrency $CONC)"
