#!/bin/bash
# Chain v2 (CPU, runs by itself after the extension array ended, whatever its outcome): verify the extensions -> freeze matrix v2 ->
# re-evaluate only the changed checkpoints of the extended runs (unchanged ones are re-attested from verified v1 output) -> aggregate.
# NO blind recovery: any anomaly writes INCOMPLETE_v2.json and stops. v1 outputs and the published package are never modified.
#SBATCH --job-name=t_chain_verify_v2
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -uo pipefail
: "${EXPECT_SHA:?}" "${T_TIP:?}" "${T_EXT_JOB:?}" "${T_TRAIN_JOBS:?}" "${T_V1_ROOT:?}" "${T_V2_ROOT:?}" "${T_EXT_RUNS:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
export PYTHONPATH="$T_TIP/src"
cd "$T_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
mkdir -p "$T_V2_ROOT/slurm"
OUT="$T_V2_ROOT"; V1M="$T_V1_ROOT/matrix_frozen.json"; M=configs/experiments/month2_t_assisted_matrix.json
EXT="${T_EXT_RUNS//:/,}"
REASONS="$("$PY" - <<PYV
import json, subprocess
m = json.load(open("$M")); v1 = {r["run_id"]: r for r in json.load(open("$V1M"))["runs"]}
root = m["out_root"]; ext = set("$EXT".split(","))
reasons = []
out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", "$T_EXT_JOB", "-o", "JobID,State,ExitCode"], capture_output=True, text=True).stdout
for line in out.strip().splitlines():
    jid, st, ec = line.split("|")
    if st != "COMPLETED" or ec != "0:0":
        reasons.append(f"slurm {jid}: {st} {ec}")
for r in m["new_runs"]:
    rid = r["run_id"]; d = f"{root}/{rid}"
    try:
        s = json.load(open(f"{d}/summary.json"))
    except Exception as e:
        reasons.append(f"{rid}: summary.json unreadable ({e})"); continue
    if s.get("status") not in ("completed", "stopped_early_plateau"):
        reasons.append(f"{rid}: status {s.get('status')} / {s.get('stop_reason')}")
    led = [json.loads(x) for x in open(f"{root}/_ledger/{rid}.jsonl").read().splitlines() if x.strip()]
    for ev in led:
        if ev.get("event") == "refused" or "fresh_restart_no_checkpoint" in json.dumps(ev):
            reasons.append(f"{rid}: ledger {ev.get('event')} {ev.get('reason','')}")
    if rid in ext:
        if not any(e.get("event") == "extension_start" for e in led):
            reasons.append(f"{rid}: no extension_start in the ledger")
        if not any(e.get("action") == "extend" for e in led if e.get("event") == "attempt_start"):
            reasons.append(f"{rid}: no extend attempt")
        if not __import__("os").path.exists(f"{d}/tranche1_epoch1000/MANIFEST.json"):
            reasons.append(f"{rid}: first tranche not preserved")
        if not (1000 < s.get("epochs_completed", 0) <= 1100):
            reasons.append(f"{rid}: epochs_completed {s.get('epochs_completed')} outside (1000, 1100]")
    else:
        import hashlib
        for name in ("best", "last", "best_seen_diagnostic"):
            h = hashlib.sha256(open(f"{d}/checkpoints/{name}.pt", "rb").read()).hexdigest()
            if h != v1[rid]["checkpoints"][name]["file_sha256"]:
                reasons.append(f"{rid}/{name}: unextended run changed since the v1 freeze")
print("\n".join(reasons))
PYV
)"
if [[ -n "$REASONS" ]]; then
  printf '{"ok": false, "stage": "verify_extension", "checked_at": "%s", "reasons": %s}\n' "$(date -Is)" "$("$PY" -c 'import json,sys;print(json.dumps(sys.stdin.read().strip().splitlines()))' <<<"$REASONS")" > "$OUT/INCOMPLETE_v2.json"
  echo "INCOMPLETE: $REASONS" >&2; exit 1
fi
"$PY" scripts/54_t_write_eval_matrix.py --out "$OUT/matrix_frozen_v2.json" || { echo '{"ok": false, "stage": "freeze_matrix_v2"}' > "$OUT/INCOMPLETE_v2.json"; exit 1; }
FROZEN="$OUT/matrix_frozen_v2.json"
RUNNING_NORMAL="$(squeue -u "$USER" -h -r -t RUNNING -p mit_normal_gpu -o '%b' | grep -ci gpu || true)"
CONC=$(( 4 - RUNNING_NORMAL )); (( CONC > 3 )) && CONC=3; (( CONC < 1 )) && CONC=1
EV=$(sbatch --parsable --account mit_amf_advanced_gpu --partition mit_normal_gpu --qos mit_amf_advanced_gpu --time 03:00:00 --array=0-11%$CONC \
  --job-name=t_eval_full_v2 --output="$OUT/slurm/eval-%A_%a.out" --error="$OUT/slurm/eval-%A_%a.err" \
  --export=ALL,T_TIP="$T_TIP",EXPECT_SHA="$EXPECT_SHA",T_EVAL_MATRIX="$FROZEN",T_EVAL_OUT="$OUT/runs",T_EVAL_MODE=full,T_EVAL_REUSE_MATRIX="$V1M",T_EVAL_REUSE_ROOT="$T_V1_ROOT/runs" "$T_TIP/scripts/engaging/t_eval.sbatch") || { echo '{"ok": false, "stage": "submit_eval_v2"}' > "$OUT/INCOMPLETE_v2.json"; exit 1; }
AG=$(sbatch --parsable --dependency=afterok:$EV --kill-on-invalid-dep=yes --output="$OUT/slurm/aggregate-%j.out" --error="$OUT/slurm/aggregate-%j.err" \
  --export=ALL,T_TIP="$T_TIP",EXPECT_SHA="$EXPECT_SHA",T_EVAL_MATRIX="$FROZEN",T_EVAL_OUT="$OUT/runs",T_PACKAGE_DEST="$OUT/package",T_TRAIN_JOBS="$T_TRAIN_JOBS",T_EVAL_JOBS="$EV" "$T_TIP/scripts/engaging/t_aggregate.sbatch") || { echo '{"ok": false, "stage": "submit_aggregate_v2"}' > "$OUT/INCOMPLETE_v2.json"; exit 1; }
printf '{"ok": true, "stage": "chain_v2_submitted", "at": "%s", "eval_array": "%s", "aggregate": "%s", "concurrency": %s, "matrix_v2_sha256": "%s", "matrix_v1_sha256": "%s", "git_sha": "%s"}\n' \
  "$(date -Is)" "$EV" "$AG" "$CONC" "$(sha256sum "$FROZEN" | cut -d' ' -f1)" "$(sha256sum "$V1M" | cut -d' ' -f1)" "$EXPECT_SHA" > "$OUT/chain_v2_submitted.json"
echo "chain v2 submitted: eval $EV aggregate $AG"
