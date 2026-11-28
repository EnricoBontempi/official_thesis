#!/bin/bash
# Submit the whole PIV main campaign ONCE: 35 trainings (2 pools) -> single continuation 1000->1200 (element-wise) -> best/last evaluation
# per run -> audit (always runs) -> strict post-processing. Real Slurm dependencies; registry of every job id written to the ledger.
#   piv_submit_campaign.sh --tip DIR --sha SHA [--dry-run]
set -uo pipefail
TIP=""; SHA=""; DRY=0
while [[ $# -gt 0 ]]; do case "$1" in --tip) TIP="$2"; shift 2;; --sha) SHA="$2"; shift 2;; --dry-run) DRY=1; shift;; *) echo "unknown $1" >&2; exit 2;; esac; done
[[ -n "$TIP" && -n "$SHA" ]] || { echo "need --tip --sha" >&2; exit 2; }
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
cd "$TIP" || exit 2
[[ "$(git rev-parse HEAD)" == "$SHA" ]] || { echo "REFUSE: HEAD != $SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
M=configs/experiments/piv_campaign_matrix.json
read -r OUT_ROOT SPEC_SHA < <("$PY" -c "import json;m=json.load(open('$M'));print(m['out_root'],m['observation_spec_sha256'])")
EVAL_ROOT=/home/bontempi/official_thesis_piv_eval; PKG=/home/bontempi/official_thesis_piv_package
LEDGER="$OUT_ROOT/_ledger"; REG="$LEDGER/campaign_registry.json"; [[ "$DRY" == 1 ]] && REG="$LEDGER/campaign_registry.dryrun.json"
[[ "$DRY" == 1 || ! -e "$REG" ]] || { echo "REFUSE: registry $REG exists (campaign already submitted)" >&2; exit 7; }
# pools (time-balanced from the F2.7 profiling; GLU on A ~83 GPU-h/run, everything else 4-9 GPU-h/run)
NORMAL="PIV-T5-piv-glu-s0,PIV-T5-piv-glu-s1,PIV-T5-piv-glu-s2,PIV-T2-unet-s0,PIV-T2-unet-s1,PIV-T2-unet-s2,PIV-T2-unet-s3,PIV-T2-unet-s4,PIV-T4-unet-s0,PIV-T4-unet-s1,PIV-T4-unet-s2,PIV-T4-unet-s3,PIV-T4-unet-s4,PIV-T7-glu-s0,PIV-T7-glu-s1,PIV-T7-glu-s2"
PREEMPT="PIV-T5-ideal-glu-s0,PIV-T5-ideal-glu-s1,PIV-T5-ideal-glu-s2,PIV-T6-piv-glu-s0,PIV-T6-piv-glu-s1,PIV-T6-piv-glu-s2,PIV-T6-ideal-glu-s0,PIV-T6-ideal-glu-s1,PIV-T6-ideal-glu-s2,PIV-T1-unet-s0,PIV-T1-unet-s1,PIV-T1-unet-s2,PIV-T1-unet-s3,PIV-T1-unet-s4,PIV-T3-unet-s0,PIV-T3-unet-s1,PIV-T3-unet-s2,PIV-T3-unet-s3,PIV-T3-unet-s4"
"$PY" - "$M" "$NORMAL" "$PREEMPT" <<'PYV' || { echo "REFUSE: pools do not partition the 35 runs" >&2; exit 2; }
import json, sys
m = json.load(open(sys.argv[1])); ids = {r["run_id"] for r in m["new_runs"]}
n, p = sys.argv[2].split(","), sys.argv[3].split(",")
assert len(set(n + p)) == len(n) + len(p) == len(ids) == 35 and set(n + p) == ids
PYV
declare -A PART ACCT QOS TLIM WALL
PART[normal]=mit_normal_gpu;  ACCT[normal]=mit_amf_advanced_gpu; QOS[normal]=mit_amf_advanced_gpu; TLIM[normal]=06:00:00;  WALL[normal]=5.4
PART[preempt]=mit_preemptable; ACCT[preempt]=mit_general;        QOS[preempt]=normal;               TLIM[preempt]=2-00:00:00; WALL[preempt]=46
sub() { if [[ "$DRY" == 1 ]]; then echo "[DRY] sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi; }
mkdir -p "$LEDGER" "$EVAL_ROOT/slurm" "$PKG/slurm" "$OUT_ROOT/slurm"
declare -A TRAIN EXT
JOBS=()
for POOL in normal preempt; do
  RUNS="${NORMAL}"; [[ $POOL == preempt ]] && RUNS="$PREEMPT"
  LIST="${RUNS//,/:}"; N=$(awk -F, '{print NF}' <<< "$RUNS")
  if [[ "$DRY" == 1 ]]; then TRAIN[$POOL]="DRYT$POOL"; else
    OUT="$(bash scripts/engaging/piv_train_submit.sh --tip "$TIP" --sha "$SHA" --pool $POOL --runs "$RUNS" --concurrency 4 --wall-hours "${WALL[$POOL]}" 2>&1)"; echo "$OUT" | tail -3
    TRAIN[$POOL]="$(echo "$OUT" | sed -n 's/^submitted .* array \([0-9]*\):.*/\1/p')"
    [[ -n "${TRAIN[$POOL]}" ]] || { echo "TRAINING SUBMISSION FAILED for $POOL" >&2; exit 3; }
  fi
  COMMON=(--account "${ACCT[$POOL]}" --partition "${PART[$POOL]}" --qos "${QOS[$POOL]}" --requeue --kill-on-invalid-dep=yes)
  EXPORT="ALL,S7_TIP=$TIP,EXPECT_SHA=$SHA,S7_OUT_ROOT=$OUT_ROOT,S7_WALL_HOURS=${WALL[$POOL]},S7_EXPECT_PARTITION=${PART[$POOL]},S7_EXPECT_ACCOUNT=${ACCT[$POOL]},S7_EXPECT_QOS=${QOS[$POOL]},S7_MATRIX=$TIP/$M,PIV_RUN_IDS=$LIST"
  EXT[$POOL]="$(sub "${COMMON[@]}" --time "${TLIM[$POOL]}" --job-name=piv_extend_$POOL --array="0-$((N-1))%4" --dependency="aftercorr:${TRAIN[$POOL]}" \
     --output="$OUT_ROOT/slurm/extend-%A_%a.out" --error="$OUT_ROOT/slurm/extend-%A_%a.err" --export="$EXPORT" "$TIP/scripts/engaging/piv_extend.sbatch")"
  JOBS+=("${TRAIN[$POOL]}" "${EXT[$POOL]}")
done
# evaluation: one job per run, element-wise after that run's continuation task; GLU-on-A (long) on the preemptible pool, the rest on normal
EVAL_IDS=(); declare -A EVAL_OF
for POOL in normal preempt; do
  RUNS="${NORMAL}"; [[ $POOL == preempt ]] && RUNS="$PREEMPT"; IFS=',' read -r -a ARR <<< "$RUNS"
  for i in "${!ARR[@]}"; do
    R="${ARR[$i]}"
    if [[ "$R" == PIV-T5-* ]]; then EP=preempt; else EP=normal; fi
    ET=06:00:00; [[ $EP == preempt ]] && ET=2-00:00:00
    ID="$(sub --account "${ACCT[$EP]}" --partition "${PART[$EP]}" --qos "${QOS[$EP]}" --requeue --kill-on-invalid-dep=yes --time "$ET" --job-name="piv_eval_$R" \
        --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G --open-mode=append --dependency="afterok:${EXT[$POOL]}_$i" \
        --output="$EVAL_ROOT/slurm/eval-$R-%j.out" --error="$EVAL_ROOT/slurm/eval-$R-%j.err" \
        --export="ALL,PIV_TIP=$TIP,EXPECT_SHA=$SHA,PIV_RUN_IDS=$R,PIV_MATRIX=$TIP/$M,PIV_EVAL_ROOT=$EVAL_ROOT,PIV_EVAL_SET=full" "$TIP/scripts/engaging/piv_eval_run.sbatch")"
    EVAL_IDS+=("$ID"); EVAL_OF[$R]="$ID"
  done
done
# manifest, audit (afterany: always runs), strict post-processing
MAN="$LEDGER/piv_manifest.json"; [[ "$DRY" == 1 ]] && MAN="$LEDGER/piv_manifest.dryrun.json"
PYTHONPATH="$TIP/src" "$PY" scripts/77_piv_write_manifest.py --eval-root "$EVAL_ROOT" --training-git-sha "$SHA" --out "$MAN" | tail -1
ALL_DEP="$(IFS=:; echo "${JOBS[*]}:${EVAL_IDS[*]}")"
ACPU=(--account mit_amf_advanced_cpu --partition mit_normal --qos mit_amf_advanced_cpu)
J_AUDIT="$(sub "${ACPU[@]}" --dependency="afterany:$ALL_DEP" --output="$PKG/slurm/freeze-%j.out" --error="$PKG/slurm/freeze-%j.err" \
   --export="ALL,PIV_TIP=$TIP,EXPECT_SHA=$SHA,PIV_MANIFEST=$MAN,PIV_DEST=$PKG" "$TIP/scripts/engaging/piv_freeze_manifest.sbatch")"
J_POST="$(sub "${ACPU[@]}" --dependency="afterany:$J_AUDIT" --output="$PKG/slurm/postproc-%j.out" --error="$PKG/slurm/postproc-%j.err" \
   --export="ALL,PIV_TIP=$TIP,EXPECT_SHA=$SHA,PIV_MANIFEST=$MAN,PIV_DEST=$PKG,PIV_POST_FLAGS=,PIV_SACCT_START=$(date +%F)" "$TIP/scripts/engaging/piv_postprocess.sbatch")"
"$PY" - "$REG" "$SHA" "$SPEC_SHA" "$DRY" "$NORMAL" "$PREEMPT" "${TRAIN[normal]}" "${TRAIN[preempt]}" "${EXT[normal]}" "${EXT[preempt]}" "$J_AUDIT" "$J_POST" "$MAN" "$OUT_ROOT" "$EVAL_ROOT" "$PKG" "$(declare -p EVAL_OF | sed 's/^declare -A EVAL_OF=//')" <<'PYR'
import json, re, sys, datetime
reg_path, sha, spec, dry, normal, preempt, tn, tp, en, ep, audit, post, man, out_root, eval_root, pkg, evof = sys.argv[1:]
ev = dict(re.findall(r'\["([^"]+)"\]="([^"]+)"', evof))
reg = {"status": "submitted" if dry == "0" else "DRY-RUN", "submitted_at": datetime.datetime.now().astimezone().isoformat(), "code_sha": sha, "observation_spec_sha256": spec,
       "pools": {"normal": {"runs": normal.split(","), "train_array": tn, "extend_array": en, "partition": "mit_normal_gpu", "account": "mit_amf_advanced_gpu", "qos": "mit_amf_advanced_gpu", "attempt_wall_hours": 5.4, "time": "06:00:00"},
                 "preempt": {"runs": preempt.split(","), "train_array": tp, "extend_array": ep, "partition": "mit_preemptable", "account": "mit_general", "qos": "normal", "attempt_wall_hours": 46, "time": "2-00:00:00"}},
       "eval_jobs": ev, "audit_job": audit, "postprocess_job": post, "manifest": man, "paths": {"training_root": out_root, "eval_root": eval_root, "package": pkg},
       "dependencies": {"extend": "aftercorr on the pool's training array (element-wise, only after the run's training task succeeds)",
                        "eval": "afterok on that run's extend task (array element)", "audit": "afterany on every training, extend and eval job", "postprocess": "afterany on audit; strict (no COMPLETE if any run missing/failed/incomplete)",
                        "invalid_dependencies": "--kill-on-invalid-dep=yes on extend and eval jobs: never-satisfied jobs are cancelled instead of left pending"}}
json.dump(reg, open(reg_path, "w"), indent=1)
print("registry", reg_path)
PYR
echo "submitted: train normal=${TRAIN[normal]} preempt=${TRAIN[preempt]} extend normal=${EXT[normal]} preempt=${EXT[preempt]} audit=$J_AUDIT post=$J_POST evals=${#EVAL_IDS[@]}"
