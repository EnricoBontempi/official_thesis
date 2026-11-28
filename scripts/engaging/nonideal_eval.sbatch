#!/bin/bash
# Block II task: one array task = one frozen best checkpoint (12 in the spec). Resources/array/out are passed by
# nonideal_submit.sh. Refuses unless the checkout is the declared SHA and clean; never overwrites an output.
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --open-mode=append
set -euo pipefail
: "${EXPECT_SHA:?}" "${NI_TIP:?}" "${NI_SPEC:?}" "${NI_OUT_ROOT:?}"
MODE="${NI_MODE:-full}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
case "$MODE" in
  full) [[ "$NI_OUT_ROOT" != *smoke* ]] || { echo "REFUSE: full with smoke out-root" >&2; exit 4; } ;;
  smoke) [[ "$NI_OUT_ROOT" == *smoke* ]] || { echo "REFUSE: smoke needs a smoke out-root" >&2; exit 4; } ;;
  *) echo "REFUSE: unknown mode $MODE" >&2; exit 4 ;;
esac
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$NI_TIP/src"
cd "$NI_TIP"
SHA_HEAD="$(git rev-parse HEAD)"
[[ "$SHA_HEAD" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD $SHA_HEAD != $EXPECT_SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
RUN_ID="$("$PY" - <<PYIDX
import os
from combustion_recon.data.nonideal_observations import load_degradation_spec
ids = [r["run_id"] for r in load_degradation_spec(os.environ["NI_SPEC"]).raw["models"]["runs"]]
sel = [x for x in os.environ.get("NI_RUN_IDS", "").split(":") if x] or ids
print(sel[int(os.environ["SLURM_ARRAY_TASK_ID"])])
PYIDX
)"
echo "[$(date -Is)] launcher host=$(hostname) job=${SLURM_JOB_ID:-na} task=${SLURM_ARRAY_TASK_ID:-na} run=$RUN_ID mode=$MODE sha=$SHA_HEAD" >&2
[[ ! -f "$NI_OUT_ROOT/CAMPAIGN_HALT.json" ]] || { echo "REFUSE: CAMPAIGN_HALT present" >&2; exit 3; }
NEEDED="$("$PY" scripts/engaging/o5_preflight_paths.py configs/experiments/month2_s7_c149_eval_matrix.json "$RUN_ID")"
[[ -n "$NEEDED" ]] || { echo "REFUSE: empty input list" >&2; exit 6; }
while read -r P; do
  timeout 120 stat -c '%s' "$P" > /dev/null 2>&1 || { echo "REFUSE: input not reachable: $P" >&2; exit 6; }
done <<< "$NEEDED"
mkdir -p "$NI_OUT_ROOT"
[[ ! -e "$NI_OUT_ROOT/$RUN_ID" ]] || { echo "REFUSE: output exists: $NI_OUT_ROOT/$RUN_ID" >&2; exit 4; }
ARGS=(scripts/60_nonideal_eval.py --spec "$NI_SPEC" --run-id "$RUN_ID" --out-root "$NI_OUT_ROOT" --eval-git-sha "$SHA_HEAD" --device cuda --mode "$MODE")
[[ -n "${NI_SMOKE_FRAMES:-}" ]] && ARGS+=(--smoke-frames "$NI_SMOKE_FRAMES")
"$PY" "${ARGS[@]}"
