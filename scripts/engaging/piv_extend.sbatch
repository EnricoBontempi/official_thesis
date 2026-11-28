#!/bin/bash
# PIV campaign: the single authorised continuation 1000 -> 1200 epochs, ONLY for runs that reached 1000 epochs without plateau.
# Runs as an array element-wise after the training array (--dependency=aftercorr). Runs that plateaued (or failed) exit 0 / are untouched.
# Resources/partition passed on the sbatch command line. Resume/requeue semantics: s7_run_task.sh (own last.pt only).
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --open-mode=append
set -uo pipefail
IDX="${SLURM_ARRAY_TASK_ID:?}"
IFS=':' read -r -a LIST <<< "${PIV_RUN_IDS:?}"
RUN="${LIST[$IDX]:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
S="$(python3 - "$S7_OUT_ROOT/$RUN/summary.json" <<'PYS'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception:
    print("none"); raise SystemExit
print("extend" if s.get("status") == "completed" and s.get("stop_reason") == "reached_max_epochs" and int(s.get("epochs_completed", -1)) == 1000 else "skip:" + str(s.get("status")))
PYS
)"
echo "[$(date -Is)] piv_extend run=$RUN decision=$S"
[[ "$S" == "extend" ]] || exit 0
export S7_TASK="$RUN" S7_MODE=campaign S7_AUTH_FLAG=--authorize-scientific-piv S7_EXTEND_N_EPOCHS=1200
export S7_EXTEND_SCOPE="this run only, n_epochs up to 1200 (single authorised continuation, decisione roadmap sec. 7.1)"
exec bash "${S7_TIP:?}/scripts/engaging/s7_run_task.sh"
