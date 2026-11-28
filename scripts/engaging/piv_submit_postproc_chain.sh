#!/bin/bash
# Mission 02 (Virtual PIV) — Slurm dependency chain helper for Sonnet integration.
# Wires real training/eval job IDs -> freeze audit -> (optional eval array) -> CPU post-processing.
# Never invents job IDs or checkpoints; supports --dry-run to inspect exact sbatch commands.
#
# Example (after trainings J1, J2 are submitted):
#   bash scripts/engaging/piv_submit_postproc_chain.sh \
#     --tip /path/to/clean_checkout --sha <HEAD_SHA> \
#     --manifest /path/to/piv_manifest.json --dest /path/to/postproc_out \
#     --train-jobs 12345:12346 --eval-cmd scripts/piv_phase4/eval_piv_run.py \
#     --run-ids RUN1:RUN2:RUN3 --dry-run
set -euo pipefail

TIP=""; SHA=""; MANIFEST=""; DEST=""
TRAIN_JOBS=""; EVAL_JOBS=""; EVAL_CMD=""; RUN_IDS=""
ARRAY_CONC="4"; POST_FLAGS=""; SACCT_START=""; DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tip) TIP="$2"; shift 2;;
    --sha) SHA="$2"; shift 2;;
    --manifest) MANIFEST="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --train-jobs) TRAIN_JOBS="$2"; shift 2;;
    --eval-jobs) EVAL_JOBS="$2"; shift 2;;
    --eval-cmd) EVAL_CMD="$2"; shift 2;;
    --run-ids) RUN_IDS="$2"; shift 2;;
    --concurrency) ARRAY_CONC="$2"; shift 2;;
    --post-flags) POST_FLAGS="$2"; shift 2;;
    --sacct-start) SACCT_START="$2"; shift 2;;
    --dry-run) DRY_RUN="1"; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

for v in TIP SHA MANIFEST DEST; do
  [[ -n "${!v}" ]] || { echo "missing required --${v,,}" >&2; exit 2; }
done

run_sbatch() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] sbatch $*" >&2
    echo "DRY_JOB_ID"
  else
    sbatch --parsable "$@"
  fi
}

mkdir -p "$DEST/slurm"

# Stage 1: freeze / status audit (depends on training jobs if supplied)
FREEZE_DEP=()
if [[ -n "$TRAIN_JOBS" ]]; then
  FREEZE_DEP=(--dependency="afterany:${TRAIN_JOBS}")
fi
J_FREEZE="$(run_sbatch "${FREEZE_DEP[@]}" \
  --output="$DEST/slurm/freeze-%j.out" --error="$DEST/slurm/freeze-%j.err" \
  --export="ALL,PIV_TIP=$TIP,EXPECT_SHA=$SHA,PIV_MANIFEST=$MANIFEST,PIV_DEST=$DEST" \
  "$TIP/scripts/engaging/piv_freeze_manifest.sbatch")"
echo "stage1_freeze=$J_FREEZE"

# Stage 2: optional GPU eval array (when --eval-cmd and --run-ids are provided)
POST_DEP_IDS=""
if [[ -n "$EVAL_CMD" && -n "$RUN_IDS" ]]; then
  N_RUNS=$(awk -F: '{print NF}' <<< "$RUN_IDS")
  J_EVAL="$(run_sbatch --dependency="afterok:${J_FREEZE}" \
    --array="0-$((N_RUNS - 1))%${ARRAY_CONC}" \
    --output="$DEST/slurm/eval-%A_%a.out" --error="$DEST/slurm/eval-%A_%a.err" \
    --export="ALL,PIV_TIP=$TIP,EXPECT_SHA=$SHA,PIV_MANIFEST=$MANIFEST,PIV_RUN_IDS=$RUN_IDS,PIV_EVAL_CMD=$EVAL_CMD" \
    "$TIP/scripts/engaging/piv_eval_array.sbatch")"
  echo "stage2_eval_array=$J_EVAL (n_runs=$N_RUNS)"
  POST_DEP_IDS="$J_EVAL"
elif [[ -n "$EVAL_JOBS" ]]; then
  POST_DEP_IDS="${J_FREEZE}:${EVAL_JOBS}"
else
  POST_DEP_IDS="$J_FREEZE"
fi

# Stage 3: CPU post-processing & diagnostic atlas
J_POST="$(run_sbatch --dependency="afterany:${POST_DEP_IDS}" \
  --output="$DEST/slurm/postproc-%j.out" --error="$DEST/slurm/postproc-%j.err" \
  --export="ALL,PIV_TIP=$TIP,EXPECT_SHA=$SHA,PIV_MANIFEST=$MANIFEST,PIV_DEST=$DEST,PIV_POST_FLAGS=$POST_FLAGS,PIV_SACCT_START=$SACCT_START" \
  "$TIP/scripts/engaging/piv_postprocess.sbatch")"
echo "stage3_postproc=$J_POST"
