#!/bin/bash
# Smoke of the downstream chain on real (tiny, non-scientific) specialist checkpoints: freeze -> specialist evaluation (smoke mode,
# a few frames, all six cells, best and last) -> readback. Output under *smoke* only.
#SBATCH --job-name=ni_smoke_eval
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
set -uo pipefail
: "${NI_TIP:?}"; : "${EXPECT_SHA:?}"; : "${PREV_UNET:?}"; : "${PREV_GLU:?}"; : "${NI_SMOKE_ROOT:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; unset PYTHONPATH; export PYTHONPATH="$NI_TIP/src"; cd "$NI_TIP"
SM="$NI_SMOKE_ROOT/eval_job${SLURM_JOB_ID:-local}"; mkdir -p "$SM"
for FAM in unet glu; do
  if [[ "$FAM" == "unet" ]]; then PREV="$PREV_UNET"; SEED=1; else PREV="$PREV_GLU"; SEED=0; fi
  RID="NI-${FAM}-on-N1G1-s${SEED}"
  "$PY" scripts/66_nonideal_spec_stage.py freeze --matrix configs/experiments/nonideal_specialist_matrix.json --smoke-root "$PREV/N1G1" --smoke-runs "$RID" --dest "$SM/freeze_$FAM" --git-sha "$EXPECT_SHA"
  # smoke freeze lists one run; the evaluator reads that entry
  "$PY" scripts/68_nonideal_specialist_eval.py --run-id "$RID" --out-root "$SM/eval_smoke_$FAM" --freeze "$SM/freeze_$FAM/freeze.json" \
     --train-root "$PREV/N1G1" --eval-git-sha "$EXPECT_SHA" --device cuda --mode smoke --smoke-frames 8
  echo "[$(date -Is)] spec-eval smoke $FAM rc=$?"
  ls "$SM/eval_smoke_$FAM/$RID/best" "$SM/eval_smoke_$FAM/$RID/last"
done
