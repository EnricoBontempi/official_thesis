#!/bin/bash
# Sync repair scripts to HOME tip and submit Gappy + S6 last/swap. No polling.
set -euo pipefail
TIP="${TIP:-$HOME/projects/official_thesis_s5}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
ROOT_LOCAL="${ROOT_LOCAL:-}"

if [[ -n "${ROOT_LOCAL}" ]]; then
  cp -f "${ROOT_LOCAL}/scripts/15_gappypod_nested_coverage_eval.py" "$TIP/scripts/"
  cp -f "${ROOT_LOCAL}/scripts/17_s6_sparse_eval_and_swap.py" "$TIP/scripts/"
  cp -f "${ROOT_LOCAL}/scripts/engaging/gappypod_nested_cpu.sbatch" "$TIP/scripts/engaging/"
  cp -f "${ROOT_LOCAL}/scripts/engaging/s6_sparse_eval.sbatch" "$TIP/scripts/engaging/"
fi

test -f "$TIP/scripts/15_gappypod_nested_coverage_eval.py"
test -f "$TIP/scripts/17_s6_sparse_eval_and_swap.py"
grep -q "chunked denorm" "$TIP/scripts/15_gappypod_nested_coverage_eval.py"
grep -q '\"p_conditioning\": bool(ds.p_conditioning)' "$TIP/scripts/17_s6_sparse_eval_and_swap.py"
grep -q "mem=128G" "$TIP/scripts/engaging/gappypod_nested_cpu.sbatch"
grep -q "mit_amf_advanced_cpu" "$TIP/scripts/engaging/gappypod_nested_cpu.sbatch"

# Verify best artifacts complete before last/swap-only relaunch
"$PY" - <<'PY'
from pathlib import Path
S6=Path("/home/bontempi/official_thesis_s6_results")
runs=["s5_sparse_unet_pon_seed0","s5_sparse_unet_poff_seed0","s5_adapted_glu_pon_seed0","s5_adapted_glu_poff_seed0"]
roles=["train_fit","seen_train_eval","val_eval"]
covs=["cov01","cov05","cov20"]
for rid in runs:
  d=S6/rid/rid
  missing=[f"per_frame_best_{c}_{r}_rep{rep}.csv.gz" for c in covs for r in roles for rep in range(3)
           if not (d/f"per_frame_best_{c}_{r}_rep{rep}.csv.gz").exists()]
  if missing:
    raise SystemExit(f"{rid} incomplete best artifacts: {missing[:3]} n={len(missing)}")
  print(rid, "best_27_ok")
print("BEST_ARTIFACTS_OK")
PY

cd "$TIP"
GAPPY_OUT="$HOME/official_thesis_gappypod_nested_1_20_amf_v2"
mkdir -p "$GAPPY_OUT"
GID=$(sbatch --parsable \
  --export=ALL,TIP="$TIP",PY="$PY",OUT_ROOT="$GAPPY_OUT" \
  "$TIP/scripts/engaging/gappypod_nested_cpu.sbatch")
echo "GAPPY_JOB=$GID OUT=$GAPPY_OUT LOG=$GAPPY_OUT/slurm_${GID}/stdout.log TIP_SLURM=$TIP/slurm-${GID}.out"

declare -a S6_IDS=()
for RUN in \
  s5_sparse_unet_pon_seed0 \
  s5_sparse_unet_poff_seed0 \
  s5_adapted_glu_pon_seed0 \
  s5_adapted_glu_poff_seed0
do
  OUT="$HOME/official_thesis_s6_results/${RUN}"
  JID=$(sbatch --parsable \
    --job-name="s6_${RUN##*_}" \
    --export=ALL,TIP="$TIP",PY="$PY",RUNS_ROOT="$HOME/official_thesis_s5_runs",ONLY_RUN_ID="$RUN",OUT_ROOT="$OUT",CHECKPOINTS="last",SKIP_FIGURES=0 \
    "$TIP/scripts/engaging/s6_sparse_eval.sbatch")
  S6_IDS+=("$JID")
  echo "S6_JOB=$JID RUN=$RUN OUT=$OUT LOG=$OUT/slurm_${JID}/stdout.log TIP_SLURM=$TIP/slurm-${JID}.out"
done
echo "S6_JOBS=${S6_IDS[*]}"
echo "SUBMIT_DONE no_polling"
