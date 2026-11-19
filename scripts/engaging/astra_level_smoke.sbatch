#!/bin/bash
# Astra A1 smoke on CPU: pytest + tiny (4 frames) neural + baselines runs into a throwaway OUT_ROOT.
# Submit with: sbatch --export=ALL,EXPECT_SHA=<sha> this_file
#SBATCH --job-name=astra_smoke
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
: "${EXPECT_SHA:?set EXPECT_SHA}"
TIP="${TIP:-$HOME/projects/official_thesis_astra_frozen}"
DATA="${DATA:-$HOME/projects/official_thesis_s5}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
GAPPY="${GAPPY:-$HOME/official_thesis_gappypod_nested_1_20_amf_v2/artifacts/gappypod_fitted_pod_p_off_on.npz}"
SMOKE_OUT="${SMOKE_OUT:-$HOME/official_thesis_astra_results/smoke_${SLURM_JOB_ID}}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset PYTHONPATH
export PYTHONPATH="$TIP/src"
cd "$TIP"
GOT_SHA="$(git rev-parse HEAD)"
[[ "$GOT_SHA" == "$EXPECT_SHA" ]] || { echo "SHA mismatch $GOT_SHA" >&2; exit 78; }
echo "== pytest =="
"$PY" -m pytest -q -p no:cacheprovider tests/test_astra_level.py tests/test_s6_swap_pconditioning.py \
  tests/test_model_input_isolation.py tests/test_u6_poff_isolation.py tests/test_baselines.py \
  tests/test_month2_baselines.py 2>&1 | tail -15
COMMON=(--cache-root "$DATA/data/processed/month2_dense_unet_cache" --mask-root "$DATA/data/processed/month2_sparse_eval_masks" \
  --stats "$DATA/data/manifests/train_norm_stats_month2_fit.json" --git-sha "$GOT_SHA" --limit-frames 4)
for R in s5_sparse_unet_pon_seed0 s5_adapted_glu_poff_seed0; do
  echo "== neural smoke $R (cpu) =="
  "$PY" scripts/18_astra_level_structure_eval.py --mode neural --run-id "$R" --runs-root "$HOME/official_thesis_s5_runs" \
    --out-root "$SMOKE_OUT" --s6-ref-root "$HOME/official_thesis_s6_results" --device cpu --batch-size 4 "${COMMON[@]}"
done
echo "== baselines smoke =="
"$PY" scripts/18_astra_level_structure_eval.py --mode baselines --out-root "$SMOKE_OUT" \
  --gappy-artifact "$GAPPY" --hist-summary "$DATA/results/tables/month2_baseline_summary.csv" "${COMMON[@]}"
echo "SMOKE_DONE $SMOKE_OUT"
