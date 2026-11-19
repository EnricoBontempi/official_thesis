#!/bin/bash
# Median9 second pass for the frozen baselines. CPU. No refit.
#SBATCH --job-name=astra_m9b
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/bontempi/official_thesis_astra_results/median9_second_pass/slurm_logs/%x-%j.out
#SBATCH --error=/home/bontempi/official_thesis_astra_results/median9_second_pass/slurm_logs/%x-%j.err
set -euo pipefail
: "${EXPECT_SHA:?set EXPECT_SHA}"
TIP="${TIP:-$HOME/projects/official_thesis_astra_median9}"
DATA="${DATA:-$HOME/projects/official_thesis_s5}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
GAPPY="${GAPPY:-$HOME/official_thesis_gappypod_nested_1_20_amf_v2/artifacts/gappypod_fitted_pod_p_off_on.npz}"
OUT_ROOT="${OUT_ROOT:-$HOME/official_thesis_astra_results/median9_second_pass}"
PRIMARY_ROOT="${PRIMARY_ROOT:-$HOME/official_thesis_astra_results/level_structure_v1}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset PYTHONPATH
export PYTHONPATH="$TIP/src"
cd "$TIP"
GOT_SHA="$(git rev-parse HEAD)"
if [[ "$GOT_SHA" != "$EXPECT_SHA" ]] || [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing: frozen tip $GOT_SHA != $EXPECT_SHA or dirty" >&2
  exit 78
fi
if [[ -e "$OUT_ROOT/baselines" ]]; then
  echo "Refusing: $OUT_ROOT/baselines already exists" >&2
  exit 78
fi
mkdir -p "$OUT_ROOT/slurm_logs"
echo "SLURM_JOB_ID=$SLURM_JOB_ID SHA=$GOT_SHA $(date -Is)"
"$PY" scripts/24_astra_median9_full_eval.py --mode baselines --out-root "$OUT_ROOT" \
  --primary-root "$PRIMARY_ROOT" \
  --cache-root "$DATA/data/processed/month2_dense_unet_cache" \
  --mask-root "$DATA/data/processed/month2_sparse_eval_masks" \
  --stats "$DATA/data/manifests/train_norm_stats_month2_fit.json" \
  --gappy-artifact "$GAPPY" --git-sha "$GOT_SHA"
echo "ASTRA_MEDIAN9_BASELINES_DONE out=$OUT_ROOT/baselines"
