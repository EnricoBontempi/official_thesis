#!/bin/bash
# Median9 second pass, one frozen S5 best per array task. No training.
# At most two of these GPU tasks run at once. Advanced QoS only.
#SBATCH --job-name=astra_m9n
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/home/bontempi/official_thesis_astra_results/median9_second_pass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/bontempi/official_thesis_astra_results/median9_second_pass/slurm_logs/%x-%A_%a.err
set -euo pipefail
: "${EXPECT_SHA:?set EXPECT_SHA}"
TIP="${TIP:-$HOME/projects/official_thesis_astra_median9}"
DATA="${DATA:-$HOME/projects/official_thesis_s5}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/official_thesis_s5_runs}"
OUT_ROOT="${OUT_ROOT:-$HOME/official_thesis_astra_results/median9_second_pass}"
PRIMARY_ROOT="${PRIMARY_ROOT:-$HOME/official_thesis_astra_results/level_structure_v1}"
RUNS=(s5_sparse_unet_pon_seed0 s5_sparse_unet_poff_seed0 s5_adapted_glu_pon_seed0 s5_adapted_glu_poff_seed0)
RUN_ID="${RUNS[$SLURM_ARRAY_TASK_ID]}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
unset PYTHONPATH
export PYTHONPATH="$TIP/src"
cd "$TIP"
GOT_SHA="$(git rev-parse HEAD)"
if [[ "$GOT_SHA" != "$EXPECT_SHA" ]] || [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing: frozen tip $GOT_SHA != $EXPECT_SHA or dirty" >&2
  exit 78
fi
if [[ -e "$OUT_ROOT/$RUN_ID" ]]; then
  echo "Refusing: $OUT_ROOT/$RUN_ID already exists" >&2
  exit 78
fi
mkdir -p "$OUT_ROOT/slurm_logs"
echo "SLURM_JOB_ID=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} RUN_ID=$RUN_ID SHA=$GOT_SHA $(date -Is)"
"$PY" -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
"$PY" scripts/24_astra_median9_full_eval.py --mode neural --run-id "$RUN_ID" \
  --runs-root "$RUNS_ROOT" --out-root "$OUT_ROOT" --primary-root "$PRIMARY_ROOT" \
  --cache-root "$DATA/data/processed/month2_dense_unet_cache" \
  --mask-root "$DATA/data/processed/month2_sparse_eval_masks" \
  --stats "$DATA/data/manifests/train_norm_stats_month2_fit.json" \
  --device cuda --batch-size 8 --git-sha "$GOT_SHA"
echo "ASTRA_MEDIAN9_NEURAL_DONE $RUN_ID out=$OUT_ROOT/$RUN_ID"
