#!/bin/bash
# Astra A2: free/oracle/AO9 from ONE forward on a frozen S5 best.pt. No training.
# Submit with: sbatch --export=ALL,RUN_ID=<s5 run id>,EXPECT_SHA=<sha> this_file
#SBATCH --job-name=astra_lvl
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
: "${RUN_ID:?set RUN_ID}"
: "${EXPECT_SHA:?set EXPECT_SHA}"
TIP="${TIP:-$HOME/projects/official_thesis_astra_frozen}"
DATA="${DATA:-$HOME/projects/official_thesis_s5}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/official_thesis_s5_runs}"
S6_REF="${S6_REF:-$HOME/official_thesis_s6_results}"
OUT_ROOT="${OUT_ROOT:-$HOME/official_thesis_astra_results/level_structure_v1}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
unset PYTHONPATH
export PYTHONPATH="$TIP/src"
cd "$TIP"
GOT_SHA="$(git rev-parse HEAD)"
if [[ "$GOT_SHA" != "$EXPECT_SHA" ]] || [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing: frozen tip $GOT_SHA != $EXPECT_SHA or dirty" >&2; exit 78
fi
mkdir -p "$OUT_ROOT/slurm_logs"
echo "SLURM_JOB_ID=$SLURM_JOB_ID RUN_ID=$RUN_ID SHA=$GOT_SHA $(date -Is)"
"$PY" -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
"$PY" scripts/18_astra_level_structure_eval.py --mode neural --run-id "$RUN_ID" \
  --runs-root "$RUNS_ROOT" --out-root "$OUT_ROOT" \
  --cache-root "$DATA/data/processed/month2_dense_unet_cache" \
  --mask-root "$DATA/data/processed/month2_sparse_eval_masks" \
  --stats "$DATA/data/manifests/train_norm_stats_month2_fit.json" \
  --s6-ref-root "$S6_REF" --device cuda --batch-size 8 --git-sha "$GOT_SHA"
echo "ASTRA_NEURAL_DONE $RUN_ID out=$OUT_ROOT/$RUN_ID"
