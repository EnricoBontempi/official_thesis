#!/bin/bash
# CPU corrections and historical T-block on the frozen C149 baseline artifact. No refit.
#SBATCH --job-name=k-baseline
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --open-mode=append
set -euo pipefail

: "${EXPECT_SHA:?EXPECT_SHA required}"
: "${K_TIP:?K_TIP required}"
: "${K_OUT:?K_OUT required}"
: "${K_MODE:?K_MODE required}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
case "$K_MODE" in
  full) [[ "$K_OUT" != *smoke* ]] || { echo "REFUSE: full output path contains smoke" >&2; exit 4; } ;;
  smoke) [[ "$K_OUT" == *smoke* ]] || { echo "REFUSE: smoke output path must contain smoke" >&2; exit 4; } ;;
  *) echo "REFUSE: mode $K_MODE" >&2; exit 4 ;;
esac
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$K_TIP/src"
cd "$K_TIP"
SHA_HEAD="$(git rev-parse HEAD)"
[[ "$SHA_HEAD" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD $SHA_HEAD != $EXPECT_SHA" >&2; exit 5; }
DIRTY="$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)"
[[ -z "$DIRTY" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
[[ ! -e "$K_OUT/provenance.json" ]] || { echo "REFUSE: $K_OUT already has provenance" >&2; exit 4; }
"$PY" scripts/51_temperature_k_baselines.py \
  --artifact "${K_ARTIFACT:-$HOME/official_thesis_s7_c149_baselines/artifacts/c149_fitted_baselines.npz}" \
  --cache-root "${K_CACHE:-$HOME/official_thesis_data/month2_dense_unet_cache_c149}" \
  --stats "${K_STATS:-$HOME/projects/official_thesis_s5/data/manifests/train_norm_stats_month2_fit.json}" \
  --expect-stats-sha256 678263fb4cb2598350b8fd69a71ce4843f747aa61c19556cf7f0f9c5c3ceac0e \
  --mask-root "${K_MASKS:-$HOME/projects/official_thesis_s5/data/processed/month2_sparse_eval_masks}" \
  --s7-baseline-root "${K_S7_BASELINES:-$HOME/official_thesis_s7_c149_baselines}" \
  --out-root "$K_OUT" \
  --eval-git-sha "$SHA_HEAD" \
  --mode "$K_MODE"
