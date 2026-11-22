#!/bin/bash
# S7 residual: swap of the FROZEN A38 GappyPOD objects (CPU only).
#SBATCH --job-name=s7_gappy_swap
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
: "${EXPECT_SHA:?}" "${S7_TIP:?}" "${S7_GSWAP_OUT:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$S7_TIP/src"
cd "$S7_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
S5=/home/bontempi/projects/official_thesis_s5
A=/home/bontempi/official_thesis_gappypod_nested_1_20_amf_v2
ARGS=(scripts/44_s7_gappy_swap_a38.py --artifact "$A/artifacts/gappypod_fitted_pod_p_off_on.npz"
      --expect-artifact-sha256 a6fbdfd2a0409cd11995247ab93db694d0c440d677f810b433daf2fe10a74ac9
      --s6-per-frame-5pct "$A/per_frame_5pct_refit.csv.gz" --cache-root "$S5/data/processed/month2_dense_unet_cache"
      --cache-hash-file "$S5/data/manifests/month2_dense_unet_cache_payload_hashes_s5_homecopy.json"
      --expect-cache-aggregate-sha256 710bab909a0efb6cc1b4b4a0d7f7dcc4889472b7adc8076abb94fd529576ab5d
      --mask-root "$S5/data/processed/month2_sparse_eval_masks"
      --expect-mask-manifest-sha256 2e2d8ff6edabf62ad56702a155140d064c0b4014d06ee96debf52811efa5c75e
      --expect-mask-aggregate-sha256 7be78f35ffea2ebc598adce8cec39dc719339aac17ecadde3137f2064891ffeb
      --stats "$S5/data/manifests/train_norm_stats_month2_fit.json"
      --expect-stats-sha256 678263fb4cb2598350b8fd69a71ce4843f747aa61c19556cf7f0f9c5c3ceac0e
      --out-root "$S7_GSWAP_OUT" --eval-git-sha "$EXPECT_SHA")
[[ "${S7_GSWAP_SMOKE:-0}" == "1" ]] && ARGS+=(--smoke)
[[ ! -e "$S7_GSWAP_OUT" ]] || { echo "REFUSE: output exists" >&2; exit 4; }
"$PY" "${ARGS[@]}"
