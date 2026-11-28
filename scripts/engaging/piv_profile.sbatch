#!/bin/bash
# F2.7 non-scientific GPU profiling smoke of one campaign cell (U-Net / GLU x IDEAL-A / PIV-A / PIV-B-in).
# Args via env: PIV_TIP, EXPECT_SHA, PIV_RUN (run id), PIV_PRESENT (presentations/epoch), PIV_VALFRAMES
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -uo pipefail
cd "${PIV_TIP:?}" || exit 2
[[ "$(git rev-parse HEAD)" == "${EXPECT_SHA:?}" ]] || { echo "REFUSE sha"; exit 5; }
PY="$HOME/envs/official_thesis/bin/python"; export PYTHONPATH="$PIV_TIP/src" PYTHONUNBUFFERED=1 PIV_PROFILE=1
M=configs/experiments/piv_campaign_matrix.json
eval "$($PY scripts/engaging/s7_plan.py row $M ${PIV_RUN:?})"
OUT=/home/bontempi/official_thesis_piv_smoke_runs
rm -rf "$OUT/$RUN_ID"
$PY scripts/16_train_sparse_scientific.py --config "$CONFIG" --authorize-scientific-piv --cache-root "$CACHE" --expect-cache-hashes "$CACHE_HASHES" \
  --mask-root "$MASKS" --expect-mask-manifest-sha256 "$EXPECT_MASK_FILE" --expect-mask-aggregate-sha256 "$EXPECT_MASK_AGG" \
  --expect-stats-sha256 "$EXPECT_STATS" --expect-cache-aggregate-sha256 "$EXPECT_CACHE_AGG" --out-root "$OUT" --run-id "$RUN_ID" --device cuda --git-sha "$EXPECT_SHA" \
  --smoke --smoke-presentations "${PIV_PRESENT:-760}" --smoke-epochs 3 --smoke-eval-every 3 --smoke-eval-frames "${PIV_VALFRAMES:-123}" 
