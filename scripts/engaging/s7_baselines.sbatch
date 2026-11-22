#!/bin/bash
# S7 baselines on the C149 pool (CPU only): control on the A38 subset, fit, evaluation, readback.
#SBATCH --job-name=s7_baselines
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
set -euo pipefail
: "${EXPECT_SHA:?}" "${S7_TIP:?}" "${S7_BASE_OUT:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$S7_TIP/src"
cd "$S7_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
[[ -z "$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)" ]] || { echo "REFUSE: dirty checkout" >&2; exit 5; }
M=configs/experiments/month2_s7_c149_eval_matrix.json
eval "$("$PY" - <<'PYM'
import json, shlex
s = json.load(open("configs/experiments/month2_s7_c149_eval_matrix.json"))["shared_inputs"]
for k, v in {"CACHE": s["c149_cache_root"], "CACHE_HASH": s["c149_hash_file"], "CACHE_AGG": s["c149_cache_aggregate_sha256"],
             "MASKS": s["mask_root"], "MASK_FILE": s["eval_mask_manifest_file_sha256"], "MASK_AGG": s["eval_mask_aggregate_payload_sha256"],
             "STATS": s["stats"], "STATS_SHA": s["stats_sha256"]}.items():
    print(f"{k}={shlex.quote(v)}")
PYM
)"
ARGS=(scripts/43_s7_baselines_c149.py --cache-root "$CACHE" --cache-hash-file "$CACHE_HASH" --expect-cache-aggregate-sha256 "$CACHE_AGG"
      --mask-root "$MASKS" --expect-mask-manifest-sha256 "$MASK_FILE" --expect-mask-aggregate-sha256 "$MASK_AGG"
      --stats "$STATS" --expect-stats-sha256 "$STATS_SHA" --out-root "$S7_BASE_OUT" --eval-git-sha "$EXPECT_SHA")
[[ "${S7_BASE_SMOKE:-0}" == "1" ]] && ARGS+=(--smoke)
[[ ! -e "$S7_BASE_OUT" ]] || { echo "REFUSE: output exists: $S7_BASE_OUT" >&2; exit 4; }
"$PY" "${ARGS[@]}"
