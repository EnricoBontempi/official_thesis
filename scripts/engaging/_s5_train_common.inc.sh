# Shared env for prepared S5 launchers (sourced by s5_*.sbatch).
# Operational time/mem remain PROVISIONAL until S4 profiling decision.
# Do NOT sbatch these until Enrico authorizes and YAML flags are flipped.

TIP="${TIP:-$HOME/orcd/pool/official_thesis}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
CACHE="${CACHE:-$HOME/orcd/pool/official_thesis_data/month2_dense_unet_cache}"
HASHES="${HASHES:-$HOME/orcd/pool/official_thesis_data/cache_payload_hashes.json}"
MASKS="${MASKS:-$HOME/orcd/pool/official_thesis_data/month2_sparse_eval_masks}"
OUT_ROOT="${OUT_ROOT:-$HOME/orcd/pool/official_thesis_runs}"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$TIP/src"

cd "$TIP"
SHA="$(cat .git_commit 2>/dev/null || git rev-parse HEAD)"
