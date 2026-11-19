#!/bin/bash
# Opus A38 seed-0 reference — O-A38-on-s0. Fresh init. Opus stream, not S5 DataLoader.
# Does not write into wave-1, C149, bridge, Astra, or S5 run directories.
#SBATCH --job-name=opus_a38_on_s0
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/bontempi/official_thesis_opus_a38s0_runs/slurm-%j.out
#SBATCH --error=/home/bontempi/official_thesis_opus_a38s0_runs/slurm-%j.err
set -euo pipefail

if [[ "${AUTHORIZE_OPUS_A38_S0:-0}" != "1" ]]; then
  echo "Refusing: set AUTHORIZE_OPUS_A38_S0=1 for the two A38 seed-0 references only." >&2
  exit 78
fi

TIP="${TIP:-$HOME/projects/official_thesis_opus_a38s0}"
S5_TIP="${S5_TIP:-$HOME/projects/official_thesis_s5}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
CACHE="${CACHE:-$S5_TIP/data/processed/month2_dense_unet_cache}"
HASHES="${HASHES:-$S5_TIP/data/manifests/month2_dense_unet_cache_payload_hashes_s5_homecopy.json}"
MASKS="${MASKS:-$S5_TIP/data/processed/month2_sparse_eval_masks}"
STATS="${STATS:-$TIP/data/manifests/train_norm_stats_month2_fit.json}"
OUT_ROOT="${OUT_ROOT:-$HOME/official_thesis_opus_a38s0_runs}"
PRIOR_ROOT="${PRIOR_ROOT:-$HOME/official_thesis_opus_runs}"
EXPECT_AGG="${EXPECT_AGG:-710bab909a0efb6cc1b4b4a0d7f7dcc4889472b7adc8076abb94fd529576ab5d}"
EXPECT_MASK="${EXPECT_MASK:-2e2d8ff6edabf62ad56702a155140d064c0b4014d06ee96debf52811efa5c75e}"
EXPECT_STATS="${EXPECT_STATS:-678263fb4cb2598350b8fd69a71ce4843f747aa61c19556cf7f0f9c5c3ceac0e}"
EXPECT_CACHE_ID="${EXPECT_CACHE_ID:-month2_dense_unet_cache_v1}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$TIP/src"
cd "$TIP"

SHA_HEAD="$(git rev-parse HEAD)"
if [[ -f .git_commit ]]; then
  SHA_FILE="$(tr -d '[:space:]' < .git_commit)"
  if [[ "$SHA_FILE" != "$SHA_HEAD" ]]; then
    echo "REFUSE: .git_commit $SHA_FILE != HEAD $SHA_HEAD" >&2
    exit 5
  fi
fi
DIRTY="$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)"
if [[ -n "$DIRTY" ]]; then
  echo "REFUSE: dirty A38 seed-0 checkout" >&2
  printf '%s\n' "$DIRTY" >&2
  exit 5
fi
SHA="$SHA_HEAD"

CFG="configs/experiments/opus_O-A38-on-s0.yaml"
RUN_ID="O-A38-on-s0"
OUT="$OUT_ROOT/$RUN_ID"
mkdir -p "$OUT_ROOT"
if [[ -e "$OUT" || -e "$PRIOR_ROOT/$RUN_ID" ]]; then
  echo "REFUSE: output already exists: $OUT or $PRIOR_ROOT/$RUN_ID" >&2
  exit 4
fi

export CACHE HASHES MASKS STATS EXPECT_AGG EXPECT_MASK EXPECT_STATS EXPECT_CACHE_ID CFG
if ! timeout 45 "$PY" - << 'PY'
import hashlib, json, os, sys
from pathlib import Path
import yaml

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

cache = Path(os.environ["CACHE"])
manifest = json.loads((cache / "manifest.json").read_text())
if manifest.get("cache_id") != os.environ["EXPECT_CACHE_ID"]:
    sys.exit(f"REFUSE: cache_id {manifest.get('cache_id')}")
hashes = json.loads(Path(os.environ["HASHES"]).read_text())
if hashes.get("aggregate_sha256") != os.environ["EXPECT_AGG"]:
    sys.exit("REFUSE: cache aggregate mismatch")
if sha(Path(os.environ["MASKS"]) / "manifest.json") != os.environ["EXPECT_MASK"]:
    sys.exit("REFUSE: mask manifest mismatch")
if sha(Path(os.environ["STATS"])) != os.environ["EXPECT_STATS"]:
    sys.exit("REFUSE: stats mismatch")
cand = yaml.safe_load(Path(os.environ["CFG"]).read_text())
tc = cand["training_candidate"]
if tc["pool_id"] != "A38" or int(tc["seed"]) != 0 or int(tc["shuffle_seed"]) != 0:
    sys.exit("REFUSE: candidate is not A38 seed 0")
if tc["p_conditioning"] is not True:
    sys.exit("REFUSE: p_on candidate has p_conditioning false")
if int(tc["presentations_per_scientific_epoch"]) != 760:
    sys.exit("REFUSE: scientific epoch is not 760")
if cand["family"] != "sparse_unet":
    sys.exit("REFUSE: family is not sparse_unet")
print("A38_S0_GATE_OK", manifest["cache_id"], hashes["aggregate_sha256"][:12])
PY
then
  echo "REFUSE: A38 seed-0 identity gate failed" >&2
  exit 6
fi

{
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION"
  echo "SLURM_JOB_ACCOUNT=$SLURM_JOB_ACCOUNT"
  echo "SLURM_JOB_QOS=$SLURM_JOB_QOS"
  echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
  echo "GIT_SHA=$SHA"
  echo "CONFIG=$CFG"
  echo "CONFIG_SHA256=$($PY -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$CFG")"
  echo "RUN_ID=$RUN_ID"
  echo "CACHE=$CACHE"
  echo "CACHE_AGGREGATE=$EXPECT_AGG"
  echo "HASHES=$HASHES"
  echo "MASKS=$MASKS"
  echo "MASK_MANIFEST_SHA256=$EXPECT_MASK"
  echo "STATS=$STATS"
  echo "STATS_SHA256=$EXPECT_STATS"
  echo "TIP=$TIP"
  echo "NOTE=A38 seed0 on Opus stream; not S5 DataLoader; wall_cap_internal=5.5h under Slurm 6h"
  date -Is
} | tee "$OUT_ROOT/slurm_context_${RUN_ID}.txt"

mkdir -p "$OUT"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"

"$PY" scripts/16_train_sparse_scientific.py \
  --config "$CFG" \
  --authorize-scientific-opus \
  --cache-root "$CACHE" \
  --expect-cache-hashes "$HASHES" \
  --mask-root "$MASKS" \
  --stats "$STATS" \
  --out-root "$OUT_ROOT" \
  --run-id "$RUN_ID" \
  --device cuda \
  --git-sha "$SHA" \
  | tee "$OUT/stdout.log"

echo "OPUS_A38_S0_DONE run=$OUT"
