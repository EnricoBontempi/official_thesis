#!/bin/bash
# Opus C149 attempt2 — O-C149-on-s0. Fresh init. No resume.
# Preserves attempt 1 (job 29613545) and does not write into its directory.
#SBATCH --job-name=opus_c149_on_s0_a2
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/bontempi/official_thesis_opus_c149_attempt2_runs/slurm-%j.out
#SBATCH --error=/home/bontempi/official_thesis_opus_c149_attempt2_runs/slurm-%j.err
set -euo pipefail

if [[ "${AUTHORIZE_OPUS_C149_ATTEMPT2:-0}" != "1" ]]; then
  echo "Refusing: set AUTHORIZE_OPUS_C149_ATTEMPT2=1 for the authorized C149 retry only." >&2
  exit 78
fi

# Inlined env: sbatch $0 is a spool copy.
TIP="${TIP:-$HOME/projects/official_thesis_opus_c149_attempt2}"
S5_TIP="${S5_TIP:-$HOME/projects/official_thesis_s5}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
CACHE="${CACHE:-/orcd/pool/005/bontempi/official_thesis_data/month2_dense_unet_cache_c149}"
HASHES="${HASHES:-$CACHE/payload_hashes.json}"
MASKS="${MASKS:-$S5_TIP/data/processed/month2_sparse_eval_masks}"
VERIFY_REPORT="${VERIFY_REPORT:-$CACHE/c149_build_verify_report.json}"
EXPECT_AGG="${EXPECT_AGG:-fa1c9358deb3bda0313cd74f9ab6855af87bc969c4069a463073d4c4120f03b4}"
OUT_ROOT="${OUT_ROOT:-$HOME/official_thesis_opus_c149_attempt2_runs}"
FAILED_OUT="${FAILED_OUT:-$HOME/official_thesis_opus_c149_runs/O-C149-on-s0}"
FAILED_JOB="${FAILED_JOB:-29613545}"
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
  echo "REFUSE: dirty attempt2 checkout" >&2
  printf '%s\n' "$DIRTY" >&2
  exit 5
fi
SHA="$SHA_HEAD"

CFG="configs/experiments/opus_O-C149-on-s0.yaml"
RUN_ID="O-C149-on-s0"
OUT="$OUT_ROOT/$RUN_ID"
mkdir -p "$OUT_ROOT"
if [[ -e "$OUT" ]]; then
  echo "REFUSE: attempt2 output already exists: $OUT" >&2
  exit 4
fi
if [[ ! -d "$FAILED_OUT" ]]; then
  echo "REFUSE: preserved failed attempt dir missing: $FAILED_OUT" >&2
  exit 4
fi

export CACHE HASHES VERIFY_REPORT EXPECT_AGG
if ! timeout 45 "$PY" - <<'PY'
import json, os, sys
from pathlib import Path
cache = Path(os.environ["CACHE"])
report_path = Path(os.environ["VERIFY_REPORT"])
hashes_path = Path(os.environ["HASHES"])
expect_agg = os.environ["EXPECT_AGG"]
missing = [str(p) for p in (cache/"manifest.json", report_path, hashes_path) if not p.is_file()]
if missing:
    sys.exit(f"REFUSE: missing C149 identity files: {missing}")
man = json.loads((cache/"manifest.json").read_text())
rep = json.loads(report_path.read_text())
hashes = json.loads(hashes_path.read_text())
want_counts = {"train_fit": 2980, "seen_train_eval": 220, "val_eval": 123}
want_pool = {"C149": 2980, "A38": 760, "B19": 380}
cache_id = "month2_dense_unet_cache_opus_c149_v1"
errors = []
if rep.get("status") != "ok":
    errors.append(f"report status {rep.get('status')!r}")
if rep.get("cache_id") != cache_id or man.get("cache_id") != cache_id:
    errors.append(f"cache_id report={rep.get('cache_id')!r} manifest={man.get('cache_id')!r}")
if rep.get("counts") != want_counts or man.get("counts") != want_counts:
    errors.append(f"counts report={rep.get('counts')!r} manifest={man.get('counts')!r}")
if rep.get("pool_index_counts") != want_pool:
    errors.append(f"pool counts {rep.get('pool_index_counts')!r}")
if rep.get("train_seen_overlap") != 0:
    errors.append(f"train/seen overlap {rep.get('train_seen_overlap')!r}")
shared = rep.get("shared_a38_vs_historical") or {}
max_abs = shared.get("max_abs_diff") or {}
if shared.get("ok") is not True or shared.get("n_shared_frames_checked") != 760 or shared.get("geometry_identical") is not True:
    errors.append(f"shared gate {shared!r}")
if set(max_abs) != {"vi_norm", "vj_norm", "target_T_norm", "target_T_K", "p_norm"} or any(float(v) != 0.0 for v in max_abs.values()):
    errors.append(f"shared diffs {max_abs!r}")
if not str(man.get("frame_protocol_path", "")).endswith("month2_frame_protocol_opus_c149.yaml"):
    errors.append(f"frame protocol {man.get('frame_protocol_path')!r}")
if hashes.get("aggregate_sha256") != expect_agg or int(hashes.get("n_files") or -1) != 20:
    errors.append(f"payload hash {hashes.get('aggregate_sha256')!r} n={hashes.get('n_files')!r}")
if Path(str(hashes.get("cache_root"))) != cache:
    errors.append(f"hash cache_root {hashes.get('cache_root')!r}")
if errors:
    sys.exit("REFUSE C149 gate: " + "; ".join(errors))
print("C149_ATTEMPT2_GATE_OK", man["counts"], hashes["aggregate_sha256"])
PY
then
  echo "REFUSE: C149 attempt2 gate failed or pool read timed out" >&2
  exit 3
fi

mkdir -p "$OUT"
{
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
  echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
  echo "SLURM_JOB_ACCOUNT=${SLURM_JOB_ACCOUNT:-}"
  echo "SLURM_JOB_QOS=${SLURM_JOB_QOS:-}"
  echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
  echo "GIT_SHA=$SHA"
  echo "CONFIG=$CFG"
  echo "CONFIG_SHA256=$("$PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$CFG")"
  echo "RUN_ID=$RUN_ID"
  echo "SCIENTIFIC_RUN_ID=O-C149-on-s0"
  echo "ATTEMPT=2"
  echo "FAILED_JOB=$FAILED_JOB"
  echo "FAILED_OUT_PRESERVED=$FAILED_OUT"
  echo "CACHE=$CACHE"
  echo "HASHES=$HASHES"
  echo "HASHES_SHA256=$("$PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$HASHES")"
  echo "VERIFY_REPORT=$VERIFY_REPORT"
  echo "MASKS=$MASKS"
  echo "TIP=$TIP"
  echo "WARM_START=false"
  echo "NOTE=opus C149 attempt2; scientific epoch 760/95; wall_cap_internal=5.5h under Slurm 6h; no resume"
  date -Is
} | tee "$OUT/slurm_context.txt"
cat > "$OUT_ROOT/LINEAGE_O-C149-on-s0.txt" <<EOF
scientific_run_id=O-C149-on-s0
attempt=2
failed_job=$FAILED_JOB
failed_out_preserved=$FAILED_OUT
failed_slurm_log_preserved=$HOME/projects/official_thesis_opus_c149/slurm-29613545.out
previous_checkout_preserved=$HOME/projects/official_thesis_opus_c149
attempt2_checkout=$TIP
attempt2_out=$OUT
git_sha=$SHA
EOF

"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"

"$PY" scripts/16_train_sparse_scientific.py \
  --config "$CFG" \
  --authorize-scientific-opus \
  --cache-root "$CACHE" \
  --expect-cache-hashes "$HASHES" \
  --mask-root "$MASKS" \
  --out-root "$OUT_ROOT" \
  --run-id "$RUN_ID" \
  --device cuda \
  --git-sha "$SHA" \
  | tee "$OUT/stdout.log"

echo "OPUS_C149_ATTEMPT2_DONE run=$OUT"
