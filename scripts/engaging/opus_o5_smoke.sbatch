#!/bin/bash
# Opus O5 SMOKE: same entrypoint and environment, few frames, non-scientific.
# Attempt2 runs only task 6 (O-C149-on-s0: C149 cache). Task 0 passed in job 29717341.
# node4108 excluded after job 29717341 task 6 waited an hour in autofs_mount_wait there.
# Never produces COMPLETE.json. Not chained to the campaign.
#SBATCH --job-name=opus_o5_smoke
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=node4108
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=6
#SBATCH --output=/home/bontempi/official_thesis_opus_o5_smoke_attempt2_runs/slurm-%A_%a.out
#SBATCH --error=/home/bontempi/official_thesis_opus_o5_smoke_attempt2_runs/slurm-%A_%a.err
set -euo pipefail

if [[ -z "${EXPECT_SHA:-}" ]]; then
  echo "Refusing: EXPECT_SHA is required." >&2
  exit 78
fi

TIP="${TIP:-$HOME/projects/official_thesis_opus_o5_v3}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
OUT_ROOT="${OUT_ROOT:-$HOME/official_thesis_opus_o5_smoke_attempt2_runs}"
if [[ "$OUT_ROOT" != *smoke* ]]; then
  echo "REFUSE: smoke OUT_ROOT must contain 'smoke', got $OUT_ROOT" >&2
  exit 4
fi
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH
export PYTHONPATH="$TIP/src"
cd "$TIP"

SHA_HEAD="$(git rev-parse HEAD)"
if [[ "$SHA_HEAD" != "$EXPECT_SHA" ]]; then
  echo "REFUSE: HEAD $SHA_HEAD != EXPECT_SHA $EXPECT_SHA" >&2
  exit 5
fi
if [[ -f .git_commit ]]; then
  SHA_FILE="$(tr -d '[:space:]' < .git_commit)"
  if [[ "$SHA_FILE" != "$SHA_HEAD" ]]; then
    echo "REFUSE: .git_commit $SHA_FILE != HEAD $SHA_HEAD" >&2
    exit 5
  fi
fi
DIRTY="$(git status --porcelain | grep -v '^[?][?] \.git_commit$' || true)"
if [[ -n "$DIRTY" ]]; then
  echo "REFUSE: dirty O5 checkout" >&2
  printf '%s\n' "$DIRTY" >&2
  exit 5
fi

mapfile -t RUNS < <("$PY" - << 'PY'
import json
from pathlib import Path
matrix = json.loads(Path("configs/experiments/opus_o5_matrix.json").read_text())
for row in matrix["runs"]:
    print(row["run_id"])
PY
)
RUN_ID="${RUNS[$SLURM_ARRAY_TASK_ID]}"

# Light accessibility check of the inputs this run consumes, taken from the matrix by
# scripts/engaging/o5_preflight_paths.py. The timeout bounds each stat only; it does not
# guarantee that a process stuck in D state can be stopped.
echo "[$(date -Is)] launcher host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-na} run=$RUN_ID" >&2
NEEDED="$("$PY" scripts/engaging/o5_preflight_paths.py configs/experiments/opus_o5_matrix.json "$RUN_ID")"
if [[ -z "$NEEDED" ]]; then
  echo "REFUSE: empty input list for $RUN_ID" >&2
  exit 6
fi
ACCESS_FAIL=0
while read -r P; do
  if timeout 120 stat -c '%s' "$P" > /dev/null 2>&1; then
    echo "[$(date -Is)] access ok: $P" >&2
  else
    echo "[$(date -Is)] ACCESS FAIL or timeout: $P" >&2
    ACCESS_FAIL=1
  fi
done <<< "$NEEDED"
if [[ "$ACCESS_FAIL" != "0" ]]; then
  echo "REFUSE: an input of $RUN_ID is not reachable from $(hostname)" >&2
  exit 6
fi
OUT="$OUT_ROOT/$RUN_ID"
mkdir -p "$OUT_ROOT"
if [[ -e "$OUT" ]]; then
  echo "REFUSE: output already exists: $OUT" >&2
  exit 4
fi

"$PY" scripts/26_opus_o5_eval.py \
  --matrix configs/experiments/opus_o5_matrix.json \
  --run-id "$RUN_ID" \
  --out-root "$OUT_ROOT" \
  --eval-git-sha "$SHA_HEAD" \
  --device cuda \
  --smoke --smoke-frames "${SMOKE_FRAMES:-6}"
