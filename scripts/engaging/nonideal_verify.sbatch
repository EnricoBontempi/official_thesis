#!/bin/bash
# CPU: verify the 12 run directories (afterany of the evaluation array). Writes VERIFY_OK.json or BLOCK_REPORT.md.
#SBATCH --job-name=ni_verify
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -uo pipefail
: "${EXPECT_SHA:?}" "${NI_TIP:?}" "${NI_SPEC:?}" "${NI_OUT_ROOT:?}" "${NI_DEST:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH; export PYTHONPATH="$NI_TIP/src"; cd "$NI_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
mkdir -p "$NI_DEST"
"$PY" scripts/61_nonideal_verify.py --spec "$NI_SPEC" --eval-root "$NI_OUT_ROOT" --dest "$NI_DEST/verify"
