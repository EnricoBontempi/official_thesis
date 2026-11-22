#!/bin/bash
# Joint post-processing of steps 2 and 3 (CPU only, no GPU, no training). Writes INDEX + tables only if
# every compatibility check passes; otherwise JOINT_BLOCCO.md and exit 3. Inputs are read, never modified.
#SBATCH --job-name=t_joint
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
: "${EXPECT_SHA:?}" "${T_TIP:?}" "${T_JOINT_DEST:?}" "${T_CURSOR_PACKAGE:?}" "${T_CURSOR_NEURAL:?}" "${T_CURSOR_BASELINE:?}" "${T_V2_ROOT:?}"
PY="${PY:-$HOME/envs/official_thesis/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
cd "$T_TIP"
[[ "$(git rev-parse HEAD)" == "$EXPECT_SHA" ]] || { echo "REFUSE: HEAD != $EXPECT_SHA" >&2; exit 5; }
"$PY" -I scripts/57_t_joint_package.py --cursor-package "$T_CURSOR_PACKAGE" --cursor-neural-root "$T_CURSOR_NEURAL" \
  --cursor-baseline-root "$T_CURSOR_BASELINE" --assisted-eval-root "$T_V2_ROOT" --dest "$T_JOINT_DEST" --git-sha "$EXPECT_SHA"
