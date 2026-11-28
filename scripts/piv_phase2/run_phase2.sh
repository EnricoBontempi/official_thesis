#!/usr/bin/env bash
# Reproduce every Phase-2 artefact (F2.0-F2.6) from a clean commit. CPU only, ~25 min; F2.4/F2.5 need network the first time.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-/home/enric/projects/official_thesis/.venv/bin/python}
OPIV=${OPIV:-$HOME/piv_qualification_data/venv_openpiv/bin/python}
export PYTHONPATH="$REPO/src"
cd "$REPO/scripts/piv_phase2"
bash setup_qualification_data.sh
$PY f20_preliminaries.py
$PY f21_write_spec.py
$PY -m pytest "$REPO/tests/test_piv_observations.py" -q
$PY f26_stats.py
$PY f26_figures.py
( cd /tmp && "$OPIV" -P "$REPO/scripts/piv_phase2/f24_neal.py" --n-pairs 120 --workers 8 )
$PY f24_analyze.py
( cd /tmp && python3 -I "$REPO/scripts/piv_phase2/f25_fetch.py" )
$PY f25_analyze.py
$PY f2_manifest.py
$PY f2_tables.py
