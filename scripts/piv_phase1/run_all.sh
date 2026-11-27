#!/usr/bin/env bash
# Reproduce every Phase-1 (F1.1-F1.3) artefact. Run from the repository root of the execution worktree.
# Requires: the gitignored payloads under $DATA_ROOT (default /home/enric/projects/official_thesis) and a Python with
# numpy, scipy, pandas, matplotlib, pyarrow, pyyaml (the project venv: $DATA_ROOT/.venv).
set -euo pipefail
PY=${PY:-/home/enric/projects/official_thesis/.venv/bin/python}
cd "$(dirname "$0")"
$PY f12_kernel_registry.py
$PY f12_grid_support.py
$PY f12b_dkern_width_matched.py
$PY f12c_b_window_content.py
$PY f12_kernel_registry.py --swap-cells --out ../../results/piv/phase1/f12_swapcells
$PY f12_grid_support.py --swap-cells --out ../../results/piv/phase1/f12_swapcells --registry ../../results/piv/phase1/f12_swapcells/kernel_registry.json
$PY f13_vref.py
$PY f11_orientation_check.py
$PY f11_analyze.py
$PY f11b_method_validation.py
$PY f11c_mean_symmetry.py
$PY f11d_symmetry_and_decorrelation.py
$PY make_tables.py
$PY make_figures.py
