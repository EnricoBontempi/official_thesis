"""Phase-2 provenance helper (does not modify the Phase-1 helpers)."""
import subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "piv_phase1"))
import common as C

DIRTY_PATHS = ["scripts/piv_phase1", "scripts/piv_phase2", "src", "tests/test_piv_observations.py",
               "configs/splits", "configs/nonideal", ":(exclude)configs/observations/piv_observation_spec_v1.yaml",
               ":(exclude)configs/observations/piv_observation_spec_v1.sha256"]


def git_dirty() -> bool:
    out = subprocess.check_output(["git", "-C", str(C.REPO), "status", "--porcelain", "--", *DIRTY_PATHS], text=True)
    return bool(out.strip())


def dump_provenance(path, extra=None):
    ex = dict(extra or {})
    ex["code_dirty_phase2_paths"] = git_dirty()
    C.dump_provenance(Path(path), ex)
