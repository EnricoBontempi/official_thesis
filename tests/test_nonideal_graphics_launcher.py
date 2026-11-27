"""The graphics sbatch is executed by Slurm from a spool copy: it must resolve the checkout from explicit input only."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

SBATCH = Path(__file__).resolve().parents[1] / "scripts" / "engaging" / "nonideal_graphics.sbatch"


def _git(cwd: Path, *a: str) -> str:
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True,
                          env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}).stdout.strip()


def _checkout(tmp: Path) -> tuple[Path, str]:
    root = tmp / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "scripts" / "70_nonideal_specialist_graphics.py").write_text("import sys\nprint('STUB', sys.argv[1:])\n")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "x")
    return root, _git(root, "rev-parse", "HEAD")


def _run(spool_copy: Path, env_extra: dict, tmp: Path) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp), "PY": shutil.which("python3") or "python3"}
    env.update({k: str(tmp / k) for k in ("NI_DEST", "NI_EVAL_ROOT", "NI_PACKAGE", "NI_FIELDS_INDEX", "NI_FREEZE")})
    env.update(env_extra)
    return subprocess.run(["bash", str(spool_copy)], cwd=tmp, capture_output=True, text=True, env=env)


def test_spool_copy_with_explicit_root_runs_expected_sha(tmp_path):
    root, sha = _checkout(tmp_path)
    spool = tmp_path / "spool" / "slurm_script"
    spool.parent.mkdir()
    shutil.copy(SBATCH, spool)
    r = _run(spool, {"NI_GRAPHICS_ROOT": str(root), "EXPECT_GRAPHICS_SHA": sha}, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "STUB" in r.stdout and f"sha={sha}" in r.stderr and f"--git-sha', '{sha}'" in r.stdout


def test_refuses_without_explicit_root_or_sha(tmp_path):
    root, sha = _checkout(tmp_path)
    spool = tmp_path / "spool" / "slurm_script"
    spool.parent.mkdir()
    shutil.copy(SBATCH, spool)
    assert _run(spool, {"EXPECT_GRAPHICS_SHA": sha}, tmp_path).returncode != 0
    assert _run(spool, {"NI_GRAPHICS_ROOT": str(root)}, tmp_path).returncode != 0


def test_refuses_wrong_sha_dirty_and_non_checkout(tmp_path):
    root, sha = _checkout(tmp_path)
    spool = tmp_path / "spool" / "slurm_script"
    spool.parent.mkdir()
    shutil.copy(SBATCH, spool)
    assert _run(spool, {"NI_GRAPHICS_ROOT": str(root), "EXPECT_GRAPHICS_SHA": "0" * 40}, tmp_path).returncode == 5
    (root / "scripts" / "dirty.py").write_text("x")
    _git(root, "add", "scripts/dirty.py")
    assert _run(spool, {"NI_GRAPHICS_ROOT": str(root), "EXPECT_GRAPHICS_SHA": sha}, tmp_path).returncode == 5
    other = tmp_path / "notgit"
    other.mkdir()
    assert _run(spool, {"NI_GRAPHICS_ROOT": str(other), "EXPECT_GRAPHICS_SHA": sha}, tmp_path).returncode == 5
