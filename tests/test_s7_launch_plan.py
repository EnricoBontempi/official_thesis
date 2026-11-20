"""S7 launch planner: fresh / resume / terminal / refuse, on disk only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from combustion_recon.training.s7_launch import (
    CHECKPOINT_KIND_COMPLETE,
    ledger_append,
    plan_attempt,
)

RUN = "S7-C149-unet-on-s1"


def _ckpt(path: Path, *, run_id: str = RUN, epoch: int = 5, complete: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_kind": CHECKPOINT_KIND_COMPLETE if complete else "partial",
            "resumable": complete,
            "epoch_state": {"epoch": epoch, "update": epoch * 95, "status": "running", "mask_epoch": epoch - 1},
            "optimizer_state_dict": {"state": {}},
            "plateau_scheduler_state_dict": {},
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": ("MT19937", [1], 1, 0, 0.0),
            "scientific_sampler_state": {"stream_state": {}},
            "extra": {"run_identity": {"run_id": run_id}},
        },
        path,
    )


def _started(root: Path, job: str = "100", restart: int = 0) -> None:
    ledger_append(root, RUN, {"event": "attempt_start", "slurm_job_id": job, "restart_count": restart})


def test_fresh_when_no_directory(tmp_path):
    p = plan_attempt(tmp_path, RUN, slurm_job_id="100", restart_count=0)
    assert p.action == "fresh" and p.resume_from is None


def test_refuses_foreign_existing_directory(tmp_path):
    (tmp_path / RUN / "checkpoints").mkdir(parents=True)
    _ckpt(tmp_path / RUN / "checkpoints" / "last.pt")
    p = plan_attempt(tmp_path, RUN, slurm_job_id="100", restart_count=0)
    assert p.action == "refuse"  # never adopts a directory no ledger attempt created


def test_refuses_same_job_without_requeue(tmp_path):
    _started(tmp_path)
    _ckpt(tmp_path / RUN / "checkpoints" / "last.pt")
    assert plan_attempt(tmp_path, RUN, slurm_job_id="100", restart_count=0).action == "refuse"


def test_requeue_resumes_from_own_last_checkpoint(tmp_path):
    _started(tmp_path)
    _ckpt(tmp_path / RUN / "checkpoints" / "last.pt", epoch=7)
    # partial / best must never be chosen
    _ckpt(tmp_path / RUN / "checkpoints" / "partial_interrupt.pt", complete=False)
    _ckpt(tmp_path / RUN / "checkpoints" / "best.pt", epoch=3)
    orphan = tmp_path / RUN / "checkpoints" / ".last.pt.tmp.123"
    orphan.write_bytes(b"torn")
    p = plan_attempt(tmp_path, RUN, slurm_job_id="100", restart_count=1)
    assert p.action == "resume"
    assert p.resume_from == str(tmp_path / RUN / "checkpoints" / "last.pt")
    assert p.details["epoch"] == 7
    assert not orphan.exists()
    assert (tmp_path / RUN / "checkpoints" / "_orphan_tmp" / ".last.pt.tmp.123").exists()


def test_terminal_summary_is_not_rerun(tmp_path):
    _started(tmp_path)
    _ckpt(tmp_path / RUN / "checkpoints" / "last.pt")
    (tmp_path / RUN / "summary.json").write_text(json.dumps({"status": "stopped_early_plateau"}))
    assert plan_attempt(tmp_path, RUN, slurm_job_id="100", restart_count=2).action == "already_terminal"


def test_wall_time_stop_is_a_resumable_continuation(tmp_path):
    _started(tmp_path)
    _ckpt(tmp_path / RUN / "checkpoints" / "last.pt", epoch=40)
    (tmp_path / RUN / "summary.json").write_text(json.dumps({"status": "stopped_wall_time"}))
    p = plan_attempt(tmp_path, RUN, slurm_job_id="100", restart_count=1)
    assert p.action == "resume" and "wall-time" in p.reason


def test_error_summary_needs_a_human(tmp_path):
    _started(tmp_path)
    _ckpt(tmp_path / RUN / "checkpoints" / "last.pt")
    (tmp_path / RUN / "summary.json").write_text(json.dumps({"status": "stopped_error"}))
    assert plan_attempt(tmp_path, RUN, slurm_job_id="100", restart_count=1).action == "refuse"


def test_no_checkpoint_is_a_labelled_fresh_restart_and_preserves_files(tmp_path):
    _started(tmp_path)
    (tmp_path / RUN).mkdir()
    (tmp_path / RUN / "init_manifest.json").write_text("{}")
    p = plan_attempt(tmp_path, RUN, slurm_job_id="100", restart_count=1)
    assert p.action == "fresh_restart_no_checkpoint"
    moved = Path(p.details["previous_attempt_moved_to"])
    assert (moved / "init_manifest.json").exists()
    assert not (tmp_path / RUN).exists()


def test_corrupt_last_checkpoint_refuses_instead_of_restarting(tmp_path):
    _started(tmp_path)
    (tmp_path / RUN / "checkpoints").mkdir(parents=True)
    (tmp_path / RUN / "checkpoints" / "last.pt").write_bytes(b"not a checkpoint")
    p = plan_attempt(tmp_path, RUN, slurm_job_id="100", restart_count=1)
    assert p.action == "refuse"
    assert (tmp_path / RUN / "checkpoints" / "last.pt").exists()


def test_checkpoint_of_another_run_is_refused(tmp_path):
    _started(tmp_path)
    _ckpt(tmp_path / RUN / "checkpoints" / "last.pt", run_id="someone_else")
    assert plan_attempt(tmp_path, RUN, slurm_job_id="100", restart_count=1).action == "refuse"
