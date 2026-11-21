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
    plan_extension,
    preserve_tranche,
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


# ---- authorised extension past the initial tranche ------------------------------------
def _finished_tranche(root: Path, *, epoch: int = 1000, status="completed", stop="reached_max_epochs", ck_status="running"):
    _started(root, job="100", restart=1)
    ck = root / RUN / "checkpoints"
    _ckpt(ck / "last.pt", epoch=epoch)
    payload = torch.load(ck / "last.pt", weights_only=False)
    payload["epoch_state"]["status"] = ck_status
    torch.save(payload, ck / "last.pt")
    _ckpt(ck / "best.pt", epoch=630)
    _ckpt(ck / "best_seen_diagnostic.pt", epoch=960)
    (root / RUN / "summary.json").write_text(
        json.dumps({"status": status, "stop_reason": stop, "epochs_completed": epoch})
    )
    (root / RUN / "eval_history.json").write_text("[]")


def test_extension_plans_for_completed_run_with_matching_last_checkpoint(tmp_path):
    _finished_tranche(tmp_path)
    p = plan_extension(tmp_path, RUN, slurm_job_id="200", restart_count=0, new_n_epochs=1100)
    assert p.action == "extend" and p.resume_from == str(tmp_path / RUN / "checkpoints" / "last.pt")
    assert p.details["prev_n_epochs"] == 1000 and p.details["new_n_epochs"] == 1100


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(status="stopped_early_plateau", stop="early_stop_x"),  # a plateau is final
        dict(status="stopped_wall_time", stop="wall_time_cap"),     # that is a plain resume
        dict(ck_status="stopped_early_plateau"),                    # terminal checkpoint
    ],
)
def test_extension_refuses_runs_that_are_not_a_finished_tranche(tmp_path, kwargs):
    _finished_tranche(tmp_path, **kwargs)
    assert plan_extension(tmp_path, RUN, slurm_job_id="200", restart_count=0, new_n_epochs=1100).action == "refuse"


def test_extension_refuses_lower_n_epochs_same_job_restart_and_second_extension(tmp_path):
    _finished_tranche(tmp_path)
    assert plan_extension(tmp_path, RUN, slurm_job_id="200", restart_count=0, new_n_epochs=1000).action == "refuse"
    assert plan_extension(tmp_path, RUN, slurm_job_id="100", restart_count=0, new_n_epochs=1100).action == "refuse"
    assert plan_extension(tmp_path, RUN, slurm_job_id="200", restart_count=1, new_n_epochs=1100).action == "refuse"
    ledger_append(tmp_path, RUN, {"event": "extension_start", "slurm_job_id": "200"})
    assert plan_extension(tmp_path, RUN, slurm_job_id="300", restart_count=0, new_n_epochs=1100).action == "refuse"


def test_extension_refuses_checkpoint_epoch_mismatch(tmp_path):
    _finished_tranche(tmp_path)
    _ckpt(tmp_path / RUN / "checkpoints" / "last.pt", epoch=990)
    assert plan_extension(tmp_path, RUN, slurm_job_id="200", restart_count=0, new_n_epochs=1100).action == "refuse"


def test_preserve_tranche_copies_byte_identical_and_keeps_originals(tmp_path):
    _finished_tranche(tmp_path)
    man = preserve_tranche(tmp_path / RUN, "tranche1_epoch1000", {"reason": "test"})
    dest = tmp_path / RUN / "tranche1_epoch1000"
    for name in ("summary.json", "eval_history.json", "checkpoints/last.pt", "checkpoints/best.pt", "checkpoints/best_seen_diagnostic.pt"):
        assert (dest / name).read_bytes() == (tmp_path / RUN / name).read_bytes()
        assert name in man["files"]
    assert (dest / "MANIFEST.json").is_file()
    with pytest.raises(FileExistsError):
        preserve_tranche(tmp_path / RUN, "tranche1_epoch1000", {})  # never overwrites a preserved tranche
