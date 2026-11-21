"""Launch planning for the S7 C149 training array (fresh start vs. resume).

The S7 array runs on a partition where jobs can be preempted and requeued.
Slurm re-executes the same batch script with the same job id and an increased
``SLURM_RESTART_COUNT``. This module decides, from files on disk only, what the
new attempt may do:

* ``fresh``                     no run directory yet; start from the seeded init.
* ``resume``                    the run's own ``checkpoints/last.pt`` is a complete
                                epoch-boundary checkpoint; continue from it.
* ``already_terminal``          ``summary.json`` records a terminal scientific
                                status; nothing to do (preempted after finishing).
* ``fresh_restart_no_checkpoint``  a previous attempt of THIS job died before
                                writing any complete checkpoint. Its files are moved
                                (never deleted) to ``_discarded_attempts`` and the
                                run starts again from the seeded init. This is
                                recorded as a fresh restart, never as a resume.
* ``extend``                    a NEW job continues a run whose summary is ``completed`` /
                                ``reached_max_epochs`` with a higher ``n_epochs`` (explicitly
                                authorised, see ``plan_extension``). The first tranche is
                                preserved byte-for-byte before anything is written.
* ``refuse``                    anything else (existing directory of an unrelated
                                submission, corrupt last.pt, stopped_error, ...).

``partial_interrupt.pt``, ``best.pt`` and ``best_seen_diagnostic.pt`` are never
resume sources.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_KIND_COMPLETE = "sparse_scientific_epoch_complete_v1"
TERMINAL_STATUSES = frozenset({"completed", "stopped_early_plateau"})
RESUMABLE_SUMMARY_STATUSES = frozenset({"stopped_wall_time"})


@dataclass
class LaunchPlan:
    action: str
    reason: str
    run_dir: str
    resume_from: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def ledger_path(out_root: Path, run_id: str) -> Path:
    return Path(out_root) / "_ledger" / f"{run_id}.jsonl"


def read_ledger(out_root: Path, run_id: str) -> list[dict[str, Any]]:
    path = ledger_path(out_root, run_id)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def ledger_append(out_root: Path, run_id: str, event: dict[str, Any]) -> None:
    path = ledger_path(out_root, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run_id": run_id, **event}
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def inspect_last_checkpoint(path: Path, *, run_id: str) -> dict[str, Any]:
    """Validate that ``path`` is a complete epoch-boundary checkpoint of ``run_id``."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    kind = payload.get("checkpoint_kind")
    if kind != CHECKPOINT_KIND_COMPLETE or not bool(payload.get("resumable")):
        raise ValueError(f"not a complete resumable checkpoint: kind={kind!r}")
    state = payload.get("epoch_state") or {}
    ident = (payload.get("extra") or {}).get("run_identity") or {}
    if ident.get("run_id") != run_id:
        raise ValueError(f"checkpoint run_id {ident.get('run_id')!r} != {run_id!r}")
    for key in ("optimizer_state_dict", "plateau_scheduler_state_dict", "torch_rng_state",
                "numpy_rng_state", "scientific_sampler_state"):
        if payload.get(key) is None:
            raise ValueError(f"checkpoint lacks {key}")
    return {
        "epoch": int(state.get("epoch", -1)),
        "update": int(state.get("update", -1)),
        "status": str(state.get("status")),
        "best_val_epoch": state.get("best_val_epoch"),
        "mask_epoch": int(state.get("mask_epoch", -1)),
    }


def _quarantine_tmp_files(run_dir: Path) -> list[str]:
    moved = []
    ck = run_dir / "checkpoints"
    if not ck.is_dir():
        return moved
    orphans = sorted(p for p in ck.iterdir() if p.name.startswith(".") and ".tmp." in p.name)
    if orphans:
        dest = ck / "_orphan_tmp"
        dest.mkdir(exist_ok=True)
        for p in orphans:
            target = dest / p.name
            shutil.move(str(p), str(target))
            moved.append(p.name)
    return moved


def plan_attempt(
    out_root: Path,
    run_id: str,
    *,
    slurm_job_id: str,
    restart_count: int,
) -> LaunchPlan:
    """Decide fresh/resume/terminal for this attempt. Never deletes data."""
    out_root = Path(out_root)
    run_dir = out_root / run_id
    prior = [r for r in read_ledger(out_root, run_id) if r.get("event") == "attempt_start"]
    same_job = [r for r in prior if str(r.get("slurm_job_id")) == str(slurm_job_id)]

    if not run_dir.exists():
        if prior and not same_job:
            return LaunchPlan(
                "refuse",
                "ledger has attempts from another job but the run directory is missing",
                str(run_dir),
            )
        return LaunchPlan("fresh", "no run directory", str(run_dir))

    if not same_job:
        return LaunchPlan(
            "refuse",
            "run directory exists but this job id never started it (no overwrite, no adoption)",
            str(run_dir),
            details={"existing_entries": sorted(p.name for p in run_dir.iterdir())[:20]},
        )
    if int(restart_count) <= 0:
        return LaunchPlan(
            "refuse",
            "run directory started by this job id but SLURM_RESTART_COUNT=0 (not a requeue)",
            str(run_dir),
        )

    summary_path = run_dir / "summary.json"
    summary_status = None
    if summary_path.is_file():
        try:
            summary_status = json.loads(summary_path.read_text()).get("status")
        except json.JSONDecodeError:
            summary_status = "unreadable"
    if summary_status in TERMINAL_STATUSES:
        return LaunchPlan(
            "already_terminal",
            f"summary.json status={summary_status}",
            str(run_dir),
            details={"summary_status": summary_status},
        )
    if summary_status is not None and summary_status not in RESUMABLE_SUMMARY_STATUSES:
        return LaunchPlan(
            "refuse",
            f"summary.json status={summary_status!r} needs a human decision",
            str(run_dir),
        )

    last = run_dir / "checkpoints" / "last.pt"
    orphans = _quarantine_tmp_files(run_dir)
    if last.is_file():
        try:
            info = inspect_last_checkpoint(last, run_id=run_id)
        except Exception as exc:  # noqa: BLE001
            return LaunchPlan(
                "refuse",
                f"last.pt present but not a valid complete checkpoint: {type(exc).__name__}: {exc}",
                str(run_dir),
            )
        if info["epoch"] < 1:
            return LaunchPlan("refuse", "last.pt epoch < 1", str(run_dir), details=info)
        reason = (
            "continuation after internal wall-time stop"
            if summary_status == "stopped_wall_time"
            else "requeue after interruption; resuming from last complete epoch boundary"
        )
        return LaunchPlan(
            "resume",
            reason,
            str(run_dir),
            resume_from=str(last),
            details={**info, "orphan_tmp_quarantined": orphans},
        )

    # No complete checkpoint: the previous attempt died before epoch 1 finished.
    n = len(list((out_root / "_discarded_attempts").glob(f"{run_id}.*"))) + 1 if (
        out_root / "_discarded_attempts"
    ).exists() else 1
    dest = out_root / "_discarded_attempts" / f"{run_id}.job{slurm_job_id}.attempt{n}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(dest))
    return LaunchPlan(
        "fresh_restart_no_checkpoint",
        "previous attempt left no complete checkpoint; restarting from the seeded init (not a resume)",
        str(run_dir),
        details={"previous_attempt_moved_to": str(dest)},
    )


# ---------------------------------------------------------------------------------------
# Authorised extension of a run that reached its initial n_epochs tranche (never plateau).
# ---------------------------------------------------------------------------------------
TRANCHE_FILES = (
    "summary.json",
    "progress.json",
    "eval_history.json",
    "curves.csv.gz",
    "stream_trace.jsonl",
    "run_identity.json",
    "effective_config.json",
    "init_manifest.json",
    "stdout.log",
)
TRANCHE_CHECKPOINTS = ("best.pt", "last.pt", "best_seen_diagnostic.pt")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plan_extension(
    out_root: Path,
    run_id: str,
    *,
    slurm_job_id: str,
    restart_count: int,
    new_n_epochs: int,
) -> LaunchPlan:
    """Plan the one authorised continuation past the initial n_epochs tranche.

    Only a run whose terminal state is ``completed``/``reached_max_epochs`` (NOT an early
    stop) and whose ``last.pt`` is the complete epoch-boundary checkpoint of that very
    epoch qualifies. The new job must not have started this run, and the run must not have
    been extended before.
    """
    out_root = Path(out_root)
    run_dir = out_root / run_id
    if int(restart_count) != 0:
        return LaunchPlan("refuse", "extension is planned only for the first attempt of its job", str(run_dir))
    if not run_dir.is_dir():
        return LaunchPlan("refuse", "extension requires an existing run directory", str(run_dir))
    rows = read_ledger(out_root, run_id)
    starts = [r for r in rows if r.get("event") == "attempt_start"]
    if not starts:
        return LaunchPlan("refuse", "ledger has no earlier attempt of this run", str(run_dir))
    if any(str(r.get("slurm_job_id")) == str(slurm_job_id) for r in starts):
        return LaunchPlan("refuse", "this job already started the run (use the requeue path)", str(run_dir))
    if any(r.get("event") == "extension_start" for r in rows):
        return LaunchPlan("refuse", "run was already extended once", str(run_dir))
    try:
        summary = json.loads((run_dir / "summary.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return LaunchPlan("refuse", f"summary.json unreadable: {exc}", str(run_dir))
    if summary.get("status") != "completed" or summary.get("stop_reason") != "reached_max_epochs":
        return LaunchPlan(
            "refuse",
            f"only completed/reached_max_epochs runs are extendable (status={summary.get('status')!r}, "
            f"stop_reason={summary.get('stop_reason')!r})",
            str(run_dir),
        )
    prev_epochs = int(summary.get("epochs_completed", -1))
    if int(new_n_epochs) <= prev_epochs:
        return LaunchPlan("refuse", f"new n_epochs {new_n_epochs} <= completed epochs {prev_epochs}", str(run_dir))
    last = run_dir / "checkpoints" / "last.pt"
    try:
        info = inspect_last_checkpoint(last, run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        return LaunchPlan("refuse", f"last.pt invalid: {type(exc).__name__}: {exc}", str(run_dir))
    if info["epoch"] != prev_epochs:
        return LaunchPlan("refuse", f"last.pt epoch {info['epoch']} != summary epochs {prev_epochs}", str(run_dir))
    if info["status"] != "running":
        return LaunchPlan("refuse", f"last.pt status {info['status']!r} is already terminal", str(run_dir))
    return LaunchPlan(
        "extend",
        f"authorised extension {prev_epochs}->{int(new_n_epochs)} epochs from the complete last.pt",
        str(run_dir),
        resume_from=str(last),
        details={**info, "prev_n_epochs": prev_epochs, "new_n_epochs": int(new_n_epochs)},
    )


def preserve_tranche(run_dir: Path, tag: str, provenance: dict[str, Any]) -> dict[str, Any]:
    """Copy (never move) the finished tranche into ``run_dir/<tag>`` with a sha256 manifest."""
    run_dir = Path(run_dir)
    dest = run_dir / tag
    dest.mkdir(parents=False, exist_ok=False)
    entries: dict[str, dict[str, Any]] = {}
    sources = [(run_dir / n, dest / n) for n in TRANCHE_FILES if (run_dir / n).is_file()]
    (dest / "checkpoints").mkdir()
    sources += [
        (run_dir / "checkpoints" / n, dest / "checkpoints" / n)
        for n in TRANCHE_CHECKPOINTS
        if (run_dir / "checkpoints" / n).is_file()
    ]
    for src, dst in sources:
        shutil.copy2(src, dst)
        sha_src, sha_dst = _sha256_file(src), _sha256_file(dst)
        if sha_src != sha_dst:
            raise RuntimeError(f"copy of {src} is not byte-identical")
        entries[str(dst.relative_to(dest))] = {"bytes": dst.stat().st_size, "sha256": sha_dst}
    manifest = {
        "tag": tag,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provenance": provenance,
        "files": entries,
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
