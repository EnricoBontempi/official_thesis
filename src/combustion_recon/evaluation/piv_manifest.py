"""Run-manifest schema, provenance validator, run-status auditor, and resumability helper.

Mission 02 ("PIV virtuale") post-processing takes an explicit **run manifest**
(``piv_postproc_manifest_v1``) instead of hardcoding paths, Slurm job IDs, or future
checkpoints.  This module provides:

* Strict schema validation and TEST-split refusal (unless an explicit R5 authorization
  artifact is verified — never active by default).
* Exact roadmap §7.2 campaign taxonomy (necessary groups ``T1``–``T7`` = 35 runs;
  conditional groups ``T8``–``T11``; Phase 3 frozen-checkpoint evaluations ``P3``).
* Mutually exclusive run-status auditing that separately reports ``missing``, ``failed``,
  ``incomplete``, ``budget_exhausted``, and ``plateau`` (training) as well as ``missing``,
  ``failed``, ``incomplete``, ``same_as_best``, and ``complete`` (evaluation).
* A-priori fixed representative frame list (frozen before seeing any training results)
  and deterministic best/worst frame selection rule.
* Idempotent stage-level resumability tracker (:class:`StageStateManager`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from combustion_recon.training import nonideal_campaign as nc

MANIFEST_SCHEMA = "piv_postproc_manifest_v1"

ALLOWED_DEV_ROLES: tuple[str, ...] = ("seen_train_eval", "val_eval")
FORBIDDEN_ROLE_PREFIXES: tuple[str, ...] = ("test",)
DEFAULT_REPLICAS: tuple[int, ...] = (0, 1, 2)
DEFAULT_CHECKPOINTS: tuple[str, ...] = ("best", "last")

ROLE_EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "seen_train_eval": {"n_frames": 220, "n_conditions": 20},
    "val_eval": {"n_frames": 123, "n_conditions": 3},
}

# Exact necessary campaign matrix from roadmap_tecnica_piv.md §7.2 (35 runs)
NECESSARY_CAMPAIGN_SPEC: tuple[dict[str, Any], ...] = (
    {"group": "T1", "family": "sparse_unet", "train_cell": "IDEAL-A", "train_config_id": "IDEAL-A", "b_position": None, "n_seeds": 5},
    {"group": "T2", "family": "sparse_unet", "train_cell": "PIV-A", "train_config_id": "PIV-A", "b_position": None, "n_seeds": 5},
    {"group": "T3", "family": "sparse_unet", "train_cell": "IDEAL-B", "train_config_id": "IDEAL-B-in", "b_position": "B-in", "n_seeds": 5},
    {"group": "T4", "family": "sparse_unet", "train_cell": "PIV-B", "train_config_id": "PIV-B-in", "b_position": "B-in", "n_seeds": 5},
    {"group": "T5", "family": "adapted_glu", "train_cell": "IDEAL-A", "train_config_id": "IDEAL-A", "b_position": None, "n_seeds": 3},
    {"group": "T5", "family": "adapted_glu", "train_cell": "PIV-A", "train_config_id": "PIV-A", "b_position": None, "n_seeds": 3},
    {"group": "T6", "family": "adapted_glu", "train_cell": "IDEAL-B", "train_config_id": "IDEAL-B-in", "b_position": "B-in", "n_seeds": 3},
    {"group": "T6", "family": "adapted_glu", "train_cell": "PIV-B", "train_config_id": "PIV-B-in", "b_position": "B-in", "n_seeds": 3},
    {"group": "T7", "family": "adapted_glu", "train_cell": "BRIDGE-5pct-IDEAL", "train_config_id": "E0", "b_position": None, "n_seeds": 3},
)

CONDITIONAL_GROUPS: frozenset[str] = frozenset({"T8", "T9", "T10", "T11"})
NECESSARY_GROUPS: frozenset[str] = frozenset({"T1", "T2", "T3", "T4", "T5", "T6", "T7"})
PHASE3_GROUPS: frozenset[str] = frozenset({"P3"})

# Evaluation scenario role classification (confirmatory vs sensitivity vs bridge)
MAIN_EVAL_CONFIGS: tuple[str, ...] = (
    "IDEAL-full",
    "IDEAL-A",
    "FILT-A",
    "PIV-A",
    "IDEAL-B-in",
    "FILT-B-in",
    "PIV-B-in",
    "IDEAL-B-down",
    "FILT-B-down",
    "PIV-B-down",
    "IDEAL-B-off",
    "FILT-B-off",
    "PIV-B-off",
)

SENSITIVITY_EVAL_CONFIGS: tuple[str, ...] = (
    "SENS-TH-A",
    "SENS-SWAP-A",
    "SENS-EDGE-A",
    "SENS-SIGLO-A",
    "SENS-SIGHI-A",
    "SENS-RHO0-A",
    "SENS-W1.0-A",
    "SENS-W1.9-A",
    "SENS-REJ-iso-flag-A",
    "SENS-REJ-iso-sub-A",
    "SENS-REJ-clu-flag-A",
    "SENS-REJ-clu-sub-A",
    "SENS-TH-B-in",
    "SENS-SWAP-B-in",
    "SENS-EDGE-B-in",
    "SENS-SIGLO-B-in",
    "SENS-SIGHI-B-in",
    "SENS-RHO0-B-in",
)

BRIDGE_AND_PHASE3_CONFIGS: tuple[str, ...] = ("E0", "E0v", "E1", "E2", "E4*", "E5*")

# A-priori fixed representative frames (selected before any training results;
# VAL: all 3 conditions at frames 400 and 1000 matching F2.6 & Mission 00;
# seen_train_eval: the 3 F2.6 v_ref-anchored TRAIN conditions at frames 1600 and 2000).
DEFAULT_REPRESENTATIVE_FRAMES: dict[str, list[dict[str, Any]]] = {
    "val_eval": [
        {"condition": "0NH3_1.05.h5", "frame_index": 400, "tag": "val_high_vref_early"},
        {"condition": "0NH3_1.05.h5", "frame_index": 1000, "tag": "val_high_vref_mid"},
        {"condition": "20NH3_1.1.h5", "frame_index": 400, "tag": "val_low_vref_early"},
        {"condition": "20NH3_1.1.h5", "frame_index": 1000, "tag": "val_low_vref_mid"},
        {"condition": "40NH3_0.9.h5", "frame_index": 400, "tag": "val_med_vref_early"},
        {"condition": "40NH3_0.9.h5", "frame_index": 1000, "tag": "val_med_vref_mid"},
    ],
    "seen_train_eval": [
        {"condition": "80NH3_1.2.h5", "frame_index": 1600, "tag": "seen_low_vref_first"},
        {"condition": "80NH3_1.2.h5", "frame_index": 2000, "tag": "seen_low_vref_last"},
        {"condition": "40NH3_1.1.h5", "frame_index": 1600, "tag": "seen_med_vref_first"},
        {"condition": "40NH3_1.1.h5", "frame_index": 2000, "tag": "seen_med_vref_last"},
        {"condition": "0NH3_0.85.h5", "frame_index": 1600, "tag": "seen_high_vref_first"},
        {"condition": "0NH3_0.85.h5", "frame_index": 2000, "tag": "seen_high_vref_last"},
    ],
}

# Run status constants (mutually exclusive)
TRAIN_STATUS_MISSING = "missing"
TRAIN_STATUS_FAILED = "failed"
TRAIN_STATUS_INCOMPLETE = "incomplete"
TRAIN_STATUS_BUDGET_EXHAUSTED = "budget_exhausted"
TRAIN_STATUS_PLATEAU = "plateau"
TRAIN_STATUS_FROZEN_EXTERNAL = "frozen_external"  # Phase 3 pre-existing checkpoints

EVAL_STATUS_MISSING = "missing"
EVAL_STATUS_FAILED = "failed"
EVAL_STATUS_INCOMPLETE = "incomplete"
EVAL_STATUS_SAME_AS_BEST = "same_as_best"
EVAL_STATUS_COMPLETE = "complete"


class ManifestError(ValueError):
    """Raised when a run manifest violates the post-processing contract."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def load_manifest(path: str | Path, *, allow_test_with_r5_auth: bool = False) -> dict[str, Any]:
    """Load and validate a ``piv_postproc_manifest_v1`` JSON or YAML file."""
    p = Path(path)
    if not p.is_file():
        raise ManifestError(f"manifest file not found: {p}")
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if p.suffix.lower() in (".yaml", ".yml") else json.loads(text)
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be a JSON/YAML mapping")
    if data.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"unexpected manifest schema {data.get('schema')!r}; expected {MANIFEST_SCHEMA!r}")

    roles = list(data.get("authorized_roles") or ALLOWED_DEV_ROLES)
    for role in roles:
        if any(str(role).lower().startswith(pref) for pref in FORBIDDEN_ROLE_PREFIXES):
            r5 = data.get("r5_test_authorization")
            if not (allow_test_with_r5_auth and isinstance(r5, dict) and r5.get("authorized") is True and r5.get("preregistration_sha256")):
                raise ManifestError(
                    f"TEST role {role!r} is strictly refused without explicit R5 authorization"
                )
        elif role not in ALLOWED_DEV_ROLES:
            raise ManifestError(f"unknown authorized role {role!r}")

    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ManifestError("manifest must contain a non-empty 'runs' list")

    seen_ids: set[str] = set()
    required_run_keys = ("run_id", "group", "family", "train_cell", "model_seed")
    for idx, r in enumerate(runs):
        if not isinstance(r, dict):
            raise ManifestError(f"runs[{idx}] must be a mapping")
        for k in required_run_keys:
            if k not in r or r[k] is None:
                raise ManifestError(f"runs[{idx}] missing required key {k!r}")
        rid = str(r["run_id"])
        if rid in seen_ids:
            raise ManifestError(f"duplicate run_id in manifest: {rid!r}")
        seen_ids.add(rid)
        if r["family"] not in ("sparse_unet", "adapted_glu", "dense_unet"):
            raise ManifestError(f"{rid}: unsupported family {r['family']!r}")
        if r.get("b_position") not in (None, "B-in", "B-down", "B-off"):
            raise ManifestError(f"{rid}: unsupported b_position {r.get('b_position')!r}")

    # Normalize optional top-level defaults without mutating caller expectations
    data.setdefault("authorized_roles", roles)
    data.setdefault("expected_replicas", list(DEFAULT_REPLICAS))
    data.setdefault("expected_checkpoints", list(DEFAULT_CHECKPOINTS))
    data.setdefault("representative_frames", DEFAULT_REPRESENTATIVE_FRAMES)
    data["_manifest_path"] = str(p.resolve())
    data["_manifest_sha256"] = sha256_file(p)
    return data


def _read_json_safe(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return (obj if isinstance(obj, dict) else None), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path.name}: {exc}"


def _audit_training_dir(
    run: Mapping[str, Any],
    *,
    expect_spec_sha: str | None = None,
    expect_git_sha: str | None = None,
    initial_epochs: int = nc.INITIAL_EPOCHS,
    final_epochs: int = nc.FINAL_EPOCHS,
    smoke: bool = False,
) -> dict[str, Any]:
    """Classify one run's training state into missing / failed / incomplete / budget_exhausted / plateau."""
    rid = str(run["run_id"])
    if run.get("group") in PHASE3_GROUPS or run.get("frozen_external"):
        ck_best = run.get("checkpoint_best_path")
        if ck_best and Path(ck_best).is_file():
            return {
                "training_status": TRAIN_STATUS_FROZEN_EXTERNAL,
                "training_valid": True,
                "training_outcome": "frozen_external_checkpoint",
                "budget_exhausted": False,
                "declared_converged": None,
                "epochs_completed": run.get("best_epoch"),
                "best_val_epoch": run.get("best_epoch"),
                "best_val_metric": None,
                "last_same_as_best": False,
                "training_problems": [],
                "training_git_sha": run.get("training_git_sha"),
                "init_state_dict_sha256": None,
                "best_checkpoint_path": str(ck_best),
                "last_checkpoint_path": None,
            }
        return {
            "training_status": TRAIN_STATUS_MISSING,
            "training_valid": False,
            "training_outcome": "missing_external_checkpoint",
            "budget_exhausted": False,
            "declared_converged": False,
            "epochs_completed": None,
            "best_val_epoch": None,
            "best_val_metric": None,
            "last_same_as_best": False,
            "training_problems": ["frozen_external checkpoint_best_path missing"],
            "training_git_sha": None,
            "init_state_dict_sha256": None,
            "best_checkpoint_path": str(ck_best or ""),
            "last_checkpoint_path": None,
        }

    tdir_raw = run.get("training_dir")
    if not tdir_raw or not Path(tdir_raw).is_dir():
        return {
            "training_status": TRAIN_STATUS_MISSING,
            "training_valid": False,
            "training_outcome": "missing_training_dir",
            "budget_exhausted": False,
            "declared_converged": False,
            "epochs_completed": None,
            "best_val_epoch": None,
            "best_val_metric": None,
            "last_same_as_best": False,
            "training_problems": [f"training_dir missing: {tdir_raw}"],
            "training_git_sha": None,
            "init_state_dict_sha256": None,
            "best_checkpoint_path": None,
            "last_checkpoint_path": None,
        }

    tdir = Path(tdir_raw)
    problems: list[str] = []
    summary, sum_err = _read_json_safe(tdir / "summary.json")
    progress, _ = _read_json_safe(tdir / "progress.json")
    ident, ident_err = _read_json_safe(tdir / "run_identity.json")
    init_man, _ = _read_json_safe(tdir / "init_manifest.json")

    if sum_err:
        problems.append(f"corrupt {sum_err}")
    if ident_err:
        problems.append(f"corrupt {ident_err}")

    best_pt = tdir / "checkpoints" / "best.pt"
    last_pt = tdir / "checkpoints" / "last.pt"
    best_exists = best_pt.is_file()
    last_exists = last_pt.is_file()

    # Determine core status from summary.json (reusing Mission 00 semantics)
    if sum_err:
        status = TRAIN_STATUS_FAILED
        outcome = "corrupt_summary"
    elif summary is None:
        status = TRAIN_STATUS_INCOMPLETE
        outcome = "no_summary_json"
        problems.append(f"summary.json missing (progress_epoch={(progress or {}).get('epoch')})")
    else:
        st = summary.get("status")
        stop = summary.get("stop_reason")
        epochs = int(summary.get("epochs_completed") or -1)
        if st == "stopped_early_plateau":
            status = TRAIN_STATUS_PLATEAU
            outcome = nc.OUTCOME_PLATEAU
        elif st == "completed" and stop == "reached_max_epochs":
            if smoke:
                status = TRAIN_STATUS_PLATEAU
                outcome = "smoke_completed"
            elif epochs == initial_epochs:
                status = TRAIN_STATUS_INCOMPLETE
                outcome = "needs_single_continuation_1000_to_1200"
                problems.append(f"reached {initial_epochs} epochs without plateau; continuation to {final_epochs} pending")
            elif epochs == final_epochs:
                status = TRAIN_STATUS_BUDGET_EXHAUSTED
                outcome = nc.OUTCOME_BUDGET
            else:
                status = TRAIN_STATUS_FAILED
                outcome = f"unexpected_completed_epochs_{epochs}"
                problems.append(f"completed at unexpected epoch count {epochs}")
        elif st == "stopped_wall_time":
            status = TRAIN_STATUS_INCOMPLETE
            outcome = "stopped_wall_time"
            problems.append("stopped by wall_time_cap; resume not completed")
        elif str(st or "").startswith("stopped_error") or str(stop or "").startswith("exception:"):
            status = TRAIN_STATUS_FAILED
            outcome = f"error:{stop}"
            problems.append(f"training error: {st} / {stop}")
        else:
            status = TRAIN_STATUS_FAILED
            outcome = f"unexpected_status:{st}"
            problems.append(f"unexpected status={st!r} stop_reason={stop!r}")

    # Check provenance & checkpoint presence for runs claiming completion
    if status in (TRAIN_STATUS_PLATEAU, TRAIN_STATUS_BUDGET_EXHAUSTED):
        if not best_exists:
            problems.append("checkpoints/best.pt missing")
            status = TRAIN_STATUS_INCOMPLETE
        if not last_exists:
            problems.append("checkpoints/last.pt missing")
            status = TRAIN_STATUS_INCOMPLETE
        if ident:
            if ident.get("run_id") and ident.get("run_id") != rid:
                problems.append(f"run_identity run_id={ident.get('run_id')!r} != {rid!r}")
                status = TRAIN_STATUS_FAILED
            if ident.get("family") and ident.get("family") != run["family"]:
                problems.append(f"run_identity family={ident.get('family')!r} != {run['family']!r}")
                status = TRAIN_STATUS_FAILED
            if expect_git_sha and ident.get("git_sha") and ident.get("git_sha") != expect_git_sha and not smoke:
                problems.append(f"training git_sha {ident.get('git_sha')} != expected {expect_git_sha}")
                status = TRAIN_STATUS_FAILED
            piv_block = ident.get("piv_observation") or ident.get("piv_campaign") or {}
            if expect_spec_sha and piv_block.get("spec_sha256") and piv_block.get("spec_sha256") != expect_spec_sha:
                problems.append("training observation spec_sha256 differs from manifest spec")
                status = TRAIN_STATUS_FAILED
            if ident.get("non_scientific_smoke") and not smoke:
                problems.append("run_identity marks a non-scientific smoke run")
                status = TRAIN_STATUS_FAILED

    epochs_done = (
        int(summary["epochs_completed"])
        if summary and summary.get("epochs_completed") is not None
        else ((progress or {}).get("epoch"))
    )
    best_epoch = summary.get("best_val_epoch") if summary else None
    best_metric = summary.get("best_val_metric") if summary else None
    if best_metric is not None and not math.isfinite(float(best_metric)):
        problems.append("non-finite best_val_metric in summary.json")
        status = TRAIN_STATUS_FAILED

    # Detect last_same_as_best via file SHA when both exist (fast, no torch load required)
    last_same = False
    if best_exists and last_exists:
        last_same = bool(
            best_epoch is not None
            and epochs_done is not None
            and int(best_epoch) == int(epochs_done)
            and sha256_file(best_pt) == sha256_file(last_pt)
        )

    is_budget = status == TRAIN_STATUS_BUDGET_EXHAUSTED
    valid = status in (TRAIN_STATUS_PLATEAU, TRAIN_STATUS_BUDGET_EXHAUSTED) and not problems
    return {
        "training_status": status,
        "training_valid": valid,
        "training_outcome": outcome,
        "budget_exhausted": is_budget,
        "declared_converged": False if is_budget else (None if valid else False),
        "epochs_completed": epochs_done,
        "best_val_epoch": best_epoch,
        "best_val_metric": best_metric,
        "clip_events": (summary or {}).get("clip_events"),
        "last_same_as_best": last_same,
        "training_problems": problems,
        "training_git_sha": (ident or {}).get("git_sha"),
        "init_state_dict_sha256": (init_man or {}).get("init_state_dict_sha256"),
        "best_checkpoint_path": str(best_pt) if best_exists else None,
        "last_checkpoint_path": str(last_pt) if last_exists else None,
    }


def _audit_eval_dir(
    run: Mapping[str, Any],
    *,
    kind: str,
    expected_configs: Sequence[str],
    authorized_roles: Sequence[str],
    expected_replicas: Sequence[int],
    expect_spec_sha: str | None = None,
) -> dict[str, Any]:
    """Classify one (run, checkpoint_kind) evaluation state."""
    edir_raw = run.get("eval_dir")
    if not edir_raw:
        return {
            "eval_status": EVAL_STATUS_MISSING,
            "eval_valid": False,
            "gate_all_pass": None,
            "n_expected_units": len(expected_configs) * len(authorized_roles) * len(expected_replicas),
            "n_present_units": 0,
            "eval_problems": ["eval_dir not specified in manifest"],
            "eval_source_dir": None,
            "eval_git_sha": None,
            "checkpoint_weights_sha256": None,
            "checkpoint_file_sha256": None,
            "checkpoint_epoch": None,
            "spec_sha256": None,
        }

    edir = Path(edir_raw)
    kdir = edir / kind
    if not kdir.is_dir():
        return {
            "eval_status": EVAL_STATUS_MISSING,
            "eval_valid": False,
            "gate_all_pass": None,
            "n_expected_units": len(expected_configs) * len(authorized_roles) * len(expected_replicas),
            "n_present_units": 0,
            "eval_problems": [f"evaluation directory missing: {kdir}"],
            "eval_source_dir": None,
            "eval_git_sha": None,
            "checkpoint_weights_sha256": None,
            "checkpoint_file_sha256": None,
            "checkpoint_epoch": None,
            "spec_sha256": None,
        }

    # Handle last == best alias cleanly without duplicate evaluation
    same_marker = kdir / "SAME_AS_BEST.json"
    if kind == "last" and same_marker.is_file():
        best_res = _audit_eval_dir(
            run,
            kind="best",
            expected_configs=expected_configs,
            authorized_roles=authorized_roles,
            expected_replicas=expected_replicas,
            expect_spec_sha=expect_spec_sha,
        )
        if best_res["eval_status"] == EVAL_STATUS_COMPLETE:
            return {
                **best_res,
                "eval_status": EVAL_STATUS_SAME_AS_BEST,
                "eval_valid": True,
                "eval_source_dir": str(edir / "best"),
            }
        return {
            **best_res,
            "eval_status": EVAL_STATUS_INCOMPLETE,
            "eval_valid": False,
            "eval_problems": ["SAME_AS_BEST points to incomplete/invalid best"] + list(best_res["eval_problems"]),
        }

    problems: list[str] = []
    gate_failed = (kdir / "GATE_FAILED.json").is_file()
    gate, gate_err = _read_json_safe(kdir / "gate.json")
    prov, prov_err = _read_json_safe(kdir / "provenance.json")
    comp, comp_err = _read_json_safe(kdir / "COMPLETE.json")

    if gate_err:
        problems.append(f"corrupt {gate_err}")
    if prov_err:
        problems.append(f"corrupt {prov_err}")
    if comp_err:
        problems.append(f"corrupt {comp_err}")

    # Count expected per-frame CSV units
    n_expected = len(expected_configs) * len(authorized_roles) * len(expected_replicas)
    n_present = 0
    missing_units: list[str] = []
    per_frame_dir = kdir / "per_frame"
    for cfg in expected_configs:
        for role in authorized_roles:
            for rep in expected_replicas:
                csv_p = per_frame_dir / cfg / role / f"rep{int(rep)}.csv"
                if csv_p.is_file():
                    n_present += 1
                else:
                    missing_units.append(f"{cfg}/{role}/rep{rep}")

    gate_ok = bool(gate.get("all_pass")) if gate is not None else None
    if gate_failed or gate_ok is False:
        problems.append("replay gate failed")
        status = EVAL_STATUS_FAILED
    elif gate_err or prov_err or comp_err:
        status = EVAL_STATUS_FAILED
    elif expect_spec_sha and prov and prov.get("spec_sha256") and prov.get("spec_sha256") != expect_spec_sha:
        problems.append(f"eval spec_sha256 {prov.get('spec_sha256')} != manifest {expect_spec_sha}")
        status = EVAL_STATUS_FAILED
    elif n_present == 0 and comp is None:
        problems.append("no per_frame CSV units found")
        status = EVAL_STATUS_MISSING
    elif n_present < n_expected or comp is None or not comp.get("ok"):
        if missing_units:
            sample = ", ".join(missing_units[:5])
            extra = f" (+{len(missing_units) - 5} more)" if len(missing_units) > 5 else ""
            problems.append(f"missing {len(missing_units)}/{n_expected} units: {sample}{extra}")
        if comp is None:
            problems.append("COMPLETE.json missing")
        elif not comp.get("ok"):
            problems.append("COMPLETE.json ok != true")
        status = EVAL_STATUS_INCOMPLETE
    elif gate_ok is None:
        problems.append("gate.json missing")
        status = EVAL_STATUS_INCOMPLETE
    else:
        status = EVAL_STATUS_COMPLETE

    return {
        "eval_status": status,
        "eval_valid": status == EVAL_STATUS_COMPLETE,
        "gate_all_pass": gate_ok,
        "n_expected_units": n_expected,
        "n_present_units": n_present,
        "eval_problems": problems,
        "eval_source_dir": str(kdir),
        "eval_git_sha": (prov or {}).get("eval_git_sha"),
        "checkpoint_weights_sha256": (prov or {}).get("checkpoint_weights_sha256"),
        "checkpoint_file_sha256": (prov or {}).get("checkpoint_file_sha256"),
        "checkpoint_epoch": (prov or {}).get("checkpoint_epoch"),
        "spec_sha256": (prov or {}).get("spec_sha256"),
    }


def audit_manifest(
    manifest: Mapping[str, Any],
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Audit every run and checkpoint in the manifest; separate missing/failed/incomplete/budget_exhausted."""
    roles = list(manifest.get("authorized_roles") or ALLOWED_DEV_ROLES)
    replicas = [int(r) for r in (manifest.get("expected_replicas") or DEFAULT_REPLICAS)]
    checkpoints = list(manifest.get("expected_checkpoints") or DEFAULT_CHECKPOINTS)
    expect_spec_sha = manifest.get("observation_spec_sha256")
    expect_train_sha = manifest.get("training_git_sha")
    default_eval_configs = list(manifest.get("eval_configs") or [])

    audit_rows: list[dict[str, Any]] = []
    by_train_status: dict[str, list[str]] = {
        TRAIN_STATUS_PLATEAU: [],
        TRAIN_STATUS_BUDGET_EXHAUSTED: [],
        TRAIN_STATUS_FROZEN_EXTERNAL: [],
        TRAIN_STATUS_INCOMPLETE: [],
        TRAIN_STATUS_FAILED: [],
        TRAIN_STATUS_MISSING: [],
    }
    by_eval_status: dict[str, list[str]] = {
        EVAL_STATUS_COMPLETE: [],
        EVAL_STATUS_SAME_AS_BEST: [],
        EVAL_STATUS_INCOMPLETE: [],
        EVAL_STATUS_FAILED: [],
        EVAL_STATUS_MISSING: [],
    }

    init_hashes: dict[tuple[str, int], set[str]] = {}
    ledger_info: dict[str, dict[str, Any]] = {}

    for run in manifest["runs"]:
        rid = str(run["run_id"])
        tr = _audit_training_dir(
            run,
            expect_spec_sha=expect_spec_sha,
            expect_git_sha=run.get("training_git_sha") or expect_train_sha,
            smoke=smoke,
        )
        by_train_status.setdefault(tr["training_status"], []).append(rid)

        if tr.get("init_state_dict_sha256"):
            init_hashes.setdefault((str(run["family"]), int(run["model_seed"])), set()).add(
                str(tr["init_state_dict_sha256"])
            )

        # Optional ledger metadata (attempts, preemptions/signals, extensions)
        led_path = run.get("ledger_path")
        if led_path and Path(led_path).is_file():
            lrows = [json.loads(ln) for ln in Path(led_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
            starts = [r for r in lrows if r.get("event") == "attempt_start"]
            ledger_info[rid] = {
                "attempts": len(starts),
                "extension": any(r.get("event") == "extension_start" for r in lrows),
                "signals": sum(1 for r in lrows if r.get("event") == "signal"),
                "partitions": sorted({str(r.get("partition")) for r in starts if r.get("partition")}),
            }
        else:
            ledger_info[rid] = {"attempts": None, "extension": None, "signals": None, "partitions": []}

        run_ckpts = list(run.get("expected_checkpoints") or (["best"] if run.get("frozen_external") else checkpoints))
        run_eval_cfgs = list(run.get("eval_configs") or default_eval_configs)

        for kind in run_ckpts:
            ev = _audit_eval_dir(
                run,
                kind=kind,
                expected_configs=run_eval_cfgs,
                authorized_roles=roles,
                expected_replicas=replicas,
                expect_spec_sha=expect_spec_sha,
            )
            by_eval_status.setdefault(ev["eval_status"], []).append(f"{rid}/{kind}")
            audit_rows.append(
                {
                    "run_id": rid,
                    "group": str(run["group"]),
                    "family": str(run["family"]),
                    "variant": str(run.get("variant", "p_on")),
                    "train_cell": str(run["train_cell"]),
                    "train_config_id": str(run.get("train_config_id", run["train_cell"])),
                    "b_position": run.get("b_position") or "",
                    "model_seed": int(run["model_seed"]),
                    "conditional": bool(run.get("conditional", run["group"] in CONDITIONAL_GROUPS)),
                    "checkpoint_kind": kind,
                    "training_status": tr["training_status"],
                    "training_valid": tr["training_valid"],
                    "training_outcome": tr["training_outcome"],
                    "budget_exhausted": tr["budget_exhausted"],
                    "declared_converged": tr["declared_converged"],
                    "epochs_completed": tr["epochs_completed"],
                    "best_val_epoch": tr["best_val_epoch"],
                    "best_val_metric": tr["best_val_metric"],
                    "clip_events": tr.get("clip_events"),
                    "last_same_as_best": tr["last_same_as_best"],
                    "training_git_sha": tr["training_git_sha"],
                    "init_state_dict_sha256": tr["init_state_dict_sha256"],
                    "training_problems": "; ".join(tr["training_problems"]),
                    "eval_status": ev["eval_status"],
                    "eval_valid": ev["eval_valid"],
                    "gate_all_pass": ev["gate_all_pass"],
                    "n_expected_units": ev["n_expected_units"],
                    "n_present_units": ev["n_present_units"],
                    "eval_source_dir": ev["eval_source_dir"] or "",
                    "eval_git_sha": ev["eval_git_sha"] or "",
                    "checkpoint_epoch": ev["checkpoint_epoch"],
                    "checkpoint_file_sha256": ev["checkpoint_file_sha256"] or "",
                    "checkpoint_weights_sha256": ev["checkpoint_weights_sha256"] or "",
                    "spec_sha256": ev["spec_sha256"] or "",
                    "eval_problems": "; ".join(ev["eval_problems"]),
                    "ledger_attempts": ledger_info[rid]["attempts"],
                    "ledger_extension": ledger_info[rid]["extension"],
                    "ledger_signals": ledger_info[rid]["signals"],
                    "training_dir": str(run.get("training_dir") or ""),
                    "eval_dir": str(run.get("eval_dir") or ""),
                    "expected_eval_configs": ",".join(run_eval_cfgs),
                }
            )

    init_mismatches = [
        f"init_state_dict_sha256 differs across cells for family={fam}, seed={sd}: {sorted(vals)}"
        for (fam, sd), vals in sorted(init_hashes.items())
        if len(vals) > 1
    ]

    df = pd.DataFrame(audit_rows)
    req_df = df[~df["conditional"]] if not df.empty else df
    n_runs = len(manifest["runs"])
    n_required_runs = sum(1 for r in manifest["runs"] if not r.get("conditional", r["group"] in CONDITIONAL_GROUPS))

    strict_complete = bool(
        not df.empty
        and not init_mismatches
        and bool(req_df["training_valid"].all())
        and bool(req_df["eval_valid"].all())
    )

    summary = {
        "schema": "piv_run_status_audit_v1",
        "manifest_path": manifest.get("_manifest_path"),
        "manifest_sha256": manifest.get("_manifest_sha256"),
        "observation_spec_sha256": expect_spec_sha,
        "n_runs_total": n_runs,
        "n_runs_required": n_required_runs,
        "n_run_checkpoints_total": len(df),
        "training_status_counts": {k: len(v) for k, v in by_train_status.items()},
        "training_runs_by_status": by_train_status,
        "eval_status_counts": {k: len(v) for k, v in by_eval_status.items()},
        "eval_units_by_status": by_eval_status,
        "init_mismatches": init_mismatches,
        "strict_complete": strict_complete,
        "n_budget_exhausted_runs": len(by_train_status[TRAIN_STATUS_BUDGET_EXHAUSTED]),
        "budget_exhausted_note": (
            "budget_exhausted runs are valid 1200-epoch artifacts included and labelled "
            "separately (hollow markers); they are NEVER declared converged."
        ),
    }
    return {"table": df, "summary": summary}


def render_status_report_md(audit: Mapping[str, Any]) -> str:
    """Human-readable Markdown status report separating missing/failed/incomplete/budget_exhausted."""
    s = audit["summary"]
    df: pd.DataFrame = audit["table"]
    lines = [
        "# Missione 02 (PIV virtuale) — Stato dei run e completezza",
        "",
        f"- **Manifest:** `{s['manifest_path']}` (sha256 `{str(s['manifest_sha256'])[:16]}...`)",
        f"- **Specifica osservazioni:** `{s['observation_spec_sha256']}`",
        f"- **Completezza stretta (pacchetto confermativo):** `{'SI' if s['strict_complete'] else 'NO (BLOCCATO / PARZIALE)'}`",
        "",
        "## Riepilogo stato training (categorie separate)",
        "",
        "| Stato training | Conteggio | Run |",
        "|---|---:|---|",
    ]
    for st in (
        TRAIN_STATUS_PLATEAU,
        TRAIN_STATUS_BUDGET_EXHAUSTED,
        TRAIN_STATUS_FROZEN_EXTERNAL,
        TRAIN_STATUS_INCOMPLETE,
        TRAIN_STATUS_FAILED,
        TRAIN_STATUS_MISSING,
    ):
        rids = s["training_runs_by_status"].get(st, [])
        lines.append(f"| `{st}` | {len(rids)} | {', '.join(rids) if rids else '—'} |")

    lines += [
        "",
        "## Riepilogo stato valutazione (`run_id/checkpoint`)",
        "",
        "| Stato valutazione | Conteggio | Unità |",
        "|---|---:|---|",
    ]
    for st in (
        EVAL_STATUS_COMPLETE,
        EVAL_STATUS_SAME_AS_BEST,
        EVAL_STATUS_INCOMPLETE,
        EVAL_STATUS_FAILED,
        EVAL_STATUS_MISSING,
    ):
        units = s["eval_units_by_status"].get(st, [])
        preview = ", ".join(units[:12]) + (f" (+{len(units) - 12} altri)" if len(units) > 12 else "")
        lines.append(f"| `{st}` | {len(units)} | {preview if units else '—'} |")

    if s["init_mismatches"]:
        lines += ["", "## Incoerenze di inizializzazione", ""] + [f"- {m}" for m in s["init_mismatches"]]

    lines += [
        "",
        "## Nota su `budget_exhausted`",
        "",
        f"> {s['budget_exhausted_note']}",
        "",
    ]
    if not df.empty:
        bad = df[~(df["training_valid"] & df["eval_valid"])]
        if not bad.empty:
            lines += [
                "## Dettaglio anomalie (mancanti, falliti, incompleti)",
                "",
                "| run_id | ckpt | training_status | eval_status | problemi |",
                "|---|---|---|---|---|",
            ]
            for _, r in bad.iterrows():
                probs = " | ".join(p for p in (r["training_problems"], r["eval_problems"]) if p)
                lines.append(
                    f"| `{r['run_id']}` | `{r['checkpoint_kind']}` | `{r['training_status']}` | `{r['eval_status']}` | {probs or '—'} |"
                )
    return "\n".join(lines) + "\n"


def select_best_worst_frames(
    per_frame: pd.DataFrame,
    *,
    role: str = "val_eval",
    checkpoint_kind: str = "best",
    replica: int = 0,
    top_k: int = 2,
) -> dict[str, Any]:
    """Explicit, deterministic selection rule for best & worst representative cases.

    Rule:
    1. Filter to ``role`` (default ``val_eval``), ``checkpoint_kind == 'best'``,
       ``replica == 0``, and each run's OWN diagonal evaluation config
       (``eval_config == train_config_id``).
    2. Average ``rmse_K`` across seeds within each ``(family, train_cell, eval_config, condition, frame_index)``.
    3. Sort deterministically by ``(mean_rmse_K, condition, frame_index)``:
       - ``best``: lowest ``top_k`` frames;
       - ``worst``: highest ``top_k`` frames (sorted descending by RMSE, tie-broken by condition, frame_index).
    """
    rule_desc = (
        f"role={role}, checkpoint={checkpoint_kind}, replica={replica}, diagonal own-config "
        f"(eval_config == train_config_id); ranked by cross-seed mean rmse_K with deterministic "
        f"tie-break (condition, frame_index); top-{top_k} best and top-{top_k} worst per (family, train_cell)."
    )
    if per_frame.empty:
        return {"rule": rule_desc, "selections": []}

    sub = per_frame[
        (per_frame["role"] == role)
        & (per_frame["checkpoint_kind"] == checkpoint_kind)
        & (per_frame["replica"].astype(int) == int(replica))
        & (per_frame["eval_config"] == per_frame["train_config_id"])
    ]
    if sub.empty:
        return {"rule": rule_desc, "selections": []}

    grp = (
        sub.groupby(["family", "train_cell", "eval_config", "condition", "frame_index"], as_index=False)
        .agg(mean_rmse_K=("rmse_K", "mean"), std_rmse_K=("rmse_K", "std"), n_seeds=("model_seed", "nunique"))
    )
    selections: list[dict[str, Any]] = []
    for (fam, cell, cfg), g in grp.groupby(["family", "train_cell", "eval_config"], sort=True):
        g_asc = g.sort_values(["mean_rmse_K", "condition", "frame_index"], ascending=[True, True, True])
        g_desc = g.sort_values(["mean_rmse_K", "condition", "frame_index"], ascending=[False, True, True])
        for rank, (_, row) in enumerate(g_asc.head(top_k).iterrows(), start=1):
            selections.append(
                {
                    "selection_kind": "best",
                    "rank": rank,
                    "family": str(fam),
                    "train_cell": str(cell),
                    "eval_config": str(cfg),
                    "role": role,
                    "replica": int(replica),
                    "condition": str(row["condition"]),
                    "frame_index": int(row["frame_index"]),
                    "mean_rmse_K": float(row["mean_rmse_K"]),
                    "std_rmse_K": float(row["std_rmse_K"]) if pd.notna(row["std_rmse_K"]) else None,
                    "n_seeds": int(row["n_seeds"]),
                }
            )
        for rank, (_, row) in enumerate(g_desc.head(top_k).iterrows(), start=1):
            selections.append(
                {
                    "selection_kind": "worst",
                    "rank": rank,
                    "family": str(fam),
                    "train_cell": str(cell),
                    "eval_config": str(cfg),
                    "role": role,
                    "replica": int(replica),
                    "condition": str(row["condition"]),
                    "frame_index": int(row["frame_index"]),
                    "mean_rmse_K": float(row["mean_rmse_K"]),
                    "std_rmse_K": float(row["std_rmse_K"]) if pd.notna(row["std_rmse_K"]) else None,
                    "n_seeds": int(row["n_seeds"]),
                }
            )
    return {"rule": rule_desc, "selections": selections}


@dataclass
class StageStateManager:
    """Resumability controller for post-processing stages without duplicating outputs."""

    dest: Path

    @property
    def state_dir(self) -> Path:
        return self.dest / ".state"

    def stage_file(self, stage: str) -> Path:
        return self.state_dir / f"{stage}.json"

    def is_up_to_date(self, stage: str, input_signature: str) -> bool:
        sf = self.stage_file(stage)
        if not sf.is_file():
            return False
        try:
            rec = json.loads(sf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if rec.get("input_signature") != input_signature or not rec.get("ok"):
            return False
        for rel, expected_sha in (rec.get("outputs") or {}).items():
            p = self.dest / rel
            if not p.is_file():
                return False
            if sha256_file(p) != expected_sha:
                return False
        return True

    def mark_done(self, stage: str, input_signature: str, output_paths: Sequence[Path]) -> dict[str, Any]:
        outputs: dict[str, str] = {}
        for p in output_paths:
            p = Path(p)
            if p.is_file():
                rel = p.relative_to(self.dest).as_posix()
                outputs[rel] = sha256_file(p)
        payload = {
            "stage": stage,
            "ok": True,
            "input_signature": input_signature,
            "outputs": outputs,
        }
        atomic_write_json(self.stage_file(stage), payload)
        return payload


__all__ = [
    "ALLOWED_DEV_ROLES",
    "BRIDGE_AND_PHASE3_CONFIGS",
    "CONDITIONAL_GROUPS",
    "DEFAULT_CHECKPOINTS",
    "DEFAULT_REPLICAS",
    "DEFAULT_REPRESENTATIVE_FRAMES",
    "EVAL_STATUS_COMPLETE",
    "EVAL_STATUS_FAILED",
    "EVAL_STATUS_INCOMPLETE",
    "EVAL_STATUS_MISSING",
    "EVAL_STATUS_SAME_AS_BEST",
    "MAIN_EVAL_CONFIGS",
    "MANIFEST_SCHEMA",
    "ManifestError",
    "NECESSARY_CAMPAIGN_SPEC",
    "NECESSARY_GROUPS",
    "PHASE3_GROUPS",
    "ROLE_EXPECTED_COUNTS",
    "SENSITIVITY_EVAL_CONFIGS",
    "StageStateManager",
    "TRAIN_STATUS_BUDGET_EXHAUSTED",
    "TRAIN_STATUS_FAILED",
    "TRAIN_STATUS_FROZEN_EXTERNAL",
    "TRAIN_STATUS_INCOMPLETE",
    "TRAIN_STATUS_MISSING",
    "TRAIN_STATUS_PLATEAU",
    "atomic_write_json",
    "atomic_write_text",
    "audit_manifest",
    "load_manifest",
    "render_status_report_md",
    "select_best_worst_frames",
    "sha256_file",
    "sha256_json",
]
