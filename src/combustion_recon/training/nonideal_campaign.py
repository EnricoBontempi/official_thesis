"""Status, validity and freeze of the 30 specialist trainings (CPU, read-only on the run directories).

A run is accepted only in a declared scientific state: ``plateau`` (early stop) or ``budget_exhausted``
(1200 epochs without plateau after the single authorised continuation). An exit code 0 alone proves nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

PLATEAU = "plateau"
NEEDS_EXTENSION = "needs_extension"      # reached the initial 1000 epochs without plateau
BUDGET_EXHAUSTED = "budget_exhausted"    # reached 1200 epochs without plateau
INCOMPLETE = "incomplete"                # wall-time stop or still running / no summary
ERROR = "error"
OUTCOME_PLATEAU = "plateau_by_operational_early_stop_rule"
OUTCOME_BUDGET = "budget_exhausted_not_declared_converged"
INITIAL_EPOCHS = 1000
FINAL_EPOCHS = 1200


def _json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def classify_run(out_root: Path, run_id: str) -> dict[str, Any]:
    d = Path(out_root) / run_id
    s = _json(d / "summary.json")
    prog = _json(d / "progress.json")
    if not d.is_dir():
        return {"run_id": run_id, "state": INCOMPLETE, "why": "run directory missing"}
    if s is None:
        return {"run_id": run_id, "state": INCOMPLETE, "why": "no summary.json",
                "progress_epoch": (prog or {}).get("epoch")}
    status, stop, epochs = s.get("status"), s.get("stop_reason"), int(s.get("epochs_completed") or -1)
    base = {"run_id": run_id, "summary_status": status, "stop_reason": stop, "epochs_completed": epochs}
    if status == "stopped_early_plateau":
        return {**base, "state": PLATEAU}
    if status == "completed" and stop == "reached_max_epochs":
        if epochs == INITIAL_EPOCHS:
            return {**base, "state": NEEDS_EXTENSION}
        if epochs == FINAL_EPOCHS:
            return {**base, "state": BUDGET_EXHAUSTED}
        return {**base, "state": ERROR, "why": f"completed at unexpected epoch count {epochs}"}
    if status == "stopped_wall_time":
        return {**base, "state": INCOMPLETE, "why": "stopped by wall time; continuation not finished"}
    return {**base, "state": ERROR, "why": f"unexpected status {status!r} / {stop!r}"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_sha256(state_dict: dict[str, Any]) -> str:
    """Same definition as scripts/38_s7_eval.py (sorted keys, raw float bytes)."""
    import numpy as np

    h = hashlib.sha256()
    for key in sorted(state_dict):
        arr = state_dict[key].detach().contiguous().cpu().numpy()
        h.update(str(key).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def validate_and_freeze_run(out_root: Path, entry: dict[str, Any], *, expect_git_sha: str, expect_spec_sha: str,
                            load_checkpoints: bool = True, smoke: bool = False) -> dict[str, Any]:
    """Validity checks of one finished run + frozen identity of best/last. Returns {valid, problems, ...}.

    ``smoke=True`` (tests of the chain on non-scientific runs only) accepts any completed state and smoke identities.
    """
    run_id = entry["run_id"]
    d = Path(out_root) / run_id
    cls = classify_run(out_root, run_id)
    if smoke and cls.get("summary_status") == "completed":
        cls = {**cls, "state": PLATEAU, "smoke_state_override": True}
    problems: list[str] = []
    if cls["state"] not in (PLATEAU, BUDGET_EXHAUSTED):
        problems.append(f"state {cls['state']}: {cls.get('why', '')}")
        return {"run_id": run_id, "valid": False, "problems": problems, **cls}
    ident = _json(d / "run_identity.json") or {}
    init = _json(d / "init_manifest.json") or {}
    spec_block = ident.get("nonideal_specialist") or {}
    if ident.get("run_id") != run_id:
        problems.append("run_identity run_id mismatch")
    if spec_block.get("regime") != entry["regime"]:
        problems.append(f"regime {spec_block.get('regime')} != {entry['regime']}")
    if ident.get("family") != entry["family"] or not ident.get("p_conditioning"):
        problems.append("family / p_on mismatch")
    if ident.get("git_sha") != expect_git_sha and expect_git_sha and not smoke:
        # continuations of the same run may run at the same frozen SHA only
        problems.append(f"git_sha {ident.get('git_sha')} != frozen {expect_git_sha}")
    if spec_block.get("degradation_spec_sha256") != expect_spec_sha:
        problems.append("degradation spec hash differs from the frozen one")
    if ident.get("non_scientific_smoke") and not smoke:
        problems.append("run_identity marks a smoke run")
    ck_dir = d / "checkpoints"
    out: dict[str, Any] = {"run_id": run_id, "family": entry["family"], "regime": entry["regime"], "seed": entry["seed"], **cls,
                           "init_state_dict_sha256": init.get("init_state_dict_sha256")}
    if load_checkpoints:
        import torch

        for kind in ("best", "last"):
            path = ck_dir / f"{kind}.pt"
            if not path.is_file():
                problems.append(f"{kind}.pt missing")
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            es = payload.get("epoch_state") or {}
            if payload.get("checkpoint_kind") != "sparse_scientific_epoch_complete_v1":
                problems.append(f"{kind}.pt is not a complete epoch checkpoint")
            sd = payload["model_state_dict"]
            if not all(bool(torch.isfinite(v).all()) for v in sd.values() if v.is_floating_point()):
                problems.append(f"{kind}.pt has non-finite weights")
            out[kind] = {"path": str(path), "epoch": int(es.get("epoch", -1)), "file_sha256": sha256_file(path),
                         "weights_sha256": weights_sha256(sd), "status": es.get("status")}
            if kind == "last":
                hist = payload.get("eval_history") or []
                out["eval_history_epochs"] = [int(e["epoch"]) for e in hist]
                vals = [e.get("val_mean_condition_rmse_K") for e in hist]
                if not hist or not all(v is not None and math.isfinite(v) for v in vals):
                    problems.append("non-finite or empty clean VAL history")
                if not all("specialist_val_mean_condition_rmse_K" in e and e.get("specialist_val_used_for_selection") is False for e in hist):
                    problems.append("specialist VAL metric missing or flagged as used for selection")
                out["best_val_epoch"] = es.get("best_val_epoch")
                out["best_val_metric"] = es.get("best_val_metric")
                out["n_evals"] = len(hist)
                out["loop_at_best"] = next((e for e in hist if int(e["epoch"]) == int(es.get("best_val_epoch", -1))), None)
                out["loop_at_last"] = next((e for e in hist if int(e["epoch"]) == int(es.get("epoch", -1))), None)
                out["specialist_val_at_best"] = (out["loop_at_best"] or {}).get("specialist_val_mean_condition_rmse_K")
                out["clip_events"] = payload.get("clip_events")
        if "best" in out and "last" in out:
            if out["best"]["epoch"] != out.get("best_val_epoch"):
                problems.append("best.pt epoch differs from the recorded best_val_epoch")
            if out["best"]["epoch"] > out["last"]["epoch"]:
                problems.append("best epoch after last epoch")
            out["last_same_as_best"] = out["best"]["weights_sha256"] == out["last"]["weights_sha256"]
            if out["loop_at_last"] is None:
                problems.append("last epoch is not an evaluation epoch (no loop reference for the replay gate)")
    # Explicit outcome label. budget_exhausted = 1200 epochs without the early-stop plateau after the single authorised
    # continuation: valid artifacts, included in the package, NEVER declared convergent. The plateau rule is an operational
    # convention, not a physical certification of convergence.
    out["training_outcome"] = OUTCOME_BUDGET if cls["state"] == BUDGET_EXHAUSTED else OUTCOME_PLATEAU
    out["budget_exhausted"] = cls["state"] == BUDGET_EXHAUSTED
    out["declared_converged"] = False if out["budget_exhausted"] else None  # plateau: converged by the operational rule only
    if cls["state"] == BUDGET_EXHAUSTED and not smoke:
        led = ledger_summary(out_root, run_id)
        if not led["extension"]:
            problems.append("1200 epochs reached without a registered continuation (ledger has no extension_start)")
        if "last" in out and out["last"]["epoch"] != FINAL_EPOCHS:
            problems.append(f"budget_exhausted but last epoch is {out['last']['epoch']} != {FINAL_EPOCHS}")
    out["valid"] = not problems
    out["problems"] = problems
    return out


def ledger_summary(out_root: Path, run_id: str) -> dict[str, Any]:
    p = Path(out_root) / "_ledger" / f"{run_id}.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else []
    starts = [r for r in rows if r.get("event") == "attempt_start"]
    return {"attempts": len(starts), "actions": [r.get("action") for r in starts],
            "extension": any(r.get("event") == "extension_start" for r in rows),
            "signals": sum(1 for r in rows if r.get("event") == "signal"),
            "partitions": sorted({str(r.get("partition")) for r in starts})}
