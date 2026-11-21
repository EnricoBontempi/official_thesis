"""Pure helpers for the S7 final evaluation (both model families). No torch, no GPU.

The scientific gate is the one fixed in roadmapsparse.md (S7): the final evaluation at 5%
must reproduce the training-loop macro RMSE within 0.05 K per run / checkpoint / role /
replica wherever a loop reference exists. A missing reference is a FAIL, never a PASS.
Per-frame deltas are diagnostics; they are reported, not thresholded.
"""

from __future__ import annotations

import gzip
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from combustion_recon.evaluation.opus_o5_plan import CHECKPOINTS, ROLES

GATE_TOLERANCE_K = 0.05
GATE_COVERAGE = 0.05
FAMILIES = ("sparse_unet", "adapted_glu")
MODES = ("full", "smoke", "parity")
# (role, replica) pairs that have a loop reference at 5%: train/seen use replica 0, VAL has three.
GATE_SETTINGS: tuple[tuple[str, int], ...] = (
    ("train_fit", 0),
    ("seen_train_eval", 0),
    ("val_eval", 0),
    ("val_eval", 1),
    ("val_eval", 2),
)
LOOP_ROLE_CSV = {
    "train_fit": "train_fit_per_frame.csv.gz",
    "seen_train_eval": "seen_train_eval_per_frame.csv.gz",
    "val_eval": "val_eval_per_frame.csv.gz",
}


def settings_for_mode(mode: str, coverages: Sequence[float], roles: Sequence[str], replicas: Sequence[int]):
    """(coverage, role, replica) tuples one run evaluates for a mode."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "parity":
        return [(GATE_COVERAGE, role, rep) for role, rep in GATE_SETTINGS]
    return [(c, r, k) for c in coverages for r in roles for k in replicas]


def loop_reference(history: Sequence[Mapping[str, Any]], epoch: int) -> dict[tuple[str, str], float] | None:
    """Loop macro RMSE at ``epoch``: keys (role, replica label). None if the epoch is absent."""
    hits = [rec for rec in history if int(rec["epoch"]) == int(epoch)]
    if len(hits) != 1:
        return None
    rec = hits[0]
    ref: dict[tuple[str, str], float] = {
        ("train_fit", "0"): float(rec["train_mean_condition_rmse_K"]),
        ("seen_train_eval", "0"): float(rec["seen_mean_condition_rmse_K"]),
        ("val_eval", "mean3"): float(rec["val_mean_condition_rmse_K"]),
    }
    for item in rec.get("val_per_replica") or []:
        ref[("val_eval", str(int(item["replica_id"])))] = float(item["mean_condition_rmse_K"])
    return ref


def gate_rows(
    *,
    checkpoint: str,
    epoch: int,
    o5_macro: Mapping[tuple[str, int], float],
    reference: Mapping[tuple[str, str], float] | None,
    tolerance_k: float = GATE_TOLERANCE_K,
) -> list[dict[str, Any]]:
    """One row per (role, replica) plus the VAL three-mask mean. Missing reference => FAIL."""
    rows: list[dict[str, Any]] = []
    wanted: list[tuple[str, str, float | None]] = []
    for role, rep in GATE_SETTINGS:
        wanted.append((role, str(rep), o5_macro.get((role, rep))))
    vals = [o5_macro.get(("val_eval", r)) for r in (0, 1, 2)]
    wanted.append(("val_eval", "mean3", None if any(v is None for v in vals) else sum(vals) / 3.0))
    for role, rep, value in wanted:
        loop_value = None if reference is None else reference.get((role, rep))
        delta = None if (value is None or loop_value is None) else float(value) - float(loop_value)
        ok = bool(delta is not None and math.isfinite(delta) and abs(delta) <= tolerance_k)
        rows.append(
            {
                "checkpoint": checkpoint,
                "epoch": int(epoch),
                "role": role,
                "replica": rep,
                "s7_eval_rmse_K": value,
                "loop_rmse_K": loop_value,
                "delta_K": delta,
                "tolerance_K": tolerance_k,
                "within_tolerance": ok,
                "reference_present": loop_value is not None,
                "loop_source": "training_loop_not_s7_eval",
            }
        )
    return rows


def summarize_gate(rows: Sequence[Mapping[str, Any]], *, n_checkpoints: int = len(CHECKPOINTS)) -> dict[str, Any]:
    expected = n_checkpoints * (len(GATE_SETTINGS) + 1)
    deltas = [abs(float(r["delta_K"])) for r in rows if r["delta_K"] is not None]
    missing = [r for r in rows if not r["reference_present"] or r["s7_eval_rmse_K"] is None]
    all_pass = bool(len(rows) == expected and not missing and all(r["within_tolerance"] for r in rows))
    return {
        "tolerance_K": GATE_TOLERANCE_K,
        "coverage": GATE_COVERAGE,
        "n_expected": expected,
        "n_comparisons": len(rows),
        "n_within_tolerance": int(sum(bool(r["within_tolerance"]) for r in rows)),
        "n_missing_reference_or_value": len(missing),
        "max_abs_delta_K": max(deltas) if deltas else None,
        "all_pass": all_pass,
    }


def read_loop_per_frame(path: Path) -> list[dict[str, Any]]:
    csv.field_size_limit(10**9)
    with gzip.open(path, "rt") as handle:
        return [
            {
                "sim_id": row["sim_id"],
                "frame_index": int(row["frame_index"]),
                "replica": int(float(row.get("mask_replica_id", 0) or 0)),
                "rmse_K": float(row["rmse_K"]),
            }
            for row in csv.DictReader(handle)
        ]


def per_frame_deltas(
    loop_rows: Sequence[Mapping[str, Any]],
    s7_rows: Sequence[Mapping[str, Any]],
    *,
    replica: int,
) -> dict[str, Any]:
    """Frame-level |delta| of rmse_K for one replica. Diagnostic only."""
    ref = {(r["sim_id"], r["frame_index"]): r["rmse_K"] for r in loop_rows if int(r["replica"]) == int(replica)}
    diffs = []
    for row in s7_rows:
        key = (str(row["sim_id"]), int(row["frame_index"]))
        if key in ref:
            diffs.append(abs(float(row["rmse_K"]) - ref[key]))
    return {
        "n_loop_frames": len(ref),
        "n_s7_frames": len(s7_rows),
        "n_matched": len(diffs),
        "max_abs_frame_delta_K": max(diffs) if diffs else None,
        "mean_abs_frame_delta_K": (sum(diffs) / len(diffs)) if diffs else None,
    }


def sha256_json(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
