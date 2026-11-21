"""S7 evaluation helpers: loop reference, the 0.05 K gate, per-frame diagnostics, mode plans."""

from __future__ import annotations

import gzip

import pytest

from combustion_recon.evaluation.opus_o5_plan import CHECKPOINTS, COVERAGES, REPLICAS, ROLES
from combustion_recon.evaluation.s7_eval_plan import (
    GATE_SETTINGS,
    GATE_TOLERANCE_K,
    gate_rows,
    loop_reference,
    per_frame_deltas,
    read_loop_per_frame,
    settings_for_mode,
    summarize_gate,
)

HIST = [
    {
        "epoch": 100,
        "train_mean_condition_rmse_K": 30.0,
        "seen_mean_condition_rmse_K": 40.0,
        "val_mean_condition_rmse_K": 41.0,
        "val_per_replica": [
            {"replica_id": 0, "mean_condition_rmse_K": 40.5},
            {"replica_id": 1, "mean_condition_rmse_K": 41.0},
            {"replica_id": 2, "mean_condition_rmse_K": 41.5},
        ],
    }
]


def _macro(shift: float = 0.0):
    return {
        ("train_fit", 0): 30.0 + shift,
        ("seen_train_eval", 0): 40.0 + shift,
        ("val_eval", 0): 40.5 + shift,
        ("val_eval", 1): 41.0 + shift,
        ("val_eval", 2): 41.5 + shift,
    }


def test_tolerance_is_the_fixed_scientific_gate():
    assert GATE_TOLERANCE_K == 0.05


def test_loop_reference_has_train_seen_val_replicas_and_mean():
    ref = loop_reference(HIST, 100)
    assert ref[("train_fit", "0")] == 30.0 and ref[("seen_train_eval", "0")] == 40.0
    assert ref[("val_eval", "1")] == 41.0 and ref[("val_eval", "mean3")] == 41.0
    assert loop_reference(HIST, 110) is None  # absent epoch


def test_gate_passes_inside_tolerance_and_fails_outside():
    ok = gate_rows(checkpoint="best", epoch=100, o5_macro=_macro(0.04), reference=loop_reference(HIST, 100))
    assert len(ok) == len(GATE_SETTINGS) + 1 and all(r["within_tolerance"] for r in ok)
    bad = gate_rows(checkpoint="best", epoch=100, o5_macro=_macro(0.06), reference=loop_reference(HIST, 100))
    assert not any(r["within_tolerance"] for r in bad)


def test_missing_reference_or_value_is_a_failure_not_a_pass():
    no_ref = gate_rows(checkpoint="best", epoch=100, o5_macro=_macro(), reference=None)
    assert not any(r["within_tolerance"] for r in no_ref) and not any(r["reference_present"] for r in no_ref)
    partial = _macro()
    partial.pop(("val_eval", 2))
    rows = gate_rows(checkpoint="best", epoch=100, o5_macro=partial, reference=loop_reference(HIST, 100))
    assert not rows[-1]["within_tolerance"]  # the VAL three-mask mean needs all replicas


def test_summary_requires_every_checkpoint_and_comparison():
    rows = []
    for name in CHECKPOINTS:
        rows += gate_rows(checkpoint=name, epoch=100, o5_macro=_macro(), reference=loop_reference(HIST, 100))
    assert summarize_gate(rows)["all_pass"] is True
    assert summarize_gate(rows[:-1])["all_pass"] is False  # one comparison absent
    rows[0]["within_tolerance"] = False
    assert summarize_gate(rows)["all_pass"] is False


def test_per_frame_deltas_and_gz_reader(tmp_path):
    path = tmp_path / "val.csv.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("sim_id,frame_index,mask_replica_id,rmse_K\na,1,0,10.0\na,2,0,20.0\na,1,1,30.0\n")
    loop = read_loop_per_frame(path)
    info = per_frame_deltas(loop, [{"sim_id": "a", "frame_index": 1, "rmse_K": 10.25},
                                   {"sim_id": "a", "frame_index": 2, "rmse_K": 20.0}], replica=0)
    assert info["n_matched"] == 2 and info["max_abs_frame_delta_K"] == pytest.approx(0.25)


def test_mode_plans():
    full = settings_for_mode("full", COVERAGES, ROLES, REPLICAS)
    assert len(full) == 27 and settings_for_mode("smoke", COVERAGES, ROLES, REPLICAS) == full
    parity = settings_for_mode("parity", COVERAGES, ROLES, REPLICAS)
    assert parity == [(0.05, r, k) for r, k in GATE_SETTINGS]
    with pytest.raises(ValueError):
        settings_for_mode("other", COVERAGES, ROLES, REPLICAS)
