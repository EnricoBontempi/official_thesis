"""Rules for the S5 selection reanalysis. No checkpoint is implied."""

from __future__ import annotations

import json
from pathlib import Path

from combustion_recon.evaluation.opus_r0_selection import (
    causal_moving_average_first_min,
    eval_points_from_s5_extract,
    first_minimum,
    leave_one_condition_out,
    r0_rows_for_points,
)

_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "results/month2/opus_data_study/r0_selection/s5_curves_source.json"
)


def test_first_minimum_keeps_earliest_tie():
    assert first_minimum([3.0, 1.0, 1.0, 0.5]) == 3
    assert first_minimum([2.0, 2.0, 1.0]) == 2
    assert first_minimum([1.0, 1.0]) == 0


def test_causal_mean_excludes_the_first_four_evals():
    values = [10.0, 0.0, 8.0, 7.0, 6.0, 5.0, 4.0, 9.0]
    index, score = causal_moving_average_first_min(values)
    # index 5: mean(0, 8, 7, 6, 5) = 5.2, the first minimum.
    # The raw minimum at index 1 is excluded.
    assert index == 5
    assert score == 5.2


def test_leave_one_condition_out_does_not_use_the_held_out_score():
    epochs = [
        {"a": 1.0, "b": 100.0, "c": 100.0},
        {"a": 50.0, "b": 2.0, "c": 2.0},
    ]
    picked = leave_one_condition_out(epochs)
    assert picked["a"]["epoch_index"] == 1
    assert picked["a"]["selection_score"] == 2.0
    assert picked["a"]["held_out_score"] == 50.0
    assert picked["b"]["epoch_index"] == 1
    assert picked["b"]["held_out_score"] == 2.0


def test_saved_s5_curve_keeps_original_best_and_includes_stop_epoch():
    payload = json.loads(_SOURCE.read_text())
    run = payload["runs"]["s5_sparse_unet_pon_seed0"]
    points = eval_points_from_s5_extract(run)
    assert points[-1]["epoch"] == 630
    assert points[-1]["source"] == "final_eval_csv+progress.json"
    rows = r0_rows_for_points("s5_sparse_unet_pon_seed0", points, summary=run["summary"])
    best = next(row for row in rows if row["rule"] == "original_best_val")
    assert best["epoch"] == 60
    assert abs(best["score"] - 42.96406039204512) < 1e-9
    smoothed = next(row for row in rows if row["rule"] == "causal_ma5_first_min")
    assert smoothed["epoch"] >= 40
