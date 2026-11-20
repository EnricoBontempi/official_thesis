"""Contracts for the prepared Opus O5 package. No GPU and no submission."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combustion_recon.evaluation.opus_o5_plan import (
    C149_CACHE_ID,
    HISTORICAL_CACHE_ID,
    LOOP_SOURCE,
    REQUIRED_RUN_IDS,
    aggregate_frames,
    loop_curve_frame,
    reuse_plan,
    shared_field_scales,
)

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs" / "experiments" / "opus_o5_matrix.json"
CURVES = ROOT / "results" / "month2" / "opus_data_study" / "o5_prep" / "loop_curves.csv"
SBATCH = ROOT / "scripts" / "engaging" / "opus_o5_eval.sbatch"
EVAL = ROOT / "scripts" / "26_opus_o5_eval.py"


def test_reuse_keeps_distinct_identities_for_equal_weights():
    plan = reuse_plan({"best": "aaa", "best_seen_diagnostic": "aaa", "last": "bbb"})
    assert plan == {"best": None, "best_seen_diagnostic": "best", "last": None}


def test_matrix_is_the_eight_opus_runs():
    matrix = json.loads(MATRIX.read_text())
    assert matrix["status"] == "prepared_not_submitted"
    assert matrix["historical_s5_is_not_a_row"] is True
    assert matrix["replay_does_not_cancel_shuffle_difference"] is True
    assert matrix["no_test_split"] is True
    ids = [row["run_id"] for row in matrix["runs"]]
    assert ids == list(REQUIRED_RUN_IDS)
    assert all("s5_" not in run_id for run_id in ids)
    assert matrix["coverages"] == [0.01, 0.05, 0.20]
    assert matrix["replicas"] == [0, 1, 2]
    assert matrix["roles"]["train_fit"] == "reference760"
    shared = matrix["shared_inputs"]
    for row in matrix["runs"]:
        assert row["status"] == "stopped_early_plateau"
        assert row["completed"] is False
        weights = {name: spec["weights_sha256"] for name, spec in row["checkpoints"].items()}
        plan = reuse_plan(weights)
        for name, spec in row["checkpoints"].items():
            assert spec["reuse_forward_from"] == plan[name]
            assert spec["file_sha256"] != spec["weights_sha256"]
        if row["pool_id"] == "C149":
            assert shared["c149_cache_id"] == C149_CACHE_ID
        else:
            assert shared["historical_cache_id"] == HISTORICAL_CACHE_ID
    on = next(row for row in matrix["runs"] if row["run_id"] == "O-A38-on-s0")
    assert on["checkpoints"]["best"]["weights_sha256"] == on["checkpoints"]["best_seen_diagnostic"]["weights_sha256"]
    assert on["checkpoints"]["best"]["file_sha256"] != on["checkpoints"]["best_seen_diagnostic"]["file_sha256"]
    assert on["checkpoints"]["best"]["epoch"] == 120
    off = next(row for row in matrix["runs"] if row["run_id"] == "O-A38-off-s0")
    assert off["checkpoints"]["best"]["epoch"] == 90
    assert off["checkpoints"]["best_seen_diagnostic"]["epoch"] == 130
    assert off["checkpoints"]["best"]["weights_sha256"] != off["checkpoints"]["best_seen_diagnostic"]["weights_sha256"]


def test_loop_curves_are_labeled_and_match_the_two_new_runs():
    frame = pd.read_csv(CURVES)
    assert set(frame["metric_source"]) == {LOOP_SOURCE}
    assert len(frame) == 515
    assert (frame["update"] == frame["epoch"] * 95).all()
    assert (frame["presentations"] == frame["epoch"] * 760).all()
    on = frame[(frame.run_id == "O-A38-on-s0") & (frame.epoch == 630)].iloc[0]
    off = frame[(frame.run_id == "O-A38-off-s0") & (frame.epoch == 660)].iloc[0]
    assert on.val_mean3_rmse_K == pytest.approx(45.49906516268322)
    assert off.val_mean3_rmse_K == pytest.approx(49.47738908243192)
    built = loop_curve_frame(
        [{"epoch": 10, "update": 950, "train_mean_condition_rmse_K": 1.0, "seen_mean_condition_rmse_K": 2.0, "val_mean_condition_rmse_K": 3.0}],
        run_id="toy",
        pool_id="A38",
        pool_size=760,
        p_conditioning=True,
        model_seed=0,
    )
    assert built.loc[0, "pool_passages"] == pytest.approx(10.0)
    assert built.loc[0, "metric_source"] == LOOP_SOURCE
    with pytest.raises(ValueError):
        loop_curve_frame(
            [{"epoch": 10, "update": 1, "train_mean_condition_rmse_K": 1, "seen_mean_condition_rmse_K": 1, "val_mean_condition_rmse_K": 1}],
            run_id="toy",
            pool_id="A38",
            pool_size=760,
            p_conditioning=True,
            model_seed=0,
        )


def test_shared_scale_is_not_per_image(tmp_path: Path):
    quiet = np.zeros((2, 2))
    loud = np.full((2, 2), 10.0)
    limits = shared_field_scales([quiet, loud], [np.full((2, 2), 1.0), np.full((2, 2), 4.0)])
    assert limits["vmin"] == 0.0
    assert limits["vmax"] == 10.0
    assert limits["emax"] == 4.0
    import importlib.util

    spec = importlib.util.spec_from_file_location("opus_o5_figures", ROOT / "scripts" / "28_opus_o5_figures.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.render(
        [
            {"run_id": "a", "sim_id": ["c0", "c1"], "pred": np.stack([quiet, loud]), "true": np.stack([quiet, loud])},
            {"run_id": "b", "sim_id": ["c0", "c1"], "pred": np.stack([loud, quiet]), "true": np.stack([quiet, loud])},
        ],
        tmp_path / "panel.png",
        "best",
    )
    assert (tmp_path / "panel.png").stat().st_size > 0


def test_aggregation_keeps_replicas_and_does_not_rank():
    rows = []
    for replica in (0, 1):
        for frame in (0, 1):
            rows.append(
                {
                    "run_id": "toy",
                    "checkpoint_kind": "best",
                    "role": "val_eval",
                    "site_fraction": 0.05,
                    "mask_replica_id": replica,
                    "sim_id": "cond",
                    "frame_index": frame,
                    "rmse_K": 10.0 + replica,
                    "mae_K": 5.0,
                    "mean_signed_error_K": -1.0,
                    "abs_bias_K": 1.0,
                    "centered_rmse_K": 4.0,
                    "high_iou": 0.5,
                    "high_rel_area_error": 0.1,
                }
            )
    tables = aggregate_frames(pd.DataFrame(rows), expected_counts={"val_eval": 2})
    assert "winner" not in tables["over_replicas"].columns
    assert set(tables["over_conditions"]["mask_replica_id"]) == {0, 1}
    assert tables["over_replicas"].loc[0, "rmse_K"] == pytest.approx(10.5)
    rows[1]["frame_index"] = 0
    with pytest.raises(ValueError):
        aggregate_frames(pd.DataFrame(rows), expected_counts={"val_eval": 2})


def test_launcher_is_advanced_gpu_and_not_submitted_by_the_script():
    text = SBATCH.read_text()
    assert "#SBATCH --array=0-7%4" in text
    assert "#SBATCH --qos=mit_amf_advanced_gpu" in text
    assert "#SBATCH --partition=mit_normal_gpu" in text
    assert "gpu:l40s:1" in text
    assert "#SBATCH --time=04:00:00" in text
    assert "AUTHORIZE_OPUS_O5" in text
    assert "preempt" not in text
    assert "mit_general" not in text
    assert "sbatch " not in text
    eval_text = EVAL.read_text()
    assert "require_eval_mask_payload=True" in eval_text
    assert "cache_indices_for_pool" in eval_text
    assert "cache_validation_kwargs" in eval_text
    assert '"test"' not in eval_text
    assert "evaluate_setting" in eval_text
