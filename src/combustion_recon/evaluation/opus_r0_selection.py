"""R0 reanalysis of recorded S5 eval curves.

The rules below only reorder epochs that were already scored on VAL.
Scheduler and early-stop used that same VAL curve, so none of these
selections is an independent holdout.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def first_minimum(values: Sequence[float]) -> int:
    """Index of the first strict minimum. Ties keep the earliest index."""
    if not values:
        raise ValueError("empty series")
    best_i = 0
    best_v = float(values[0])
    for i, raw in enumerate(values):
        value = float(raw)
        if value < best_v:
            best_i = i
            best_v = value
    return best_i


def causal_moving_average_first_min(
    values: Sequence[float],
    *,
    window: int = 5,
    exclude_first: int = 4,
) -> tuple[int, float]:
    """First minimum of a causal trailing mean.

    The smoothed value at index i is the mean of values[i-window+1 : i+1].
    Indices below ``exclude_first`` cannot be selected. With the roadmap
    defaults the first eligible index is 4, which is also the first index
    that has five observations.
    """
    if window < 1:
        raise ValueError("window must be positive")
    if exclude_first < 0:
        raise ValueError("exclude_first must be non-negative")
    start = max(window - 1, exclude_first)
    if len(values) <= start:
        raise ValueError(
            f"need more than {start} evals for window={window} exclude_first={exclude_first}"
        )
    smoothed: list[float] = []
    indices: list[int] = []
    for i in range(start, len(values)):
        chunk = [float(v) for v in values[i - window + 1 : i + 1]]
        smoothed.append(sum(chunk) / float(window))
        indices.append(i)
    pick = first_minimum(smoothed)
    return indices[pick], smoothed[pick]


def leave_one_condition_out(
    epoch_condition_scores: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float | int]]:
    """For each held-out condition, select on the mean of the other conditions.

    ``epoch_condition_scores[i][condition]`` is the score at eval i.
    The returned index is into that sequence. The held-out score is reported
    at the selected index and is not part of the selection score.
    """
    if not epoch_condition_scores:
        raise ValueError("no epochs")
    conditions = list(epoch_condition_scores[0].keys())
    if len(conditions) < 2:
        raise ValueError("leave-one-condition-out needs at least two conditions")
    for row in epoch_condition_scores:
        if list(row.keys()) != conditions:
            raise ValueError("condition keys differ across epochs")
    out: dict[str, dict[str, float | int]] = {}
    for held in conditions:
        others = [name for name in conditions if name != held]
        selection = [
            sum(float(row[name]) for name in others) / float(len(others))
            for row in epoch_condition_scores
        ]
        index = first_minimum(selection)
        out[held] = {
            "epoch_index": index,
            "selection_score": float(selection[index]),
            "held_out_score": float(epoch_condition_scores[index][held]),
        }
    return out


def _condition_mean(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["sim_id"]), []).append(float(row["rmse_K"]))
    return {sim: sum(vals) / float(len(vals)) for sim, vals in grouped.items()}


def eval_points_from_s5_extract(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Join eval_history.json rows with per-condition CSV means.

    The final early-stop eval is absent from the historical JSON and is
    appended from the persisted CSV plus progress.json. Train RMSE at that
    final epoch is left missing when the train CSV is absent.
    """
    by_epoch: dict[int, list[Mapping[str, Any]]] = {}
    for row in run["val_condition_means"]:
        by_epoch.setdefault(int(row["epoch"]), []).append(row)
    points: list[dict[str, Any]] = []
    for rec in run["history"]:
        epoch = int(rec["epoch"])
        cond_rows = by_epoch.get(epoch)
        if not cond_rows:
            raise ValueError(f"missing VAL condition rows for epoch {epoch}")
        points.append(
            {
                "epoch": epoch,
                "update": int(rec["update"]),
                "val_mean": float(rec["val_mean_condition_rmse_K"]),
                "seen_mean": (
                    None
                    if rec.get("seen_mean_condition_rmse_K") is None
                    else float(rec["seen_mean_condition_rmse_K"])
                ),
                "train_mean": (
                    None
                    if rec.get("train_mean_condition_rmse_K") is None
                    else float(rec["train_mean_condition_rmse_K"])
                ),
                "lr": None if rec.get("lr") is None else float(rec["lr"]),
                "conditions": _condition_mean(cond_rows),
                "source": "eval_history.json+val_csv",
            }
        )
    tail = run["tail"]
    tail_epoch = int(tail["epoch"])
    if any(p["epoch"] == tail_epoch for p in points):
        raise ValueError(f"tail epoch {tail_epoch} already present in history")
    tail_rows = by_epoch.get(tail_epoch)
    if not tail_rows:
        raise ValueError(f"missing VAL condition rows for tail epoch {tail_epoch}")
    progress = run["progress_final"]
    seen = tail.get("seen_macro")
    points.append(
        {
            "epoch": tail_epoch,
            "update": int(progress["update"]),
            "val_mean": float(tail["val_macro"]),
            "seen_mean": None if seen is None else float(seen),
            "train_mean": None,
            "lr": None if progress.get("lr") is None else float(progress["lr"]),
            "conditions": _condition_mean(tail_rows),
            "source": "final_eval_csv+progress.json",
        }
    )
    points.sort(key=lambda row: int(row["epoch"]))
    return points


def r0_rows_for_points(
    run_id: str,
    points: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Original best/last plus the three roadmap rules. No new checkpoints."""
    if not points:
        raise ValueError(f"{run_id}: no eval points")
    by_epoch = {int(p["epoch"]): p for p in points}
    best_epoch = int(summary["best_val_epoch"])
    last_epoch = int(summary["epochs_completed"])
    if best_epoch not in by_epoch or last_epoch not in by_epoch:
        raise ValueError(f"{run_id}: best/last epoch missing from the curve")
    rows: list[dict[str, Any]] = []

    def emit(rule: str, point: Mapping[str, Any], *, score_name: str, score: float, note: str) -> None:
        cond = dict(point["conditions"])
        rows.append(
            {
                "run_id": run_id,
                "rule": rule,
                "epoch": int(point["epoch"]),
                "update": int(point["update"]),
                "score_name": score_name,
                "score": float(score),
                "val_mean_condition_rmse_K": float(point["val_mean"]),
                "seen_mean_condition_rmse_K": point["seen_mean"],
                "train_mean_condition_rmse_K": point["train_mean"],
                "lr": point["lr"],
                "conditions": sorted(cond),
                "condition_rmse_K": cond,
                "source": point["source"],
                "note": note,
            }
        )

    best = by_epoch[best_epoch]
    emit(
        "original_best_val",
        best,
        score_name="val_mean_condition_rmse_K",
        score=float(summary["best_val_metric"]),
        note="checkpoint best.pt; scheduler and early-stop used this VAL curve",
    )
    last = by_epoch[last_epoch]
    emit(
        "original_last",
        last,
        score_name="val_mean_condition_rmse_K",
        score=float(last["val_mean"]),
        note="early-stop boundary; weights in last.pt; train CSV for this epoch is absent",
    )
    seen_values = [p["seen_mean"] for p in points]
    if any(v is None for v in seen_values):
        raise ValueError(f"{run_id}: seen score missing; refusing to skip an eval")
    seen_index = first_minimum([float(v) for v in seen_values])
    seen_point = points[seen_index]
    emit(
        "min_seen",
        seen_point,
        score_name="seen_mean_condition_rmse_K",
        score=float(seen_point["seen_mean"]),
        note="first minimum of seen; seen was not the training selection metric",
    )
    val_values = [float(p["val_mean"]) for p in points]
    ma_index, ma_score = causal_moving_average_first_min(val_values)
    emit(
        "causal_ma5_first_min",
        points[ma_index],
        score_name="causal_mean_last_5_val",
        score=ma_score,
        note="first minimum of the causal 5-eval VAL mean; first four evals excluded",
    )
    loo = leave_one_condition_out([p["conditions"] for p in points])
    for held, picked in loo.items():
        point = points[int(picked["epoch_index"])]
        emit(
            f"loo_holdout_{held}",
            point,
            score_name="mean_of_other_val_conditions",
            score=float(picked["selection_score"]),
            note=(
                f"held-out condition RMSE {float(picked['held_out_score']):.6f}; "
                "selection used the other recorded VAL conditions, not a retraining"
            ),
        )
    return rows
