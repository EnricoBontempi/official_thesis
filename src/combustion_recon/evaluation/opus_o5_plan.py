"""Planning helpers for the Opus O5 evaluation.

These functions do not train, refit, select a checkpoint, or rank a run.
Loop-curve rows stay labeled as training-loop metrics.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

CHECKPOINTS: tuple[str, ...] = ("best", "best_seen_diagnostic", "last")
ROLES: tuple[str, ...] = ("train_fit", "seen_train_eval", "val_eval")
COVERAGES: tuple[float, ...] = (0.01, 0.05, 0.20)
REPLICAS: tuple[int, ...] = (0, 1, 2)
ROLE_N_FRAMES = {
    "train_fit": 760,
    "seen_train_eval": 220,
    "val_eval": 123,
}
ROLE_LABEL = {
    "train_fit": "reference760",
    "seen_train_eval": "seen220",
    "val_eval": "VAL123",
}
CRITERIA = {
    "best": "first_absolute_min_val_mean_condition_rmse_K",
    "best_seen_diagnostic": "min_seen_replica0_mean_condition_rmse_K",
    "last": "early_stop_epoch",
}
LOOP_SOURCE = "training_loop_not_o5"
PRESENTATIONS_PER_EPOCH = 760
UPDATES_PER_EPOCH = 95
REQUIRED_RUN_IDS = (
    "O-B19-on-s0",
    "O-B19-off-s0",
    "O-A38-on-s0",
    "O-A38-off-s0",
    "O-A38-on-s1",
    "O-A38-off-s1",
    "O-C149-on-s0",
    "O-C149-off-s0",
)
HISTORICAL_CACHE_ID = "month2_dense_unet_cache_v1"
C149_CACHE_ID = "month2_dense_unet_cache_opus_c149_v1"
CORE_METRICS = (
    "rmse_K",
    "mae_K",
    "mean_signed_error_K",
    "abs_bias_K",
    "centered_rmse_K",
    "high_rmse_K",
    "high_mae_K",
    "high_iou",
    "high_rel_area_error",
    "non_high_rmse_K",
    "peak_rmse_K",
    "jet_rmse_K",
    "frac_omega_high_true",
    "frac_omega_high_pred",
)
GROUP_KEYS = (
    "run_id",
    "checkpoint_kind",
    "role",
    "site_fraction",
    "mask_replica_id",
    "sim_id",
)
IDENTITY_KEYS = (
    "pool_id",
    "model_seed",
    "p_conditioning",
    "epoch",
    "criterion",
    "file_sha256",
    "weights_sha256",
    "shared_forward_from",
)


def reuse_plan(weight_sha_by_checkpoint: Mapping[str, str]) -> dict[str, str | None]:
    """Map a later checkpoint onto the first identical weight hash.

    The returned source is the checkpoint whose forward may be copied.
    A null source means this checkpoint needs its own forward. Identities
    stay distinct even when the source is not null.
    """
    missing = [name for name in CHECKPOINTS if name not in weight_sha_by_checkpoint]
    if missing:
        raise KeyError(f"weight hashes missing for {missing}")
    first: dict[str, str] = {}
    plan: dict[str, str | None] = {}
    for name in CHECKPOINTS:
        digest = str(weight_sha_by_checkpoint[name])
        if not digest:
            raise ValueError(f"empty weight hash for {name}")
        if digest in first:
            plan[name] = first[digest]
        else:
            first[digest] = name
            plan[name] = None
    return plan


def loop_curve_frame(
    history: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    pool_id: str,
    pool_size: int,
    p_conditioning: bool,
    model_seed: int,
) -> pd.DataFrame:
    """One row per recorded eval. Metrics are the training loop, not O5."""
    if int(pool_size) <= 0:
        raise ValueError("pool_size must be positive")
    rows: list[dict[str, Any]] = []
    for rec in history:
        epoch = int(rec["epoch"])
        update = int(rec["update"])
        expected_update = epoch * UPDATES_PER_EPOCH
        if update != expected_update:
            raise ValueError(
                f"{run_id} epoch {epoch}: update {update} != {expected_update}"
            )
        presentations = epoch * PRESENTATIONS_PER_EPOCH
        rows.append(
            {
                "run_id": run_id,
                "pool_id": pool_id,
                "p_conditioning": bool(p_conditioning),
                "model_seed": int(model_seed),
                "epoch": epoch,
                "update": update,
                "presentations": presentations,
                "pool_size": int(pool_size),
                "pool_passages": presentations / float(pool_size),
                "train_rep0_rmse_K": float(rec["train_mean_condition_rmse_K"]),
                "seen_rep0_rmse_K": float(rec["seen_mean_condition_rmse_K"]),
                "val_mean3_rmse_K": float(rec["val_mean_condition_rmse_K"]),
                "metric_source": LOOP_SOURCE,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"{run_id} has an empty eval history")
    if frame["metric_source"].ne(LOOP_SOURCE).any():
        raise ValueError("loop curve source label was overwritten")
    return frame


def shared_field_scales(
    trues: Sequence[np.ndarray],
    signed_errors: Sequence[np.ndarray],
) -> dict[str, float]:
    """One temperature scale and one symmetric error scale for a whole panel.

    Limits come from every array together. A panel does not keep its own
    vmin/vmax.
    """
    if not trues or not signed_errors:
        raise ValueError("shared scales need truth and signed-error arrays")
    truth = [np.asarray(item, dtype=np.float64) for item in trues]
    errors = [np.asarray(item, dtype=np.float64) for item in signed_errors]
    vmin = float(min(float(item.min()) for item in truth))
    vmax = float(max(float(item.max()) for item in truth))
    emax = float(max(float(np.max(np.abs(item))) for item in errors))
    if not all(math.isfinite(v) for v in (vmin, vmax, emax)):
        raise ValueError("non-finite shared scale")
    if vmax <= vmin:
        vmax = vmin + 1.0
    if emax <= 0.0:
        emax = 1.0
    return {"vmin": vmin, "vmax": vmax, "emax": emax}


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"frame table missing columns: {missing}")


def aggregate_frames(
    frame: pd.DataFrame,
    *,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Condition means, then means over conditions, then over replicas.

    No ranking column is added. RMSE must be finite on every frame.
    """
    expected = dict(expected_counts or ROLE_N_FRAMES)
    _require_columns(
        frame,
        (
            *GROUP_KEYS,
            "frame_index",
            "rmse_K",
            "mae_K",
            "mean_signed_error_K",
            "abs_bias_K",
            "centered_rmse_K",
            "high_iou",
            "high_rel_area_error",
        ),
    )
    if not np.isfinite(frame["rmse_K"].to_numpy(dtype=np.float64)).all():
        raise ValueError("rmse_K has a non-finite value")
    metrics = [name for name in CORE_METRICS if name in frame.columns]
    work = frame.copy()
    for key in ("site_fraction", "mask_replica_id", "epoch", "model_seed"):
        if key in work.columns:
            work[key] = pd.to_numeric(work[key])
    grouped_sizes = work.groupby(list(GROUP_KEYS), dropna=False)["frame_index"]
    sizes = grouped_sizes.nunique()
    dupes = grouped_sizes.size()
    if int((dupes - sizes).sum()) != 0:
        raise ValueError("duplicate frame_index inside a condition group")
    for (role, count) in sizes.groupby(level="role"):
        wanted = int(expected[str(role)])
        if not bool((count == wanted).all()):
            raise ValueError(
                f"{role} expected {wanted} frames per condition, found {sorted(set(count.tolist()))}"
            )
    identity = [name for name in IDENTITY_KEYS if name in work.columns]
    by_condition = (
        work.groupby(list(GROUP_KEYS) + identity, dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    by_condition["n_frames"] = (
        work.groupby(list(GROUP_KEYS) + identity, dropna=False)["frame_index"]
        .nunique()
        .to_numpy()
    )
    by_condition["table"] = "condition_mean"
    over_conditions = (
        by_condition.groupby(
            [name for name in (*GROUP_KEYS, *identity) if name != "sim_id" and name in by_condition.columns],
            dropna=False,
        )[metrics]
        .mean()
        .reset_index()
    )
    over_conditions["table"] = "mean_over_conditions"
    replica_keys = [
        name
        for name in over_conditions.columns
        if name not in metrics and name not in ("table", "mask_replica_id")
    ]
    over_replicas = (
        over_conditions.groupby(replica_keys, dropna=False)[metrics].mean().reset_index()
    )
    over_replicas["table"] = "mean_over_replicas"
    for table in (by_condition, over_conditions, over_replicas):
        if "winner" in table.columns or "rank" in table.columns:
            raise ValueError("aggregation must not rank runs")
    return {
        "by_condition": by_condition,
        "over_conditions": over_conditions,
        "over_replicas": over_replicas,
    }
