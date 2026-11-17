"""Epoch-based dense U-Net training loop (U4 path; run requires explicit authorization).

Resume is epoch-boundary only. Complete-epoch checkpoints are distinct from
partial interrupt dumps and never overwrite a valid resumable ``last.pt``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.evaluation.month2_evaluator import (
    aggregate_frame_then_condition_then_split,
    denormalize_temperature,
    evaluate_frame_prediction,
)
from combustion_recon.training.dense_cache import collate_dense_batch
from combustion_recon.training.trainer import build_optimizer, mse_loss, set_seed, write_json

CHECKPOINT_KIND_COMPLETE = "dense_unet_epoch_complete_v2"
CHECKPOINT_KIND_PARTIAL = "dense_unet_epoch_partial_v2"

# Scientific identity keys that must match on resume (n_epochs handled separately).
_RESUME_LOCKED_CFG_KEYS = (
    "batch_size",
    "lr",
    "weight_decay",
    "warmup_updates",
    "plateau_factor",
    "plateau_patience_evals",
    "plateau_threshold_rel",
    "plateau_min_lr",
    "grad_clip_norm",
    "seed",
    "eval_every_epochs",
    "p_conditioning",
    "base_channels",
    "allow_lr_fallback_3e4",
    "num_workers",
    "early_stop_min_epochs",
    "early_stop_patience_evals",
    "early_stop_rel_improvement",
    "schedule_kind",
    "allow_tf32",
    "eval_train_fit",
)


@dataclass
class EpochTrainConfig:
    """Scientific epoch-loop settings authorized by Enrico for Engaging U4."""

    n_epochs: int = 1000
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_updates: int = 950
    plateau_factor: float = 0.5
    plateau_patience_evals: int = 5
    plateau_threshold_rel: float = 0.001
    plateau_min_lr: float = 1e-5
    grad_clip_norm: float = 1.0
    seed: int = 0
    device: str = "cuda"
    eval_every_epochs: int = 10
    wall_time_cap_s: float = 6 * 3600
    p_conditioning: bool = True
    base_channels: int = 32
    allow_lr_fallback_3e4: bool = False
    num_workers: int = 0
    early_stop_min_epochs: int = 200
    early_stop_patience_evals: int = 15
    early_stop_rel_improvement: float = 0.001
    schedule_kind: str = "warmup_then_reduce_on_plateau"
    allow_tf32: bool = False
    eval_train_fit: bool = True
    # Kept only so older YAML keys can be rejected explicitly upstream.
    warmup_fraction: float | None = None
    cosine_eta_min: float | None = None


@dataclass
class EpochTrainState:
    epoch: int = 0  # completed full epochs only
    update: int = 0
    samples_seen: int = 0
    best_val_metric: float | None = None  # true absolute first minimum
    best_val_epoch: int | None = None
    status: str = "not_started"  # running|completed|stopped_*
    stop_reason: str | None = None
    early_stop_ref_metric: float | None = None
    early_stop_bad_evals: int = 0
    plateau_num_bad_evals_at_last_step: int = 0
    last_lr: float | None = None


class ResumeIncompatibilityError(ValueError):
    """Checkpoint scientific identity does not match the requested run."""


def updates_per_epoch(n_samples: int, batch_size: int) -> int:
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return int(math.ceil(n_samples / batch_size))


def total_scheduled_updates(n_epochs: int, n_samples: int, batch_size: int) -> int:
    return int(n_epochs) * updates_per_epoch(n_samples, batch_size)


def should_evaluate_epoch(epoch_1based: int, n_epochs: int, every: int = 10) -> bool:
    if epoch_1based < 1:
        return False
    if epoch_1based == int(n_epochs):
        return True
    return epoch_1based % int(every) == 0


def relative_improved(
    new: float,
    reference: float,
    *,
    rel: float,
    mode: str = "min",
) -> bool:
    """True if ``new`` improves on ``reference`` by at least relative ``rel``."""
    if not (math.isfinite(new) and math.isfinite(reference)):
        return False
    if mode != "min":
        raise ValueError("only mode=min is supported")
    if reference == 0.0:
        return new < reference - abs(rel)
    return new < reference * (1.0 - float(rel))


def select_best_val(
    history: list[dict[str, Any]],
    *,
    metric_key: str = "val_mean_condition_rmse_K",
) -> dict[str, Any] | None:
    """First minimum of VAL mean-condition RMSE (strict < updates; ties keep earlier)."""
    best: dict[str, Any] | None = None
    for rec in history:
        if metric_key not in rec or rec[metric_key] is None:
            continue
        score = float(rec[metric_key])
        if not math.isfinite(score):
            continue
        if best is None or score < float(best[metric_key]):
            best = dict(rec)
    return best


def make_epoch_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, torch.Generator | None]:
    g = generator
    if shuffle:
        if g is None:
            g = torch.Generator()
            g.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_dense_batch,
        generator=g if shuffle else None,
    )
    return loader, g


def apply_tf32_policy(*, allow_tf32: bool) -> dict[str, Any]:
    """Disable TF32 for the first scientific comparison unless explicitly allowed."""
    recorded: dict[str, Any] = {"allow_tf32": bool(allow_tf32)}
    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
            recorded["cuda_matmul_allow_tf32"] = bool(
                torch.backends.cuda.matmul.allow_tf32
            )
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
            recorded["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    return recorded


def _state_from_mapping(raw: dict[str, Any]) -> EpochTrainState:
    allowed = {f.name for f in fields(EpochTrainState)}
    return EpochTrainState(**{k: raw[k] for k in allowed if k in raw})


def save_epoch_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    plateau_scheduler: ReduceLROnPlateau | None,
    state: EpochTrainState,
    cfg: EpochTrainConfig,
    checkpoint_kind: str,
    loader_generator: torch.Generator | None = None,
    eval_history: list[dict[str, Any]] | None = None,
    curves: list[dict[str, Any]] | None = None,
    clip_events: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_kind": checkpoint_kind,
        "resumable": checkpoint_kind == CHECKPOINT_KIND_COMPLETE,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "plateau_scheduler_state_dict": (
            plateau_scheduler.state_dict() if plateau_scheduler is not None else None
        ),
        "epoch_state": asdict(state),
        "epoch_config": asdict(cfg),
        "eval_history": list(eval_history or []),
        "curves": list(curves or []),
        "clip_events": int(clip_events),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "extra": extra or {},
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if loader_generator is not None:
        payload["dataloader_generator_state"] = loader_generator.get_state()
    torch.save(payload, path)


def load_epoch_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    plateau_scheduler: ReduceLROnPlateau | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
    require_resumable: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    kind = payload.get("checkpoint_kind")
    resumable = bool(payload.get("resumable", kind == CHECKPOINT_KIND_COMPLETE))
    if require_resumable and not resumable:
        raise ResumeIncompatibilityError(
            f"checkpoint {path} is not resumable (kind={kind!r})"
        )
    if require_resumable and kind not in {CHECKPOINT_KIND_COMPLETE, "dense_unet_epoch_v1"}:
        # v1 accepted only if explicitly marked resumable; prefer v2 complete.
        if kind != CHECKPOINT_KIND_COMPLETE:
            raise ResumeIncompatibilityError(
                f"unsupported checkpoint_kind for resume: {kind!r}"
            )
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if plateau_scheduler is not None and payload.get("plateau_scheduler_state_dict"):
        plateau_scheduler.load_state_dict(payload["plateau_scheduler_state_dict"])
    if restore_rng:
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])
        if "numpy_rng_state" in payload:
            np.random.set_state(payload["numpy_rng_state"])
        if "cuda_rng_state_all" in payload and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
            except Exception:
                pass
    return payload


def assert_resume_compatible(
    *,
    payload: dict[str, Any],
    cfg: EpochTrainConfig,
    run_identity: dict[str, Any],
    allow_extend_max_epochs: bool = False,
) -> dict[str, Any]:
    """Reject resume when data/arch/scientific settings disagree."""
    ck_cfg_raw = payload.get("epoch_config") or {}
    mismatches: list[str] = []
    for key in _RESUME_LOCKED_CFG_KEYS:
        if key not in ck_cfg_raw:
            mismatches.append(f"missing_cfg:{key}")
            continue
        left = ck_cfg_raw[key]
        right = getattr(cfg, key)
        if isinstance(right, float) or isinstance(left, float):
            if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15):
                mismatches.append(f"{key}:{left!r}!={right!r}")
        elif left != right:
            mismatches.append(f"{key}:{left!r}!={right!r}")

    ck_epochs = int(ck_cfg_raw.get("n_epochs", cfg.n_epochs))
    if int(cfg.n_epochs) < ck_epochs:
        mismatches.append(f"n_epochs_decreased:{cfg.n_epochs}<{ck_epochs}")
    elif int(cfg.n_epochs) > ck_epochs and not allow_extend_max_epochs:
        mismatches.append(
            f"n_epochs_extension_requires_flag:{ck_epochs}->{cfg.n_epochs}"
        )

    extra = payload.get("extra") or {}
    ck_identity = extra.get("run_identity") or {}
    for key in (
        "dense_config_sha256",
        "u4_candidate_sha256",
        "model_config_sha256",
        "stats_sha256",
        "cache_payload_sha256",
        "cache_data_fingerprints",
    ):
        if key not in ck_identity or key not in run_identity:
            continue
        if ck_identity[key] != run_identity[key]:
            mismatches.append(f"identity:{key}")

    ck_arch = (ck_identity.get("architecture") or {}).get("base_channels")
    if ck_arch is not None and int(ck_arch) != int(cfg.base_channels):
        mismatches.append(f"architecture.base_channels:{ck_arch}!={cfg.base_channels}")

    if mismatches:
        raise ResumeIncompatibilityError(
            "resume rejected; incompatible checkpoint: " + "; ".join(mismatches)
        )
    return {
        "n_epochs_from_checkpoint": ck_epochs,
        "n_epochs_requested": int(cfg.n_epochs),
        "extended_max_epochs": int(cfg.n_epochs) > ck_epochs,
        "allow_extend_max_epochs": bool(allow_extend_max_epochs),
    }


def assert_u4_candidate_contract(cand: dict[str, Any]) -> dict[str, Any]:
    """Consume required U4 candidate fields or reject unknowns / forbidden schedules."""
    required_top = {
        "experiment_id",
        "paired_experiment",
        "status",
        "scientific_training_authorized",
        "budget_definitively_approved",
        "u4_executed",
        "inherits_scientific_contract",
        "model",
        "training_candidate",
        "artifacts",
        "decision",
    }
    unknown_top = sorted(set(cand) - required_top - {"command_template", "notes"})
    if unknown_top:
        raise ValueError(f"unknown U4 candidate top-level keys: {unknown_top}")

    if cand.get("experiment_id") != "month2_dense_unet_u4_engaging_v1":
        raise ValueError(f"unexpected experiment_id: {cand.get('experiment_id')!r}")

    tc = cand.get("training_candidate")
    if not isinstance(tc, dict):
        raise ValueError("training_candidate must be a mapping")

    allowed_tc = {
        "temporal_mode",
        "task_mode",
        "K",
        "site_fraction",
        "p_conditioning",
        "seed",
        "device",
        "precision",
        "allow_tf32",
        "batch_size",
        "n_epochs",
        "lr",
        "weight_decay",
        "warmup_updates",
        "schedule_kind",
        "plateau",
        "grad_clip_norm",
        "eval_every_epochs",
        "eval_roles",
        "wall_time_cap_hours",
        "wall_time_cap_configurable",
        "allow_automatic_lr_fallback_3e4",
        "num_workers",
        "init",
        "selection",
        "also_evaluate",
        "early_stopping",
        "checkpoint",
        "resume_policy",
    }
    forbidden_tc = {
        "warmup_fraction_of_updates",
        "cosine_eta_min",
        "early_stopping_performance",
    }
    unknown_tc = sorted(set(tc) - allowed_tc)
    if unknown_tc:
        raise ValueError(f"unknown training_candidate keys: {unknown_tc}")
    present_forbidden = sorted(set(tc) & forbidden_tc)
    if present_forbidden:
        raise ValueError(
            "rejected obsolete schedule/early-stop keys in training_candidate: "
            f"{present_forbidden}"
        )

    if tc.get("schedule_kind") != "warmup_then_reduce_on_plateau":
        raise ValueError(f"unsupported schedule_kind: {tc.get('schedule_kind')!r}")
    if tc.get("precision") != "fp32":
        raise ValueError("precision must be fp32")
    if bool(tc.get("allow_tf32", False)):
        raise ValueError("allow_tf32 must be false for this first comparison")
    if bool(tc.get("allow_automatic_lr_fallback_3e4", False)):
        raise ValueError("automatic lr fallback must remain false")
    if int(tc.get("K", -1)) != 0 or float(tc.get("site_fraction", -1)) != 1.0:
        raise ValueError("dense free K=0 site_fraction=1.0 required")
    if tc.get("temporal_mode") != "single_frame":
        raise ValueError("temporal_mode must be single_frame")
    if int(tc.get("warmup_updates", -1)) != 950:
        raise ValueError("warmup_updates must be 950")
    if int(tc.get("n_epochs", -1)) != 1000:
        raise ValueError("n_epochs max initial budget must be 1000")

    plateau = tc.get("plateau") or {}
    for k, expected in {
        "mode": "min",
        "factor": 0.5,
        "patience_evals": 5,
        "threshold_rel": 0.001,
        "min_lr": 1e-5,
        "metric": "val_mean_condition_rmse_K",
    }.items():
        if k not in plateau:
            raise ValueError(f"plateau.{k} required")
        if k in {"factor", "threshold_rel", "min_lr"}:
            if not math.isclose(float(plateau[k]), float(expected), rel_tol=0, abs_tol=1e-15):
                raise ValueError(f"plateau.{k} must be {expected}")
        elif plateau[k] != expected:
            raise ValueError(f"plateau.{k} must be {expected!r}")

    early = tc.get("early_stopping") or {}
    for k, expected in {
        "enabled": True,
        "min_epochs": 200,
        "patience_evals": 15,
        "rel_improvement": 0.001,
        "reset_patience_on_lr_reduce": True,
        "metric": "val_mean_condition_rmse_K",
        "note": "operational_convention_not_physical_certification",
    }.items():
        if k not in early:
            raise ValueError(f"early_stopping.{k} required")
        if k == "rel_improvement":
            if not math.isclose(float(early[k]), float(expected), rel_tol=0, abs_tol=1e-15):
                raise ValueError(f"early_stopping.{k} must be {expected}")
        elif early[k] != expected:
            raise ValueError(f"early_stopping.{k} must be {expected!r}")

    ckpt = tc.get("checkpoint") or {}
    if ckpt.get("boundary") != "epoch_complete_only":
        raise ValueError("checkpoint.boundary must be epoch_complete_only")
    if ckpt.get("partial_overwrite_last") is not False:
        raise ValueError("checkpoint.partial_overwrite_last must be false")

    return {
        "experiment_id": cand["experiment_id"],
        "consumed_training_candidate_keys": sorted(tc.keys()),
        "schedule_kind": tc["schedule_kind"],
        "n_epochs": int(tc["n_epochs"]),
        "warmup_updates": int(tc["warmup_updates"]),
        "early_stopping": dict(early),
        "plateau": dict(plateau),
    }


class EpochDenseTrainer:
    """Train by epochs over a fixed train dataset; select on VAL mean-condition RMSE."""

    def __init__(
        self,
        model: nn.Module,
        *,
        cfg: EpochTrainConfig,
        normalizer: Month2FitOnlyNormalizer,
        n_train_samples: int,
    ):
        if cfg.schedule_kind != "warmup_then_reduce_on_plateau":
            raise ValueError(f"unsupported schedule_kind: {cfg.schedule_kind}")
        if cfg.allow_lr_fallback_3e4:
            raise ValueError("allow_lr_fallback_3e4 must be False")
        if cfg.warmup_fraction is not None or cfg.cosine_eta_min is not None:
            raise ValueError("cosine/fraction schedule fields are rejected for U4 Engaging")

        self.model = model
        self.cfg = cfg
        self.normalizer = normalizer
        self.device = torch.device(cfg.device)
        self.tf32_policy = apply_tf32_policy(allow_tf32=cfg.allow_tf32)
        self.model.to(self.device)
        self.n_train_samples = int(n_train_samples)
        self.u_per_epoch = updates_per_epoch(self.n_train_samples, cfg.batch_size)
        self.total_updates = total_scheduled_updates(
            cfg.n_epochs, self.n_train_samples, cfg.batch_size
        )
        # Dummy TrainConfig-compatible optimizer builder.
        from combustion_recon.training.trainer import TrainConfig

        tcfg = TrainConfig(
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            grad_clip_norm=cfg.grad_clip_norm,
            max_updates=max(1, self.total_updates),
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            device=cfg.device,
            wall_time_cap_s=cfg.wall_time_cap_s,
        )
        self.optimizer = build_optimizer(model, tcfg)
        self.plateau_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(cfg.plateau_factor),
            patience=int(cfg.plateau_patience_evals),
            threshold=float(cfg.plateau_threshold_rel),
            threshold_mode="rel",
            min_lr=float(cfg.plateau_min_lr),
        )
        self.state = EpochTrainState(last_lr=float(cfg.lr))
        self.curves: list[dict[str, Any]] = []
        self.eval_history: list[dict[str, Any]] = []
        self.clip_events = 0
        self.loader_generator: torch.Generator | None = None
        self._set_lr(0.0)  # start of warmup before first step

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)
        self.state.last_lr = float(lr)

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _apply_warmup_lr(self) -> None:
        # After optimizer.step, state.update is already incremented.
        step = int(self.state.update)
        w = max(1, int(self.cfg.warmup_updates))
        if step <= w:
            self._set_lr(float(self.cfg.lr) * (float(step) / float(w)))
        elif step == w + 1 and self._current_lr() < float(self.cfg.lr):
            # Ensure full base lr immediately after warmup window.
            self._set_lr(float(self.cfg.lr))

    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()

    def _one_train_step(self, batch: dict[str, Any]) -> tuple[float, float] | str:
        """Returns (loss, grad_norm) or a stop_reason string."""
        x = batch["input_tensor"].to(self.device)
        y = batch["target_T_norm"].to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        pred = self.model(x)
        loss = mse_loss(pred, y)
        if not torch.isfinite(loss):
            return "non_finite_loss"
        loss.backward()
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return "non_finite_gradients"
        total_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_sq += float(p.grad.detach().float().pow(2).sum().cpu())
        grad_norm = float(total_sq**0.5)
        if not math.isfinite(grad_norm):
            return "non_finite_grad_norm"
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        if grad_norm > self.cfg.grad_clip_norm:
            self.clip_events += 1
        self.optimizer.step()
        self.state.update += 1
        self.state.samples_seen += int(x.shape[0])
        self._apply_warmup_lr()
        return float(loss.detach().cpu()), grad_norm

    @torch.no_grad()
    def evaluate_loader(
        self,
        loader: DataLoader,
        *,
        role: str,
        method: str = "DenseUNet",
        variant: str = "p_on",
        model_seed: int = 0,
        extra_tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.eval_mode()
        rows: list[dict[str, Any]] = []
        sigma = float(self.normalizer.temperature_std)
        tags = dict(extra_tags or {})
        for batch in loader:
            x = batch["input_tensor"].to(self.device)
            pred_n = self.model(x).detach().cpu().numpy()
            true_K = batch["target_T_K"].numpy()
            for i, meta in enumerate(batch["metadata"]):
                pred_K = denormalize_temperature(pred_n[i, 0], self.normalizer)
                row = evaluate_frame_prediction(
                    pred_K,
                    true_K[i, 0],
                    train_sigma_T=sigma,
                    sim_id=str(meta["sim_id"]),
                    split=str(meta["split"]),
                    role=role,
                    frame_index=int(meta["frame_index"]),
                    method=method,
                    variant=variant,
                    p_conditioning=bool(self.cfg.p_conditioning),
                    K=0,
                    site_fraction=1.0,
                    model_seed=int(model_seed),
                    mask_replica_id=0,
                    layout_id="dense_full_grid",
                    task_mode="temperature_free",
                    base_channels=int(self.cfg.base_channels),
                    **tags,
                )
                rows.append(row)
        agg = aggregate_frame_then_condition_then_split(rows)
        mean_cond = None
        sel = agg["selection_mean_condition_rmse"]
        if role == "val_eval" and sel is not None and not getattr(sel, "empty", True):
            mean_cond = float(sel["mean_condition_rmse_K"].iloc[0])
        elif not agg["per_condition"].empty and "rmse_K" in agg["per_condition"].columns:
            mean_cond = float(agg["per_condition"]["rmse_K"].mean())
        return {
            "rows": rows,
            "aggregation": agg,
            "mean_condition_rmse_K": mean_cond,
        }

    def _persist_eval_tables(
        self,
        out_dir: Path,
        *,
        epoch: int,
        role: str,
        eval_out: dict[str, Any],
    ) -> None:
        ed = out_dir / "evals" / f"epoch_{epoch:04d}"
        ed.mkdir(parents=True, exist_ok=True)
        rows = eval_out["rows"]
        if rows:
            pd.DataFrame(rows).to_csv(
                ed / f"{role}_per_frame.csv.gz", index=False, compression="gzip"
            )
        agg = eval_out["aggregation"]
        if not agg["per_condition"].empty:
            agg["per_condition"].to_csv(
                ed / f"{role}_per_condition.csv.gz", index=False, compression="gzip"
            )
        if not agg["per_split"].empty:
            agg["per_split"].to_csv(
                ed / f"{role}_per_split.csv.gz", index=False, compression="gzip"
            )

    def _write_progress(self, out_dir: Path, *, last_train_loss: float, epoch_losses: list[float], t0: float) -> None:
        write_json(
            out_dir / "progress.json",
            {
                "epoch": self.state.epoch,
                "update": self.state.update,
                "samples_seen": self.state.samples_seen,
                "status": self.state.status,
                "stop_reason": self.state.stop_reason,
                "best_val_metric": self.state.best_val_metric,
                "best_val_epoch": self.state.best_val_epoch,
                "early_stop_bad_evals": self.state.early_stop_bad_evals,
                "early_stop_ref_metric": self.state.early_stop_ref_metric,
                "last_train_loss": last_train_loss,
                "epoch_mean_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                "lr": self._current_lr(),
                "elapsed_s": time.perf_counter() - t0,
                "tf32_policy": self.tf32_policy,
            },
        )

    def fit(
        self,
        train_ds: Dataset,
        *,
        seen_loader: DataLoader | None,
        val_loader: DataLoader,
        out_dir: Path,
        run_identity: dict[str, Any],
        resume_path: str | Path | None = None,
        train_eval_loader: DataLoader | None = None,
        allow_extend_max_epochs: bool = False,
        start_epoch: int = 0,
    ) -> dict[str, Any]:
        """Run epoch loop. Resume only from complete-epoch checkpoints."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if len(train_ds) != self.n_train_samples:
            raise ValueError(
                f"train_ds length {len(train_ds)} != n_train_samples {self.n_train_samples}"
            )

        write_json(out_dir / "run_identity.json", run_identity)

        if resume_path is not None:
            payload = load_epoch_checkpoint(
                resume_path,
                model=self.model,
                optimizer=self.optimizer,
                plateau_scheduler=self.plateau_scheduler,
                map_location=self.device,
                restore_rng=True,
                require_resumable=True,
            )
            resume_meta = assert_resume_compatible(
                payload=payload,
                cfg=self.cfg,
                run_identity=run_identity,
                allow_extend_max_epochs=allow_extend_max_epochs,
            )
            self.state = _state_from_mapping(payload.get("epoch_state") or {})
            self.eval_history = list(payload.get("eval_history") or [])
            self.curves = list(payload.get("curves") or [])
            self.clip_events = int(payload.get("clip_events") or 0)
            start_epoch = int(self.state.epoch)
            if "dataloader_generator_state" in payload:
                self.loader_generator = torch.Generator()
                self.loader_generator.set_state(payload["dataloader_generator_state"])
            run_identity = {
                **run_identity,
                "resumed_from": str(resume_path),
                "resume_meta": resume_meta,
                "resume_shuffle_state_restored": "dataloader_generator_state" in payload,
            }
            write_json(out_dir / "run_identity.json", run_identity)

        self.state.status = "running"
        self.state.stop_reason = None
        t0 = time.perf_counter()
        wall_cap = float(self.cfg.wall_time_cap_s)
        last_train_loss = float("nan")
        samples_this_epoch_expected = self.n_train_samples
        stop_now: str | None = None

        try:
            for epoch in range(start_epoch, int(self.cfg.n_epochs)):
                if (time.perf_counter() - t0) >= wall_cap:
                    self.state.status = "stopped_wall_time"
                    self.state.stop_reason = "wall_time_cap"
                    break

                epoch_seed = int(self.cfg.seed) + 100_003 * (epoch + 1)
                train_loader, self.loader_generator = make_epoch_loader(
                    train_ds,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    seed=epoch_seed,
                    generator=None,
                )
                self.train_mode()
                epoch_losses: list[float] = []
                n_seen_epoch = 0
                stop_now = None

                for batch in train_loader:
                    if (time.perf_counter() - t0) >= wall_cap:
                        stop_now = "wall_time_cap"
                        break
                    step_out = self._one_train_step(batch)
                    if isinstance(step_out, str):
                        stop_now = step_out
                        break
                    loss_v, grad_norm = step_out
                    last_train_loss = loss_v
                    epoch_losses.append(loss_v)
                    n_seen_epoch += int(batch["input_tensor"].shape[0])
                    self.curves.append(
                        {
                            "epoch": epoch + 1,
                            "update": self.state.update,
                            "loss": loss_v,
                            "lr": self._current_lr(),
                            "grad_norm_pre_clip": grad_norm,
                            "completed_epochs": self.state.epoch,
                        }
                    )

                if stop_now is not None:
                    # Mid-epoch interrupt: do NOT increment completed epochs;
                    # do NOT overwrite resumable last.pt.
                    self.state.status = (
                        "stopped_wall_time"
                        if stop_now == "wall_time_cap"
                        else f"stopped_{stop_now}"
                    )
                    self.state.stop_reason = stop_now
                    save_epoch_checkpoint(
                        ckpt_dir / "partial_interrupt.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        plateau_scheduler=self.plateau_scheduler,
                        state=self.state,
                        cfg=self.cfg,
                        checkpoint_kind=CHECKPOINT_KIND_PARTIAL,
                        loader_generator=self.loader_generator,
                        eval_history=self.eval_history,
                        curves=self.curves,
                        clip_events=self.clip_events,
                        extra={
                            "run_identity": run_identity,
                            "partial_epoch_index_0based": epoch,
                            "samples_seen_this_epoch": n_seen_epoch,
                            "note": "not resumable; last.pt preserved if present",
                        },
                    )
                    self._write_progress(
                        out_dir,
                        last_train_loss=last_train_loss,
                        epoch_losses=epoch_losses,
                        t0=t0,
                    )
                    break

                if n_seen_epoch != samples_this_epoch_expected:
                    self.state.status = "stopped_error"
                    self.state.stop_reason = (
                        f"incomplete_epoch_sample_coverage: "
                        f"seen {n_seen_epoch} != {samples_this_epoch_expected}"
                    )
                    save_epoch_checkpoint(
                        ckpt_dir / "partial_interrupt.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        plateau_scheduler=self.plateau_scheduler,
                        state=self.state,
                        cfg=self.cfg,
                        checkpoint_kind=CHECKPOINT_KIND_PARTIAL,
                        loader_generator=self.loader_generator,
                        eval_history=self.eval_history,
                        curves=self.curves,
                        clip_events=self.clip_events,
                        extra={"run_identity": run_identity},
                    )
                    break

                # Full epoch completed.
                self.state.epoch = epoch + 1

                if should_evaluate_epoch(
                    self.state.epoch, self.cfg.n_epochs, self.cfg.eval_every_epochs
                ):
                    variant = "p_on" if self.cfg.p_conditioning else "p_off"
                    eval_rec: dict[str, Any] = {
                        "epoch": self.state.epoch,
                        "update": self.state.update,
                        "lr": self._current_lr(),
                    }
                    if self.cfg.eval_train_fit and train_eval_loader is not None:
                        train_out = self.evaluate_loader(
                            train_eval_loader,
                            role="train_fit",
                            variant=variant,
                            model_seed=self.cfg.seed,
                            extra_tags={"run_id": run_identity.get("run_id")},
                        )
                        eval_rec["train_mean_condition_rmse_K"] = train_out[
                            "mean_condition_rmse_K"
                        ]
                        self._persist_eval_tables(
                            out_dir, epoch=self.state.epoch, role="train_fit", eval_out=train_out
                        )
                    if seen_loader is not None:
                        seen_out = self.evaluate_loader(
                            seen_loader,
                            role="seen_train_eval",
                            variant=variant,
                            model_seed=self.cfg.seed,
                            extra_tags={"run_id": run_identity.get("run_id")},
                        )
                        eval_rec["seen_mean_condition_rmse_K"] = seen_out[
                            "mean_condition_rmse_K"
                        ]
                        self._persist_eval_tables(
                            out_dir,
                            epoch=self.state.epoch,
                            role="seen_train_eval",
                            eval_out=seen_out,
                        )
                    val_out = self.evaluate_loader(
                        val_loader,
                        role="val_eval",
                        variant=variant,
                        model_seed=self.cfg.seed,
                        extra_tags={"run_id": run_identity.get("run_id")},
                    )
                    eval_rec["val_mean_condition_rmse_K"] = val_out["mean_condition_rmse_K"]
                    self._persist_eval_tables(
                        out_dir, epoch=self.state.epoch, role="val_eval", eval_out=val_out
                    )
                    self.eval_history.append(eval_rec)

                    score = val_out["mean_condition_rmse_K"]
                    lr_before = self._current_lr()
                    if score is not None and math.isfinite(float(score)):
                        score_f = float(score)
                        # Absolute best (any strict improvement).
                        if (
                            self.state.best_val_metric is None
                            or score_f < float(self.state.best_val_metric)
                        ):
                            self.state.best_val_metric = score_f
                            self.state.best_val_epoch = int(self.state.epoch)
                            save_epoch_checkpoint(
                                ckpt_dir / "best.pt",
                                model=self.model,
                                optimizer=self.optimizer,
                                plateau_scheduler=self.plateau_scheduler,
                                state=self.state,
                                cfg=self.cfg,
                                checkpoint_kind=CHECKPOINT_KIND_COMPLETE,
                                loader_generator=self.loader_generator,
                                eval_history=self.eval_history,
                                curves=self.curves,
                                clip_events=self.clip_events,
                                extra={
                                    "selection_metric": "val_mean_condition_rmse_K",
                                    "selection_value": self.state.best_val_metric,
                                    "run_identity": run_identity,
                                },
                            )

                        # Early-stop relative counter (separate from absolute best).
                        if self.state.early_stop_ref_metric is None:
                            self.state.early_stop_ref_metric = score_f
                            self.state.early_stop_bad_evals = 0
                        elif relative_improved(
                            score_f,
                            float(self.state.early_stop_ref_metric),
                            rel=self.cfg.early_stop_rel_improvement,
                        ):
                            self.state.early_stop_ref_metric = score_f
                            self.state.early_stop_bad_evals = 0
                        else:
                            self.state.early_stop_bad_evals += 1

                        # Plateau LR after warmup only.
                        if self.state.update >= int(self.cfg.warmup_updates):
                            self.plateau_scheduler.step(score_f)
                            lr_after = self._current_lr()
                            self.state.last_lr = lr_after
                            if lr_after < lr_before - 1e-16:
                                # Reset early-stop patience when LR is reduced.
                                self.state.early_stop_bad_evals = 0
                                eval_rec["lr_reduced"] = True
                                eval_rec["lr_before"] = lr_before
                                eval_rec["lr_after"] = lr_after
                            else:
                                eval_rec["lr_reduced"] = False

                        eval_rec["early_stop_bad_evals"] = self.state.early_stop_bad_evals
                        eval_rec["early_stop_ref_metric"] = self.state.early_stop_ref_metric
                        eval_rec["best_val_metric"] = self.state.best_val_metric

                    self.train_mode()

                    # Early stopping decision after eval.
                    if (
                        self.state.epoch >= int(self.cfg.early_stop_min_epochs)
                        and self.state.early_stop_bad_evals
                        >= int(self.cfg.early_stop_patience_evals)
                    ):
                        self.state.status = "stopped_early_plateau"
                        self.state.stop_reason = (
                            "early_stop_no_rel_improvement_"
                            f"{self.cfg.early_stop_rel_improvement}_"
                            f"for_{self.state.early_stop_bad_evals}_evals"
                        )
                        save_epoch_checkpoint(
                            ckpt_dir / "last.pt",
                            model=self.model,
                            optimizer=self.optimizer,
                            plateau_scheduler=self.plateau_scheduler,
                            state=self.state,
                            cfg=self.cfg,
                            checkpoint_kind=CHECKPOINT_KIND_COMPLETE,
                            loader_generator=self.loader_generator,
                            eval_history=self.eval_history,
                            curves=self.curves,
                            clip_events=self.clip_events,
                            extra={"run_identity": run_identity},
                        )
                        self._write_progress(
                            out_dir,
                            last_train_loss=last_train_loss,
                            epoch_losses=epoch_losses,
                            t0=t0,
                        )
                        break

                # Always refresh resumable last.pt after a completed epoch.
                save_epoch_checkpoint(
                    ckpt_dir / "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    plateau_scheduler=self.plateau_scheduler,
                    state=self.state,
                    cfg=self.cfg,
                    checkpoint_kind=CHECKPOINT_KIND_COMPLETE,
                    loader_generator=self.loader_generator,
                    eval_history=self.eval_history,
                    curves=self.curves,
                    clip_events=self.clip_events,
                    extra={"run_identity": run_identity},
                )
                write_json(out_dir / "eval_history.json", self.eval_history)
                if self.curves:
                    pd.DataFrame(self.curves).to_csv(
                        out_dir / "curves.csv.gz", index=False, compression="gzip"
                    )
                self._write_progress(
                    out_dir,
                    last_train_loss=last_train_loss,
                    epoch_losses=epoch_losses,
                    t0=t0,
                )

                if self.state.status.startswith("stopped_"):
                    break
            else:
                self.state.status = "completed"
                self.state.stop_reason = "reached_max_epochs"

        except Exception as exc:  # noqa: BLE001 — must record failure state
            self.state.status = "stopped_error"
            self.state.stop_reason = f"exception:{type(exc).__name__}:{exc}"
            try:
                save_epoch_checkpoint(
                    ckpt_dir / "partial_interrupt.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    plateau_scheduler=self.plateau_scheduler,
                    state=self.state,
                    cfg=self.cfg,
                    checkpoint_kind=CHECKPOINT_KIND_PARTIAL,
                    loader_generator=self.loader_generator,
                    eval_history=self.eval_history,
                    curves=self.curves,
                    clip_events=self.clip_events,
                    extra={"run_identity": run_identity, "error": str(exc)},
                )
            except Exception:
                pass
            raise

        selection = select_best_val(self.eval_history)
        improving_at_cap = (
            self.state.status == "completed"
            and self.state.best_val_epoch is not None
            and int(self.state.best_val_epoch) >= int(self.cfg.n_epochs) - int(self.cfg.eval_every_epochs)
            and self.state.early_stop_bad_evals < int(self.cfg.early_stop_patience_evals)
        )
        summary = {
            "status": self.state.status,
            "stop_reason": self.state.stop_reason,
            "completed": self.state.status == "completed",
            "epochs_completed": self.state.epoch,
            "n_epochs_planned": self.cfg.n_epochs,
            "updates": self.state.update,
            "total_updates_planned": self.total_updates,
            "updates_per_epoch": self.u_per_epoch,
            "samples_seen": self.state.samples_seen,
            "clip_events": self.clip_events,
            "clip_frequency": self.clip_events / max(1, self.state.update),
            "best_val_metric": self.state.best_val_metric,
            "best_val_epoch": self.state.best_val_epoch,
            "early_stop_ref_metric": self.state.early_stop_ref_metric,
            "early_stop_bad_evals": self.state.early_stop_bad_evals,
            "last_lr": self._current_lr(),
            "selection": selection,
            "elapsed_s": time.perf_counter() - t0,
            "allow_lr_fallback_3e4": self.cfg.allow_lr_fallback_3e4,
            "schedule_kind": self.cfg.schedule_kind,
            "tf32_policy": self.tf32_policy,
            "scientific_run": True,
            "budget_open_question_if_still_improving_at_1000": improving_at_cap,
            "checkpoints": {
                "last": str(ckpt_dir / "last.pt") if (ckpt_dir / "last.pt").exists() else None,
                "best": str(ckpt_dir / "best.pt") if (ckpt_dir / "best.pt").exists() else None,
                "partial_interrupt": (
                    str(ckpt_dir / "partial_interrupt.pt")
                    if (ckpt_dir / "partial_interrupt.pt").exists()
                    else None
                ),
            },
        }
        write_json(out_dir / "summary.json", summary)
        write_json(out_dir / "eval_history.json", self.eval_history)
        if self.curves:
            pd.DataFrame(self.curves).to_csv(
                out_dir / "curves.csv.gz", index=False, compression="gzip"
            )
        # Snapshot best/last metric tables pointers.
        write_json(
            out_dir / "selection.json",
            {
                "best": selection,
                "best_val_epoch": self.state.best_val_epoch,
                "best_val_metric": self.state.best_val_metric,
                "last_epoch": self.state.epoch,
                "last_eval": self.eval_history[-1] if self.eval_history else None,
            },
        )
        return summary
