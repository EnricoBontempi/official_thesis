"""Scientific sparse epoch runner for U-Net / adapted GLU (S4 path).

Reuses verified schedule/checkpoint helpers from ``epoch_loop`` (warmup,
ReduceLROnPlateau, absolute-best vs significant early-stop, complete vs partial
checkpoints). Diagnostic ``SparseEpochTrainer`` remains separate and incomplete.

Scientific training runs are gated by YAML ``scientific_training_authorized``
and CLI authorization flags; S4 profiling configs keep that flag false.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.evaluation.month2_evaluator import (
    aggregate_frame_then_condition_then_split,
    denormalize_temperature,
    evaluate_frame_prediction,
)
from combustion_recon.training.epoch_loop import (
    EpochTrainConfig,
    EpochTrainState,
    ResumeIncompatibilityError,
    apply_tf32_policy,
    relative_improved,
    select_best_val,
    should_evaluate_epoch,
    total_scheduled_updates,
    updates_per_epoch,
)
from combustion_recon.training.scientific_epoch_sampler import (
    ListSampler,
    ScientificPresentationStream,
)
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset, collate_sparse_batch
from combustion_recon.training.sparse_epoch_trainer import make_sparse_loader
from combustion_recon.training.trainer import build_optimizer, mse_loss, set_seed, write_json

Family = Literal["sparse_unet", "adapted_glu"]

CHECKPOINT_KIND_COMPLETE = "sparse_scientific_epoch_complete_v1"
CHECKPOINT_KIND_PARTIAL = "sparse_scientific_epoch_partial_v1"


def _adapted_glu_cls():
    from combustion_recon.models.glu_reconstructor import AdaptedDeterministicGLU

    return AdaptedDeterministicGLU


def _is_adapted_glu(model: nn.Module) -> bool:
    return isinstance(model, _adapted_glu_cls())

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
    "family",
    "site_fraction",
    "train_mask_seed",
    "eval_mask_seed",
    "layout_id",
    "val_replicas",
    "presentations_per_scientific_epoch",
    "pool_id",
    "require_eval_mask_payload",
    "amp_enabled",
    "torch_compile",
)


@dataclass
class SparseScientificConfig(EpochTrainConfig):
    """Scientific sparse settings (defaults match dense U4 schedule)."""

    family: Family = "sparse_unet"
    site_fraction: float = 0.05
    train_mask_seed: int = 20261118
    eval_mask_seed: int = 20261116
    layout_id: str = "random_sparse_primary"
    val_replicas: tuple[int, ...] = (0, 1, 2)
    shuffle_seed: int = 0
    decode_chunk_size: int = 512
    decode_chunked_eval: bool = True
    microbatch_size: int | None = None  # None → full batch_size; accumulation to batch_size
    wall_time_cap_s: float = 6 * 3600
    # Opus scientific-epoch contract. None → historical one-pass-over-pool (S5).
    presentations_per_scientific_epoch: int | None = None
    pool_id: str | None = None
    require_eval_mask_payload: bool = False
    amp_enabled: bool = False
    torch_compile: bool = False
    save_best_seen_diagnostic: bool = False


@dataclass
class SparseScientificState(EpochTrainState):
    mask_epoch: int = 0  # last set_epoch value applied to train dataset
    sampler_cycle: int = 0
    sampler_pos: int = 0
    sampler_presentations_served: int = 0


def _state_from_mapping(raw: dict[str, Any]) -> SparseScientificState:
    allowed = {f.name for f in fields(SparseScientificState)}
    return SparseScientificState(**{k: raw[k] for k in allowed if k in raw})


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write checkpoint via temp file + os.replace to avoid truncated .pt files."""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def save_sparse_scientific_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    plateau_scheduler: ReduceLROnPlateau | None,
    state: SparseScientificState,
    cfg: SparseScientificConfig,
    checkpoint_kind: str,
    loader_generator: torch.Generator | None = None,
    eval_history: list[dict[str, Any]] | None = None,
    curves: list[dict[str, Any]] | None = None,
    clip_events: int = 0,
    extra: dict[str, Any] | None = None,
    model_state_dict: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = model_state_dict if model_state_dict is not None else model.state_dict()
    payload: dict[str, Any] = {
        "checkpoint_kind": checkpoint_kind,
        "resumable": checkpoint_kind == CHECKPOINT_KIND_COMPLETE,
        "model_state_dict": weights,
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
        "python_random_state": None,
        "extra": extra or {},
    }
    # Explicit python RNG for continuity tests when used.
    import random

    payload["python_random_state"] = random.getstate()
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if loader_generator is not None:
        payload["dataloader_generator_state"] = loader_generator.get_state()
    if _is_adapted_glu(model):
        payload["glu_geometry_meta"] = dict(model.geometry_meta)
    if extra and "scientific_sampler_state" in extra:
        payload["scientific_sampler_state"] = extra["scientific_sampler_state"]
    _atomic_torch_save(payload, path)


def load_sparse_scientific_checkpoint(
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
    if require_resumable and kind != CHECKPOINT_KIND_COMPLETE:
        raise ResumeIncompatibilityError(
            f"unsupported checkpoint_kind for sparse scientific resume: {kind!r}"
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
        if payload.get("python_random_state") is not None:
            import random

            random.setstate(payload["python_random_state"])
        if "cuda_rng_state_all" in payload and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
            except Exception:
                pass
    return payload


def assert_sparse_resume_compatible(
    *,
    payload: dict[str, Any],
    cfg: SparseScientificConfig,
    run_identity: dict[str, Any],
    allow_extend_max_epochs: bool = False,
) -> dict[str, Any]:
    ck_cfg_raw = payload.get("epoch_config") or {}
    mismatches: list[str] = []
    for key in _RESUME_LOCKED_CFG_KEYS:
        if key not in ck_cfg_raw:
            mismatches.append(f"missing_cfg:{key}")
            continue
        left = ck_cfg_raw[key]
        right = getattr(cfg, key)
        if key == "val_replicas":
            if tuple(left) != tuple(right):
                mismatches.append(f"{key}:{left!r}!={right!r}")
            continue
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
        "protocol_sha256",
        "scientific_candidate_sha256",
        "model_config_sha256",
        "stats_sha256",
        "cache_payload_sha256",
        "mask_payload_sha256",
        "eval_mask_manifest_file_sha256",
        "eval_mask_aggregate_payload_sha256",
        "family",
    ):
        if key not in ck_identity or key not in run_identity:
            continue
        if ck_identity[key] != run_identity[key]:
            mismatches.append(f"identity:{key}")

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


def assert_sparse_scientific_candidate_contract(cand: dict[str, Any]) -> dict[str, Any]:
    """Consume sparse scientific candidate YAML; refuse unknown / forbidden keys."""
    required_top = {
        "experiment_id",
        "status",
        "scientific_training_authorized",
        "profiling_diagnostic_only",
        "inherits_protocol",
        "model",
        "training_candidate",
        "artifacts",
        "decision",
    }
    unknown_top = sorted(
        set(cand)
        - required_top
        - {
            "command_template",
            "notes",
            "family",
            "s5_submission_authorized",
            "opus_submission_authorized",
            "s7_submission_authorized",
            "operational_s7",
            "operational_provisional_pending_s4_profiling",
            "operational_from_s4_profiling_candidate",
            "operational_from_s5_recipe",
        }
    )
    if unknown_top:
        raise ValueError(f"unknown sparse scientific top-level keys: {unknown_top}")

    if bool(cand.get("scientific_training_authorized")) and bool(
        cand.get("profiling_diagnostic_only")
    ):
        raise ValueError("cannot be both scientific_training_authorized and profiling_diagnostic_only")

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
        "shuffle_seed",
        "train_mask_seed",
        "eval_mask_seed",
        "layout_id",
        "val_replicas",
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
        "family",
        "decode_chunk_size",
        "decode_chunked_eval",
        "microbatch_size",
        "pool_id",
        "presentations_per_scientific_epoch",
        "require_eval_mask_payload",
        "amp",
        "torch_compile",
        "save_best_seen_diagnostic",
    }
    unknown_tc = sorted(set(tc) - allowed_tc)
    if unknown_tc:
        raise ValueError(f"unknown training_candidate keys: {unknown_tc}")

    if tc.get("schedule_kind") != "warmup_then_reduce_on_plateau":
        raise ValueError(f"unsupported schedule_kind: {tc.get('schedule_kind')!r}")
    if tc.get("precision") != "fp32":
        raise ValueError("precision must be fp32")
    if bool(tc.get("allow_tf32", False)):
        raise ValueError("allow_tf32 must be false")
    if bool(tc.get("amp", False)):
        raise ValueError("amp must be false (explicit fp32)")
    if bool(tc.get("torch_compile", False)):
        raise ValueError("torch_compile must be false")
    if bool(tc.get("allow_automatic_lr_fallback_3e4", False)):
        raise ValueError("automatic lr fallback must remain false")
    if int(tc.get("K", -1)) != 0:
        raise ValueError("K must be 0")
    if not math.isclose(float(tc.get("site_fraction", -1)), 0.05, rel_tol=0, abs_tol=1e-15):
        raise ValueError("first sparse scientific coverage is 0.05 only")
    if tc.get("temporal_mode") != "single_frame":
        raise ValueError("temporal_mode must be single_frame")
    if int(tc.get("warmup_updates", -1)) != 950:
        raise ValueError("warmup_updates must be 950")
    if int(tc.get("batch_size", -1)) != 8:
        raise ValueError("batch_size must be 8")
    if int(tc.get("n_epochs", -1)) != 1000:
        raise ValueError("n_epochs initial tranche must be 1000")
    if tc.get("presentations_per_scientific_epoch") is not None:
        if int(tc["presentations_per_scientific_epoch"]) != 760:
            raise ValueError("presentations_per_scientific_epoch must be 760 when set")
    if tc.get("pool_id") is not None and str(tc["pool_id"]) not in {"A38", "B19", "C149"}:
        raise ValueError("pool_id must be A38|B19|C149 when set")

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

    family = str(tc.get("family") or cand.get("family") or "")
    if family not in {"sparse_unet", "adapted_glu"}:
        raise ValueError(f"family must be sparse_unet|adapted_glu, got {family!r}")

    return {
        "experiment_id": cand["experiment_id"],
        "family": family,
        "scientific_training_authorized": bool(cand.get("scientific_training_authorized")),
        "profiling_diagnostic_only": bool(cand.get("profiling_diagnostic_only")),
        "n_epochs": int(tc["n_epochs"]),
        "warmup_updates": int(tc["warmup_updates"]),
        "site_fraction": float(tc["site_fraction"]),
        "p_conditioning": bool(tc["p_conditioning"]),
    }


def config_from_candidate(
    cand: dict[str, Any],
    *,
    device: str,
    wall_time_cap_s: float | None = None,
    overrides: dict[str, Any] | None = None,
) -> SparseScientificConfig:
    """Build runtime config from candidate YAML (scientific defaults)."""
    meta = assert_sparse_scientific_candidate_contract(cand)
    tc = cand["training_candidate"]
    plateau = tc["plateau"]
    early = tc["early_stopping"]
    cfg = SparseScientificConfig(
        n_epochs=int(tc["n_epochs"]),
        batch_size=int(tc["batch_size"]),
        lr=float(tc["lr"]),
        weight_decay=float(tc["weight_decay"]),
        warmup_updates=int(tc["warmup_updates"]),
        plateau_factor=float(plateau["factor"]),
        plateau_patience_evals=int(plateau["patience_evals"]),
        plateau_threshold_rel=float(plateau["threshold_rel"]),
        plateau_min_lr=float(plateau["min_lr"]),
        grad_clip_norm=float(tc["grad_clip_norm"]),
        seed=int(tc["seed"]),
        shuffle_seed=int(tc.get("shuffle_seed", tc["seed"])),
        device=device,
        eval_every_epochs=int(tc["eval_every_epochs"]),
        wall_time_cap_s=float(
            wall_time_cap_s
            if wall_time_cap_s is not None
            else float(tc.get("wall_time_cap_hours", 6.0)) * 3600.0
        ),
        p_conditioning=bool(tc["p_conditioning"]),
        base_channels=int((cand.get("model") or {}).get("base_channels", 32)),
        allow_lr_fallback_3e4=False,
        num_workers=int(tc.get("num_workers", 0)),
        early_stop_min_epochs=int(early["min_epochs"]),
        early_stop_patience_evals=int(early["patience_evals"]),
        early_stop_rel_improvement=float(early["rel_improvement"]),
        schedule_kind="warmup_then_reduce_on_plateau",
        allow_tf32=False,
        eval_train_fit=True,
        family=meta["family"],  # type: ignore[arg-type]
        site_fraction=float(tc["site_fraction"]),
        train_mask_seed=int(tc["train_mask_seed"]),
        eval_mask_seed=int(tc["eval_mask_seed"]),
        layout_id=str(tc["layout_id"]),
        val_replicas=tuple(int(x) for x in tc.get("val_replicas", [0, 1, 2])),
        decode_chunk_size=int(tc.get("decode_chunk_size", 512)),
        decode_chunked_eval=bool(tc.get("decode_chunked_eval", True)),
        microbatch_size=(
            None if tc.get("microbatch_size") in (None, "null") else int(tc["microbatch_size"])
        ),
        presentations_per_scientific_epoch=(
            None
            if tc.get("presentations_per_scientific_epoch") in (None, "null")
            else int(tc["presentations_per_scientific_epoch"])
        ),
        pool_id=(None if tc.get("pool_id") in (None, "null") else str(tc["pool_id"])),
        require_eval_mask_payload=bool(tc.get("require_eval_mask_payload", False)),
        amp_enabled=bool(tc.get("amp", False)),
        torch_compile=bool(tc.get("torch_compile", False)),
        save_best_seen_diagnostic=bool(tc.get("save_best_seen_diagnostic", False)),
    )
    if cfg.amp_enabled or cfg.torch_compile:
        raise ValueError("amp and torch_compile must remain disabled")
    if overrides:
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"unknown SparseScientificConfig override: {k}")
            setattr(cfg, k, v)
    return cfg


class SparseScientificTrainer:
    """Epoch trainer shared by sparse U-Net and adapted GLU."""

    def __init__(
        self,
        model: nn.Module,
        *,
        cfg: SparseScientificConfig,
        normalizer: Month2FitOnlyNormalizer,
        n_train_samples: int,
        query_coords_phys: torch.Tensor | None = None,
    ):
        if cfg.schedule_kind != "warmup_then_reduce_on_plateau":
            raise ValueError(f"unsupported schedule_kind: {cfg.schedule_kind}")
        if cfg.allow_lr_fallback_3e4:
            raise ValueError("allow_lr_fallback_3e4 must be False")
        if cfg.family == "adapted_glu" and not _is_adapted_glu(model):
            raise TypeError("adapted_glu family requires AdaptedDeterministicGLU")
        if cfg.family == "sparse_unet" and _is_adapted_glu(model):
            raise TypeError("sparse_unet family cannot use AdaptedDeterministicGLU")

        self.model = model
        self.cfg = cfg
        self.normalizer = normalizer
        self.device = torch.device(cfg.device)
        self.tf32_policy = apply_tf32_policy(allow_tf32=cfg.allow_tf32)
        if cfg.amp_enabled or cfg.torch_compile:
            raise ValueError("amp/torch_compile must be false for sparse scientific training")
        self.model.to(self.device)
        # n_train_samples = pool size (distinct frames). Scientific presentations
        # may exceed pool size (B19); updates/epoch use presentations when set.
        self.n_train_samples = int(n_train_samples)
        self.presentations_per_epoch = int(
            cfg.presentations_per_scientific_epoch
            if cfg.presentations_per_scientific_epoch is not None
            else n_train_samples
        )
        self.use_scientific_stream = cfg.presentations_per_scientific_epoch is not None
        self.u_per_epoch = updates_per_epoch(self.presentations_per_epoch, cfg.batch_size)
        self.total_updates = total_scheduled_updates(
            cfg.n_epochs, self.presentations_per_epoch, cfg.batch_size
        )
        self.presentation_stream: ScientificPresentationStream | None = None
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
        self.state = SparseScientificState(last_lr=float(cfg.lr))
        self.curves: list[dict[str, Any]] = []
        self.eval_history: list[dict[str, Any]] = []
        self.clip_events = 0
        self.loader_generator: torch.Generator | None = None
        # Smoke-only fault injection (None in every scientific run): called as
        # fault_hook(epoch_index_0based, batches_done_in_epoch) before each step.
        self.fault_hook: Any = None
        # Per-epoch trace of the presentation stream, consumed by smoke checks.
        self.stream_trace: list[dict[str, Any]] = []
        if query_coords_phys is not None:
            self.query_coords_phys = query_coords_phys.to(self.device)
        else:
            self.query_coords_phys = None
        self._set_lr(0.0)

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)
        self.state.last_lr = float(lr)

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _apply_warmup_lr(self) -> None:
        step = int(self.state.update)
        w = max(1, int(self.cfg.warmup_updates))
        if step <= w:
            self._set_lr(float(self.cfg.lr) * (float(step) / float(w)))
        elif step == w + 1 and self._current_lr() < float(self.cfg.lr):
            self._set_lr(float(self.cfg.lr))

    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()

    def _forward_norm(self, batch: dict[str, Any], *, decode_chunked: bool | None = None) -> torch.Tensor:
        if self.cfg.family == "sparse_unet":
            x = batch["input_tensor"].to(self.device)
            return self.model(x)
        # GLU
        assert self.query_coords_phys is not None
        tokens = batch["glu_tokens"]
        B = len(tokens)
        coords = torch.stack(
            [
                torch.from_numpy(np.asarray(t["coords_phys"], dtype=np.float32))
                for t in tokens
            ],
            dim=0,
        ).to(self.device)
        vals = torch.stack(
            [
                torch.from_numpy(np.asarray(t["values_norm"], dtype=np.float32))
                for t in tokens
            ],
            dim=0,
        ).to(self.device)
        sites = torch.stack(
            [
                torch.from_numpy(np.asarray(t["site_index"], dtype=np.int64))
                for t in tokens
            ],
            dim=0,
        ).to(self.device)
        if "p_norm_tensor" not in batch:
            raise KeyError("GLU batches require p_norm_tensor (attached by trainer)")
        p = batch["p_norm_tensor"].to(self.device)
        if not self.cfg.p_conditioning:
            p = torch.zeros_like(p)
        Y = self.query_coords_phys.unsqueeze(0).expand(B, -1, -1).contiguous()
        use_chunk = (
            self.cfg.decode_chunked_eval if decode_chunked is None else bool(decode_chunked)
        )
        pred = self.model(
            coords,
            vals,
            sites,
            p,
            Y,
            p_conditioning=self.cfg.p_conditioning,
            decode_chunked=use_chunk,
            chunk_size=self.cfg.decode_chunk_size,
            return_flat=False,
        )
        return pred

    def _one_train_step(self, batch: dict[str, Any]) -> tuple[float, float] | str:
        y = batch["target_T_norm"].to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        mb = self.cfg.microbatch_size
        if mb is None or mb >= int(y.shape[0]):
            pred = self._forward_norm(batch, decode_chunked=False)
            loss = mse_loss(pred, y)
            if not torch.isfinite(loss):
                return "non_finite_loss"
            loss.backward()
            loss_v = float(loss.detach().cpu())
        else:
            # Gradient accumulation to effective batch_size (equal weighting by samples).
            n = int(y.shape[0])
            total_loss = 0.0
            for start in range(0, n, int(mb)):
                sl = slice(start, min(start + int(mb), n))
                sub = {k: (v[sl] if torch.is_tensor(v) else v) for k, v in batch.items()}
                if "metadata" in batch:
                    sub["metadata"] = batch["metadata"][sl]
                if "glu_tokens" in batch:
                    sub["glu_tokens"] = batch["glu_tokens"][sl]
                pred = self._forward_norm(sub, decode_chunked=False)
                # Scale so sum of microbatch grads == full-batch mean MSE grad.
                loss_mb = mse_loss(pred, y[sl]) * (pred.shape[0] / float(n))
                if not torch.isfinite(loss_mb):
                    return "non_finite_loss"
                loss_mb.backward()
                total_loss += float(loss_mb.detach().cpu())
            loss_v = total_loss

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
        self.state.samples_seen += int(y.shape[0])
        self._apply_warmup_lr()
        return loss_v, grad_norm

    @torch.no_grad()
    def evaluate_dataset(
        self,
        ds: SparseUNetCacheDataset,
        *,
        role: str,
        replica_id: int,
        method: str,
        variant: str,
        model_seed: int,
        batch_size: int = 8,
        extra_tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.eval_mode()
        ds.replica_id = int(replica_id)
        ds.mask_mode = "evaluation"
        loader, _ = make_sparse_loader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            shuffle_seed=0,
            num_workers=0,
        )
        rows: list[dict[str, Any]] = []
        sigma = float(self.normalizer.temperature_std)
        tags = dict(extra_tags or {})
        for batch in loader:
            # Attach p_norm for GLU from metadata when available via dataset p_norm array
            if self.cfg.family == "adapted_glu":
                p_rows = []
                for meta in batch["metadata"]:
                    idx = int(meta["cache_index"])
                    p_rows.append(
                        [float(ds.p_norm[idx, 0]), float(ds.p_norm[idx, 1])]
                    )
                batch["p_norm_tensor"] = torch.tensor(p_rows, dtype=torch.float32)
            pred_n = self._forward_norm(batch, decode_chunked=True).detach().cpu().numpy()
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
                    site_fraction=float(self.cfg.site_fraction),
                    model_seed=int(model_seed),
                    mask_replica_id=int(replica_id),
                    layout_id=str(self.cfg.layout_id),
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
            "replica_id": int(replica_id),
        }

    def evaluate_val_replicas(
        self,
        val_ds: SparseUNetCacheDataset,
        *,
        method: str,
        variant: str,
        model_seed: int,
        extra_tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mean over configured VAL replicas of mean-condition RMSE (selection metric)."""
        per_rep = []
        all_rows: list[dict[str, Any]] = []
        for rid in self.cfg.val_replicas:
            out = self.evaluate_dataset(
                val_ds,
                role="val_eval",
                replica_id=int(rid),
                method=method,
                variant=variant,
                model_seed=model_seed,
                extra_tags=extra_tags,
            )
            per_rep.append(
                {
                    "replica_id": int(rid),
                    "mean_condition_rmse_K": out["mean_condition_rmse_K"],
                }
            )
            all_rows.extend(out["rows"])
        scores = [r["mean_condition_rmse_K"] for r in per_rep if r["mean_condition_rmse_K"] is not None]
        mean_over_replicas = float(np.mean(scores)) if scores else None
        return {
            "per_replica": per_rep,
            "mean_condition_rmse_K": mean_over_replicas,
            "rows": all_rows,
        }

    def _persist_eval_tables(
        self,
        out_dir: Path,
        *,
        epoch: int,
        role: str,
        rows: list[dict[str, Any]],
    ) -> None:
        ed = out_dir / "evals" / f"epoch_{epoch:04d}"
        ed.mkdir(parents=True, exist_ok=True)
        if rows:
            pd.DataFrame(rows).to_csv(
                ed / f"{role}_per_frame.csv.gz", index=False, compression="gzip"
            )

    def _write_progress(
        self, out_dir: Path, *, last_train_loss: float, epoch_losses: list[float], t0: float
    ) -> None:
        write_json(
            out_dir / "progress.json",
            {
                "epoch": self.state.epoch,
                "mask_epoch": self.state.mask_epoch,
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
                "family": self.cfg.family,
            },
        )

    def _trace_epoch(self, epoch: int, epoch_indices: Sequence[int]) -> None:
        """Record the presentation order of one scientific epoch (smoke/audit)."""
        import hashlib

        blob = ",".join(str(int(i)) for i in epoch_indices).encode()
        rec = {
            "epoch_index_0based": int(epoch),
            "n_presentations": len(epoch_indices),
            "indices_sha256": hashlib.sha256(blob).hexdigest(),
            "first8": [int(i) for i in list(epoch_indices)[:8]],
            "stream_cycle": int(self.state.sampler_cycle),
            "stream_pos": int(self.state.sampler_pos),
        }
        self.stream_trace.append(rec)
        path = getattr(self, "stream_trace_path", None)
        if path is not None:
            with open(path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def fit(
        self,
        train_ds: SparseUNetCacheDataset,
        *,
        val_ds: SparseUNetCacheDataset,
        out_dir: Path,
        run_identity: dict[str, Any],
        seen_ds: SparseUNetCacheDataset | None = None,
        train_eval_ds: SparseUNetCacheDataset | None = None,
        resume_path: str | Path | None = None,
        allow_extend_max_epochs: bool = False,
    ) -> dict[str, Any]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if len(train_ds) != self.n_train_samples:
            raise ValueError(
                f"train_ds length {len(train_ds)} != n_train_samples {self.n_train_samples}"
            )
        if train_ds.mask_mode != "training":
            raise ValueError("train_ds.mask_mode must be 'training'")

        if self.cfg.family == "adapted_glu" and self.query_coords_phys is None:
            from combustion_recon.models.glu_reconstructor import make_synthetic_query_grid

            self.query_coords_phys = make_synthetic_query_grid(train_ds.x, train_ds.y).to(
                self.device
            )

        write_json(out_dir / "run_identity.json", run_identity)
        start_epoch = 0

        if self.use_scientific_stream:
            self.presentation_stream = ScientificPresentationStream(
                pool_size=self.n_train_samples,
                shuffle_seed=int(self.cfg.shuffle_seed),
                presentations_per_epoch=self.presentations_per_epoch,
            )

        if resume_path is not None:
            payload = load_sparse_scientific_checkpoint(
                resume_path,
                model=self.model,
                optimizer=self.optimizer,
                plateau_scheduler=self.plateau_scheduler,
                map_location=self.device,
                restore_rng=True,
                require_resumable=True,
            )
            resume_meta = assert_sparse_resume_compatible(
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
            if self.use_scientific_stream:
                samp = payload.get("scientific_sampler_state")
                if samp is None:
                    raise ResumeIncompatibilityError(
                        "resume rejected: scientific_sampler_state missing from checkpoint"
                    )
                assert self.presentation_stream is not None
                self.presentation_stream.load_state_dict(samp)
            train_ds.set_epoch(self.state.mask_epoch)
            # The best-seen diagnostic tracker lives in run_identity (mutated in
            # place during training). Restore it from the checkpoint, otherwise a
            # resumed run would overwrite best_seen_diagnostic.pt with a worse epoch.
            ck_ident = (payload.get("extra") or {}).get("run_identity") or {}
            for _k in ("_best_seen_metric", "_best_seen_epoch"):
                if _k in ck_ident:
                    run_identity[_k] = ck_ident[_k]
            resumed_status = str(self.state.status)
            run_identity = {
                **run_identity,
                "resumed_from": str(resume_path),
                "resume_checkpoint_epoch": int(self.state.epoch),
                "resume_checkpoint_update": int(self.state.update),
                "resume_checkpoint_status": resumed_status,
                "resume_meta": resume_meta,
                "resume_shuffle_state_restored": "dataloader_generator_state" in payload,
                "resume_mask_epoch": self.state.mask_epoch,
                "resume_scientific_sampler_restored": bool(
                    payload.get("scientific_sampler_state")
                ),
            }
            write_json(out_dir / "run_identity.json", run_identity)

        method = "SparseUNet" if self.cfg.family == "sparse_unet" else "AdaptedGLU"
        variant = "p_on" if self.cfg.p_conditioning else "p_off"
        # A checkpoint written at an early-stop epoch already carries the terminal
        # decision; resuming it must finalize the run, not train past the stop.
        already_stopped = (
            resume_path is not None
            and str(self.state.status) == "stopped_early_plateau"
        )
        if not already_stopped:
            self.state.status = "running"
            self.state.stop_reason = None
        t0 = time.perf_counter()
        wall_cap = float(self.cfg.wall_time_cap_s)
        last_train_loss = float("nan")
        stop_now: str | None = None

        def _ckpt_extra(base: dict[str, Any] | None = None) -> dict[str, Any]:
            out = dict(base or {})
            out["run_identity"] = run_identity
            out["presentations_per_scientific_epoch"] = self.presentations_per_epoch
            out["pool_size"] = self.n_train_samples
            out["updates_per_scientific_epoch"] = self.u_per_epoch
            out["amp_enabled"] = False
            out["torch_compile"] = False
            out["allow_tf32"] = False
            out["precision"] = "fp32"
            if self.presentation_stream is not None:
                out["scientific_sampler_state"] = self.presentation_stream.state_dict()
            return out

        try:
            loop_epochs = (
                range(0) if already_stopped else range(start_epoch, int(self.cfg.n_epochs))
            )
            for epoch in loop_epochs:
                if (time.perf_counter() - t0) >= wall_cap:
                    self.state.status = "stopped_wall_time"
                    self.state.stop_reason = "wall_time_cap"
                    break

                # Mask epoch = scientific epoch index (0-based). Shared by all
                # presentations of the same frame within this scientific epoch.
                train_ds.set_epoch(epoch)
                self.state.mask_epoch = int(epoch)

                if self.use_scientific_stream:
                    assert self.presentation_stream is not None
                    epoch_indices = self.presentation_stream.next_scientific_epoch()
                    self.state.sampler_cycle = int(self.presentation_stream.state.cycle)
                    self.state.sampler_pos = int(self.presentation_stream.state.pos)
                    self.state.sampler_presentations_served = int(
                        self.presentation_stream.state.presentations_served
                    )
                    self._trace_epoch(epoch, epoch_indices)
                    train_loader = DataLoader(
                        train_ds,
                        batch_size=self.cfg.batch_size,
                        sampler=ListSampler(epoch_indices),
                        shuffle=False,
                        num_workers=self.cfg.num_workers,
                        collate_fn=collate_sparse_batch,
                    )
                    self.loader_generator = None
                else:
                    epoch_seed = int(self.cfg.shuffle_seed) + 100_003 * (epoch + 1)
                    train_loader, self.loader_generator = make_sparse_loader(
                        train_ds,
                        batch_size=self.cfg.batch_size,
                        shuffle=True,
                        shuffle_seed=epoch_seed,
                        num_workers=self.cfg.num_workers,
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
                    if self.fault_hook is not None:
                        self.fault_hook(int(epoch), len(epoch_losses))
                    if self.cfg.family == "adapted_glu":
                        p_rows = []
                        for meta in batch["metadata"]:
                            idx = int(meta["cache_index"])
                            p_rows.append(
                                [float(train_ds.p_norm[idx, 0]), float(train_ds.p_norm[idx, 1])]
                            )
                        batch["p_norm_tensor"] = torch.tensor(p_rows, dtype=torch.float32)
                    step_out = self._one_train_step(batch)
                    if isinstance(step_out, str):
                        stop_now = step_out
                        break
                    loss_v, grad_norm = step_out
                    last_train_loss = loss_v
                    epoch_losses.append(loss_v)
                    n_seen_epoch += int(batch["target_T_norm"].shape[0])
                    self.curves.append(
                        {
                            "epoch": epoch + 1,
                            "update": self.state.update,
                            "loss": loss_v,
                            "lr": self._current_lr(),
                            "grad_norm_pre_clip": grad_norm,
                            "completed_epochs": self.state.epoch,
                            "mask_epoch": self.state.mask_epoch,
                        }
                    )

                if stop_now is not None:
                    self.state.status = (
                        "stopped_wall_time"
                        if stop_now == "wall_time_cap"
                        else f"stopped_{stop_now}"
                    )
                    self.state.stop_reason = stop_now
                    save_sparse_scientific_checkpoint(
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

                if n_seen_epoch != self.presentations_per_epoch:
                    self.state.status = "stopped_error"
                    self.state.stop_reason = (
                        f"incomplete_epoch_sample_coverage: "
                        f"seen {n_seen_epoch} != {self.presentations_per_epoch}"
                    )
                    save_sparse_scientific_checkpoint(
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
                        extra=_ckpt_extra(),
                    )
                    break

                self.state.epoch = epoch + 1

                if should_evaluate_epoch(
                    self.state.epoch, self.cfg.n_epochs, self.cfg.eval_every_epochs
                ):
                    eval_rec: dict[str, Any] = {
                        "epoch": self.state.epoch,
                        "update": self.state.update,
                        "lr": self._current_lr(),
                        "mask_epoch": self.state.mask_epoch,
                    }
                    if self.cfg.eval_train_fit and train_eval_ds is not None:
                        # Diagnostic only — never used for selection.
                        tr_out = self.evaluate_dataset(
                            train_eval_ds,
                            role="train_fit",
                            replica_id=0,
                            method=method,
                            variant=variant,
                            model_seed=self.cfg.seed,
                            extra_tags={"run_id": run_identity.get("run_id"), "selection": False},
                        )
                        eval_rec["train_mean_condition_rmse_K"] = tr_out["mean_condition_rmse_K"]
                        self._persist_eval_tables(
                            out_dir, epoch=self.state.epoch, role="train_fit", rows=tr_out["rows"]
                        )
                    selected_best_seen_weights: dict[str, Any] | None = None
                    if seen_ds is not None:
                        seen_out = self.evaluate_dataset(
                            seen_ds,
                            role="seen_train_eval",
                            replica_id=0,
                            method=method,
                            variant=variant,
                            model_seed=self.cfg.seed,
                            extra_tags={"run_id": run_identity.get("run_id"), "selection": False},
                        )
                        eval_rec["seen_mean_condition_rmse_K"] = seen_out["mean_condition_rmse_K"]
                        self._persist_eval_tables(
                            out_dir,
                            epoch=self.state.epoch,
                            role="seen_train_eval",
                            rows=seen_out["rows"],
                        )
                        if self.cfg.save_best_seen_diagnostic:
                            seen_score = seen_out["mean_condition_rmse_K"]
                            if seen_score is not None and math.isfinite(float(seen_score)):
                                prev = run_identity.get("_best_seen_metric")
                                if prev is None or float(seen_score) < float(prev):
                                    run_identity["_best_seen_metric"] = float(seen_score)
                                    run_identity["_best_seen_epoch"] = int(self.state.epoch)
                                    selected_best_seen_weights = {
                                        k: v.detach().cpu().clone()
                                        for k, v in self.model.state_dict().items()
                                    }

                    val_out = self.evaluate_val_replicas(
                        val_ds,
                        method=method,
                        variant=variant,
                        model_seed=self.cfg.seed,
                        extra_tags={"run_id": run_identity.get("run_id"), "selection": True},
                    )
                    eval_rec["val_mean_condition_rmse_K"] = val_out["mean_condition_rmse_K"]
                    eval_rec["val_per_replica"] = val_out["per_replica"]
                    self._persist_eval_tables(
                        out_dir, epoch=self.state.epoch, role="val_eval", rows=val_out["rows"]
                    )
                    self.eval_history.append(eval_rec)

                    score = val_out["mean_condition_rmse_K"]
                    lr_before = self._current_lr()
                    # Defer best.pt write until AFTER early-stop/plateau/history updates
                    # so resumable best carries coherent epoch-boundary state. Weights
                    # remain those of the absolute-best selection (snapshot below).
                    selected_best_weights: dict[str, Any] | None = None
                    if score is not None and math.isfinite(float(score)):
                        score_f = float(score)
                        if self.state.best_val_metric is None or score_f < float(
                            self.state.best_val_metric
                        ):
                            self.state.best_val_metric = score_f
                            self.state.best_val_epoch = int(self.state.epoch)
                            selected_best_weights = {
                                k: v.detach().cpu().clone()
                                for k, v in self.model.state_dict().items()
                            }

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

                        if self.state.update >= int(self.cfg.warmup_updates):
                            self.plateau_scheduler.step(score_f)
                            lr_after = self._current_lr()
                            self.state.last_lr = lr_after
                            if lr_after < lr_before - 1e-16:
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

                    if selected_best_weights is not None:
                        save_sparse_scientific_checkpoint(
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
                            model_state_dict=selected_best_weights,
                            extra=_ckpt_extra(
                                {
                                    "selection_metric": "val_mean_condition_rmse_K",
                                    "selection_value": self.state.best_val_metric,
                                    "selection_epoch": self.state.best_val_epoch,
                                    "best_weights_are_absolute_min_selection": True,
                                    "epoch_boundary_state_after_scheduler_counters": True,
                                }
                            ),
                        )
                    if selected_best_seen_weights is not None:
                        # Diagnostic only — does not change scheduler/early-stop/primary.
                        save_sparse_scientific_checkpoint(
                            ckpt_dir / "best_seen_diagnostic.pt",
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
                            model_state_dict=selected_best_seen_weights,
                            extra=_ckpt_extra(
                                {
                                    "selection_metric": "seen_mean_condition_rmse_K",
                                    "selection_value": run_identity.get("_best_seen_metric"),
                                    "selection_epoch": run_identity.get("_best_seen_epoch"),
                                    "diagnostic_only": True,
                                    "does_not_affect_scheduler_or_primary": True,
                                }
                            ),
                        )

                    if self.state.status == "stopped_early_plateau":
                        save_sparse_scientific_checkpoint(
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
                            extra=_ckpt_extra(),
                        )
                        # Persist standalone history including the final early-stop eval
                        # (previously omitted on this path; last.pt still held it).
                        write_json(out_dir / "eval_history.json", self.eval_history)
                        self._write_progress(
                            out_dir,
                            last_train_loss=last_train_loss,
                            epoch_losses=epoch_losses,
                            t0=t0,
                        )
                        break

                save_sparse_scientific_checkpoint(
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
                    extra=_ckpt_extra(),
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
                if not already_stopped:
                    self.state.status = "completed"
                    self.state.stop_reason = "reached_max_epochs"

        except Exception as exc:  # noqa: BLE001
            self.state.status = "stopped_error"
            self.state.stop_reason = f"exception:{type(exc).__name__}:{exc}"
            try:
                save_sparse_scientific_checkpoint(
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
        summary = {
            "status": self.state.status,
            "stop_reason": self.state.stop_reason,
            "completed": self.state.status == "completed",
            "epochs_completed": self.state.epoch,
            "n_epochs_planned": self.cfg.n_epochs,
            "updates": self.state.update,
            "total_updates_planned": self.total_updates,
            "updates_per_epoch": self.u_per_epoch,
            "presentations_per_scientific_epoch": self.presentations_per_epoch,
            "pool_size": self.n_train_samples,
            "pool_id": self.cfg.pool_id,
            "use_scientific_stream": self.use_scientific_stream,
            "samples_seen": self.state.samples_seen,
            "best_val_metric": self.state.best_val_metric,
            "best_val_epoch": self.state.best_val_epoch,
            "best_seen_diagnostic_metric": run_identity.get("_best_seen_metric"),
            "best_seen_diagnostic_epoch": run_identity.get("_best_seen_epoch"),
            "early_stop_ref_metric": self.state.early_stop_ref_metric,
            "early_stop_bad_evals": self.state.early_stop_bad_evals,
            "selection": selection,
            "family": self.cfg.family,
            "clip_events": self.clip_events,
            "elapsed_s": time.perf_counter() - t0,
            "tf32_policy": self.tf32_policy,
            "precision": "fp32",
            "amp_enabled": False,
            "torch_compile": False,
        }
        write_json(out_dir / "summary.json", summary)
        return summary
