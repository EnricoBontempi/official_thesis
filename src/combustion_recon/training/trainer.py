"""Minimal dense U-Net trainer (MSE on normalized T; checkpointable)."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.evaluation.month2_evaluator import (
    aggregate_frame_then_condition_then_split,
    denormalize_temperature,
    evaluate_frame_prediction,
)
from combustion_recon.models.unet import DenseUNet
from combustion_recon.training.dense_cache import collate_dense_batch


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    cosine_eta_min: float = 1e-5
    grad_clip_norm: float = 1.0
    max_updates: int = 500
    batch_size: int = 8
    seed: int = 0
    device: str = "cpu"
    wall_time_cap_s: float | None = 30 * 60


@dataclass
class TrainState:
    update: int = 0
    samples_seen: int = 0
    best_val_metric: float | None = None


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(math.ceil(cfg.warmup_fraction * cfg.max_updates)))
    total = max(cfg.max_updates, warmup_steps + 1)
    eta_min_ratio = cfg.cosine_eta_min / cfg.lr if cfg.lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        # step is 0-indexed after each scheduler.step()
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def save_checkpoint(
    path: str | Path,
    *,
    model: DenseUNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    state: TrainState,
    cfg: TrainConfig,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_state": asdict(state),
        "train_config": asdict(cfg),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: DenseUNet,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if restore_rng:
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])
        if "numpy_rng_state" in payload:
            np.random.set_state(payload["numpy_rng_state"])
    return payload


class DenseUNetTrainer:
    def __init__(
        self,
        model: DenseUNet,
        *,
        cfg: TrainConfig,
        normalizer: Month2FitOnlyNormalizer | None = None,
    ):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.state = TrainState()
        self.normalizer = normalizer
        self.history: list[dict[str, Any]] = []
        self.clip_events = 0
        self.grad_norm_pre_clip: list[float] = []

    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval_mode()
        return self.model(x.to(self.device))

    def train_updates(
        self,
        loader: DataLoader,
        *,
        max_updates: int | None = None,
        wall_time_cap_s: float | None = None,
        log_every: int = 10,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run up to ``max_updates`` optimizer steps (cycles the loader as needed)."""
        self.train_mode()
        max_u = int(max_updates if max_updates is not None else self.cfg.max_updates)
        wall_cap = wall_time_cap_s if wall_time_cap_s is not None else self.cfg.wall_time_cap_s
        t0 = time.perf_counter()
        it = iter(loader)
        last_loss = float("nan")
        first_loss = None
        stopped_reason = "max_updates"

        while self.state.update < max_u:
            if wall_cap is not None and (time.perf_counter() - t0) >= float(wall_cap):
                stopped_reason = "wall_time_cap"
                break
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            x = batch["input_tensor"].to(self.device)
            y = batch["target_T_norm"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(x)
            loss = mse_loss(pred, y)
            if not torch.isfinite(loss):
                stopped_reason = "non_finite_loss"
                last_loss = float(loss.detach().cpu())
                break
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip_norm
                )
            )
            self.grad_norm_pre_clip.append(grad_norm)
            if grad_norm > self.cfg.grad_clip_norm:
                self.clip_events += 1
            self.optimizer.step()
            self.scheduler.step()
            self.state.update += 1
            self.state.samples_seen += int(x.shape[0])
            last_loss = float(loss.detach().cpu())
            if first_loss is None:
                first_loss = last_loss
            rec = {
                "update": self.state.update,
                "loss": last_loss,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "grad_norm_pre_clip": grad_norm,
            }
            self.history.append(rec)
            if callback is not None and (self.state.update % log_every == 0):
                callback(rec)

        elapsed = time.perf_counter() - t0
        return {
            "updates": self.state.update,
            "samples_seen": self.state.samples_seen,
            "first_loss": first_loss,
            "last_loss": last_loss,
            "elapsed_s": elapsed,
            "stopped_reason": stopped_reason,
            "clip_events": self.clip_events,
            "clip_frequency": (self.clip_events / max(1, self.state.update)),
        }

    @torch.no_grad()
    def evaluate_loader(
        self,
        loader: DataLoader,
        *,
        role: str,
        method: str = "DenseUNet",
        variant: str = "p_on",
        model_seed: int = 0,
        velocity_control: str = "aligned",
    ) -> dict[str, Any]:
        """Evaluate in Kelvin via fit-only denormalization; fail-fast aggregation."""
        if self.normalizer is None:
            raise ValueError("normalizer required for Kelvin evaluation")
        self.eval_mode()
        rows: list[dict[str, Any]] = []
        sigma = float(self.normalizer.temperature_std)
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
                    p_conditioning=bool(variant == "p_on"),
                    K=0,
                    site_fraction=1.0,
                    model_seed=int(model_seed),
                    mask_replica_id=0,
                    layout_id="dense_full_grid",
                    task_mode="temperature_free",
                    velocity_control=velocity_control,
                )
                rows.append(row)
        agg = aggregate_frame_then_condition_then_split(rows)
        return {"rows": rows, "aggregation": agg}


def make_loader(dataset, *, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_dense_batch,
        generator=g if shuffle else None,
    )


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
