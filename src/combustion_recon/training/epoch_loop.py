"""Epoch-based dense U-Net training loop (U4 path; run requires explicit authorization)."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.evaluation.month2_evaluator import (
    aggregate_frame_then_condition_then_split,
    denormalize_temperature,
    evaluate_frame_prediction,
)
from combustion_recon.training.dense_cache import collate_dense_batch
from combustion_recon.training.trainer import (
    TrainConfig,
    build_optimizer,
    build_scheduler,
    mse_loss,
    set_seed,
    write_json,
)


@dataclass
class EpochTrainConfig:
    """Scientific epoch-loop settings (candidate until authorized)."""

    n_epochs: int = 200
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    cosine_eta_min: float = 1e-5
    grad_clip_norm: float = 1.0
    seed: int = 0
    device: str = "cpu"
    eval_every_epochs: int = 10
    wall_time_cap_s: float = 6 * 3600
    p_conditioning: bool = True
    base_channels: int = 32
    allow_lr_fallback_3e4: bool = False  # must stay False unless separately decided
    num_workers: int = 0


@dataclass
class EpochTrainState:
    epoch: int = 0  # completed epochs
    update: int = 0
    samples_seen: int = 0
    best_val_metric: float | None = None
    best_val_epoch: int | None = None
    status: str = "not_started"  # running|completed|stopped_*
    stop_reason: str | None = None


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


def save_epoch_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    state: EpochTrainState,
    cfg: EpochTrainConfig,
    loader_generator: torch.Generator | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_kind": "dense_unet_epoch_v1",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch_state": asdict(state),
        "epoch_config": asdict(cfg),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "extra": extra or {},
    }
    if loader_generator is not None:
        payload["dataloader_generator_state"] = loader_generator.get_state()
    torch.save(payload, path)


def load_epoch_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
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
        self.model = model
        self.cfg = cfg
        self.normalizer = normalizer
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.n_train_samples = int(n_train_samples)
        self.u_per_epoch = updates_per_epoch(self.n_train_samples, cfg.batch_size)
        self.total_updates = total_scheduled_updates(
            cfg.n_epochs, self.n_train_samples, cfg.batch_size
        )
        # Reuse update-based scheduler wiring via TrainConfig.max_updates = total.
        tcfg = TrainConfig(
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            warmup_fraction=cfg.warmup_fraction,
            cosine_eta_min=cfg.cosine_eta_min,
            grad_clip_norm=cfg.grad_clip_norm,
            max_updates=self.total_updates,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            device=cfg.device,
            wall_time_cap_s=cfg.wall_time_cap_s,
        )
        self.optimizer = build_optimizer(model, tcfg)
        self.scheduler = build_scheduler(self.optimizer, tcfg)
        self.state = EpochTrainState()
        self.curves: list[dict[str, Any]] = []
        self.eval_history: list[dict[str, Any]] = []
        self.clip_events = 0
        self.loader_generator: torch.Generator | None = None

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
        self.scheduler.step()
        self.state.update += 1
        self.state.samples_seen += int(x.shape[0])
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

    def fit(
        self,
        train_ds: Dataset,
        *,
        seen_loader: DataLoader | None,
        val_loader: DataLoader,
        out_dir: Path,
        run_identity: dict[str, Any],
        resume_path: str | Path | None = None,
        start_epoch: int = 0,
    ) -> dict[str, Any]:
        """Run epoch loop. Does not claim completion unless all epochs finish cleanly."""
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
                scheduler=self.scheduler,
                map_location=self.device,
                restore_rng=True,
            )
            st = payload.get("epoch_state") or {}
            self.state = EpochTrainState(**{k: st[k] for k in asdict(EpochTrainState()) if k in st})
            start_epoch = int(self.state.epoch)
            if "dataloader_generator_state" in payload:
                self.loader_generator = torch.Generator()
                self.loader_generator.set_state(payload["dataloader_generator_state"])
            # Exact shuffle continuity after resume is restored from saved generator
            # + epoch index; declared only when those fields are present.
            run_identity = {
                **run_identity,
                "resumed_from": str(resume_path),
                "resume_shuffle_state_restored": "dataloader_generator_state" in payload,
            }
            write_json(out_dir / "run_identity.json", run_identity)

        self.state.status = "running"
        self.state.stop_reason = None
        t0 = time.perf_counter()
        wall_cap = float(self.cfg.wall_time_cap_s)
        last_train_loss = float("nan")
        samples_this_epoch_expected = self.n_train_samples

        try:
            for epoch in range(start_epoch, int(self.cfg.n_epochs)):
                if (time.perf_counter() - t0) >= wall_cap:
                    self.state.status = "stopped_wall_time"
                    self.state.stop_reason = "wall_time_cap"
                    break

                # Fresh epoch loader; seed folded with epoch for reproducible order.
                # Generator created once then advanced by DataLoader across epochs when resumed.
                epoch_seed = int(self.cfg.seed) + 100_003 * (epoch + 1)
                train_loader, self.loader_generator = make_epoch_loader(
                    train_ds,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    seed=epoch_seed,
                    generator=None,  # new generator each epoch from epoch_seed
                )
                self.train_mode()
                epoch_losses: list[float] = []
                n_seen_epoch = 0
                stop_now: str | None = None

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
                            "lr": float(self.optimizer.param_groups[0]["lr"]),
                            "grad_norm_pre_clip": grad_norm,
                        }
                    )

                if n_seen_epoch != samples_this_epoch_expected and stop_now is None:
                    # Must visit all and only train samples each full epoch.
                    stop_now = "incomplete_epoch_sample_coverage"
                    self.state.status = "stopped_error"
                    self.state.stop_reason = (
                        f"{stop_now}: seen {n_seen_epoch} != {samples_this_epoch_expected}"
                    )
                    break

                self.state.epoch = epoch + 1

                if stop_now is not None:
                    self.state.status = (
                        "stopped_wall_time"
                        if stop_now == "wall_time_cap"
                        else f"stopped_{stop_now}"
                    )
                    self.state.stop_reason = stop_now
                    break

                # Eval cadence
                if should_evaluate_epoch(
                    self.state.epoch, self.cfg.n_epochs, self.cfg.eval_every_epochs
                ):
                    variant = "p_on" if self.cfg.p_conditioning else "p_off"
                    eval_rec: dict[str, Any] = {
                        "epoch": self.state.epoch,
                        "update": self.state.update,
                    }
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
                        # seen never selects
                    val_out = self.evaluate_loader(
                        val_loader,
                        role="val_eval",
                        variant=variant,
                        model_seed=self.cfg.seed,
                        extra_tags={"run_id": run_identity.get("run_id")},
                    )
                    eval_rec["val_mean_condition_rmse_K"] = val_out["mean_condition_rmse_K"]
                    self.eval_history.append(eval_rec)

                    # Selection: first minimum on VAL
                    score = val_out["mean_condition_rmse_K"]
                    if score is not None and math.isfinite(float(score)):
                        if (
                            self.state.best_val_metric is None
                            or float(score) < float(self.state.best_val_metric)
                        ):
                            self.state.best_val_metric = float(score)
                            self.state.best_val_epoch = int(self.state.epoch)
                            save_epoch_checkpoint(
                                ckpt_dir / "best.pt",
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                state=self.state,
                                cfg=self.cfg,
                                loader_generator=self.loader_generator,
                                extra={
                                    "selection_metric": "val_mean_condition_rmse_K",
                                    "selection_value": self.state.best_val_metric,
                                    "run_identity": run_identity,
                                },
                            )
                    # Restore train mode before further updates
                    self.train_mode()

                # Always refresh last checkpoint after each completed epoch
                save_epoch_checkpoint(
                    ckpt_dir / "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    state=self.state,
                    cfg=self.cfg,
                    loader_generator=self.loader_generator,
                    extra={"run_identity": run_identity},
                )
                write_json(
                    out_dir / "progress.json",
                    {
                        "epoch": self.state.epoch,
                        "update": self.state.update,
                        "samples_seen": self.state.samples_seen,
                        "status": self.state.status,
                        "best_val_metric": self.state.best_val_metric,
                        "best_val_epoch": self.state.best_val_epoch,
                        "last_train_loss": last_train_loss,
                        "epoch_mean_loss": float(np.mean(epoch_losses))
                        if epoch_losses
                        else None,
                        "elapsed_s": time.perf_counter() - t0,
                    },
                )

            else:
                # for-loop exhausted without break
                self.state.status = "completed"
                self.state.stop_reason = None

        except Exception as exc:  # noqa: BLE001 — must record failure state
            self.state.status = "stopped_error"
            self.state.stop_reason = f"exception:{type(exc).__name__}:{exc}"
            try:
                save_epoch_checkpoint(
                    ckpt_dir / "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    state=self.state,
                    cfg=self.cfg,
                    loader_generator=self.loader_generator,
                    extra={"run_identity": run_identity, "error": str(exc)},
                )
            except Exception:
                pass
            raise

        # Final last save if stopped mid-run
        if self.state.status != "completed":
            save_epoch_checkpoint(
                ckpt_dir / "last.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                state=self.state,
                cfg=self.cfg,
                loader_generator=self.loader_generator,
                extra={"run_identity": run_identity},
            )

        selection = select_best_val(self.eval_history)
        summary = {
            "status": self.state.status,
            "stop_reason": self.state.stop_reason,
            "completed": self.state.status == "completed",
            "epochs_completed": self.state.epoch,
            "n_epochs_planned": self.cfg.n_epochs,
            "updates": self.state.update,
            "total_updates_planned": self.total_updates,
            "samples_seen": self.state.samples_seen,
            "clip_events": self.clip_events,
            "clip_frequency": self.clip_events / max(1, self.state.update),
            "best_val_metric": self.state.best_val_metric,
            "best_val_epoch": self.state.best_val_epoch,
            "selection": selection,
            "elapsed_s": time.perf_counter() - t0,
            "allow_lr_fallback_3e4": self.cfg.allow_lr_fallback_3e4,
            "scientific_run": True,
        }
        write_json(out_dir / "summary.json", summary)
        write_json(out_dir / "eval_history.json", self.eval_history)
        if self.curves:
            pd.DataFrame(self.curves).to_csv(out_dir / "curves.csv.gz", index=False, compression="gzip")
        return summary
