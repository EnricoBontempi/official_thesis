"""Minimal epoch trainer for sparse U-Net diagnostics (S2).

Calls ``dataset.set_epoch`` every epoch (including after resume).
Complete-epoch checkpoints are distinct from partial dumps.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from combustion_recon.models.unet import SparseUNet
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset, collate_sparse_batch
from combustion_recon.training.trainer import mse_loss, set_seed

CHECKPOINT_KIND_COMPLETE = "sparse_unet_epoch_complete_v1"
CHECKPOINT_KIND_PARTIAL = "sparse_unet_epoch_partial_v1"


@dataclass
class SparseEpochConfig:
    n_epochs: int = 20
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    model_seed: int = 0
    shuffle_seed: int = 0
    train_mask_seed: int = 20261118
    device: str = "cpu"
    p_conditioning: bool = True
    num_workers: int = 0
    wall_time_cap_s: float | None = None


@dataclass
class SparseEpochState:
    epoch: int = 0  # completed full epochs
    update: int = 0
    samples_seen: int = 0
    best_train_loss: float | None = None
    status: str = "not_started"
    stop_reason: str | None = None


def make_sparse_loader(
    ds: SparseUNetCacheDataset,
    *,
    batch_size: int,
    shuffle: bool,
    shuffle_seed: int,
    num_workers: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, torch.Generator | None]:
    g = generator
    if shuffle and g is None:
        g = torch.Generator()
        g.manual_seed(int(shuffle_seed))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_sparse_batch,
        generator=g if shuffle else None,
    )
    return loader, g if shuffle else None


def save_sparse_checkpoint(
    path: Path,
    *,
    kind: str,
    model: SparseUNet,
    optimizer: torch.optim.Optimizer,
    state: SparseEpochState,
    cfg: SparseEpochConfig,
    history: list[dict[str, Any]],
    loader_generator: torch.Generator | None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_kind": kind,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_state": asdict(state),
        "train_config": asdict(cfg),
        "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "extra": extra or {},
    }
    if loader_generator is not None:
        payload["dataloader_generator_state"] = loader_generator.get_state()
    torch.save(payload, path)


def load_sparse_checkpoint(
    path: Path,
    *,
    model: SparseUNet,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    kind = payload.get("checkpoint_kind")
    if kind not in {CHECKPOINT_KIND_COMPLETE, CHECKPOINT_KIND_PARTIAL}:
        raise ValueError(f"unexpected checkpoint kind {kind!r}")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if restore_rng:
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])
        if "numpy_rng_state" in payload:
            np.random.set_state(payload["numpy_rng_state"])
    return payload


class SparseEpochTrainer:
    def __init__(
        self,
        model: SparseUNet,
        train_ds: SparseUNetCacheDataset,
        *,
        cfg: SparseEpochConfig,
        out_dir: str | Path,
    ):
        self.model = model
        self.train_ds = train_ds
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        set_seed(cfg.model_seed)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.state = SparseEpochState()
        self.history: list[dict[str, Any]] = []
        self.loader_generator: torch.Generator | None = None
        self.clip_events = 0

    def train(
        self,
        *,
        resume_from: Path | None = None,
        extend_n_epochs: int | None = None,
    ) -> dict[str, Any]:
        if resume_from is not None:
            payload = load_sparse_checkpoint(
                resume_from, model=self.model, optimizer=self.optimizer, map_location=self.device
            )
            if payload.get("checkpoint_kind") != CHECKPOINT_KIND_COMPLETE:
                raise ValueError("resume requires a complete-epoch checkpoint")
            st = payload["train_state"]
            self.state = SparseEpochState(**st)
            self.history = list(payload.get("history") or [])
            if "dataloader_generator_state" in payload:
                self.loader_generator = torch.Generator()
                self.loader_generator.set_state(payload["dataloader_generator_state"])
            # Critical: restore mask epoch after resume
            self.train_ds.set_epoch(self.state.epoch)
            if extend_n_epochs is not None:
                if int(extend_n_epochs) < self.cfg.n_epochs:
                    raise ValueError("extend_n_epochs must be >= configured n_epochs")
                self.cfg.n_epochs = int(extend_n_epochs)

        train_loader, self.loader_generator = make_sparse_loader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            shuffle_seed=self.cfg.shuffle_seed,
            num_workers=self.cfg.num_workers,
            generator=self.loader_generator,
        )

        t0 = time.perf_counter()
        self.state.status = "running"
        target_epochs = int(self.cfg.n_epochs)

        while self.state.epoch < target_epochs:
            if self.cfg.wall_time_cap_s is not None:
                if (time.perf_counter() - t0) >= float(self.cfg.wall_time_cap_s):
                    self.state.status = "stopped_wall_time"
                    self.state.stop_reason = "wall_time_cap"
                    break

            # Masks depend on epoch — must run every epoch including post-resume.
            self.train_ds.set_epoch(self.state.epoch)
            self.model.train()
            epoch_losses: list[float] = []

            for batch in train_loader:
                x = batch["input_tensor"].to(self.device)
                y = batch["target_T_norm"].to(self.device)
                if not self.cfg.p_conditioning:
                    if not torch.all(x[:, -2:] == 0):
                        raise RuntimeError("p_off violated in training batch")
                self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(x)
                loss = mse_loss(pred, y)
                if not torch.isfinite(loss):
                    self.state.status = "stopped_non_finite"
                    self.state.stop_reason = "non_finite_loss"
                    break
                loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip_norm
                    )
                )
                if grad_norm > self.cfg.grad_clip_norm:
                    self.clip_events += 1
                self.optimizer.step()
                self.state.update += 1
                self.state.samples_seen += int(x.shape[0])
                epoch_losses.append(float(loss.detach().cpu()))

            if self.state.stop_reason == "non_finite_loss":
                break

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            self.state.epoch += 1  # completed
            if self.state.best_train_loss is None or mean_loss < self.state.best_train_loss:
                self.state.best_train_loss = mean_loss
            rec = {
                "epoch": self.state.epoch,
                "mean_train_loss": mean_loss,
                "update": self.state.update,
                "mask_epoch_used": self.state.epoch - 1,
            }
            self.history.append(rec)

            save_sparse_checkpoint(
                self.out_dir / "last.pt",
                kind=CHECKPOINT_KIND_COMPLETE,
                model=self.model,
                optimizer=self.optimizer,
                state=self.state,
                cfg=self.cfg,
                history=self.history,
                loader_generator=self.loader_generator,
                extra={"mask_epoch": self.state.epoch},
            )

        if self.state.status == "running":
            self.state.status = "completed"
            self.state.stop_reason = "n_epochs"
        (self.out_dir / "history.json").write_text(
            json.dumps({"history": self.history, "state": asdict(self.state)}, indent=2) + "\n"
        )
        return {
            "state": asdict(self.state),
            "history": self.history,
            "elapsed_s": time.perf_counter() - t0,
            "clip_events": self.clip_events,
            "out_dir": str(self.out_dir),
        }
