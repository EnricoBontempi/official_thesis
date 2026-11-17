"""Synthetic tests for U4 epoch-loop runner (no scientific training)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.training.epoch_loop import (
    EpochDenseTrainer,
    EpochTrainConfig,
    EpochTrainState,
    load_epoch_checkpoint,
    make_epoch_loader,
    save_epoch_checkpoint,
    select_best_val,
    should_evaluate_epoch,
    total_scheduled_updates,
    updates_per_epoch,
)


class TinyDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class TinyDenseDataset(torch.utils.data.Dataset):
    """Synthetic dense samples (H=W=16) with Month-2-like metadata."""

    def __init__(self, n: int, role: str, *, n_conditions: int = 2, seed: int = 0):
        self.n = n
        self.role = role
        self.split = "val" if role == "val_eval" else "train"
        rng = np.random.default_rng(seed)
        self.x = rng.normal(size=(n, 6, 16, 16)).astype(np.float32)
        self.y = rng.normal(size=(n, 1, 16, 16)).astype(np.float32)
        # Physical T around mean so denormalize path is finite
        self.y_K = (2000.0 + 10.0 * rng.normal(size=(n, 1, 16, 16))).astype(np.float32)
        self.conds = [f"C{i % n_conditions}.h5" for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "model_inputs": {
                "input_tensor": torch.from_numpy(self.x[i].copy()),
                "p_conditioning": True,
            },
            "targets": {
                "temperature_normalized": torch.from_numpy(self.y[i].copy()),
                "temperature": torch.from_numpy(self.y_K[i].copy()),
            },
            "metadata": {
                "sim_id": self.conds[i],
                "frame_index": int(i * 40),
                "split": self.split,
                "role": self.role,
            },
        }


def test_should_evaluate_and_selection_tiebreak():
    assert should_evaluate_epoch(10, 200, 10)
    assert should_evaluate_epoch(200, 200, 10)
    assert not should_evaluate_epoch(11, 200, 10)
    assert should_evaluate_epoch(25, 25, 10)
    hist = [
        {"epoch": 10, "val_mean_condition_rmse_K": 5.0},
        {"epoch": 20, "val_mean_condition_rmse_K": 5.0},  # tie → keep first
        {"epoch": 30, "val_mean_condition_rmse_K": 4.0},
    ]
    best = select_best_val(hist)
    assert best is not None
    assert best["epoch"] == 30
    hist2 = [
        {"epoch": 10, "val_mean_condition_rmse_K": 3.0},
        {"epoch": 20, "val_mean_condition_rmse_K": 3.0},
    ]
    best2 = select_best_val(hist2)
    assert best2["epoch"] == 10


def test_updates_scheduling_matches_full_epoch_visits():
    assert updates_per_epoch(760, 8) == 95
    assert total_scheduled_updates(200, 760, 8) == 19000
    assert updates_per_epoch(7, 4) == 2


def test_epoch_loop_visits_all_samples_and_eval_cadence(tmp_path: Path):
    set_n = 8
    train = TinyDenseDataset(set_n, "train_fit", n_conditions=2, seed=0)
    seen = TinyDenseDataset(4, "seen_train_eval", n_conditions=2, seed=1)
    val = TinyDenseDataset(4, "val_eval", n_conditions=2, seed=2)
    cfg = EpochTrainConfig(
        n_epochs=5,
        batch_size=4,
        lr=1e-3,
        seed=0,
        device="cpu",
        eval_every_epochs=2,
        wall_time_cap_s=120.0,
        base_channels=8,
    )
    normalizer = Month2FitOnlyNormalizer()
    trainer = EpochDenseTrainer(
        TinyDenseNet(), cfg=cfg, normalizer=normalizer, n_train_samples=set_n
    )
    seen_loader, _ = make_epoch_loader(seen, batch_size=4, shuffle=False, seed=0)
    val_loader, _ = make_epoch_loader(val, batch_size=4, shuffle=False, seed=0)
    summary = trainer.fit(
        train,
        seen_loader=seen_loader,
        val_loader=val_loader,
        out_dir=tmp_path / "run",
        run_identity={"run_id": "synth", "seed": 0},
    )
    assert summary["completed"] is True
    assert summary["status"] == "completed"
    assert summary["epochs_completed"] == 5
    assert summary["updates"] == total_scheduled_updates(5, set_n, 4)
    assert summary["samples_seen"] == 5 * set_n
    # eval at epochs 2,4,5
    epochs_eval = [e["epoch"] for e in trainer.eval_history]
    assert epochs_eval == [2, 4, 5]
    assert (tmp_path / "run" / "checkpoints" / "last.pt").exists()
    assert (tmp_path / "run" / "checkpoints" / "best.pt").exists()
    # train mode restored after evals
    assert trainer.model.training is True


def test_checkpoint_reload_and_wall_stop(tmp_path: Path):
    train = TinyDenseDataset(8, "train_fit", seed=3)
    val = TinyDenseDataset(4, "val_eval", seed=4)
    cfg = EpochTrainConfig(
        n_epochs=3,
        batch_size=4,
        seed=3,
        device="cpu",
        eval_every_epochs=1,
        wall_time_cap_s=1e-9,  # force wall stop immediately
        base_channels=8,
    )
    normalizer = Month2FitOnlyNormalizer()
    trainer = EpochDenseTrainer(
        TinyDenseNet(), cfg=cfg, normalizer=normalizer, n_train_samples=8
    )
    val_loader, _ = make_epoch_loader(val, batch_size=4, shuffle=False, seed=0)
    summary = trainer.fit(
        train,
        seen_loader=None,
        val_loader=val_loader,
        out_dir=tmp_path / "wall",
        run_identity={"run_id": "wall"},
    )
    assert summary["completed"] is False
    assert summary["status"] == "stopped_wall_time"
    assert (tmp_path / "wall" / "checkpoints" / "last.pt").exists()

    # Save/load roundtrip predictions
    model = TinyDenseNet()
    cfg2 = EpochTrainConfig(n_epochs=2, batch_size=4, seed=0, device="cpu", eval_every_epochs=1)
    tr = EpochDenseTrainer(model, cfg=cfg2, normalizer=normalizer, n_train_samples=8)
    # train one short completed run for a real checkpoint
    cfg3 = EpochTrainConfig(
        n_epochs=1,
        batch_size=4,
        seed=0,
        device="cpu",
        eval_every_epochs=1,
        wall_time_cap_s=60.0,
    )
    tr2 = EpochDenseTrainer(
        TinyDenseNet(), cfg=cfg3, normalizer=normalizer, n_train_samples=8
    )
    val_loader, _ = make_epoch_loader(val, batch_size=4, shuffle=False, seed=0)
    tr2.fit(
        train,
        seen_loader=None,
        val_loader=val_loader,
        out_dir=tmp_path / "ckpt",
        run_identity={"run_id": "ckpt"},
    )
    x = torch.from_numpy(train.x[:2])
    with torch.no_grad():
        p0 = tr2.model(x).clone()
    model2 = TinyDenseNet()
    load_epoch_checkpoint(tmp_path / "ckpt" / "checkpoints" / "last.pt", model=model2)
    model2.eval()
    with torch.no_grad():
        p1 = model2(x)
    assert torch.allclose(p0, p1)


def test_nonfinite_grad_sets_stopped_status(tmp_path: Path):
    train = TinyDenseDataset(4, "train_fit", seed=5)
    val = TinyDenseDataset(2, "val_eval", seed=6)
    cfg = EpochTrainConfig(
        n_epochs=2,
        batch_size=2,
        seed=5,
        device="cpu",
        eval_every_epochs=1,
        wall_time_cap_s=60.0,
    )
    normalizer = Month2FitOnlyNormalizer()
    model = TinyDenseNet()
    trainer = EpochDenseTrainer(
        model, cfg=cfg, normalizer=normalizer, n_train_samples=4
    )
    handle = model.net[-1].weight.register_hook(
        lambda g: torch.full_like(g, float("nan"))
    )
    val_loader, _ = make_epoch_loader(val, batch_size=2, shuffle=False, seed=0)
    try:
        summary = trainer.fit(
            train,
            seen_loader=None,
            val_loader=val_loader,
            out_dir=tmp_path / "nan",
            run_identity={"run_id": "nan"},
        )
    finally:
        handle.remove()
    assert summary["completed"] is False
    assert "non_finite" in str(summary["status"])


def test_cli_refuses_u4_without_authorization():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/07_train_reconstructor.py"),
            "train-u4",
            "--run-id",
            "should_not_run",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "authorize-scientific-u4" in (proc.stderr + proc.stdout)
