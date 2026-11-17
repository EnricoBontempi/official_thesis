"""Synthetic tests for U4 epoch-loop runner (resume + schedule; no scientific training)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.training.epoch_loop import (
    CHECKPOINT_KIND_COMPLETE,
    CHECKPOINT_KIND_PARTIAL,
    EpochDenseTrainer,
    EpochTrainConfig,
    ResumeIncompatibilityError,
    assert_u4_candidate_contract,
    load_epoch_checkpoint,
    make_epoch_loader,
    relative_improved,
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


def _cfg(**kwargs) -> EpochTrainConfig:
    base = dict(
        n_epochs=5,
        batch_size=4,
        lr=1e-3,
        seed=0,
        device="cpu",
        eval_every_epochs=2,
        wall_time_cap_s=120.0,
        base_channels=8,
        warmup_updates=2,
        early_stop_min_epochs=1000,
        early_stop_patience_evals=15,
        schedule_kind="warmup_then_reduce_on_plateau",
        allow_tf32=False,
        eval_train_fit=True,
    )
    base.update(kwargs)
    return EpochTrainConfig(**base)


def test_should_evaluate_and_selection_tiebreak():
    assert should_evaluate_epoch(10, 200, 10)
    assert should_evaluate_epoch(200, 200, 10)
    assert not should_evaluate_epoch(11, 200, 10)
    hist = [
        {"epoch": 10, "val_mean_condition_rmse_K": 5.0},
        {"epoch": 20, "val_mean_condition_rmse_K": 5.0},
        {"epoch": 30, "val_mean_condition_rmse_K": 4.0},
    ]
    assert select_best_val(hist)["epoch"] == 30
    hist2 = [
        {"epoch": 10, "val_mean_condition_rmse_K": 3.0},
        {"epoch": 20, "val_mean_condition_rmse_K": 3.0},
    ]
    assert select_best_val(hist2)["epoch"] == 10


def test_relative_improved():
    assert relative_improved(0.998, 1.0, rel=0.001)
    assert not relative_improved(0.9995, 1.0, rel=0.001)


def test_updates_scheduling_matches_full_epoch_visits():
    assert updates_per_epoch(760, 8) == 95
    assert total_scheduled_updates(1000, 760, 8) == 95000


def test_epoch_loop_visits_all_samples_and_eval_cadence(tmp_path: Path):
    set_n = 8
    train = TinyDenseDataset(set_n, "train_fit", n_conditions=2, seed=0)
    seen = TinyDenseDataset(4, "seen_train_eval", n_conditions=2, seed=1)
    val = TinyDenseDataset(4, "val_eval", n_conditions=2, seed=2)
    cfg = _cfg(n_epochs=5, eval_every_epochs=2)
    normalizer = Month2FitOnlyNormalizer()
    trainer = EpochDenseTrainer(
        TinyDenseNet(), cfg=cfg, normalizer=normalizer, n_train_samples=set_n
    )
    train_eval, _ = make_epoch_loader(train, batch_size=4, shuffle=False, seed=0)
    seen_loader, _ = make_epoch_loader(seen, batch_size=4, shuffle=False, seed=0)
    val_loader, _ = make_epoch_loader(val, batch_size=4, shuffle=False, seed=0)
    summary = trainer.fit(
        train,
        seen_loader=seen_loader,
        val_loader=val_loader,
        train_eval_loader=train_eval,
        out_dir=tmp_path / "run",
        run_identity={"run_id": "synth", "seed": 0},
    )
    assert summary["completed"] is True
    assert summary["epochs_completed"] == 5
    assert summary["updates"] == total_scheduled_updates(5, set_n, 4)
    assert [e["epoch"] for e in trainer.eval_history] == [2, 4, 5]
    assert (tmp_path / "run" / "checkpoints" / "last.pt").exists()
    assert (tmp_path / "run" / "checkpoints" / "best.pt").exists()
    assert (tmp_path / "run" / "evals" / "epoch_0002" / "val_eval_per_frame.csv.gz").exists()
    payload = torch.load(tmp_path / "run" / "checkpoints" / "last.pt", weights_only=False)
    assert payload["checkpoint_kind"] == CHECKPOINT_KIND_COMPLETE
    assert payload["resumable"] is True


def test_wall_interrupt_does_not_increment_epoch_nor_overwrite_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    train = TinyDenseDataset(8, "train_fit", seed=3)
    val = TinyDenseDataset(4, "val_eval", seed=4)
    normalizer = Month2FitOnlyNormalizer()
    # Complete one epoch first to create a valid last.pt
    cfg1 = _cfg(n_epochs=1, eval_every_epochs=1, wall_time_cap_s=60.0, seed=3)
    tr1 = EpochDenseTrainer(
        TinyDenseNet(), cfg=cfg1, normalizer=normalizer, n_train_samples=8
    )
    val_loader, _ = make_epoch_loader(val, batch_size=4, shuffle=False, seed=0)
    tr1.fit(
        train,
        seen_loader=None,
        val_loader=val_loader,
        out_dir=tmp_path / "wall",
        run_identity={"run_id": "wall"},
    )
    last_path = tmp_path / "wall" / "checkpoints" / "last.pt"
    assert last_path.exists()
    before = last_path.read_bytes()
    before_epoch = torch.load(last_path, weights_only=False)["epoch_state"]["epoch"]
    assert before_epoch == 1

    # Resume and abort after the first train step of the next epoch.
    cfg2 = _cfg(n_epochs=5, eval_every_epochs=1, wall_time_cap_s=60.0, seed=3)
    tr2 = EpochDenseTrainer(
        TinyDenseNet(), cfg=cfg2, normalizer=normalizer, n_train_samples=8
    )
    identity = {
        "run_id": "wall",
        "dense_config_sha256": "x",
        "u4_candidate_sha256": "x",
        "model_config_sha256": "x",
        "stats_sha256": "x",
        "cache_payload_sha256": "x",
        "cache_data_fingerprints": {},
        "architecture": {"base_channels": 8},
    }
    real_step = tr2._one_train_step
    seen_steps = {"n": 0}

    def step_then_wall(batch):
        seen_steps["n"] += 1
        if seen_steps["n"] == 1:
            return real_step(batch)
        return "wall_time_cap"

    monkeypatch.setattr(tr2, "_one_train_step", step_then_wall)
    summary = tr2.fit(
        train,
        seen_loader=None,
        val_loader=val_loader,
        out_dir=tmp_path / "wall",
        run_identity=identity,
        resume_path=last_path,
        allow_extend_max_epochs=True,
    )
    assert summary["completed"] is False
    assert summary["status"] == "stopped_wall_time"
    assert summary["epochs_completed"] == 1  # not incremented on partial epoch
    assert last_path.read_bytes() == before  # not overwritten
    partial = tmp_path / "wall" / "checkpoints" / "partial_interrupt.pt"
    assert partial.exists()
    part = torch.load(partial, weights_only=False)
    assert part["checkpoint_kind"] == CHECKPOINT_KIND_PARTIAL
    assert part["resumable"] is False


def test_continuous_vs_resume_at_epoch_boundary(tmp_path: Path):
    from combustion_recon.training.trainer import set_seed

    train = TinyDenseDataset(8, "train_fit", seed=0)
    val = TinyDenseDataset(4, "val_eval", seed=1)
    normalizer = Month2FitOnlyNormalizer()
    identity = {
        "run_id": "cmp",
        "dense_config_sha256": "d",
        "u4_candidate_sha256": "u",
        "model_config_sha256": "m",
        "stats_sha256": "s",
        "cache_payload_sha256": "c",
        "cache_data_fingerprints": {"k": "v"},
        "architecture": {"base_channels": 8},
    }
    val_loader, _ = make_epoch_loader(val, batch_size=4, shuffle=False, seed=0)
    train_eval, _ = make_epoch_loader(train, batch_size=4, shuffle=False, seed=0)

    set_seed(0)
    cfg_full = _cfg(n_epochs=4, eval_every_epochs=2, warmup_updates=3, seed=0)
    cont = EpochDenseTrainer(
        TinyDenseNet(), cfg=cfg_full, normalizer=normalizer, n_train_samples=8
    )
    cont.fit(
        train,
        seen_loader=None,
        val_loader=val_loader,
        train_eval_loader=train_eval,
        out_dir=tmp_path / "continuous",
        run_identity=identity,
    )
    with torch.no_grad():
        x = torch.from_numpy(train.x[:2])
        pred_cont = cont.model(x).clone()

    set_seed(0)
    cfg_a = _cfg(n_epochs=2, eval_every_epochs=2, warmup_updates=3, seed=0)
    a = EpochDenseTrainer(
        TinyDenseNet(), cfg=cfg_a, normalizer=normalizer, n_train_samples=8
    )
    a.fit(
        train,
        seen_loader=None,
        val_loader=val_loader,
        train_eval_loader=train_eval,
        out_dir=tmp_path / "resume",
        run_identity=identity,
    )
    last = tmp_path / "resume" / "checkpoints" / "last.pt"
    assert torch.load(last, weights_only=False)["epoch_state"]["epoch"] == 2

    cfg_b = _cfg(n_epochs=4, eval_every_epochs=2, warmup_updates=3, seed=0)
    b = EpochDenseTrainer(
        TinyDenseNet(), cfg=cfg_b, normalizer=normalizer, n_train_samples=8
    )
    b.fit(
        train,
        seen_loader=None,
        val_loader=val_loader,
        train_eval_loader=train_eval,
        out_dir=tmp_path / "resume",
        run_identity=identity,
        resume_path=last,
        allow_extend_max_epochs=True,
    )
    with torch.no_grad():
        pred_res = b.model(x)
    assert torch.allclose(pred_cont, pred_res, rtol=1e-5, atol=1e-6)
    assert b.state.epoch == 4
    assert b.state.update == cont.state.update


def test_resume_rejects_incompatible_science(tmp_path: Path):
    train = TinyDenseDataset(8, "train_fit", seed=0)
    val = TinyDenseDataset(4, "val_eval", seed=1)
    normalizer = Month2FitOnlyNormalizer()
    identity = {
        "run_id": "rej",
        "dense_config_sha256": "d",
        "u4_candidate_sha256": "u",
        "model_config_sha256": "m",
        "stats_sha256": "s",
        "cache_payload_sha256": "c",
        "cache_data_fingerprints": {},
        "architecture": {"base_channels": 8},
    }
    cfg = _cfg(n_epochs=1, eval_every_epochs=1)
    tr = EpochDenseTrainer(
        TinyDenseNet(), cfg=cfg, normalizer=normalizer, n_train_samples=8
    )
    val_loader, _ = make_epoch_loader(val, batch_size=4, shuffle=False, seed=0)
    tr.fit(
        train,
        seen_loader=None,
        val_loader=val_loader,
        out_dir=tmp_path / "rej",
        run_identity=identity,
    )
    last = tmp_path / "rej" / "checkpoints" / "last.pt"
    bad = _cfg(n_epochs=2, eval_every_epochs=1, lr=3e-4)
    tr2 = EpochDenseTrainer(
        TinyDenseNet(), cfg=bad, normalizer=normalizer, n_train_samples=8
    )
    with pytest.raises(ResumeIncompatibilityError):
        tr2.fit(
            train,
            seen_loader=None,
            val_loader=val_loader,
            out_dir=tmp_path / "rej",
            run_identity=identity,
            resume_path=last,
            allow_extend_max_epochs=True,
        )


def test_nonfinite_grad_sets_stopped_status(tmp_path: Path):
    train = TinyDenseDataset(4, "train_fit", seed=5)
    val = TinyDenseDataset(2, "val_eval", seed=6)
    cfg = _cfg(n_epochs=2, batch_size=2, seed=5, eval_every_epochs=1, wall_time_cap_s=60.0)
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
    assert (tmp_path / "nan" / "checkpoints" / "partial_interrupt.pt").exists()
    assert not (tmp_path / "nan" / "checkpoints" / "last.pt").exists()


def test_u4_candidate_contract_consumes_yaml():
    root = Path(__file__).resolve().parents[1]
    cand = yaml.safe_load(
        (root / "configs/experiments/month2_dense_unet_u4_candidate.yaml").read_text()
    )
    out = assert_u4_candidate_contract(cand)
    assert out["n_epochs"] == 1000
    assert out["warmup_updates"] == 950


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
