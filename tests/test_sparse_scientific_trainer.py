"""Synthetic tests for sparse scientific runner (resume/schedule; not S5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.models.glu_reconstructor import AdaptedDeterministicGLU, make_synthetic_query_grid
from combustion_recon.models.unet import SparseUNet
from combustion_recon.training.epoch_loop import ResumeIncompatibilityError
from combustion_recon.training.sparse_scientific_loop import (
    CHECKPOINT_KIND_COMPLETE,
    CHECKPOINT_KIND_PARTIAL,
    SparseScientificConfig,
    SparseScientificTrainer,
    assert_sparse_scientific_candidate_contract,
    load_sparse_scientific_checkpoint,
)
from combustion_recon.training.sparse_view import CanonicalSparseObservation
from combustion_recon.training.trainer import set_seed

ROOT = Path(__file__).resolve().parents[1]
H = W = 32  # U-Net has 4× downsample; 8×8 collapses


class TinySparseDS:
    """Duck-typed sparse dataset for scientific runner continuity tests."""

    def __init__(self, n: int, role: str, *, seed: int = 0, n_sites: int = 16):
        self.n = n
        self.role = role
        self.split = "train" if role == "train_fit" else ("val" if role == "val_eval" else "seen")
        self.mask_mode = "training" if role == "train_fit" else "evaluation"
        self.epoch = 0
        self.replica_id = 0
        self.site_fraction = 0.05
        rng = np.random.default_rng(seed)
        self.x = np.linspace(-1, 1, W)
        self.y = np.linspace(-1, 1, H)
        self.p_norm = rng.normal(size=(n, 2)).astype(np.float64)
        self._x = rng.normal(size=(n, 7, H, W)).astype(np.float32)
        self._y = rng.normal(size=(n, 1, H, W)).astype(np.float32)
        self._yK = 1500 + 50 * self._y
        self.n_sites = n_sites
        self.conds = [f"c{i%2}" for i in range(n)]
        self.epochs_seen: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.epochs_seen.append(self.epoch)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        sites = np.arange(self.n_sites, dtype=np.int64)
        # Depend on epoch so resume mask epoch is observable
        vals = np.stack(
            [
                self._x[i, 0].ravel()[: self.n_sites] + 0.01 * self.epoch,
                self._x[i, 1].ravel()[: self.n_sites],
            ],
            axis=1,
        ).astype(np.float32)
        coords = np.stack(
            [np.tile(self.x, H)[: self.n_sites], np.repeat(self.y, W)[: self.n_sites]],
            axis=1,
        ).astype(np.float64)
        mask = np.zeros((H, W), dtype=bool)
        mask.ravel()[sites] = True
        obs = CanonicalSparseObservation(
            sites=sites,
            velocity_coords=coords,
            velocity_values_norm=vals.astype(np.float64),
            velocity_values_phys=vals.astype(np.float64),
            mask=mask,
            site_fraction=0.05,
            layout_id="random_sparse_primary",
            velocity_mask_seed=0,
            mask_hash="synth",
            n_sites=self.n_sites,
            H=H,
            W=W,
        )
        return {
            "model_inputs": {
                "input_tensor": torch.from_numpy(self._x[i].copy()),
                "p_conditioning": True,
            },
            "targets": {
                "temperature_normalized": torch.from_numpy(self._y[i].copy()),
                "temperature": torch.from_numpy(self._yK[i].copy()),
            },
            "observation": obs,
            "glu_tokens": obs.to_glu_tokens(),
            "baseline_physical": obs.to_baseline_physical(),
            "metadata": {
                "sim_id": self.conds[i],
                "frame_index": int(i * 10),
                "split": self.split,
                "role": self.role,
                "cache_index": i,
            },
        }


def _cfg(**kwargs) -> SparseScientificConfig:
    base = dict(
        n_epochs=5,
        batch_size=4,
        lr=1e-3,
        seed=0,
        shuffle_seed=0,
        device="cpu",
        eval_every_epochs=2,
        wall_time_cap_s=120.0,
        base_channels=8,
        warmup_updates=2,
        early_stop_min_epochs=1000,
        early_stop_patience_evals=15,
        schedule_kind="warmup_then_reduce_on_plateau",
        allow_tf32=False,
        eval_train_fit=False,
        family="sparse_unet",
        site_fraction=0.05,
        train_mask_seed=20261118,
        eval_mask_seed=20261116,
        layout_id="random_sparse_primary",
        val_replicas=(0, 1),
        plateau_patience_evals=1,
        plateau_threshold_rel=0.001,
    )
    base.update(kwargs)
    return SparseScientificConfig(**base)


def test_scientific_candidate_contract_consumes_yaml():
    cand = yaml.safe_load(
        (ROOT / "configs/experiments/month2_sparse_scientific_candidate.yaml").read_text()
    )
    out = assert_sparse_scientific_candidate_contract(cand)
    assert out["warmup_updates"] == 950
    assert out["scientific_training_authorized"] is False
    assert out["n_epochs"] == 1000


def test_sparse_unet_visits_and_eval_cadence(tmp_path: Path):
    train = TinySparseDS(8, "train_fit", seed=0)
    val = TinySparseDS(4, "val_eval", seed=1)
    cfg = _cfg(n_epochs=5, eval_every_epochs=2, base_channels=8)
    model = SparseUNet(base_channels=8, groups=1, init_seed=0)
    normalizer = Month2FitOnlyNormalizer()
    tr = SparseScientificTrainer(model, cfg=cfg, normalizer=normalizer, n_train_samples=8)
    summary = tr.fit(
        train,
        val_ds=val,
        out_dir=tmp_path / "run",
        run_identity={"run_id": "synth", "family": "sparse_unet"},
    )
    assert summary["completed"] is True
    assert summary["epochs_completed"] == 5
    assert [e["epoch"] for e in tr.eval_history] == [2, 4, 5]
    assert (tmp_path / "run" / "checkpoints" / "last.pt").exists()
    payload = torch.load(tmp_path / "run" / "checkpoints" / "last.pt", weights_only=False)
    assert payload["checkpoint_kind"] == CHECKPOINT_KIND_COMPLETE
    assert payload["resumable"] is True
    assert train.epochs_seen == [0, 1, 2, 3, 4]


def test_continuous_vs_resume_compares_lr_counters_rng(tmp_path: Path):
    train = TinySparseDS(8, "train_fit", seed=0)
    val = TinySparseDS(4, "val_eval", seed=1)
    normalizer = Month2FitOnlyNormalizer()
    identity = {
        "run_id": "cmp",
        "family": "sparse_unet",
        "protocol_sha256": "p",
        "scientific_candidate_sha256": "s",
        "model_config_sha256": "m",
        "stats_sha256": "st",
        "cache_payload_sha256": "c",
        "mask_payload_sha256": "mk",
    }

    set_seed(0)
    cfg_full = _cfg(n_epochs=4, eval_every_epochs=2, warmup_updates=3, seed=0)
    cont = SparseScientificTrainer(
        SparseUNet(base_channels=8, groups=1, init_seed=0),
        cfg=cfg_full,
        normalizer=normalizer,
        n_train_samples=8,
    )
    cont.fit(train, val_ds=val, out_dir=tmp_path / "continuous", run_identity=identity)
    with torch.no_grad():
        x = torch.from_numpy(train._x[:2])
        pred_cont = cont.model(x).clone()
    cont_lr = cont._current_lr()
    cont_update = cont.state.update
    cont_bad = cont.state.early_stop_bad_evals

    set_seed(0)
    train2 = TinySparseDS(8, "train_fit", seed=0)
    cfg_a = _cfg(n_epochs=2, eval_every_epochs=2, warmup_updates=3, seed=0)
    a = SparseScientificTrainer(
        SparseUNet(base_channels=8, groups=1, init_seed=0),
        cfg=cfg_a,
        normalizer=normalizer,
        n_train_samples=8,
    )
    a.fit(train2, val_ds=val, out_dir=tmp_path / "resume", run_identity=identity)
    last = tmp_path / "resume" / "checkpoints" / "last.pt"
    mid = torch.load(last, weights_only=False)
    assert mid["epoch_state"]["epoch"] == 2

    cfg_b = _cfg(n_epochs=4, eval_every_epochs=2, warmup_updates=3, seed=0)
    b = SparseScientificTrainer(
        SparseUNet(base_channels=8, groups=1, init_seed=0),
        cfg=cfg_b,
        normalizer=normalizer,
        n_train_samples=8,
    )
    b.fit(
        train2,
        val_ds=val,
        out_dir=tmp_path / "resume",
        run_identity=identity,
        resume_path=last,
        allow_extend_max_epochs=True,
    )
    with torch.no_grad():
        pred_res = b.model(x)
    assert torch.allclose(pred_cont, pred_res, rtol=1e-5, atol=1e-6)
    assert b.state.epoch == 4
    assert b.state.update == cont_update
    assert abs(b._current_lr() - cont_lr) < 1e-15
    assert b.state.early_stop_bad_evals == cont_bad
    assert len(b.eval_history) == len(cont.eval_history)


def test_resume_rejects_incompatible_and_partial(tmp_path: Path):
    train = TinySparseDS(8, "train_fit", seed=0)
    val = TinySparseDS(4, "val_eval", seed=1)
    normalizer = Month2FitOnlyNormalizer()
    identity = {"run_id": "rej", "family": "sparse_unet"}
    cfg = _cfg(n_epochs=1, eval_every_epochs=1)
    tr = SparseScientificTrainer(
        SparseUNet(base_channels=8, groups=1, init_seed=0),
        cfg=cfg,
        normalizer=normalizer,
        n_train_samples=8,
    )
    tr.fit(train, val_ds=val, out_dir=tmp_path / "rej", run_identity=identity)
    last = tmp_path / "rej" / "checkpoints" / "last.pt"
    bad = _cfg(n_epochs=2, eval_every_epochs=1, lr=3e-4)
    tr2 = SparseScientificTrainer(
        SparseUNet(base_channels=8, groups=1, init_seed=0),
        cfg=bad,
        normalizer=normalizer,
        n_train_samples=8,
    )
    with pytest.raises(ResumeIncompatibilityError):
        tr2.fit(
            train,
            val_ds=val,
            out_dir=tmp_path / "rej",
            run_identity=identity,
            resume_path=last,
            allow_extend_max_epochs=True,
        )


def test_warmup_then_plateau_and_early_stop_counters(tmp_path: Path):
    """Exercise warmup completion, LR reduce, and early-stop counters (synthetic)."""
    train = TinySparseDS(8, "train_fit", seed=0)
    val = TinySparseDS(4, "val_eval", seed=2)
    normalizer = Month2FitOnlyNormalizer()
    # Small warmup; plateau patience 0 effectively via patience_evals=0? patience min 1
    cfg = _cfg(
        n_epochs=6,
        eval_every_epochs=1,
        warmup_updates=2,
        plateau_patience_evals=1,
        early_stop_min_epochs=3,
        early_stop_patience_evals=2,
        early_stop_rel_improvement=0.5,  # hard to improve → accumulate bad evals
        batch_size=4,
    )
    tr = SparseScientificTrainer(
        SparseUNet(base_channels=8, groups=1, init_seed=0),
        cfg=cfg,
        normalizer=normalizer,
        n_train_samples=8,
    )
    # Force flat VAL by stubbing evaluate_val_replicas
    tr.evaluate_val_replicas = lambda *a, **k: {  # type: ignore[method-assign]
        "per_replica": [{"replica_id": 0, "mean_condition_rmse_K": 10.0}],
        "mean_condition_rmse_K": 10.0,
        "rows": [],
    }
    summary = tr.fit(
        train,
        val_ds=val,
        out_dir=tmp_path / "sched",
        run_identity={"run_id": "sched", "family": "sparse_unet"},
    )
    assert tr.state.update >= 2
    # After warmup, LR should reach base then possibly reduce
    assert tr.state.epoch >= 3
    assert summary["status"] in {"stopped_early_plateau", "completed"}
    if summary["status"] == "stopped_early_plateau":
        assert tr.state.early_stop_bad_evals >= 2


def test_glu_geometry_persisted_in_scientific_checkpoint(tmp_path: Path):
    train = TinySparseDS(4, "train_fit", seed=0, n_sites=8)
    val = TinySparseDS(2, "val_eval", seed=1, n_sites=8)
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    model = AdaptedDeterministicGLU(
        height=H, width=W, All_dim=32, top_k=4, domain_x=x, domain_y=y, decode_chunked_eval=True
    )
    cfg = _cfg(
        n_epochs=1,
        eval_every_epochs=1,
        family="adapted_glu",
        batch_size=2,
        base_channels=32,
        warmup_updates=1,
    )
    normalizer = Month2FitOnlyNormalizer()
    q = make_synthetic_query_grid(x, y)
    tr = SparseScientificTrainer(
        model, cfg=cfg, normalizer=normalizer, n_train_samples=4, query_coords_phys=q
    )
    # Stub VAL to keep test light
    tr.evaluate_val_replicas = lambda *a, **k: {  # type: ignore[method-assign]
        "per_replica": [{"replica_id": 0, "mean_condition_rmse_K": 1.0}],
        "mean_condition_rmse_K": 1.0,
        "rows": [],
    }
    tr.fit(
        train,
        val_ds=val,
        out_dir=tmp_path / "glu",
        run_identity={"run_id": "glu", "family": "adapted_glu"},
    )
    last = tmp_path / "glu" / "checkpoints" / "last.pt"
    payload = torch.load(last, weights_only=False)
    assert "glu_geometry_meta" in payload
    assert "domain_center" in payload["model_state_dict"]
    assert "domain_L" in payload["model_state_dict"]
    model2 = AdaptedDeterministicGLU(
        height=H, width=W, All_dim=32, top_k=4, domain_x=x, domain_y=y
    )
    load_sparse_scientific_checkpoint(last, model=model2, require_resumable=True)
    assert torch.allclose(model2.domain_center, model.domain_center)
    assert torch.allclose(model2.domain_L, model.domain_L)
