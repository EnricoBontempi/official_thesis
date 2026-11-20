"""S7 resume contract: scientific stream, GLU, best-seen tracker, early-stop checkpoint.

Synthetic data only (CPU). These tests cover what preemptable execution relies on:
a run killed in the middle of an epoch resumes from ``last.pt`` and ends in the
same state as an uninterrupted run, including the persistent presentation stream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.models.glu_reconstructor import (
    AdaptedDeterministicGLU,
    make_synthetic_query_grid,
)
from combustion_recon.models.unet import SparseUNet
from combustion_recon.training.sparse_scientific_loop import SparseScientificTrainer
from combustion_recon.training.trainer import set_seed

from test_sparse_scientific_trainer import H, W, TinySparseDS, _cfg, _identity


class Preempted(BaseException):
    """Stands in for SIGKILL: not an Exception, so fit() cannot clean up."""


def _model(family: str):
    if family == "sparse_unet":
        return SparseUNet(base_channels=8, groups=1, init_seed=0)
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    set_seed(0)
    return AdaptedDeterministicGLU(
        height=H, width=W, All_dim=32, top_k=4, domain_x=x, domain_y=y, decode_chunked_eval=True
    )


def _trainer(family: str, cfg, n_pool: int):
    kwargs = {}
    if family == "adapted_glu":
        x = np.linspace(-1, 1, W)
        y = np.linspace(-1, 1, H)
        kwargs["query_coords_phys"] = make_synthetic_query_grid(x, y)
    return SparseScientificTrainer(
        _model(family), cfg=cfg, normalizer=Month2FitOnlyNormalizer(), n_train_samples=n_pool, **kwargs
    )


def _science_cfg(family: str, **kw):
    base = dict(
        family=family,
        n_epochs=4,
        eval_every_epochs=2,
        batch_size=2,
        base_channels=8 if family == "sparse_unet" else 32,
        warmup_updates=3,
        presentations_per_scientific_epoch=12,  # pool 8 -> crosses a cycle mid-epoch
        pool_id=None,
        val_replicas=(0,),
        plateau_patience_evals=1,
        early_stop_min_epochs=1000,
        save_best_seen_diagnostic=True,
    )
    base.update(kw)
    return _cfg(**base)


def _ds(role: str, n: int, seed: int):
    return TinySparseDS(n, role, seed=seed, n_sites=8)


def _state_sig(tr):
    return {
        "epoch": tr.state.epoch,
        "update": tr.state.update,
        "samples_seen": tr.state.samples_seen,
        "mask_epoch": tr.state.mask_epoch,
        "lr": tr._current_lr(),
        "bad": tr.state.early_stop_bad_evals,
        "best": (tr.state.best_val_metric, tr.state.best_val_epoch),
        "cycle_pos": (tr.state.sampler_cycle, tr.state.sampler_pos),
        "history_epochs": [e["epoch"] for e in tr.eval_history],
        "curves": len(tr.curves),
    }


@pytest.mark.parametrize("family", ["sparse_unet", "adapted_glu"])
def test_killed_mid_epoch_resume_equals_uninterrupted(family, tmp_path: Path):
    ident = _identity("s7_resume")
    ident["family"] = family

    # Uninterrupted reference.
    set_seed(0)
    cont = _trainer(family, _science_cfg(family), 8)
    cont.stream_trace_path = tmp_path / "cont_trace.jsonl"
    cont.fit(_ds("train_fit", 8, 0), val_ds=_ds("val_eval", 4, 1), seen_ds=_ds("seen_train_eval", 4, 2),
             out_dir=tmp_path / "cont", run_identity=dict(ident))

    # Interrupted: hard stop inside epoch index 2 after one batch (epochs 0-1 complete).
    set_seed(0)
    a = _trainer(family, _science_cfg(family), 8)
    a.stream_trace_path = tmp_path / "res_trace.jsonl"

    def _kill(epoch: int, batches_done: int) -> None:
        if epoch == 2 and batches_done == 1:
            raise Preempted()

    a.fault_hook = _kill
    with pytest.raises(Preempted):
        a.fit(_ds("train_fit", 8, 0), val_ds=_ds("val_eval", 4, 1), seen_ds=_ds("seen_train_eval", 4, 2),
              out_dir=tmp_path / "res", run_identity=dict(ident))
    last = tmp_path / "res" / "checkpoints" / "last.pt"
    mid = torch.load(last, weights_only=False)
    assert mid["epoch_state"]["epoch"] == 2  # epoch 3 was never completed
    assert not (tmp_path / "res" / "summary.json").exists()

    # New process: fresh trainer, same config, resume from last.pt (no extension flag).
    set_seed(12345)  # different global RNG: resume must restore, not inherit
    b = _trainer(family, _science_cfg(family), 8)
    b.stream_trace_path = tmp_path / "res_trace.jsonl"
    train_b = _ds("train_fit", 8, 0)
    b.fit(train_b, val_ds=_ds("val_eval", 4, 1), seen_ds=_ds("seen_train_eval", 4, 2),
          out_dir=tmp_path / "res", run_identity=dict(ident), resume_path=last)

    assert _state_sig(b) == _state_sig(cont)
    for k, v in cont.model.state_dict().items():
        assert torch.equal(v, b.model.state_dict()[k]), k
    # The persistent stream is the same through the interruption: epochs 0-1 once,
    # epoch 2 served twice (killed + replayed), epoch 3 once.
    import json

    def _trace(path):
        return [json.loads(x) for x in path.read_text().splitlines()]

    ct, rt = _trace(tmp_path / "cont_trace.jsonl"), _trace(tmp_path / "res_trace.jsonl")
    assert [r["epoch_index_0based"] for r in rt] == [0, 1, 2, 2, 3]
    by_epoch = {r["epoch_index_0based"]: r["indices_sha256"] for r in ct}
    for r in rt:
        assert r["indices_sha256"] == by_epoch[r["epoch_index_0based"]]
    # Masks: set_epoch is re-applied after resume.
    # (the restore call replays the last completed mask epoch, 1, then 2 and 3 follow)
    assert train_b.epochs_seen[-2:] == [2, 3]
    s_cont = json.loads((tmp_path / "cont" / "summary.json").read_text())
    s_res = json.loads((tmp_path / "res" / "summary.json").read_text())
    assert s_res["epochs_completed"] == s_cont["epochs_completed"] == 4
    assert s_res["updates"] == s_cont["updates"]
    assert s_res["best_seen_diagnostic_epoch"] == s_cont["best_seen_diagnostic_epoch"]


def test_best_seen_tracker_survives_resume(tmp_path: Path):
    """A resumed run must not overwrite best_seen_diagnostic with a worse epoch."""
    ident = _identity("s7_bestseen")
    seen_scores = {2: 1.0, 4: 5.0, 6: 6.0}

    def make(n_epochs: int):
        tr = _trainer("sparse_unet", _science_cfg("sparse_unet", n_epochs=n_epochs), 8)
        real = tr.evaluate_dataset

        def _eval(ds, *, role, **kw):
            out = real(ds, role=role, **kw)
            if role == "seen_train_eval":
                out["mean_condition_rmse_K"] = seen_scores[tr.state.epoch]
            return out

        tr.evaluate_dataset = _eval  # type: ignore[method-assign]
        return tr

    set_seed(0)
    a = make(4)
    a.fit(_ds("train_fit", 8, 0), val_ds=_ds("val_eval", 4, 1), seen_ds=_ds("seen_train_eval", 4, 2),
          out_dir=tmp_path / "r", run_identity=dict(ident))
    first = torch.load(tmp_path / "r" / "checkpoints" / "best_seen_diagnostic.pt", weights_only=False)
    assert first["extra"]["selection_epoch"] == 2

    b = make(6)
    b.fit(_ds("train_fit", 8, 0), val_ds=_ds("val_eval", 4, 1), seen_ds=_ds("seen_train_eval", 4, 2),
          out_dir=tmp_path / "r", run_identity=dict(ident),
          resume_path=tmp_path / "r" / "checkpoints" / "last.pt", allow_extend_max_epochs=True)
    after = torch.load(tmp_path / "r" / "checkpoints" / "best_seen_diagnostic.pt", weights_only=False)
    assert after["extra"]["selection_epoch"] == 2
    import json

    assert json.loads((tmp_path / "r" / "summary.json").read_text())["best_seen_diagnostic_epoch"] == 2


def test_resume_from_early_stop_checkpoint_finalizes_without_training(tmp_path: Path):
    ident = _identity("s7_early")
    cfg = _science_cfg(
        "sparse_unet", n_epochs=8, eval_every_epochs=1, early_stop_min_epochs=1,
        early_stop_patience_evals=1, warmup_updates=1,
    )
    set_seed(0)
    a = _trainer("sparse_unet", cfg, 8)
    a.evaluate_val_replicas = lambda *x, **k: {  # type: ignore[method-assign]
        "per_replica": [{"replica_id": 0, "mean_condition_rmse_K": 3.0}],
        "mean_condition_rmse_K": 3.0, "rows": [],
    }
    summ = a.fit(_ds("train_fit", 8, 0), val_ds=_ds("val_eval", 4, 1), out_dir=tmp_path / "e",
                 run_identity=dict(ident))
    assert summ["status"] == "stopped_early_plateau"
    (tmp_path / "e" / "summary.json").unlink()  # preempted before the summary was written

    b = _trainer("sparse_unet", cfg, 8)
    b.evaluate_val_replicas = a.evaluate_val_replicas  # type: ignore[method-assign]
    s2 = b.fit(_ds("train_fit", 8, 0), val_ds=_ds("val_eval", 4, 1), out_dir=tmp_path / "e",
               run_identity=dict(ident), resume_path=tmp_path / "e" / "checkpoints" / "last.pt")
    assert s2["status"] == "stopped_early_plateau"
    assert s2["epochs_completed"] == summ["epochs_completed"]
    assert s2["updates"] == summ["updates"]
    assert (tmp_path / "e" / "summary.json").exists()


def test_partial_and_best_are_not_resume_sources(tmp_path: Path):
    from combustion_recon.training.s7_launch import inspect_last_checkpoint

    ident = _identity("s7_src")
    set_seed(0)
    a = _trainer("sparse_unet", _science_cfg("sparse_unet", n_epochs=2), 8)
    a.fit(_ds("train_fit", 8, 0), val_ds=_ds("val_eval", 4, 1), out_dir=tmp_path / "p",
          run_identity=dict(ident))
    ck = tmp_path / "p" / "checkpoints"
    info = inspect_last_checkpoint(ck / "last.pt", run_id="s7_src")
    assert info["epoch"] == 2
    with pytest.raises(ValueError):
        inspect_last_checkpoint(ck / "last.pt", run_id="other_run")


def test_rng_states_are_moved_to_cpu_before_restore(tmp_path: Path, monkeypatch):
    """With map_location=cuda the payload RNG states are CUDA tensors; torch needs CPU ones."""
    from combustion_recon.training import sparse_scientific_loop as loop

    ident = _identity("s7_rng")
    set_seed(0)
    a = _trainer("sparse_unet", _science_cfg("sparse_unet", n_epochs=2), 8)
    a.fit(_ds("train_fit", 8, 0), val_ds=_ds("val_eval", 4, 1), out_dir=tmp_path / "g", run_identity=dict(ident))

    moved = []

    class FakeDeviceTensor:
        """Stands in for a CUDA tensor: usable only after .cpu()."""

        def __init__(self, t):
            self.t = t

        def cpu(self):
            moved.append(1)
            return self.t

    real_load = torch.load

    def fake_load(path, *args, **kwargs):
        p = real_load(path, *args, **{**kwargs, "map_location": "cpu"})
        p["torch_rng_state"] = FakeDeviceTensor(p["torch_rng_state"])
        return p

    seen = {}
    real_set = torch.set_rng_state

    def strict_set(state):
        assert isinstance(state, torch.Tensor) and state.device.type == "cpu"
        seen["ok"] = True
        real_set(state)

    monkeypatch.setattr(loop.torch, "load", fake_load)
    monkeypatch.setattr(loop.torch, "set_rng_state", strict_set)
    tr = _trainer("sparse_unet", _science_cfg("sparse_unet", n_epochs=2), 8)
    loop.load_sparse_scientific_checkpoint(tmp_path / "g" / "checkpoints" / "last.pt", model=tr.model)
    assert seen.get("ok") and moved
