"""Dense U-Net / cache / trainer isolation and checkpoint tests (U1–U2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.evaluation.month2_evaluator import (
    MixedConfigurationError,
    aggregate_frame_then_condition_then_split,
    denormalize_temperature,
    evaluate_frame_prediction,
)
from combustion_recon.models.unet import DenseUNet, build_dense_unet_from_config, load_unet_config
from combustion_recon.training.dense_cache import (
    CHANNEL_ORDER,
    DenseUNetCacheDataset,
    build_dense_cache,
    build_dense_input_tensor,
    scale_xy_to_minus1_1,
    subset_indices_for_overfit,
)
from combustion_recon.training.snapshot_swap import build_within_condition_swap_pairs
from combustion_recon.training.trainer import (
    DenseUNetTrainer,
    TrainConfig,
    load_checkpoint,
    make_loader,
    save_checkpoint,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "processed" / "month2_dense_unet_cache"
HAS_RAW = (ROOT / "data/raw/combustion/hf_dataset/numerical").exists()


def test_unet_config_loads_and_refuses_unknown_keys(tmp_path: Path):
    cfg = load_unet_config()
    assert cfg["input"]["channels"] == 6
    bad = tmp_path / "bad.yaml"
    bad.write_text("architecture: dense_unet_concat_p\nfilm: true\ninput: {}\n")
    with pytest.raises(ValueError, match="unsupported keys|missing required"):
        load_unet_config(bad)


def test_unet_forward_shape_and_param_count():
    set_seed(0)
    model = build_dense_unet_from_config(init_seed=0, base_channels=32)
    x = torch.randn(2, 6, 128, 128)
    y = model(x)
    assert y.shape == (2, 1, 128, 128)
    n = model.count_parameters()
    # Opus estimate ~4.9M for b=32; allow broad band, record exact.
    assert 3_000_000 < n < 8_000_000
    # Head not all-zero
    assert float(model.head.weight.detach().abs().max()) > 0


def test_unet_gradients_flow_to_early_blocks():
    set_seed(0)
    model = DenseUNet(base_channels=16, init_seed=0)
    x = torch.randn(1, 6, 128, 128, requires_grad=False)
    y = torch.zeros(1, 1, 128, 128)
    pred = model(x)
    loss = torch.mean((pred - y) ** 2)
    loss.backward()
    assert model.enc1.block[0].conv.weight.grad is not None
    assert torch.isfinite(model.enc1.block[0].conv.weight.grad).all()
    assert float(model.enc1.block[0].conv.weight.grad.abs().sum()) > 0
    assert model.head.weight.grad is not None


def test_mutation_isolation_forward_ignores_target_vk_metadata():
    set_seed(1)
    model = build_dense_unet_from_config(init_seed=1, base_channels=16)
    model.eval()
    x = torch.randn(1, 6, 128, 128)
    with torch.no_grad():
        y0 = model(x).clone()
        # Mutate tensors that must NOT be inputs
        _t = torch.randn(1, 1, 128, 128)
        _vk = torch.randn(1, 1, 128, 128)
        _meta = {"sim_id": "fake", "frame_index": 999}
        y1 = model(x)
    assert torch.allclose(y0, y1)


def test_positive_sensitivity_to_vi_and_p():
    set_seed(2)
    model = build_dense_unet_from_config(init_seed=2, base_channels=16)
    model.eval()
    x = torch.zeros(1, 6, 128, 128)
    x[:, 0] = 0.1
    x[:, 4] = 0.2
    x[:, 5] = -0.3
    with torch.no_grad():
        y0 = model(x).clone()
        x_vi = x.clone()
        x_vi[:, 0] = x_vi[:, 0] + 0.5
        y_vi = model(x_vi)
        x_p = x.clone()
        x_p[:, 4] = x_p[:, 4] + 1.0
        y_p = model(x_p)
        x_poff = x.clone()
        x_poff[:, 4] = 0.0
        x_poff[:, 5] = 0.0
        y_poff = model(x_poff)
    assert not torch.allclose(y0, y_vi)
    assert not torch.allclose(y0, y_p)
    assert not torch.allclose(y0, y_poff)


def test_checkpoint_reload_predictions_match(tmp_path: Path):
    set_seed(3)
    model = build_dense_unet_from_config(init_seed=3, base_channels=16)
    cfg = TrainConfig(max_updates=2, batch_size=1, device="cpu", seed=3)
    trainer = DenseUNetTrainer(model, cfg=cfg)
    # one synthetic step
    x = torch.randn(1, 6, 128, 128)
    y = torch.randn(1, 1, 128, 128)
    trainer.train_mode()
    trainer.optimizer.zero_grad(set_to_none=True)
    loss = torch.mean((model(x) - y) ** 2)
    loss.backward()
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.state.update = 1
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        state=trainer.state,
        cfg=cfg,
        extra={"stats_hash": "dummy"},
    )
    with torch.no_grad():
        p0 = model(x).clone()
    model2 = build_dense_unet_from_config(init_seed=99, base_channels=16)
    load_checkpoint(path, model=model2, restore_rng=False)
    model2.eval()
    with torch.no_grad():
        p1 = model2(x)
    assert torch.allclose(p0, p1, atol=0, rtol=0)


def test_evaluator_kelvin_and_failfast_aggregation():
    norm = Month2FitOnlyNormalizer()
    rng = np.random.default_rng(0)
    true = 1800 + 50 * rng.standard_normal((128, 128))
    pred_n = (true - norm.temperature_mean) / norm.temperature_std
    pred_K = denormalize_temperature(pred_n, norm)
    assert np.allclose(pred_K, true, rtol=0, atol=1e-5)
    row_a = evaluate_frame_prediction(
        pred_K,
        true,
        train_sigma_T=norm.temperature_std,
        sim_id="0NH3_0.85.h5",
        split="train",
        role="train_fit",
        frame_index=0,
        method="DenseUNet",
        variant="p_on",
        model_seed=0,
        site_fraction=1.0,
        K=0,
        task_mode="temperature_free",
    )
    row_b = dict(row_a)
    row_b["role"] = "val_eval"
    row_b["sim_id"] = "80NH3_0.9.h5"
    row_b["split"] = "val"
    with pytest.raises(MixedConfigurationError):
        aggregate_frame_then_condition_then_split([row_a, row_b])
    # Homogeneous OK
    row_c = dict(row_a)
    row_c["frame_index"] = 40
    row_c["sim_id"] = "0NH3_0.9.h5"
    agg = aggregate_frame_then_condition_then_split([row_a, row_c])
    assert "per_condition" in agg


def test_snapshot_swap_pairing_no_fixed_points():
    frames = []
    for sim in ("A.h5", "B.h5"):
        for fi in range(0, 200, 40):
            frames.append(
                {
                    "sim_id": sim,
                    "frame_index": fi,
                    "cache_index": fi,
                    "role": "val_eval",
                }
            )
    pairs = build_within_condition_swap_pairs(frames, role="val_eval")
    assert len(pairs) == len(frames)
    for p in pairs:
        assert p["src_frame_index"] != p["dest_frame_index"]


@pytest.mark.skipif(not HAS_RAW, reason="local raw shards required")
def test_build_or_reuse_dense_cache_contract():
    manifest = build_dense_cache(cache_root=CACHE_ROOT, overwrite=False)
    assert manifest["counts"]["train_fit"] == 760
    assert manifest["counts"]["seen_train_eval"] == 220
    assert manifest["counts"]["val_eval"] == 123
    assert manifest["test_payloads"] == "none"
    assert abs(manifest["moments_train_fit"]["train_fit_T_norm_mean"]) < 0.05
    ds = DenseUNetCacheDataset("train_fit", cache_root=CACHE_ROOT, p_conditioning=True)
    item = ds[0]
    x = item["model_inputs"]["input_tensor"]
    assert x.shape == (6, 128, 128)
    assert list(item["model_inputs"]["channel_order"]) == list(CHANNEL_ORDER)
    # Target separate; T not in input tensor construction path
    assert "temperature" not in item["model_inputs"]
    y = item["targets"]["temperature_normalized"]
    assert y.shape == (1, 128, 128)
    # p_off zeros channels 4/5
    ds_off = DenseUNetCacheDataset("train_fit", cache_root=CACHE_ROOT, p_conditioning=False)
    x_off = ds_off[0]["model_inputs"]["input_tensor"]
    assert torch.count_nonzero(x_off[4]) == 0
    assert torch.count_nonzero(x_off[5]) == 0
    assert not torch.allclose(x[4], x_off[4])


@pytest.mark.skipif(not HAS_RAW, reason="local raw shards required")
def test_cache_matches_dataset_sample_and_overfit_subset():
    from combustion_recon.data.month2_dataset import Month2ReconDataset
    from combustion_recon.data.shard_access import ShardStore

    if not (CACHE_ROOT / "manifest.json").exists():
        build_dense_cache(cache_root=CACHE_ROOT, overwrite=False)
    ds_cache = DenseUNetCacheDataset("train_fit", cache_root=CACHE_ROOT)
    store = ShardStore(max_cached_frames=8)
    ds = Month2ReconDataset(
        "train_fit",
        task_mode="temperature_free",
        K=0,
        site_fraction=1.0,
        p_conditioning=True,
        normalizer=Month2FitOnlyNormalizer(),
        store=store,
        max_cached_frames=8,
    )
    for idx in (0, 10, 100):
        s = ds[idx]
        c = ds_cache[idx]
        dens = s["model_inputs"]["dense_values"]
        assert np.allclose(c["model_inputs"]["input_tensor"][0].numpy(), dens[0], atol=1e-5)
        assert np.allclose(c["model_inputs"]["input_tensor"][1].numpy(), dens[1], atol=1e-5)
        assert np.allclose(
            c["targets"]["temperature_normalized"][0].numpy(),
            s["targets"]["temperature_normalized"],
            atol=1e-5,
        )
        assert bool(s["model_inputs"]["mask_velocity"].all())
    idxs = subset_indices_for_overfit(ds_cache.frames, n_frames=8, seed=0)
    assert len(idxs) == 8
    sims = {ds_cache.frames[i]["sim_id"] for i in idxs}
    assert 2 <= len(sims) <= 4


def test_scale_xy_range():
    x = np.linspace(-0.03, 0.03, 128)
    y = np.linspace(0.06, 0.16, 128)
    xs, ys = scale_xy_to_minus1_1(x, y)
    assert xs.shape == (128, 128)
    assert abs(float(xs.min()) + 1.0) < 1e-5
    assert abs(float(xs.max()) - 1.0) < 1e-5
    assert abs(float(ys.min()) + 1.0) < 1e-5
    assert abs(float(ys.max()) - 1.0) < 1e-5
    packed = build_dense_input_tensor(
        vi_norm=np.zeros((128, 128), np.float32),
        vj_norm=np.zeros((128, 128), np.float32),
        x_scaled=xs,
        y_scaled=ys,
        nh3_norm=0.5,
        phi_norm=-0.25,
        p_conditioning=True,
    )
    assert packed.shape == (6, 128, 128)
    assert np.allclose(packed[4], 0.5)
