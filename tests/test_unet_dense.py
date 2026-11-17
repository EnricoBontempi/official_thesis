"""Dense U-Net / cache / trainer isolation and checkpoint tests (U1–U2)."""

from __future__ import annotations

import json
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
    DenseCacheError,
    DenseContractError,
    DenseUNetCacheDataset,
    assert_dense_scientific_contract,
    build_dense_cache,
    build_dense_input_tensor,
    dense_fields_to_input_tensor,
    refresh_dense_cache_manifest_metadata,
    scale_xy_to_minus1_1,
    subset_indices_for_overfit,
    validate_dense_cache_manifest,
)
from combustion_recon.data.month2_protocol import load_yaml
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


def _synthetic_dense_fields(seed: int = 0) -> dict:
    """Synthetic cache-like fields (no production raw/cache mutation)."""
    rng = np.random.default_rng(seed)
    H = W = 128
    x = np.linspace(-0.03, 0.03, W)
    y = np.linspace(0.06, 0.16, H)
    xs, ys = scale_xy_to_minus1_1(x, y)
    return {
        "vi_norm": rng.normal(0, 1, size=(H, W)).astype(np.float32),
        "vj_norm": rng.normal(0, 1, size=(H, W)).astype(np.float32),
        "x_scaled": xs,
        "y_scaled": ys,
        "nh3_norm": float(rng.normal()),
        "phi_norm": float(rng.normal()),
        "temperature": (1800 + 50 * rng.normal(size=(H, W))).astype(np.float64),
        "temperature_normalized": rng.normal(0, 1, size=(H, W)).astype(np.float32),
        "velocity_k": rng.normal(0, 1, size=(H, W)).astype(np.float32),
        "metadata": {"sim_id": "synth_A.h5", "frame_index": 0, "role": "train_fit"},
    }


def test_mutation_isolation_sample_path_to_forward():
    """Real path: mutate forbidden sample fields → rebuild input → forward unchanged.

    Distinct from upstream ``InputIsolationProbe`` / observation-pack tests, which
    cover Month2ReconDataset sparse packs rather than the dense U-Net adapter.
    """
    set_seed(1)
    model = build_dense_unet_from_config(init_seed=1, base_channels=16)
    model.eval()
    base = _synthetic_dense_fields(1)

    x0 = dense_fields_to_input_tensor(
        vi_norm=base["vi_norm"],
        vj_norm=base["vj_norm"],
        x_scaled=base["x_scaled"],
        y_scaled=base["y_scaled"],
        nh3_norm=base["nh3_norm"],
        phi_norm=base["phi_norm"],
        p_conditioning=True,
        temperature=base["temperature"],
        temperature_normalized=base["temperature_normalized"],
        velocity_k=base["velocity_k"],
        metadata=base["metadata"],
    )
    with torch.no_grad():
        y0 = model(torch.from_numpy(x0)[None])

    mut = dict(base)
    mut["temperature"] = base["temperature"] + 500.0
    mut["temperature_normalized"] = base["temperature_normalized"] + 3.0
    mut["velocity_k"] = base["velocity_k"] + 10.0
    mut["metadata"] = {
        "sim_id": "FAKE.h5",
        "frame_index": 9999,
        "role": "val_eval",
        "split": "val",
        "time": 1e9,
    }
    x1 = dense_fields_to_input_tensor(
        vi_norm=mut["vi_norm"],
        vj_norm=mut["vj_norm"],
        x_scaled=mut["x_scaled"],
        y_scaled=mut["y_scaled"],
        nh3_norm=mut["nh3_norm"],
        phi_norm=mut["phi_norm"],
        p_conditioning=True,
        temperature=mut["temperature"],
        temperature_normalized=mut["temperature_normalized"],
        velocity_k=mut["velocity_k"],
        metadata=mut["metadata"],
    )
    assert np.array_equal(x0, x1)
    with torch.no_grad():
        y1 = model(torch.from_numpy(x1)[None])
    assert torch.allclose(y0, y1)

    # p_off: changing source p must not change the zeroed input / prediction.
    x_off0 = dense_fields_to_input_tensor(
        vi_norm=base["vi_norm"],
        vj_norm=base["vj_norm"],
        x_scaled=base["x_scaled"],
        y_scaled=base["y_scaled"],
        nh3_norm=base["nh3_norm"],
        phi_norm=base["phi_norm"],
        p_conditioning=False,
    )
    x_off1 = dense_fields_to_input_tensor(
        vi_norm=base["vi_norm"],
        vj_norm=base["vj_norm"],
        x_scaled=base["x_scaled"],
        y_scaled=base["y_scaled"],
        nh3_norm=base["nh3_norm"] + 5.0,
        phi_norm=base["phi_norm"] - 3.0,
        p_conditioning=False,
    )
    assert np.array_equal(x_off0, x_off1)
    assert np.count_nonzero(x_off0[4]) == 0 and np.count_nonzero(x_off0[5]) == 0
    with torch.no_grad():
        y_off0 = model(torch.from_numpy(x_off0)[None])
        y_off1 = model(torch.from_numpy(x_off1)[None])
    assert torch.allclose(y_off0, y_off1)

    # Positive controls on allowed observations.
    x_vi = dense_fields_to_input_tensor(
        vi_norm=base["vi_norm"] + 0.75,
        vj_norm=base["vj_norm"],
        x_scaled=base["x_scaled"],
        y_scaled=base["y_scaled"],
        nh3_norm=base["nh3_norm"],
        phi_norm=base["phi_norm"],
        p_conditioning=True,
    )
    x_p = dense_fields_to_input_tensor(
        vi_norm=base["vi_norm"],
        vj_norm=base["vj_norm"],
        x_scaled=base["x_scaled"],
        y_scaled=base["y_scaled"],
        nh3_norm=base["nh3_norm"] + 1.5,
        phi_norm=base["phi_norm"],
        p_conditioning=True,
    )
    assert not np.array_equal(x0, x_vi)
    assert not np.array_equal(x0, x_p)
    with torch.no_grad():
        y_vi = model(torch.from_numpy(x_vi)[None])
        y_p = model(torch.from_numpy(x_p)[None])
    assert not torch.allclose(y0, y_vi)
    assert not torch.allclose(y0, y_p)


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


def test_trainer_refuses_nonfinite_gradients_before_step():
    set_seed(5)
    trainer = DenseUNetTrainer(
        build_dense_unet_from_config(init_seed=5, base_channels=16),
        cfg=TrainConfig(max_updates=1, batch_size=1, device="cpu", seed=5),
    )

    class _Ok(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, i):
            return {
                "model_inputs": {
                    "input_tensor": torch.randn(6, 128, 128),
                    "p_conditioning": True,
                },
                "targets": {
                    "temperature_normalized": torch.randn(1, 128, 128),
                    "temperature": torch.randn(1, 128, 128),
                },
                "metadata": {
                    "sim_id": "x",
                    "frame_index": 0,
                    "split": "train",
                    "role": "train_fit",
                },
            }

    def _nan_grad(grad):
        return torch.full_like(grad, float("nan"))

    handle = trainer.model.head.weight.register_hook(_nan_grad)
    try:
        out = trainer.train_updates(
            make_loader(_Ok(), batch_size=1, shuffle=False, seed=5), max_updates=1
        )
    finally:
        handle.remove()
    assert out["stopped_reason"] == "non_finite_gradients"
    assert out["updates"] == 0
    assert torch.isfinite(torch.tensor(out["last_loss"]))


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


def test_dense_yaml_contract_consume_and_reject(tmp_path: Path):
    cfg = load_yaml(ROOT / "configs/experiments/month2_dense_unet.yaml")
    contract = assert_dense_scientific_contract(cfg)
    assert contract["site_fraction"] == 1.0
    assert contract["K"] == 0
    bad = dict(cfg)
    bad["inputs"] = dict(cfg["inputs"])
    bad["inputs"]["site_fraction"] = 0.05
    with pytest.raises(DenseContractError, match="site_fraction"):
        assert_dense_scientific_contract(bad)
    bad2 = dict(cfg)
    bad2["experiment_id"] = "month2_baselines_sparse"
    with pytest.raises(DenseContractError, match="experiment_id"):
        assert_dense_scientific_contract(bad2)


def test_validate_manifest_rejects_altered_frames_and_stats(tmp_path: Path):
    # Minimal fake cache tree
    root = tmp_path / "cache"
    for role, n in (("train_fit", 2),):
        rd = root / role
        rd.mkdir(parents=True)
        np.save(rd / "vi_norm.npy", np.zeros((n, 128, 128), np.float32))
        np.save(rd / "vj_norm.npy", np.zeros((n, 128, 128), np.float32))
        np.save(rd / "target_T_norm.npy", np.zeros((n, 128, 128), np.float32))
        np.save(rd / "target_T_K.npy", np.zeros((n, 128, 128), np.float32))
        np.save(rd / "p_norm.npy", np.zeros((n, 2), np.float32))
        frames = [
            {
                "cache_index": 0,
                "sim_id": "WRONG.h5",
                "frame_index": 0,
                "role": role,
                "split": "train",
                "nh3_fraction": 0.0,
                "phi": 0.85,
                "equivalence_ratio": 0.85,
            },
            {
                "cache_index": 1,
                "sim_id": "WRONG2.h5",
                "frame_index": 40,
                "role": role,
                "split": "train",
                "nh3_fraction": 0.0,
                "phi": 0.85,
                "equivalence_ratio": 0.85,
            },
        ]
        (rd / "frames.json").write_text(json.dumps(frames))
    x = np.linspace(-1, 1, 128)
    y = np.linspace(-1, 1, 128)
    xs, ys = scale_xy_to_minus1_1(x, y)
    np.savez_compressed(root / "geometry.npz", x=x, y=y, x_scaled=xs, y_scaled=ys)
    geom_sha = __import__("hashlib").sha256((root / "geometry.npz").read_bytes()).hexdigest()
    # Use real stats/protocol fingerprints so those checks pass before identity fails
    from combustion_recon.data.month2_protocol import (
        scientific_protocol_fingerprint,
        sha256_file,
        load_frame_protocol,
    )

    stats_path = ROOT / "data/manifests/train_norm_stats_month2_fit.json"
    manifest = {
        "cache_id": "month2_dense_unet_cache_v1",
        "test_payloads": "none",
        "task_mode": "temperature_free",
        "K": 0,
        "site_fraction": 1.0,
        "channel_order": list(CHANNEL_ORDER),
        "counts": {"train_fit": 2},
        "stats_path": "data/manifests/train_norm_stats_month2_fit.json",
        "geometry_sha256": geom_sha,
        "data_fingerprints": {
            "stats_sha256": sha256_file(stats_path),
            "frame_protocol_scientific_sha256": scientific_protocol_fingerprint(
                load_frame_protocol()
            ),
            "splits_sha256": sha256_file(ROOT / "configs/splits/splits_interp_extrap.yaml"),
            "geometry_sha256": geom_sha,
        },
        "fingerprints": {
            "stats_sha256": sha256_file(stats_path),
            "frame_protocol_scientific_sha256": scientific_protocol_fingerprint(
                load_frame_protocol()
            ),
            "splits_sha256": sha256_file(ROOT / "configs/splits/splits_interp_extrap.yaml"),
            "geometry_sha256": geom_sha,
        },
    }
    with pytest.raises(DenseCacheError, match="count mismatch|sim_id"):
        validate_dense_cache_manifest(manifest, cache_root=root, require_roles=("train_fit",))

    # Altered stats fingerprint must fail even with matching counts skipped
    manifest2 = dict(manifest)
    manifest2["counts"] = {"train_fit": 760}  # still wrong payload length
    # Force stats mismatch
    manifest2["data_fingerprints"] = dict(manifest["data_fingerprints"])
    manifest2["data_fingerprints"]["stats_sha256"] = "0" * 64
    manifest2["fingerprints"] = dict(manifest2["data_fingerprints"])
    with pytest.raises(DenseCacheError, match="stats fingerprint|count|frames.json length"):
        validate_dense_cache_manifest(manifest2, cache_root=root, require_roles=("train_fit",))


def test_model_fingerprint_change_does_not_invalidate_data_cache():
    """Width/budget live in model_fingerprints; data reuse ignores them."""
    if not (CACHE_ROOT / "manifest.json").exists():
        pytest.skip("dense cache not built")
    refresh_dense_cache_manifest_metadata(CACHE_ROOT)
    m = json.loads((CACHE_ROOT / "manifest.json").read_text())
    assert "data_fingerprints" in m
    assert "model_fingerprints" in m
    # Mutate only model fingerprint in-memory; validation must still pass.
    m2 = json.loads(json.dumps(m))
    m2["model_fingerprints"]["model_config_sha256"] = "deadbeef" * 8
    validate_dense_cache_manifest(m2, cache_root=CACHE_ROOT, require_roles=("train_fit",))


@pytest.mark.skipif(not HAS_RAW, reason="local raw shards required")
def test_refresh_manifest_without_rebuild():
    before = (CACHE_ROOT / "train_fit" / "vi_norm.npy").stat().st_mtime
    out = refresh_dense_cache_manifest_metadata(CACHE_ROOT)
    after = (CACHE_ROOT / "train_fit" / "vi_norm.npy").stat().st_mtime
    assert before == after
    assert out["cache_schema_version"] == 2
    assert "NOT locally recomputed" in out["shard_sha256_policy"]
    ds = DenseUNetCacheDataset("train_fit", cache_root=CACHE_ROOT)
    assert len(ds) == 760
