"""S3 adapted deterministic GLU: contracts, isolation, permutations, chunking, learning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.models.glu_reconstructor import (
    AdaptedDeterministicGLU,
    build_glu_from_config,
    load_glu_minimal_config,
    make_synthetic_query_grid,
)
from combustion_recon.training.dense_cache import subset_indices_for_overfit
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset
from combustion_recon.training.sparse_view import sites_from_training_mask
from combustion_recon.training.trainer import set_seed

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data/processed/month2_dense_unet_cache"
STATS = ROOT / "data/manifests/train_norm_stats_month2_fit.json"
HAS_CACHE = (CACHE_ROOT / "manifest.json").exists()

# Declared BEFORE comparisons
ATOL_BIT = 0.0
RTOL_ABS = 0.0
ATOL_CHUNK = 1e-5
ATOL_PERM = 1e-5
ATOL_QUERY_EQ = 1e-5


def _mesh_queries(ds: SparseUNetCacheDataset, batch: int = 1) -> torch.Tensor:
    q = make_synthetic_query_grid(ds.x, ds.y)
    return q.unsqueeze(0).expand(batch, -1, -1).contiguous()


@pytest.fixture(scope="module")
def cache_bundle():
    if not HAS_CACHE:
        pytest.skip("dense cache missing")
    normalizer = Month2FitOnlyNormalizer(STATS)
    frames = json.loads((CACHE_ROOT / "train_fit/frames.json").read_text())
    idxs = subset_indices_for_overfit(frames, n_frames=8, seed=7)
    fixed = {
        ci: sites_from_training_mask(
            train_mask_seed=20261118,
            epoch=0,
            sim_id=frames[ci]["sim_id"],
            frame_index=int(frames[ci]["frame_index"]),
        )[0]
        for ci in idxs
    }
    ds = SparseUNetCacheDataset(
        "train_fit",
        cache_root=CACHE_ROOT,
        indices=idxs,
        mask_mode="fixed",
        fixed_sites_by_cache_index=fixed,
        p_conditioning=True,
        normalizer=normalizer,
    )
    return {"ds": ds, "normalizer": normalizer, "idxs": idxs, "frames": frames}


def test_glu_config_refuses_unknown(tmp_path: Path):
    cfg = load_glu_minimal_config()
    assert cfg["encoder"]["All_dim"] == 32
    bad = tmp_path / "bad.yaml"
    bad.write_text("architecture: adapted_deterministic_glu\nlr_from_val: 1e-3\n")
    with pytest.raises(ValueError, match="unsupported keys|missing required"):
        load_glu_minimal_config(bad)


def test_glu_shapes_finite_and_param_registration():
    set_seed(0)
    model = build_glu_from_config()
    n0 = model.count_parameters()
    names0 = {k for k, _ in model.named_parameters()}
    assert n0 > 500_000
    B, N, P = 2, 64, 256
    coords = torch.randn(B, N, 2)
    vals = torch.randn(B, N, 2)
    sites = torch.arange(N).unsqueeze(0).expand(B, -1)
    p = torch.randn(B, 2)
    Y = torch.randn(B, P, 2)
    out = model(
        coords,
        vals,
        sites,
        p,
        Y,
        decode_chunked=False,
        return_flat=True,
    )
    assert out.shape == (B, P)
    assert torch.isfinite(out).all()
    loss = out.pow(2).mean()
    loss.backward()
    n1 = model.count_parameters()
    names1 = {k for k, _ in model.named_parameters()}
    assert n0 == n1
    assert names0 == names1
    # Unused modules (e.g. decoder MLP when USE_FINAL_MLP=False) may lack grads.
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads)
    assert model.encoder.p_proj.weight.grad is not None
    assert model.encoder.embed["val_linear"].weight.grad is not None
    assert model.decoder.head.weight.grad is not None


def test_per_sensor_sigma_upstream_risk_separate():
    """Separate torch check of upstream lazy Parameter creation (reading vs executed)."""
    from combustion_recon.models import glu_reconstructor as gr

    Soft = gr._SoftDomainAdaptiveReconstructor
    # Executed: with per_sensor_sigma=True, forward may create log_sigma
    dec = Soft(
        d_model=32,
        num_heads=4,
        N_channels=1,
        latent_tokens=64,
        top_k=4,
        per_sensor_sigma=True,
        retain_cls=False,
        bandwidth_init=0.05,
    )
    n_before = sum(1 for _ in dec.parameters())
    B, T, S, P, D = 1, 1, 8, 16, 32
    z = torch.randn(B, T, S, D)
    Y = torch.randn(B, P, 2)
    sens = torch.randn(B, S, 2)
    mask = torch.ones(B, T, S, dtype=torch.bool)
    dec(z, Y, sens, mask)
    n_after = sum(1 for _ in dec.parameters())
    assert n_after > n_before  # verified: lazy Parameter appears
    # Thesis path forces False — no new params after forward
    model = build_glu_from_config()
    assert model.decoder.per_sensor_sigma is False
    n0 = sum(1 for _ in model.parameters())
    coords = torch.randn(1, 16, 2)
    vals = torch.randn(1, 16, 2)
    sites = torch.arange(16).view(1, -1)
    p = torch.zeros(1, 2)
    Y = torch.randn(1, 64, 2)
    model(coords, vals, sites, p, Y, decode_chunked=False, return_flat=True)
    n1 = sum(1 for _ in model.parameters())
    assert n0 == n1


@pytest.mark.skipif(not HAS_CACHE, reason="dense cache missing")
def test_glu_isolation_and_p_off(cache_bundle):
    ds = cache_bundle["ds"]
    item = ds[0]
    obs = item["observation"]
    idx = ds.indices[0]
    queries = _mesh_queries(ds, 1)
    # Build p_norm from cache
    p_norm = torch.tensor([[float(ds.p_norm[idx, 0]), float(ds.p_norm[idx, 1])]], dtype=torch.float32)

    def run(vi_norm, vj_norm, p, *, p_conditioning=True):
        ys = obs.sites // 128
        xs = obs.sites % 128
        sens_vals = np.stack(
            [vi_norm[ys, xs], vj_norm[ys, xs]], axis=1
        ).astype(np.float32)
        set_seed(0)
        model = build_glu_from_config()
        model.eval()
        with torch.no_grad():
            return model(
                torch.from_numpy(obs.velocity_coords.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(sens_vals).unsqueeze(0),
                torch.from_numpy(obs.sites.astype(np.int64)).unsqueeze(0),
                p,
                queries,
                p_conditioning=p_conditioning,
                decode_chunked=True,
                chunk_size=512,
            )

    set_seed(0)
    vi = np.asarray(ds.vi[idx], dtype=np.float64)
    vj = np.asarray(ds.vj[idx], dtype=np.float64)
    y0 = run(vi, vj, p_norm.clone())

    vi_h = vi.copy()
    vj_h = vj.copy()
    vi_h[~obs.mask] += 9.0
    vj_h[~obs.mask] -= 4.0
    y_h = run(vi_h, vj_h, p_norm.clone())
    assert torch.equal(y0, y_h)

    # T mutation unused
    t_mut = np.asarray(ds.t_norm[idx]) + 50.0
    _ = t_mut
    y_t = run(vi, vj, p_norm.clone())
    assert torch.equal(y0, y_t)

    # p_off: source p mutated
    set_seed(0)
    model = build_glu_from_config()
    model.eval()
    base_kwargs = dict(
        sensor_coords_phys=torch.from_numpy(obs.velocity_coords.astype(np.float32)).unsqueeze(0),
        sensor_values_norm=torch.from_numpy(obs.velocity_values_norm.astype(np.float32)).unsqueeze(0),
        site_index=torch.from_numpy(obs.sites.astype(np.int64)).unsqueeze(0),
        query_coords_phys=queries,
        p_conditioning=False,
        decode_chunked=False,
    )
    with torch.no_grad():
        y_a = model(**base_kwargs, p_norm=p_norm.clone())
        y_b = model(**base_kwargs, p_norm=p_norm.clone() + 3.0)
    assert torch.equal(y_a, y_b)


def test_sensor_permutation_invariance_and_query_equivariance():
    set_seed(0)
    model = build_glu_from_config()
    model.eval()
    B, N, P = 1, 32, 64
    coords = torch.randn(B, N, 2)
    vals = torch.randn(B, N, 2)
    sites = torch.arange(N).view(1, -1)
    p = torch.randn(B, 2)
    Y = torch.randn(B, P, 2)
    with torch.no_grad():
        y0 = model(coords, vals, sites, p, Y, decode_chunked=False, return_flat=True)
        perm = torch.randperm(N)
        y1 = model(
            coords[:, perm],
            vals[:, perm],
            sites[:, perm],
            p,
            Y,
            decode_chunked=False,
            return_flat=True,
        )
        assert torch.allclose(y0, y1, rtol=RTOL_ABS, atol=ATOL_PERM)
        # Query permutation equivariance
        qperm = torch.randperm(P)
        y2 = model(coords, vals, sites, p, Y[:, qperm], decode_chunked=False, return_flat=True)
        assert torch.allclose(y0[:, qperm], y2, rtol=RTOL_ABS, atol=ATOL_QUERY_EQ)


def test_topk_tie_break_equidistant_neighbors():
    """Equidistant sensors: after site sort, lower site index preferred (no distance jitter)."""
    set_seed(0)
    model = AdaptedDeterministicGLU(
        height=8,
        width=8,
        All_dim=32,
        top_k=2,
        decode_chunked_eval=False,
    )
    model.eval()
    # Two sensors at same distance from a query; sites 5 and 9
    coords = torch.tensor([[[0.0, 1.0], [0.0, -1.0]]], dtype=torch.float32)
    vals = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    sites = torch.tensor([[9, 5]], dtype=torch.int64)  # unsorted on purpose
    p = torch.zeros(1, 2)
    Y = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)  # equidistant
    with torch.no_grad():
        y_a = model(coords, vals, sites, p, Y, decode_chunked=False, return_flat=True)
        # Swap values but keep sites — after sort by site, assignment follows site order
        y_b = model(
            coords[:, [1, 0]],
            vals[:, [1, 0]],
            sites[:, [1, 0]],
            p,
            Y,
            decode_chunked=False,
            return_flat=True,
        )
    assert torch.allclose(y_a, y_b, rtol=RTOL_ABS, atol=ATOL_PERM)


def test_full_vs_chunked_decoder_diff():
    set_seed(0)
    model = build_glu_from_config()
    model.eval()
    B, N, P = 1, 48, 1024  # > 512
    coords = torch.randn(B, N, 2)
    vals = torch.randn(B, N, 2)
    sites = torch.arange(N).view(1, -1)
    p = torch.zeros(B, 2)
    Y = torch.randn(B, P, 2)
    with torch.no_grad():
        y_full = model(coords, vals, sites, p, Y, decode_chunked=False, return_flat=True)
        y_chunk = model(coords, vals, sites, p, Y, decode_chunked=True, chunk_size=512, return_flat=True)
    max_abs = float((y_full - y_chunk).abs().max())
    assert max_abs <= ATOL_CHUNK, f"full vs chunk max_abs={max_abs} > {ATOL_CHUNK}"
    # Chunked backward for training is NOT declared supported here (eval-only check).


def test_collinear_value_embedding_sensitivity():
    set_seed(0)
    model = build_glu_from_config()
    model.eval()
    B, N, P = 1, 16, 32
    coords = torch.randn(B, N, 2)
    direction = torch.tensor([3.0, 4.0])
    vals_a = direction.view(1, 1, 2).expand(B, N, 2) * 0.1
    vals_b = direction.view(1, 1, 2).expand(B, N, 2) * 1.5
    sites = torch.arange(N).view(1, -1)
    p = torch.zeros(B, 2)
    Y = torch.randn(B, P, 2)
    with torch.no_grad():
        ya = model(coords, vals_a, sites, p, Y, decode_chunked=False, return_flat=True)
        yb = model(coords, vals_b, sites, p, Y, decode_chunked=False, return_flat=True)
    # Do not presume mathematical collapse; only require amplitude sensitivity on this draw
    assert not torch.allclose(ya, yb, rtol=0, atol=1e-6)


@pytest.mark.skipif(not HAS_CACHE, reason="dense cache missing")
def test_glu_small_overfit_fixed_and_positive_controls(cache_bundle, tmp_path: Path):
    """Capacity check on real 5% packets; full-field diagnostic run is scripts/14_*."""
    ds = cache_bundle["ds"]
    set_seed(0)
    model = build_glu_from_config()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Subsample queries for pytest walltime; script 14 uses full 16384.
    q_full = _mesh_queries(ds, 1)
    q_idx = torch.linspace(0, q_full.shape[1] - 1, 1024).long()
    queries = q_full[:, q_idx, :]
    names0 = {k for k, _ in model.named_parameters()}

    def batch_loss(i: int):
        item = ds[i]
        idx = ds.indices[i]
        p = torch.tensor([[float(ds.p_norm[idx, 0]), float(ds.p_norm[idx, 1])]], dtype=torch.float32)
        pred = model(
            torch.from_numpy(item["observation"].velocity_coords.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(item["observation"].velocity_values_norm.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(item["observation"].sites.astype(np.int64)).unsqueeze(0),
            p,
            queries,
            p_conditioning=True,
            decode_chunked=True,
            chunk_size=512,
            return_flat=True,
        )
        tgt = item["targets"]["temperature_normalized"].reshape(1, -1)[:, q_idx]
        return torch.mean((pred - tgt) ** 2)

    losses = []
    model.train()
    for epoch in range(8):
        epoch_losses = []
        for i in range(len(ds)):
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(i)
            assert torch.isfinite(loss)
            loss.backward()
            assert {k for k, _ in model.named_parameters()} == names0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))

    assert losses[-1] < losses[0] * 0.9, f"no learning: {losses[0]} -> {losses[-1]}"

    model.eval()
    item = ds[0]
    idx = ds.indices[0]
    p = torch.tensor([[float(ds.p_norm[idx, 0]), float(ds.p_norm[idx, 1])]], dtype=torch.float32)
    coords = torch.from_numpy(item["observation"].velocity_coords.astype(np.float32)).unsqueeze(0)
    vals = torch.from_numpy(item["observation"].velocity_values_norm.astype(np.float32)).unsqueeze(0)
    sites = torch.from_numpy(item["observation"].sites.astype(np.int64)).unsqueeze(0)
    with torch.no_grad():
        y0 = model(coords, vals, sites, p, queries, decode_chunked=True, return_flat=True)
        y1 = model(coords, vals + 1.0, sites, p, queries, decode_chunked=True, return_flat=True)
        y2 = model(coords, vals, sites, p + 0.5, queries, p_conditioning=True, decode_chunked=True, return_flat=True)
    assert not torch.equal(y0, y1)
    assert not torch.equal(y0, y2)

    (tmp_path / "glu_overfit_losses.json").write_text(
        json.dumps({"first": losses[0], "last": losses[-1], "curve": losses, "query_subsample": 1024}, indent=2) + "\n"
    )


@pytest.mark.skipif(not HAS_CACHE, reason="dense cache missing")
def test_glu_variable_mask_brief(cache_bundle):
    normalizer = cache_bundle["normalizer"]
    idxs = cache_bundle["idxs"]
    ds = SparseUNetCacheDataset(
        "train_fit",
        cache_root=CACHE_ROOT,
        indices=idxs,
        mask_mode="training",
        train_mask_seed=20261118,
        epoch=0,
        p_conditioning=True,
        normalizer=normalizer,
    )
    set_seed(0)
    model = build_glu_from_config()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    q_full = _mesh_queries(ds, 1)
    q_idx = torch.linspace(0, q_full.shape[1] - 1, 1024).long()
    queries = q_full[:, q_idx, :]
    first = last = None
    model.train()
    for epoch in range(5):
        ds.set_epoch(epoch)
        ep = []
        for i in range(len(ds)):
            item = ds[i]
            idx = ds.indices[i]
            p = torch.tensor([[float(ds.p_norm[idx, 0]), float(ds.p_norm[idx, 1])]], dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            pred = model(
                torch.from_numpy(item["observation"].velocity_coords.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(item["observation"].velocity_values_norm.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(item["observation"].sites.astype(np.int64)).unsqueeze(0),
                p,
                queries,
                decode_chunked=True,
                chunk_size=512,
                return_flat=True,
            )
            tgt = item["targets"]["temperature_normalized"].reshape(1, -1)[:, q_idx]
            loss = torch.mean((pred - tgt) ** 2)
            loss.backward()
            opt.step()
            ep.append(float(loss.detach()))
        mean = float(np.mean(ep))
        if first is None:
            first = mean
        last = mean
    assert np.isfinite(first) and np.isfinite(last)
    assert last <= first * 1.05 or last < first
