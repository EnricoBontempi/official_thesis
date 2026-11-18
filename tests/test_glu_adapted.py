"""S3 adapted deterministic GLU: contracts, isolation, tie-break, domain geometry."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pytest
import torch

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.models.glu_reconstructor import (
    AdaptedDeterministicGLU,
    build_glu_from_config,
    domain_geometry_from_axes,
    load_glu_minimal_config,
    make_synthetic_query_grid,
    topk_distance_then_site_index,
)
from combustion_recon.training.dense_cache import subset_indices_for_overfit
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset
from combustion_recon.training.sparse_view import (
    build_canonical_observation,
    sites_from_training_mask,
)
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
ATOL_GEOM_PRED = 1e-5
ATOL_RELOAD = 1e-6

_ENV = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "device_under_test": "cpu",
    "note": "CPU-only checks; no claim of universal CPU/GPU bit-reproducibility",
}


def _synth_domain(*, half: float = 2.0, n: int = 8):
    x = np.linspace(-half, half, n)
    y = np.linspace(-half, half, n)
    return x, y


def _build_synth(**kwargs):
    x, y = _synth_domain()
    return build_glu_from_config(domain_x=x, domain_y=y, height=8, width=8, **kwargs)


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


def test_glu_requires_domain_geometry():
    with pytest.raises(ValueError, match="domain geometry required"):
        AdaptedDeterministicGLU(height=8, width=8)  # type: ignore[call-arg]


def test_glu_shapes_finite_and_param_registration():
    set_seed(0)
    model = _build_synth()
    n0 = model.count_parameters()
    names0 = {k for k, _ in model.named_parameters()}
    assert n0 > 500_000
    B, N, P = 2, 64, 256
    coords = torch.randn(B, N, 2) * 0.5
    vals = torch.randn(B, N, 2)
    sites = torch.arange(N).unsqueeze(0).expand(B, -1)
    p = torch.randn(B, 2)
    Y = torch.randn(B, P, 2) * 0.5
    out = model(coords, vals, sites, p, Y, decode_chunked=False, return_flat=True)
    assert out.shape == (B, P)
    assert torch.isfinite(out).all()
    loss = out.pow(2).mean()
    loss.backward()
    assert n0 == model.count_parameters()
    assert names0 == {k for k, _ in model.named_parameters()}
    assert model.encoder.p_proj.weight.grad is not None
    assert "domain_center" in model.state_dict()
    assert "domain_L" in model.state_dict()


def test_per_sensor_sigma_upstream_risk_separate():
    from combustion_recon.models import glu_reconstructor as gr

    Soft = gr._SoftDomainAdaptiveReconstructor
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
    dec(
        torch.randn(B, T, S, D),
        torch.randn(B, P, 2),
        torch.randn(B, S, 2),
        torch.ones(B, T, S, dtype=torch.bool),
    )
    assert sum(1 for _ in dec.parameters()) > n_before
    model = _build_synth()
    assert model.decoder.per_sensor_sigma is False
    n0 = sum(1 for _ in model.parameters())
    model(
        torch.randn(1, 16, 2) * 0.5,
        torch.randn(1, 16, 2),
        torch.arange(16).view(1, -1),
        torch.zeros(1, 2),
        torch.randn(1, 64, 2) * 0.5,
        decode_chunked=False,
        return_flat=True,
    )
    assert sum(1 for _ in model.parameters()) == n0


def test_topk_helper_expected_indices_and_permutations():
    """N>k, more equidistant candidates than slots; check expected indices."""
    # Query at origin; sensors on a circle (equal distance) with distinct site ids
    # sites: 30,10,40,20 at angles — all |r|=1 → equidistant. top_k=2 → sites 10,20
    coords = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]], dtype=torch.float32
    )
    sites = torch.tensor([[30, 10, 40, 20]], dtype=torch.int64)
    Y = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
    d = torch.cdist(Y, coords)  # all ones
    top = topk_distance_then_site_index(d, sites, k=2)
    # After selection in sensor-slot space: slots whose site_index are 10 and 20
    selected_sites = sites[0, top[0, 0]].tolist()
    assert selected_sites == [10, 20], selected_sites

    # Permute sensor order: same site ids selected
    perm = torch.tensor([2, 0, 3, 1])
    top2 = topk_distance_then_site_index(d[:, :, perm], sites[:, perm], k=2)
    selected2 = sites[0, perm][top2[0, 0]].tolist()
    assert selected2 == [10, 20]

    # Unequal distances: nearer sensor wins even with larger site_index
    coords_u = torch.tensor(
        [[[0.5, 0.0], [2.0, 0.0], [2.0, 0.0]]], dtype=torch.float32
    )
    sites_u = torch.tensor([[99, 1, 2]], dtype=torch.int64)
    d_u = torch.cdist(Y, coords_u)
    top_u = topk_distance_then_site_index(d_u, sites_u, k=1)
    assert int(sites_u[0, top_u[0, 0, 0]]) == 99


def test_topk_tie_break_in_decoder_path_with_predictions():
    set_seed(0)
    x, y = _synth_domain()
    model = AdaptedDeterministicGLU(
        height=8,
        width=8,
        All_dim=32,
        top_k=2,
        decode_chunked_eval=False,
        domain_x=x,
        domain_y=y,
    )
    model.eval()
    model.decoder._record_top_idx = True
    # 4 equidistant sensors, k=2; sites 40,10,30,20 → expect sites 10,20
    coords = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]], dtype=torch.float32
    )
    vals = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]], dtype=torch.float32
    )
    sites = torch.tensor([[40, 10, 30, 20]], dtype=torch.int64)
    p = torch.zeros(1, 2)
    Y = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
    with torch.no_grad():
        y_a = model(coords, vals, sites, p, Y, decode_chunked=False, return_flat=True)
        top_a = model.decoder.last_top_idx
        assert top_a is not None
        # After canonical sort, sites order becomes [10,20,30,40] → slots 0,1
        selected_sites_a = torch.sort(sites[0]).values[top_a[0, 0]].tolist()
        # sites sorted: 10,20,30,40; top slots should be 0 and 1
        assert top_a[0, 0].tolist() == [0, 1], top_a[0, 0].tolist()

        perm = torch.tensor([2, 0, 3, 1])
        y_b = model(
            coords[:, perm],
            vals[:, perm],
            sites[:, perm],
            p,
            Y,
            decode_chunked=False,
            return_flat=True,
        )
        top_b = model.decoder.last_top_idx
        assert top_b is not None
        assert top_b[0, 0].tolist() == [0, 1]
    assert torch.allclose(y_a, y_b, rtol=RTOL_ABS, atol=ATOL_PERM)

    # Unequal distances dominate site_index
    set_seed(0)
    model2 = AdaptedDeterministicGLU(
        height=8, width=8, All_dim=32, top_k=1, decode_chunked_eval=False, domain_x=x, domain_y=y
    )
    model2.eval()
    model2.decoder._record_top_idx = True
    coords_u = torch.tensor([[[0.2, 0.0], [1.5, 0.0]]], dtype=torch.float32)
    vals_u = torch.tensor([[[9.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
    sites_u = torch.tensor([[50, 1]], dtype=torch.int64)
    with torch.no_grad():
        model2(coords_u, vals_u, sites_u, p, Y, decode_chunked=False, return_flat=True)
        # After sort by site: [1,50] → nearer (0.2) is site 50 at slot 1
        assert model2.decoder.last_top_idx is not None
        assert model2.decoder.last_top_idx[0, 0].tolist() == [1]


@pytest.mark.skipif(not HAS_CACHE, reason="dense cache missing")
def test_glu_isolation_target_path_and_p_off(cache_bundle):
    """Mutate T on the real canonical sample path; GLU inputs/pred unchanged."""
    ds = cache_bundle["ds"]
    item = ds[0]
    obs = item["observation"]
    idx = ds.indices[0]
    queries = _mesh_queries(ds, 1)
    p_norm = torch.tensor([[float(ds.p_norm[idx, 0]), float(ds.p_norm[idx, 1])]], dtype=torch.float32)

    vi = np.asarray(ds.vi[idx], dtype=np.float64)
    vj = np.asarray(ds.vj[idx], dtype=np.float64)
    t0 = np.asarray(ds.t_norm[idx], dtype=np.float64)
    vi_p, vj_p = ds._phys_velocities(idx)

    def glu_inputs_from_fields(*, vi_n, vj_n, t_field):
        can = build_canonical_observation(
            sites=obs.sites,
            vi_phys=vi_p,
            vj_phys=vj_p,
            vi_norm=vi_n,
            vj_norm=vj_n,
            x_coords=ds.x,
            y_coords=ds.y,
            site_fraction=0.05,
            layout_id="random_sparse_primary",
            velocity_mask_seed=obs.velocity_mask_seed,
        )
        # Target field is not consumed by canonic observation / GLU features.
        _ = t_field
        return {
            "sensor_coords_phys": torch.from_numpy(can.velocity_coords.astype(np.float32)).unsqueeze(0),
            "sensor_values_norm": torch.from_numpy(can.velocity_values_norm.astype(np.float32)).unsqueeze(0),
            "site_index": torch.from_numpy(can.sites.astype(np.int64)).unsqueeze(0),
            "target_T_norm": t_field.copy(),
        }

    set_seed(0)
    model = build_glu_from_config()  # cache geometry default
    model.eval()
    pack0 = glu_inputs_from_fields(vi_n=vi, vj_n=vj, t_field=t0)
    with torch.no_grad():
        y0 = model(
            pack0["sensor_coords_phys"],
            pack0["sensor_values_norm"],
            pack0["site_index"],
            p_norm,
            queries,
            decode_chunked=True,
            chunk_size=512,
        )

    t_mut = t0 + 50.0
    pack1 = glu_inputs_from_fields(vi_n=vi, vj_n=vj, t_field=t_mut)
    assert not np.array_equal(pack0["target_T_norm"], pack1["target_T_norm"])
    assert np.array_equal(pack0["sensor_values_norm"].numpy(), pack1["sensor_values_norm"].numpy())
    assert np.array_equal(pack0["site_index"].numpy(), pack1["site_index"].numpy())
    assert np.array_equal(pack0["sensor_coords_phys"].numpy(), pack1["sensor_coords_phys"].numpy())
    with torch.no_grad():
        y1 = model(
            pack1["sensor_coords_phys"],
            pack1["sensor_values_norm"],
            pack1["site_index"],
            p_norm,
            queries,
            decode_chunked=True,
            chunk_size=512,
        )
    assert torch.equal(y0, y1)

    # Hidden velocity isolation via same path
    vi_h = vi.copy()
    vj_h = vj.copy()
    vi_h[~obs.mask] += 9.0
    vj_h[~obs.mask] -= 4.0
    pack_h = glu_inputs_from_fields(vi_n=vi_h, vj_n=vj_h, t_field=t0)
    assert np.array_equal(pack0["sensor_values_norm"].numpy(), pack_h["sensor_values_norm"].numpy())
    with torch.no_grad():
        y_h = model(
            pack_h["sensor_coords_phys"],
            pack_h["sensor_values_norm"],
            pack_h["site_index"],
            p_norm,
            queries,
            decode_chunked=True,
            chunk_size=512,
        )
    assert torch.equal(y0, y_h)

    # p_off: mutate source p, rebuild with p_conditioning=False
    with torch.no_grad():
        y_a = model(
            pack0["sensor_coords_phys"],
            pack0["sensor_values_norm"],
            pack0["site_index"],
            p_norm.clone(),
            queries,
            p_conditioning=False,
            decode_chunked=False,
        )
        y_b = model(
            pack0["sensor_coords_phys"],
            pack0["sensor_values_norm"],
            pack0["site_index"],
            p_norm.clone() + 3.0,
            queries,
            p_conditioning=False,
            decode_chunked=False,
        )
    assert torch.equal(y_a, y_b)


def test_sensor_permutation_invariance_and_query_equivariance():
    set_seed(0)
    model = _build_synth()
    model.eval()
    B, N, P = 1, 32, 64
    coords = torch.randn(B, N, 2) * 0.5
    vals = torch.randn(B, N, 2)
    sites = torch.arange(N).view(1, -1)
    p = torch.randn(B, 2)
    Y = torch.randn(B, P, 2) * 0.5
    with torch.no_grad():
        y0 = model(coords, vals, sites, p, Y, decode_chunked=False, return_flat=True)
        perm = torch.randperm(N)
        y1 = model(
            coords[:, perm], vals[:, perm], sites[:, perm], p, Y, decode_chunked=False, return_flat=True
        )
        assert torch.allclose(y0, y1, rtol=RTOL_ABS, atol=ATOL_PERM)
        qperm = torch.randperm(P)
        y2 = model(coords, vals, sites, p, Y[:, qperm], decode_chunked=False, return_flat=True)
        assert torch.allclose(y0[:, qperm], y2, rtol=RTOL_ABS, atol=ATOL_QUERY_EQ)


@pytest.mark.skipif(not HAS_CACHE, reason="dense cache missing")
def test_fixed_domain_geometry_invariants(cache_bundle, tmp_path: Path):
    """A–E geometry checks with declared tolerances; environment recorded."""
    ds = cache_bundle["ds"]
    set_seed(0)
    model = build_glu_from_config()
    model.eval()
    geom0 = dict(model.geometry_meta)
    item0 = ds[0]
    item1 = ds[1]
    q_full = _mesh_queries(ds, 1)
    subset = torch.tensor([0, 17, 128, 500, 1024, 4096, 8000, 16000])
    subset = subset[subset < q_full.shape[1]]
    q_sub = q_full[:, subset, :]

    def pack(item, queries, batch_expand: int = 1):
        idx = item["metadata"]["cache_index"]
        p = torch.tensor([[float(ds.p_norm[idx, 0]), float(ds.p_norm[idx, 1])]], dtype=torch.float32)
        coords = torch.from_numpy(item["observation"].velocity_coords.astype(np.float32)).unsqueeze(0)
        vals = torch.from_numpy(item["observation"].velocity_values_norm.astype(np.float32)).unsqueeze(0)
        sites = torch.from_numpy(item["observation"].sites.astype(np.int64)).unsqueeze(0)
        if batch_expand > 1:
            coords = coords.expand(batch_expand, -1, -1).contiguous()
            vals = vals.expand(batch_expand, -1, -1).contiguous()
            sites = sites.expand(batch_expand, -1).contiguous()
            p = p.expand(batch_expand, -1).contiguous()
            queries = queries.expand(batch_expand, -1, -1).contiguous()
        return coords, vals, sites, p, queries

    with torch.no_grad():
        # A: full-grid prediction restricted vs direct subset queries
        y_full = model(*pack(item0, q_full), decode_chunked=False, return_flat=True)
        y_sub = model(*pack(item0, q_sub), decode_chunked=False, return_flat=True)
        diff_a = float((y_full[:, subset] - y_sub).abs().max())

        # B: singleton vs batched with another sample
        y_single = model(*pack(item0, q_sub), decode_chunked=False, return_flat=True)
        # build batch of two different layouts
        c0, v0, s0, p0, q0 = pack(item0, q_sub)
        c1, v1, s1, p1, q1 = pack(item1, q_sub)
        # Homogeneous N required — both 5% fixed → same N=819
        y_batch = model(
            torch.cat([c0, c1], 0),
            torch.cat([v0, v1], 0),
            torch.cat([s0, s1], 0),
            torch.cat([p0, p1], 0),
            torch.cat([q0, q1], 0),
            decode_chunked=False,
            return_flat=True,
        )
        diff_b = float((y_single[0] - y_batch[0]).abs().max())

        # C: center/L invariant to observed layout
        _ = model(*pack(item1, q_sub), decode_chunked=False, return_flat=True)
        geom1 = dict(model.geometry_meta)
        assert geom0["center"] == geom1["center"]
        assert geom0["L"] == geom1["L"]

        # D: full vs chunk
        y_chunk = model(*pack(item0, q_full), decode_chunked=True, chunk_size=512, return_flat=True)
        diff_d = float((y_full - y_chunk).abs().max())

        # E: checkpoint reload
        ckpt = tmp_path / "glu_geom.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt)
        model2 = build_glu_from_config()
        model2.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
        model2.eval()
        assert model2.geometry_meta["center"] == geom0["center"]
        assert model2.geometry_meta["L"] == geom0["L"]
        y_reload = model2(*pack(item0, q_sub), decode_chunked=False, return_flat=True)
        diff_e = float((y_sub - y_reload).abs().max())

    assert diff_a <= ATOL_GEOM_PRED, f"A max_abs={diff_a}"
    assert diff_b <= ATOL_GEOM_PRED, f"B max_abs={diff_b}"
    assert diff_d <= ATOL_CHUNK, f"D max_abs={diff_d}"
    assert diff_e <= ATOL_RELOAD, f"E max_abs={diff_e}"

    report = {
        "environment": _ENV,
        "tolerances": {
            "ATOL_GEOM_PRED": ATOL_GEOM_PRED,
            "ATOL_CHUNK": ATOL_CHUNK,
            "ATOL_RELOAD": ATOL_RELOAD,
            "RTOL": RTOL_ABS,
        },
        "measured_max_abs": {"A_subset": diff_a, "B_batch": diff_b, "D_chunk": diff_d, "E_reload": diff_e},
        "geometry": geom0,
    }
    (tmp_path / "geometry_invariants.json").write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.skipif(not HAS_CACHE, reason="dense cache missing")
def test_prior_fullgrid_geometry_equivalence_to_domain():
    """Prior diagnostic full-grid queries: sensors∪full_queries ≡ domain axes.

    Correct limit of the old call-site helper: when the query set already spans the
    full mesh extrema, center/L match the cache-axis domain map. Chunking that
    mapped q *before* splitting queries therefore did not invalidate prior
    full-vs-chunk diagnostics. Subset-only sensors∪queries can differ.
    """
    g = np.load(CACHE_ROOT / "geometry.npz")
    domain = domain_geometry_from_axes(g["x"], g["y"], source="cache")
    xx, yy = np.meshgrid(g["x"], g["y"], indexing="xy")
    queries = np.stack([xx.ravel(), yy.ravel()], axis=1)
    sensors = queries[::20]
    pts = np.concatenate([sensors, queries], axis=0)
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    L_old = max(xmax - xmin, ymax - ymin)
    c_old = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)
    assert abs(L_old - domain["L"]) <= 1e-15
    assert abs(c_old[0] - domain["center"][0]) <= 1e-15
    assert abs(c_old[1] - domain["center"][1]) <= 1e-15

    # Corner subset would have differed under the removed call-site helper
    corner = queries[(queries[:, 0] < domain["center"][0]) & (queries[:, 1] < domain["center"][1])]
    L_corner = max(
        float(corner[:, 0].max() - corner[:, 0].min()),
        float(corner[:, 1].max() - corner[:, 1].min()),
    )
    assert abs(L_corner - domain["L"]) > 1e-3


def test_full_vs_chunked_decoder_diff():
    set_seed(0)
    model = _build_synth()
    model.eval()
    B, N, P = 1, 48, 1024
    coords = torch.randn(B, N, 2) * 0.5
    vals = torch.randn(B, N, 2)
    sites = torch.arange(N).view(1, -1)
    p = torch.zeros(B, 2)
    Y = torch.randn(B, P, 2) * 0.5
    with torch.no_grad():
        y_full = model(coords, vals, sites, p, Y, decode_chunked=False, return_flat=True)
        y_chunk = model(coords, vals, sites, p, Y, decode_chunked=True, chunk_size=512, return_flat=True)
    assert float((y_full - y_chunk).abs().max()) <= ATOL_CHUNK


def test_collinear_value_embedding_sensitivity():
    set_seed(0)
    model = _build_synth()
    model.eval()
    B, N, P = 1, 16, 32
    coords = torch.randn(B, N, 2) * 0.5
    direction = torch.tensor([3.0, 4.0])
    vals_a = direction.view(1, 1, 2).expand(B, N, 2) * 0.1
    vals_b = direction.view(1, 1, 2).expand(B, N, 2) * 1.5
    sites = torch.arange(N).view(1, -1)
    p = torch.zeros(B, 2)
    Y = torch.randn(B, P, 2) * 0.5
    with torch.no_grad():
        ya = model(coords, vals_a, sites, p, Y, decode_chunked=False, return_flat=True)
        yb = model(coords, vals_b, sites, p, Y, decode_chunked=False, return_flat=True)
    assert not torch.allclose(ya, yb, rtol=0, atol=1e-6)


@pytest.mark.skipif(not HAS_CACHE, reason="dense cache missing")
def test_glu_small_overfit_fixed_and_positive_controls(cache_bundle, tmp_path: Path):
    ds = cache_bundle["ds"]
    set_seed(0)
    model = build_glu_from_config()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
