"""CPU tests for the degradation operators (Block I). No model, no VAL/TEST, no temperature."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from combustion_recon.data import nonideal_observations as ni
from combustion_recon.data.month2_observations import observation_mask_seed, training_mask_seed
from combustion_recon.data.nonideal_observations import (
    DegradationError,
    degrade_observation,
    glu_batching_policy,
    load_degradation_spec,
    load_velocity_scale,
    parse_degradation_spec,
)
from combustion_recon.data.sparse_eval_masks import DEFAULT_PAYLOAD_ROOT
from combustion_recon.training.sparse_view import build_canonical_observation

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/nonideal/degradation_spec_v1.yaml"
STATS = ROOT / "data/manifests/train_norm_stats_month2_fit.json"
STATS_SHA = "678263fb4cb2598350b8fd69a71ce4843f747aa61c19556cf7f0f9c5c3ceac0e"
X = np.linspace(-0.0369067, 0.0368875, 128)
Y = np.linspace(0.0628528, 0.1591344, 128)
SIM = "0NH3_1.h5"


@pytest.fixture(scope="module")
def spec():
    return load_degradation_spec(SPEC)


@pytest.fixture(scope="module")
def scale():
    return load_velocity_scale(STATS, expected_sha256=STATS_SHA)


def make_obs(seed=0, n=819, mean=(-0.059971279319043884, 4.272026277961735), std=(5.673608459887564, 11.909608552613038)):
    rng = np.random.default_rng(seed)
    sites = np.sort(rng.choice(128 * 128, size=n, replace=False)).astype(np.int64)
    vi_n = rng.standard_normal((128, 128))
    vj_n = rng.standard_normal((128, 128))
    vi_p = vi_n * std[0] + mean[0]
    vj_p = vj_n * std[1] + mean[1]
    return build_canonical_observation(
        sites=sites, vi_phys=vi_p, vj_phys=vj_p, vi_norm=vi_n, vj_norm=vj_n,
        x_coords=X, y_coords=Y, site_fraction=0.05, layout_id="random_sparse_primary",
        velocity_mask_seed=123,
    )


def run(obs, spec, scale, cell, fr=0, rep=0, sim=SIM):
    return degrade_observation(obs, spec.cells[cell], spec, scale=scale, sim_id=sim,
                               frame_index=fr, replica_id=rep, x_m=X, y_m=Y)


# ------------------------------------------------------------------ spec

def test_spec_parses_and_is_frozen_matrix(spec):
    assert list(spec.cells) == ["R0", "N1", "G1", "N1G1", "G1-eq", "N2"]
    assert spec.cells["N1"].noise_sigma_ms == 0.5 and spec.cells["N2"].noise_sigma_ms == 1.5
    assert spec.cells["N1G1"].gap == "g1" and spec.cells["G1-eq"].gap == "g1_eq"
    assert spec.cells["R0"].noise_sigma_ms is None and spec.cells["R0"].gap is None
    assert spec.g1.area_fraction == 0.25 and spec.g1.min_surviving_sites >= 16
    assert spec.degradation_seed not in ni.HISTORICAL_SEEDS


def test_spec_stats_hash_and_zscore_match_frozen_file(spec, scale):
    raw = spec.raw["provenance"]["normalizer"]
    assert raw["stats_sha256"] == STATS_SHA == scale.stats_sha256
    z = raw["velocity_zscore_ms"]
    assert (z["Velocity_i"]["std"], z["Velocity_j"]["std"]) == scale.std
    assert (z["Velocity_i"]["mean"], z["Velocity_j"]["mean"]) == scale.mean


def test_spec_models_and_roles(spec):
    runs = spec.raw["models"]["runs"]
    assert len(runs) == 6 and sum(r["family"] == "sparse_unet" for r in runs) == 3
    assert all(len(r["best_file_sha256"]) == 64 for r in runs)
    assert spec.raw["roles"]["forbidden"] == ["test"]


@pytest.mark.parametrize("mutate,msg", [
    (lambda r: r["operators"]["noise"].update(unit="normalized"), "m/s"),
    (lambda r: r["operators"]["noise"].update(sigma_norm=0.1), "m/s"),
    (lambda r: r["seeds"].update(degradation_seed=20261116), "historical"),
    (lambda r: r["seeds"].update(degradation_seed=20261118), "historical"),
    (lambda r: r["roles"]["evaluation"].append("test_eval"), "forbidden role"),
    (lambda r: r["operators"]["g1"].update(dependence_on_temperature=True), "temperature"),
    (lambda r: r["operators"]["g1"]["rejection"].update(min_surviving_sites=4), "top_k"),
    (lambda r: r["operators"]["g1"].update(family="blobs"), "union_of_discs"),
    (lambda r: r["seeds"].update(namespace="x"), "namespace"),
    (lambda r: r["cells"].append({"id": "N1", "noise_sigma_ms": 0.1, "gap": None}), "duplicate"),
    (lambda r: r["cells"].append({"id": "Z", "noise_sigma_ms": 0.1, "gap": "g1_eq"}), "no noise"),
    (lambda r: r["cells"].append({"id": "Z", "noise_sigma_ms": -1.0, "gap": None}), "sigma"),
    (lambda r: r["cells"].append({"id": "Z", "gap": "t_dependent"}), "unknown gap"),
])
def test_spec_validation_rejects(mutate, msg):
    raw = copy.deepcopy(yaml.safe_load(SPEC.read_text()))
    mutate(raw)
    with pytest.raises(DegradationError, match=msg):
        parse_degradation_spec(raw)


def test_stats_hash_mismatch_is_rejected():
    with pytest.raises(DegradationError, match="sha256"):
        load_velocity_scale(STATS, expected_sha256="0" * 64)


# ------------------------------------------------------------------ identity / units

def test_identity_returns_original_packet(spec, scale):
    obs = make_obs()
    out = run(obs, spec, scale, "R0")
    assert out.obs is obs
    assert out.removed_sites.size == 0 and out.region_mask is None
    for k in ("sites", "velocity_coords", "velocity_values_phys", "velocity_values_norm", "mask"):
        assert np.array_equal(getattr(out.obs, k), getattr(obs, k))


def test_zero_sigma_is_bit_exact_identity_on_values(spec, scale):
    obs = make_obs()
    zero = ni.CellSpec("N0", 0.0, None)
    out = degrade_observation(obs, zero, spec, scale=scale, sim_id=SIM, frame_index=3,
                              replica_id=1, x_m=X, y_m=Y)
    assert np.array_equal(out.obs.velocity_values_norm, obs.velocity_values_norm)
    assert np.array_equal(out.obs.velocity_values_phys, obs.velocity_values_phys)
    assert np.array_equal(out.obs.sites, obs.sites)


def test_noise_conversion_both_components(spec, scale):
    obs = make_obs()
    out = run(obs, spec, scale, "N1", fr=5, rep=2)
    z = ni.standardized_noise(spec.degradation_seed, SIM, 5, 2)[obs.sites]
    noise = 0.5 * z
    assert np.allclose(out.obs.velocity_values_phys - obs.velocity_values_phys, noise, atol=1e-12)
    # explicit (phys + noise - mean) / std, per component, with the frozen stats
    mean, std = np.array(scale.mean), np.array(scale.std)
    expect = (obs.velocity_values_phys + noise - mean) / std
    assert np.allclose(out.obs.velocity_values_norm, expect, atol=1e-12)
    # the same physical sigma weighs twice as much on vi than vj in normalized units
    d = out.obs.velocity_values_norm - obs.velocity_values_norm
    ratio = np.std(d[:, 0]) / np.std(d[:, 1])
    assert abs(ratio - std[1] / std[0]) < 0.15
    assert abs(0.5 / std[0] - 0.0881) < 1e-3 and abs(0.5 / std[1] - 0.0420) < 1e-3


def test_empirical_noise_sigma_is_in_ms(spec, scale):
    obs = make_obs(n=4000)
    out = degrade_observation(obs, spec.cells["N2"], spec, scale=scale, sim_id=SIM,
                              frame_index=0, replica_id=0, x_m=X, y_m=Y)
    d = out.obs.velocity_values_phys - obs.velocity_values_phys
    assert np.allclose(d.std(axis=0), 1.5, rtol=0.05)


@pytest.mark.skipif(not STATS.exists(), reason="stats missing")
def test_matches_month2_normalizer_if_constructible(spec, scale):
    try:
        from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
        nz = Month2FitOnlyNormalizer(STATS)
    except Exception as exc:  # protocol files may be absent in a bare checkout
        pytest.skip(f"normalizer not constructible: {exc}")
    v = np.array([[1.0, 2.0], [3.0, -4.0]])
    ni_ = nz.normalize_field("Velocity_i", v[:, 0])
    nj_ = nz.normalize_field("Velocity_j", v[:, 1])
    mine = (v - np.array(scale.mean)) / np.array(scale.std)
    assert np.allclose(np.stack([ni_, nj_], 1), mine, atol=1e-12)


# ------------------------------------------------------------------ determinism / coupling

def test_determinism_and_replica_variation(spec, scale):
    obs = make_obs()
    a, b = run(obs, spec, scale, "N1G1", 7, 1), run(obs, spec, scale, "N1G1", 7, 1)
    assert np.array_equal(a.obs.sites, b.obs.sites)
    assert np.array_equal(a.obs.velocity_values_norm, b.obs.velocity_values_norm)
    c = run(obs, spec, scale, "N1G1", 7, 2)
    assert not np.array_equal(a.obs.sites, c.obs.sites)
    d = run(obs, spec, scale, "N1G1", 8, 1)
    assert not np.array_equal(a.obs.sites, d.obs.sites)


def test_n1_and_n1g1_share_noise_on_survivors(spec, scale):
    obs = make_obs()
    n1, n1g1, g1 = (run(obs, spec, scale, c, 4, 1) for c in ("N1", "N1G1", "G1"))
    pos = np.searchsorted(obs.sites, n1g1.obs.sites)
    assert np.array_equal(n1.obs.sites[pos], n1g1.obs.sites)
    assert np.array_equal(n1.obs.velocity_values_norm[pos], n1g1.obs.velocity_values_norm)
    assert np.array_equal(n1.obs.velocity_values_phys[pos], n1g1.obs.velocity_values_phys)
    # same removed sites for G1 and N1G1; G1 values untouched
    assert np.array_equal(g1.removed_sites, n1g1.removed_sites)
    assert np.array_equal(g1.obs.velocity_values_norm, obs.velocity_values_norm[pos])


def test_n2_is_scaled_n1_same_realization(spec, scale):
    obs = make_obs()
    n1, n2 = run(obs, spec, scale, "N1", 2, 0), run(obs, spec, scale, "N2", 2, 0)
    d1 = n1.obs.velocity_values_phys - obs.velocity_values_phys
    d2 = n2.obs.velocity_values_phys - obs.velocity_values_phys
    assert np.allclose(d2, 3.0 * d1, atol=1e-12)


def test_g1_eq_same_count_and_different_sites(spec, scale):
    obs = make_obs()
    for fr in range(6):
        g1, eq = run(obs, spec, scale, "G1", fr, 0), run(obs, spec, scale, "G1-eq", fr, 0)
        assert eq.meta["n_removed"] == g1.meta["n_removed"] > 0
        assert eq.obs.n_sites == g1.obs.n_sites
        assert not np.array_equal(eq.removed_sites, g1.removed_sites)
        assert eq.region_mask is None
        assert np.array_equal(eq.obs.velocity_values_norm, obs.velocity_values_norm[np.isin(obs.sites, eq.obs.sites)])


def test_g1_eq_removal_is_not_contiguous_like_g1(spec, scale):
    """Descriptive sanity: G1 removed sites cluster; uniform removal does not (nearest-removed distance)."""
    obs = make_obs()
    def mean_nn(rem):
        yx = np.stack([rem // 128 * 0.758, rem % 128 * 0.581], 1)
        d = np.linalg.norm(yx[:, None] - yx[None], axis=2); np.fill_diagonal(d, np.inf)
        return d.min(1).mean()
    g = [mean_nn(run(obs, spec, scale, "G1", f, 0).removed_sites) for f in range(5)]
    e = [mean_nn(run(obs, spec, scale, "G1-eq", f, 0).removed_sites) for f in range(5)]
    assert np.mean(g) < np.mean(e)


# ------------------------------------------------------------------ alignment / validity

def test_alignment_of_coordinates_values_validity(spec, scale):
    obs = make_obs()
    out = run(obs, spec, scale, "N1G1", 9, 0).obs
    pos = np.searchsorted(obs.sites, out.sites)
    assert np.array_equal(obs.sites[pos], out.sites)
    assert np.array_equal(out.velocity_coords, obs.velocity_coords[pos])
    assert out.n_sites == out.sites.size == out.velocity_values_norm.shape[0] == out.velocity_coords.shape[0]
    assert np.array_equal(np.flatnonzero(out.mask), out.sites)
    ys, xs = out.sites // 128, out.sites % 128
    assert np.allclose(out.velocity_coords[:, 0], X[xs]) and np.allclose(out.velocity_coords[:, 1], Y[ys])
    assert out.mask_hash != obs.mask_hash and out.velocity_mask_seed == obs.velocity_mask_seed


def test_unet_and_glu_views_have_no_filling(spec, scale):
    obs = make_obs()
    deg = run(obs, spec, scale, "N1G1", 1, 0)
    xs = np.zeros((128, 128), np.float32)
    grid = deg.obs.to_unet_grid(x_scaled=xs, y_scaled=xs, nh3_norm=0.1, phi_norm=0.2)
    removed = np.zeros(128 * 128, bool); removed[deg.removed_sites] = True
    removed = removed.reshape(128, 128)
    assert deg.removed_sites.size > 0
    assert np.all(grid[0][removed] == 0) and np.all(grid[1][removed] == 0) and np.all(grid[2][removed] == 0)
    # unobserved cells that were never sites are also 0 / Mv=0
    assert grid[2].sum() == deg.obs.n_sites
    assert np.all(grid[2][np.unravel_index(deg.obs.sites, (128, 128))] == 1)
    tok = deg.obs.to_glu_tokens()
    assert tok["site_index"].size == deg.obs.n_sites and not np.isin(deg.removed_sites, tok["site_index"]).any()
    assert tok["values_norm"].shape == (deg.obs.n_sites, 2)


# ------------------------------------------------------------------ no T / independence

def test_no_temperature_anywhere():
    banned = {"t", "t_k", "t_norm", "temperature", "target_t", "target_t_k", "target_t_norm"}
    for fn in (ni.degrade_observation, ni.g1_region, ni.g1_remove, ni.g1_eq_removed, ni.standardized_noise):
        assert not banned & {n.lower() for n in inspect.signature(fn).parameters}
    src = inspect.getsource(ni)
    import re
    assert not re.search(r"\b(target_t\w*|t_norm|t_k)\b", src.lower())


def test_region_independent_of_values(spec, scale):
    a, b = make_obs(seed=1), make_obs(seed=1)
    b2 = build_canonical_observation(
        sites=b.sites, vi_phys=np.zeros((128, 128)), vj_phys=np.ones((128, 128)),
        vi_norm=np.zeros((128, 128)), vj_norm=np.ones((128, 128)), x_coords=X, y_coords=Y,
        site_fraction=0.05, layout_id="random_sparse_primary", velocity_mask_seed=1,
    )
    ra, rb = run(a, spec, scale, "G1", 3, 0), run(b2, spec, scale, "G1", 3, 0)
    assert np.array_equal(ra.region_mask, rb.region_mask)
    assert np.array_equal(ra.removed_sites, rb.removed_sites)


# ------------------------------------------------------------------ G1 properties

def test_g1_area_bounds_and_border_clipping(spec):
    for fr in range(10):
        r = ni.g1_region(spec.g1, spec_seed=spec.degradation_seed, sim_id=SIM, frame_index=fr,
                         replica_id=0, x_m=X, y_m=Y)
        assert r.n_cells >= int(np.ceil(0.25 * 128 * 128))
        assert r.n_cells < 0.25 * 128 * 128 * 1.05  # overshoot at most one disc
        assert r.mask.shape == (128, 128) and r.mask.dtype == bool
        assert abs(r.area_fraction_effective - r.n_cells / 16384) < 1e-12


def test_g1_discs_are_within_diameter_range(spec):
    p = spec.g1
    # single-disc case: tiny target so the loop stops after one disc
    tiny = ni.G1Params(area_fraction=1e-4, diameter_min_mm=p.diameter_min_mm,
                       diameter_max_mm=p.diameter_max_mm, max_attempts=3, max_discs=5,
                       min_surviving_sites=16)
    for fr in range(20):
        r = ni.g1_region(tiny, spec_seed=1, sim_id=SIM, frame_index=fr, replica_id=0, x_m=X, y_m=Y)
        assert r.n_discs == 1
        ys, xs = np.nonzero(r.mask)
        ext_x = (xs.max() - xs.min() + 1) * 0.581
        ext_y = (ys.max() - ys.min() + 1) * 0.758
        assert max(ext_x, ext_y) <= p.diameter_max_mm + 1.0


def test_g1_rejection_and_failure_modes(spec):
    sites = np.arange(10, dtype=np.int64) * 7  # 10 sites: survivors can never reach 16
    with pytest.raises(DegradationError, match="no acceptable region"):
        ni.g1_remove(spec.g1, sites, spec_seed=1, sim_id=SIM, frame_index=0, replica_id=0, x_m=X, y_m=Y)
    capped = ni.G1Params(0.25, 3.0, 10.0, 3, 2, 16)
    with pytest.raises(DegradationError, match="max_discs"):
        ni.g1_region(capped, spec_seed=1, sim_id=SIM, frame_index=0, replica_id=0, x_m=X, y_m=Y)
    with pytest.raises(DegradationError, match="invalid n_remove"):
        ni.g1_eq_removed(np.arange(50), 0, spec_seed=1, sim_id=SIM, frame_index=0, replica_id=0)
    with pytest.raises(DegradationError, match="invalid n_remove"):
        ni.g1_eq_removed(np.arange(50), 50, spec_seed=1, sim_id=SIM, frame_index=0, replica_id=0)


def test_unsorted_or_empty_sites_rejected(spec, scale):
    obs = make_obs()
    bad = type(obs)(**{**obs.__dict__, "sites": obs.sites[::-1].copy()})
    with pytest.raises(DegradationError, match="sorted"):
        run(bad, spec, scale, "N1")
    small = make_obs(n=12)
    with pytest.raises(DegradationError):
        run(small, spec, scale, "G1")


def test_retry_attempt_changes_seed(spec):
    a = ni.g1_region(spec.g1, spec_seed=1, sim_id=SIM, frame_index=0, replica_id=0, x_m=X, y_m=Y, attempt=0)
    b = ni.g1_region(spec.g1, spec_seed=1, sim_id=SIM, frame_index=0, replica_id=0, x_m=X, y_m=Y, attempt=1)
    assert a.seed != b.seed and not np.array_equal(a.mask, b.mask)


# ------------------------------------------------------------------ namespaces / batching

def test_namespaces_do_not_collide(spec):
    seen = set()
    for fr in range(50):
        for rep in range(3):
            hist = observation_mask_seed(mask_seed=20261116, sim_id=SIM, frame_index=fr,
                                         layout_id="random_sparse_primary", replica_id=rep)
            trn = training_mask_seed(train_mask_seed=20261118, epoch=rep, sim_id=SIM, frame_index=fr,
                                     layout_id="random_sparse_primary")
            seen |= {hist, trn}
            for op in ("noise", "g1", "g1_eq"):
                assert ni.degradation_seed_for(spec.degradation_seed, op, SIM, fr, rep) not in seen
    ops = {ni.degradation_seed_for(1, op, SIM, 0, 0) for op in ("noise", "g1", "g1_eq")}
    assert len(ops) == 3


def test_glu_batching_policy():
    assert glu_batching_policy([819] * 20) == [list(range(8)), list(range(8, 16)), list(range(16, 20))]
    assert glu_batching_policy([600, 610, 605]) == [[0], [1], [2]]
    assert glu_batching_policy([5, 5, 6, 6, 6]) == [[0, 1], [2, 3, 4]]
    assert glu_batching_policy([]) == []


# ------------------------------------------------------------------ real TRAIN geometry and masks

NESTED = DEFAULT_PAYLOAD_ROOT / "train_fit_nested_sites.npz"
GEOM = ROOT / "data/processed/month2_dense_unet_cache_c149/geometry.npz"


@pytest.mark.skipif(not (NESTED.exists() and GEOM.exists()), reason="TRAIN payload not available")
def test_g1_on_train_masks_qualification_subset(spec):
    g = np.load(GEOM)
    d = np.load(NESTED, allow_pickle=True)
    idx = np.arange(0, len(d["sim_id"]), 57)
    for i in idx:
        sites = np.sort(d["sites_5pct"][i].astype(np.int64))
        reg, rem = ni.g1_remove(spec.g1, sites, spec_seed=spec.degradation_seed, sim_id=str(d["sim_id"][i]),
                                frame_index=int(d["frame_index"][i]), replica_id=int(d["replica_id"][i]),
                                x_m=g["x"], y_m=g["y"])
        assert sites.size == 819 and reg.attempt == 0
        assert sites.size - rem.sum() >= spec.g1.min_surviving_sites
        assert 0 < rem.sum() < sites.size


@pytest.mark.skipif(not (NESTED.exists() and GEOM.exists()), reason="TRAIN payload not available")
def test_real_train_frames_identity_and_noise_conversion(spec, scale):
    cache = ROOT / "data/processed/month2_dense_unet_cache_c149/train_fit"
    if not (cache / "vi_norm.npy").exists():
        pytest.skip("train_fit cache not available")
    vi = np.load(cache / "vi_norm.npy", mmap_mode="r")
    vj = np.load(cache / "vj_norm.npy", mmap_mode="r")
    frames = json.loads((cache / "frames.json").read_text())
    d = np.load(NESTED, allow_pickle=True)
    g = np.load(GEOM)
    checked = 0
    for k in (0, 500, 1200):
        meta = frames[k]
        sim, fr = meta["sim_id"], int(meta["frame_index"])
        hit = np.flatnonzero((d["sim_id"] == sim) & (d["frame_index"] == fr) & (d["replica_id"] == 0))
        if hit.size != 1:
            continue
        sites = np.sort(d["sites_5pct"][hit[0]].astype(np.int64))
        vin, vjn = np.asarray(vi[k], np.float64), np.asarray(vj[k], np.float64)
        vip = vin * scale.std[0] + scale.mean[0]
        vjp = vjn * scale.std[1] + scale.mean[1]
        obs = build_canonical_observation(sites=sites, vi_phys=vip, vj_phys=vjp, vi_norm=vin, vj_norm=vjn,
                                          x_coords=g["x"], y_coords=g["y"], site_fraction=0.05,
                                          layout_id="random_sparse_primary", velocity_mask_seed=0)
        ident = degrade_observation(obs, spec.cells["R0"], spec, scale=scale, sim_id=sim,
                                    frame_index=fr, replica_id=0, x_m=g["x"], y_m=g["y"])
        assert ident.obs is obs
        out = degrade_observation(obs, spec.cells["N1G1"], spec, scale=scale, sim_id=sim,
                                  frame_index=fr, replica_id=0, x_m=g["x"], y_m=g["y"])
        pos = np.searchsorted(sites, out.obs.sites)
        dn = out.obs.velocity_values_norm - obs.velocity_values_norm[pos]
        dp = out.obs.velocity_values_phys - obs.velocity_values_phys[pos]
        assert np.allclose(dn * np.array(scale.std), dp, atol=1e-9)
        checked += 1
    assert checked >= 1
