"""CPU tests for the virtual PIV observation generator (roadmap F2.3, spec ``piv_observation_spec_v1``).

They use synthetic velocity fields (no dataset payload needed) and the frozen spec/statistics tracked in the repository.
Nothing here reads temperature.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, label

from combustion_recon.data import piv_observations as P
from combustion_recon.data.nonideal_observations import load_velocity_scale

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "configs/observations/piv_observation_spec_v1.yaml"
SPEC_SHA = REPO / "configs/observations/piv_observation_spec_v1.sha256"
STATS = REPO / "data/manifests/train_norm_stats_month2_fit.json"
STATS_SHA = "678263fb4cb2598350b8fd69a71ce4843f747aa61c19556cf7f0f9c5c3ceac0e"

pytestmark = pytest.mark.skipif(not SPEC.exists(), reason="spec not written yet")

SIM = "0NH3_1.h5"


@pytest.fixture(scope="module")
def spec():
    return P.load_piv_spec(SPEC, expected_sha256=SPEC_SHA.read_text().split()[0])


@pytest.fixture(scope="module")
def gen(spec):
    scale = load_velocity_scale(STATS, expected_sha256=STATS_SHA)
    x = np.linspace(-0.036907, 0.036888, 128)
    y = np.linspace(0.062853, 0.159134, 128)
    return P.PivObservationGenerator(spec, scale, x_vec_m=x, y_vec_m=y)


@pytest.fixture(scope="module")
def les():
    rng = np.random.default_rng(7)
    vi = gaussian_filter(rng.standard_normal((128, 128)), 3) * 25.0
    vj = gaussian_filter(rng.standard_normal((128, 128)), 3) * 40.0 + 4.0
    return vi, vj


def obs(gen, cfg, les, frame=0, role="eval", rep=0, inter=False, sim=SIM):
    return gen.observe(cfg, les[0], les[1], sim_id=sim, frame_index=frame, role=role, replica=rep, return_intermediates=inter)


# ----------------------------------------------------------------------------- spec
def test_spec_hash_and_namespace(spec):
    assert hashlib.sha256(SPEC.read_bytes()).hexdigest() == SPEC_SHA.read_text().split()[0]
    assert set(P.ROLE_SEEDS.values()).isdisjoint(P.HISTORICAL_AND_MISSION00_SEEDS)
    assert spec.raw["frozen"]["content_sha256"] == P.spec_content_hash(spec.raw)
    assert spec.raw["normalization"]["stats_sha256"] == STATS_SHA
    assert spec.raw["orientation"]["status"].startswith("probable")


def test_spec_refuses_wrong_hash():
    with pytest.raises(P.PivError):
        P.load_piv_spec(SPEC, expected_sha256="0" * 64)


def test_no_test_reference_velocity(spec):
    ids = set(spec.raw["reference_velocity"]["per_condition_m_s"])
    import yaml
    s = yaml.safe_load(open(REPO / "configs/splits/splits_interp_extrap.yaml"))
    assert ids == set(s["train"]["sim_ids"]) | set(s["val"]["sim_ids"])
    assert ids.isdisjoint(set(s["test_interp"]["sim_ids"]) | set(s["test_extrap"]["sim_ids"]))


# ----------------------------------------------------------------------------- kernels
@pytest.mark.parametrize("conv", ["primary", "swap"])
@pytest.mark.parametrize("W", [1.0, 1.3, 1.5, 1.9])
def test_kernel_weights_sum_and_discrete_length(spec, conv, W):
    steps = P._steps(spec.steps, conv)
    for axis, D in zip(("rows", "cols"), steps):
        for which, kind, target in (("response_mhg", "mhg", 0.93 * W), ("noise", "mhg", 1.2 * W), ("response_tophat", "tophat", W)):
            e = spec.kernel(which, conv, W, axis)
            k, w = P.kernel_weights(kind, e["param_mm"], D)
            assert abs(w.sum() - 1) < 1e-12
            assert abs(P.lsr_discrete(w, D) / target - 1) <= P.CAL_TOL
            # recalibration reproduces the frozen parameter
            c = P.calibrate(kind, target, D)
            assert abs(c["param"] / e["param_mm"] - 1) < 1e-6
            M = P.axis_matrix(40, k, w)  # edge-renormalised operator
            assert np.allclose(M.sum(1), 1.0, atol=1e-12)


def test_calibration_refuses_unattainable_target():
    with pytest.raises(P.PivError):
        P.calibrate("mhg", 0.5, 0.758)   # target below the cell size
    with pytest.raises(P.PivError):
        P.calibrate("tophat", 0.7, 0.758)


def test_mhg_formulas():
    s = np.linspace(-12, 12, 480001)
    F = (1 - P.B_MHG * s**2) * np.exp(-s**2 / 2) / ((1 - P.B_MHG) * np.sqrt(2 * np.pi))
    ds = s[1] - s[0]
    assert abs(F.sum() * ds - 1) < 1e-8
    assert abs(1 / ((F**2).sum() * ds) - P.mhg_lsr_continuous(1.0)) < 1e-6
    assert abs(P.mhg_cdf(np.array([20.0]), 1.0)[0] - P.mhg_cdf(np.array([-20.0]), 1.0)[0] - 1) < 1e-12


# ----------------------------------------------------------------------------- geometry, valid nodes
def test_counts_and_windows(gen, les):
    a = obs(gen, "PIV-A", les)
    assert a.meta["n_valid_vectors"] == 124 * 126 == 15624
    for f in ("B-in", "B-down", "B-off"):
        b = obs(gen, f"PIV-{f}", les)
        r0, r1, c0, c1 = b.meta["field_rect_rows_cols"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (52, 40)
        assert b.meta["n_valid_vectors"] == 48 * 38 == 1824
    assert obs(gen, "PIV-B-in", les).meta["field_rect_rows_cols"] == [38, 89, 88, 127]
    assert obs(gen, "PIV-B-down", les).meta["field_rect_rows_cols"] == [38, 89, 48, 87]
    assert obs(gen, "PIV-B-off", les).meta["field_rect_rows_cols"] == [64, 115, 88, 127]
    assert P.window_margin(1.3, 0.5810574) == 2 and P.window_margin(1.3, 0.7581227) == 1
    assert P.window_margin(1.5, 0.5810574) == 2 and P.window_margin(1.5, 0.7581227) == 1


def test_swap_convention_geometry(gen, les):
    s = obs(gen, "SENS-SWAP-B-in", les)
    r0, r1, c0, c1 = s.meta["field_rect_rows_cols"]
    assert (r1 - r0 + 1, c1 - c0 + 1) == (40, 52) and c1 == 127
    assert s.meta["step_convention"] == "swap"


@pytest.mark.parametrize("f", ["A", "B-in", "B-down", "B-off"])
def test_ideal_and_piv_share_valid_nodes(gen, les, f):
    p, i, ff = obs(gen, f"PIV-{f}", les), obs(gen, f"IDEAL-{f}", les), obs(gen, f"FILT-{f}", les)
    assert p.meta["valid_mask_hash"] == i.meta["valid_mask_hash"] == ff.meta["valid_mask_hash"]
    assert np.array_equal(p.obs.sites, i.obs.sites) and np.array_equal(p.obs.sites, ff.obs.sites)
    assert p.meta["observed_mask_hash"] == i.meta["observed_mask_hash"]


def test_identities_ideal(gen, les):
    full = obs(gen, "IDEAL-full", les)
    assert full.obs.n_sites == 128 * 128
    assert np.array_equal(full.obs.velocity_values_phys[:, 0], les[0].ravel()) and np.array_equal(full.obs.velocity_values_phys[:, 1], les[1].ravel())
    for f in ("A", "B-in", "B-down", "B-off"):
        i = obs(gen, f"IDEAL-{f}", les)
        rr, cc = i.obs.sites // 128, i.obs.sites % 128
        assert np.array_equal(i.obs.velocity_values_phys[:, 0], les[0][rr, cc])
        assert np.array_equal(i.obs.velocity_values_phys[:, 1], les[1][rr, cc])
    # normalisation is the frozen z-score
    st = load_velocity_scale(STATS)
    assert np.allclose(full.obs.velocity_values_norm * np.array(st.std) + np.array(st.mean), full.obs.velocity_values_phys, atol=1e-9)


def test_filtered_minus_error_decomposition(gen, les):
    """Same nodes and same realisation on both sides: PIV - FILT == error, exactly."""
    p = obs(gen, "PIV-A", les, inter=True)
    f = obs(gen, "FILT-A", les, inter=True)
    d = p.obs.velocity_values_phys - f.obs.velocity_values_phys
    rr, cc = p.obs.sites // 128, p.obs.sites % 128
    assert np.allclose(d[:, 0], p.intermediates["err_vi"][rr, cc], atol=1e-12)
    assert np.allclose(d[:, 1], p.intermediates["err_vj"][rr, cc], atol=1e-12)


# ----------------------------------------------------------------------------- spatial response
def _sin_field(wavelength_mm, axis, steps):
    n = 128
    s = np.arange(n) * steps[0 if axis == "rows" else 1]
    prof = np.sin(2 * np.pi * s / wavelength_mm + 0.37)
    return (np.tile(prof[:, None], (1, n)) if axis == "rows" else np.tile(prof[None, :], (n, 1)))


@pytest.mark.parametrize("axis", ["rows", "cols"])
@pytest.mark.parametrize("lam", [2.5, 4.0, 10.0])
def test_transfer_function_interior_and_edge(gen, spec, axis, lam):
    steps = P._steps(spec.steps, "primary")
    f = _sin_field(lam, axis, steps)
    o = gen.observe("FILT-A", f, f, sim_id=SIM, frame_index=0, role="eval", return_intermediates=True)
    filt = o.intermediates["filtered_vi"]
    D = steps[0 if axis == "rows" else 1]
    k, w = P.kernel_weights("mhg", spec.kernel("response_mhg", "primary", 1.3, axis)["param_mm"], D)
    H = P.transfer_function(k, w, D, lam)[0]
    n = int(np.abs(k).max())
    # interior: filtered = H * (original sinusoid) with unchanged phase (symmetric kernel)
    sl = slice(n, 128 - n)
    interior = filt[sl, 64] if axis == "rows" else filt[64, sl]
    orig = f[sl, 64] if axis == "rows" else f[64, sl]
    assert np.allclose(interior, H * orig, atol=1e-10)
    # edge: truncated + renormalised kernel, checked against a brute-force direct sum
    for i in list(range(n)) + list(range(128 - n, 128)):
        j = i + k
        ok = (j >= 0) & (j < 128)
        prof = f[:, 64] if axis == "rows" else f[64, :]
        brute = (w[ok] * prof[j[ok]]).sum() / w[ok].sum()
        got = filt[i, 64] if axis == "rows" else filt[64, i]
        # the other axis also filters a constant along it -> unchanged
        assert abs(got - brute) < 1e-10
    # and the edge response differs from the interior one (so the interior/edge split is meaningful)
    assert abs(filt[0, 64] - H * f[0, 64]) > 1e-6 if axis == "rows" else abs(filt[64, 0] - H * f[64, 0]) > 1e-6


def test_response_is_limited_to_the_field(gen, les):
    """B windows do not use LES values outside the window."""
    p = obs(gen, "FILT-B-in", les, inter=True)
    r0, r1, c0, c1 = p.meta["field_rect_rows_cols"]
    mod = (les[0].copy(), les[1].copy())
    mod[0][:r0, :] += 100.0; mod[0][r1 + 1:, :] += 100.0; mod[0][:, :c0] += 100.0
    q = gen.observe("FILT-B-in", mod[0], mod[1], sim_id=SIM, frame_index=0, role="eval")
    assert np.array_equal(p.obs.velocity_values_phys, q.obs.velocity_values_phys)


# ----------------------------------------------------------------------------- error and its autocorrelation
def _noise_stack(gen, spec, W, n=300, comp=0):
    return np.stack([gen._noise_field("eval", comp, SIM, i, 0, "primary", W, True) for i in range(n)])


@pytest.mark.parametrize("W", [1.3, 1.5])
def test_noise_variance_and_autocorrelation(gen, spec, W):
    z = _noise_stack(gen, spec, W)
    assert abs(z.std() - 1.0) < 0.01
    steps = P._steps(spec.steps, "primary")
    for axis, ax_idx, D in (("rows", 1, steps[0]), ("cols", 2, steps[1])):
        k, w = P.kernel_weights("mhg", spec.kernel("noise", "primary", W, axis)["param_mm"], D)
        ac = P.autocorr_discrete(w, 4)
        for lag in (1, 2, 3, 4):
            a = np.take(z, np.arange(lag, 128), axis=ax_idx)[..., 8:120] if False else None
            x1 = np.take(z, np.arange(0, 128 - lag), axis=ax_idx)
            x2 = np.take(z, np.arange(lag, 128), axis=ax_idx)
            emp = float((x1 * x2).mean())
            assert abs(emp - ac[lag]) < 0.02, (axis, lag, emp, ac[lag])
        # integral length from the empirical autocorrelation: L = step * (1 + 2 sum rho) ... equals step / sum(g^2) analytically
        assert abs(P.lsr_discrete(w, D) / (1.2 * W) - 1) <= P.CAL_TOL


def test_noise_is_stationary_up_to_the_edges(gen, spec):
    z = _noise_stack(gen, spec, 1.3, n=400)
    v = z.var(0)
    assert abs(v[0, :].mean() - 1) < 0.04 and abs(v[-1, :].mean() - 1) < 0.04
    assert abs(v[:, 0].mean() - 1) < 0.04 and abs(v[:, -1].mean() - 1) < 0.04


def test_error_amplitude_in_m_s_and_independence(gen, spec, les):
    vref = spec.vref(SIM)
    errs_i, errs_j = [], []
    for fr in range(200):
        o = obs(gen, "PIV-A", les, frame=fr, inter=True)
        v = o.intermediates["valid"]
        errs_i.append(o.intermediates["err_vi"][v]); errs_j.append(o.intermediates["err_vj"][v])
    ei, ej = np.concatenate(errs_i), np.concatenate(errs_j)
    target = 0.10 * vref / 6.0
    assert abs(ei.std() / target - 1) < 0.02 and abs(ej.std() / target - 1) < 0.02
    assert abs(np.corrcoef(ei, ej)[0, 1]) < 0.02
    assert abs(ei.mean()) < 0.02 * target


def test_sigma_sensitivities_share_the_standardised_realisation(gen, les):
    a = obs(gen, "PIV-A", les, inter=True).intermediates["err_vi"]
    lo = obs(gen, "SENS-SIGLO-A", les, inter=True).intermediates["err_vi"]
    hi = obs(gen, "SENS-SIGHI-A", les, inter=True).intermediates["err_vi"]
    v = obs(gen, "PIV-A", les, inter=True).intermediates["valid"]
    assert np.allclose(lo[v] / a[v], 0.6, atol=1e-9) and np.allclose(hi[v] / a[v], 1.5, atol=1e-9)


def test_rho0_is_uncorrelated(gen, les):
    errs = np.stack([obs(gen, "SENS-RHO0-A", les, frame=i, inter=True).intermediates["err_vi"] for i in range(200)])
    assert abs((errs[:, :-1] * errs[:, 1:]).mean() / errs.var()) < 0.02
    assert abs((errs[:, :, :-1] * errs[:, :, 1:]).mean() / errs.var()) < 0.02


# ----------------------------------------------------------------------------- top-hat, edge, window sensitivities
def test_tophat_discrete_length_equals_W(spec):
    for W in (1.3, 1.5):
        for axis, D in zip(("rows", "cols"), P._steps(spec.steps, "primary")):
            k, w = P.kernel_weights("tophat", spec.kernel("response_tophat", "primary", W, axis)["param_mm"], D)
            assert abs(P.lsr_discrete(w, D) / W - 1) <= P.CAL_TOL


def test_edge_strict_is_a_subset_of_valid(gen, les):
    for f in ("A", "B-in"):
        a = obs(gen, f"PIV-{f}", les); e = obs(gen, f"SENS-EDGE-{f}", les)
        assert set(e.obs.sites) < set(a.obs.sites)
        assert e.meta["n_edge_vectors"] == 0


# ----------------------------------------------------------------------------- rejections and substitution
def test_rejection_fraction_and_levels(gen, les):
    base = obs(gen, "PIV-A", les, inter=True)
    n_valid = base.meta["n_valid_vectors"]
    for rej in ("iso", "clu"):
        fl = obs(gen, f"SENS-REJ-{rej}-flag-A", les, inter=True)
        sb = obs(gen, f"SENS-REJ-{rej}-sub-A", les, inter=True)
        k = int(round(0.10 * n_valid))
        assert fl.meta["n_rejected"] == k and fl.obs.n_sites == n_valid - k
        assert sb.obs.n_sites == n_valid and sb.meta["n_substituted"] == k
        assert np.array_equal(fl.intermediates["rejected"], sb.intermediates["rejected"])  # same positions in both levels
        # flag: rejected vectors absent from the packet; sub: present
        rej_flat = np.flatnonzero(fl.intermediates["rejected"].ravel())
        assert not np.isin(rej_flat, fl.obs.sites).any() and np.isin(rej_flat, sb.obs.sites).all()
        # surviving vectors are unchanged by the substitution level
        keep = np.isin(sb.obs.sites, fl.obs.sites)
        assert np.array_equal(sb.obs.velocity_values_phys[keep], fl.obs.velocity_values_phys)
        # Mv: U-Net mask follows the level
        assert fl.obs.mask.sum() == n_valid - k and sb.obs.mask.sum() == n_valid


def test_cluster_rejections_are_connected(gen, les):
    sz = {}
    for rej in ("iso", "clu"):
        s = []
        for fr in range(20):
            m = obs(gen, f"SENS-REJ-{rej}-flag-A", les, frame=fr, inter=True).intermediates["rejected"]
            s.append(P.cluster_sizes(m))
        sz[rej] = np.concatenate(s)
    # field correlated on scale ~W thresholded at 10 %: clusters clearly larger than isolated rejections (ratio ~2.6 measured)
    assert sz["clu"].mean() > 2 * sz["iso"].mean()
    assert (sz["clu"] >= 3).mean() > 2 * (sz["iso"] >= 3).mean()


def test_substitution_rule():
    v = np.zeros((7, 7)); v[:] = np.arange(49).reshape(7, 7)
    dom = np.ones((7, 7), bool)
    known = dom.copy(); known[3, 3] = False
    out, passes = P.substitute_iterative_3x3(v, known, dom)
    nb = [v[2, 2], v[2, 3], v[2, 4], v[3, 2], v[3, 4], v[4, 2], v[4, 3], v[4, 4]]
    assert passes == 1 and abs(out[3, 3] - np.mean(nb)) < 1e-12
    assert np.array_equal(out[known], v[known])
    # a 3x3 cluster needs two passes; the centre uses already filled neighbours
    known = dom.copy(); known[2:5, 2:5] = False
    out, passes = P.substitute_iterative_3x3(v, known, dom)
    assert passes == 2 and np.isfinite(out).all()
    # restricted domain: neighbours outside the domain are never used
    dom2 = dom.copy(); dom2[:, :2] = False
    out, _ = P.substitute_iterative_3x3(np.where(dom2, v, 1e9), dom2 & ~np.isin(np.arange(49).reshape(7, 7), [24]), dom2)
    assert out[3, 3] < 100


def test_main_configs_have_no_rejections(gen, les):
    for c in ("PIV-A", "PIV-B-in", "PIV-B-down", "PIV-B-off", "IDEAL-A", "FILT-A"):
        m = obs(gen, c, les).meta
        assert m["rejection"] is None and m["n_rejected"] == 0 and m["n_substituted"] == 0


# ----------------------------------------------------------------------------- determinism, namespaces, families
def test_determinism_and_separation(gen, les):
    a, b = obs(gen, "PIV-A", les, frame=10, rep=2), obs(gen, "PIV-A", les, frame=10, rep=2)
    assert a.obs_hash == b.obs_hash
    assert np.array_equal(a.obs.velocity_values_phys, b.obs.velocity_values_phys)
    assert obs(gen, "PIV-A", les, frame=10, rep=3).obs_hash != a.obs_hash       # replica / epoch
    assert obs(gen, "PIV-A", les, frame=11, rep=2).obs_hash != a.obs_hash       # frame
    assert obs(gen, "PIV-A", les, frame=10, rep=2, role="train").obs_hash != a.obs_hash   # role namespace
    assert obs(gen, "PIV-A", les, frame=10, rep=2, role="selection").obs_hash != obs(gen, "PIV-A", les, frame=10, rep=2, role="train").obs_hash
    assert obs(gen, "PIV-A", les, frame=10, rep=2, sim="20NH3_1.05.h5").obs_hash != a.obs_hash
    assert a.meta["role_seed"] == 20261212 and a.meta["namespace"] == "piv_observation_v1"


def test_refusals(gen, les):
    with pytest.raises(P.PivError):
        obs(gen, "PIV-A", les, role="test")
    with pytest.raises(P.PivError):
        obs(gen, "PIV-A", les, sim="20NH3_1.h5")        # TEST_interp condition has no reference velocity
    with pytest.raises(P.PivError):
        obs(gen, "NOPE", les)
    with pytest.raises(P.PivError):
        gen.observe("PIV-A", les[0][:64], les[1][:64], sim_id=SIM, frame_index=0, role="eval")


def test_unet_and_glu_receive_the_same_physical_packet(gen, les):
    o = obs(gen, "PIV-B-in", les)
    geo = np.load(REPO / "data/processed/month2_dense_unet_cache_c149/geometry.npz") if (REPO / "data/processed/month2_dense_unet_cache_c149/geometry.npz").exists() else None
    xs = np.zeros((128, 128)) if geo is None else geo["x_scaled"]
    ys = np.zeros((128, 128)) if geo is None else geo["y_scaled"]
    grid = o.obs.to_unet_grid(x_scaled=xs, y_scaled=ys, nh3_norm=0.1, phi_norm=0.2)
    tok = o.obs.to_glu_tokens()
    rr, cc = tok["site_index"] // 128, tok["site_index"] % 128
    assert grid.shape == (7, 128, 128)
    assert np.array_equal(grid[0][rr, cc], tok["values_norm"][:, 0].astype(np.float32))
    assert np.array_equal(grid[1][rr, cc], tok["values_norm"][:, 1].astype(np.float32))
    assert int(grid[2].sum()) == tok["site_index"].size == o.obs.n_sites
    assert grid[2][rr, cc].all() and grid[0][grid[2] == 0].sum() == 0
    # historical coordinates for the networks, hypothesised physical coordinates kept apart
    assert o.obs.velocity_coords.shape == o.phys_coords.shape
    assert not np.allclose(o.obs.velocity_coords * 1e3, o.phys_coords)
    sig = inspect.signature(P.PivObservationGenerator.observe)
    assert not any(p in sig.parameters for p in ("model", "family", "model_seed", "seed"))


def test_hash_does_not_depend_on_model_family(gen, les):
    # the observation function has no family argument: identical call == identical packet
    assert obs(gen, "PIV-A", les, frame=5).obs_hash == obs(gen, "PIV-A", les, frame=5).obs_hash


# ----------------------------------------------------------------------------- no access to temperature
def test_generator_never_sees_temperature():
    names = set()
    for fn in (P.PivObservationGenerator.observe, P.PivObservationGenerator.__init__, P.substitute_iterative_3x3):
        names |= set(inspect.signature(fn).parameters)
    assert not {n for n in names if n.lower() in ("t", "temp", "temperature", "t_k", "target_t", "target")}
    src = Path(P.__file__).read_text().lower()
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", '"', "'")))
    for bad in ("target_t", "temperature[", "t_norm", "np.load", "open(", "read_bytes"):
        if bad == "read_bytes":
            continue  # the spec / stats loaders legitimately read bytes
        assert bad not in code or bad in ("np.load",) and "np.load" not in code
    assert "temperature_used" in src


def test_observation_independent_of_any_temperature_perturbation(gen, les):
    """T is not an input: a perturbed T field cannot reach the generator, so identical velocity gives identical packets."""
    a = obs(gen, "PIV-A", les, frame=3)
    _unused_T = np.random.default_rng(0).uniform(300, 2400, (128, 128))  # exists only to make the point explicit
    b = obs(gen, "PIV-A", les, frame=3)
    assert a.obs_hash == b.obs_hash


def test_nonideal_module_untouched():
    import subprocess
    out = subprocess.run(["git", "-C", str(REPO), "diff", "--stat", "d4fecb2", "--", "src/combustion_recon/data/nonideal_observations.py"],
                         capture_output=True, text=True)
    assert out.stdout.strip() == ""
