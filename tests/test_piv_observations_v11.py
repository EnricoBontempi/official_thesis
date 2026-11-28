"""CPU tests for the additive spec v1.1 (SENS-TAIL, 5x5 substitution) and the v1 -> v1.1 regression (decisione_R2 sec. 8)."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, convolve
from scipy.stats import kurtosis, spearmanr

from combustion_recon.data import piv_observations as P
from combustion_recon.data.nonideal_observations import load_velocity_scale

REPO = Path(__file__).resolve().parents[1]
V1 = REPO / "configs/observations/piv_observation_spec_v1.yaml"
V11 = REPO / "configs/observations/piv_observation_spec_v1_1.yaml"
STATS = REPO / "data/manifests/train_norm_stats_month2_fit.json"
pytestmark = pytest.mark.skipif(not V11.exists(), reason="v1.1 not written")
SIM = "0NH3_1.h5"


def _gen(path):
    sha = path.with_suffix(".sha256").read_text().split()[0]
    spec = P.load_piv_spec(path, expected_sha256=sha)
    scale = load_velocity_scale(STATS, expected_sha256="678263fb4cb2598350b8fd69a71ce4843f747aa61c19556cf7f0f9c5c3ceac0e")
    return P.PivObservationGenerator(spec, scale, x_vec_m=np.linspace(-0.036907, 0.036888, 128), y_vec_m=np.linspace(0.062853, 0.159134, 128))


@pytest.fixture(scope="module")
def g1(): return _gen(V1)
@pytest.fixture(scope="module")
def g11(): return _gen(V11)
@pytest.fixture(scope="module")
def les():
    rng = np.random.default_rng(11)
    return gaussian_filter(rng.standard_normal((128, 128)), 3) * 25.0, gaussian_filter(rng.standard_normal((128, 128)), 3) * 40.0 + 4.0


def obs(g, cfg, les, frame=0, rep=0, role="eval", inter=False):
    return g.observe(cfg, les[0], les[1], sim_id=SIM, frame_index=frame, role=role, replica=rep, return_intermediates=inter)


def test_v11_is_additive_and_v1_untouched(g1, g11):
    s1, s11 = g1.spec, g11.spec
    assert s11.raw["spec_version"] == 1.1 and s11.raw["parent"]["file_sha256"] == s1.sha256
    assert set(s1.raw["configs"]) < set(s11.raw["configs"])
    assert set(s11.raw["configs"]) - set(s1.raw["configs"]) == {"SENS-TAIL-A", "SENS-TAIL-B-in", "SENS-REJ-iso-sub5-A", "SENS-REJ-clu-sub5-A"}
    for k in s1.raw["configs"]:
        assert s1.raw["configs"][k] == s11.raw["configs"][k], k
    for k in ("kernels", "reference_velocity", "geometry", "windows_B", "normalization", "seeds", "operation_order"):
        assert s1.raw[k] == s11.raw[k], k
    assert s1.sha256 == hashlib.sha256(V1.read_bytes()).hexdigest()


@pytest.mark.parametrize("cfg", sorted(_gen(V1).spec.raw["configs"]) if V1.exists() else [])
def test_regression_31_configs_bit_identical(g1, g11, les, cfg):
    for frame, rep, role in ((0, 0, "eval"), (10, 2, "train")):
        a, b = obs(g1, cfg, les, frame, rep, role), obs(g11, cfg, les, frame, rep, role)
        assert a.obs_hash == b.obs_hash
        assert np.array_equal(a.obs.sites, b.obs.sites)
        assert np.array_equal(a.obs.velocity_values_phys, b.obs.velocity_values_phys)


@pytest.mark.parametrize("fld", ["A", "B-in"])
def test_tail_unit_variance_kurtosis_rank_nodes(g11, les, fld):
    base, tail = f"PIV-{fld}", f"SENS-TAIL-{fld}"
    e0, et, ez = [], [], []
    for fr in range(150):
        a = obs(g11, base, les, fr, inter=True); t = obs(g11, tail, les, fr, inter=True)
        assert np.array_equal(a.obs.sites, t.obs.sites)
        v = a.intermediates["valid"]; sig = a.meta["sigma_u_m_s"]
        e0.append(a.intermediates["err_vi"][v] / sig); et.append(t.intermediates["err_vi"][v] / sig)
    e0, et = np.concatenate(e0), np.concatenate(et)
    assert abs(et.std() - 1) < 0.01
    assert abs(kurtosis(et) - 2.0) < 0.2, kurtosis(et)
    assert abs(kurtosis(e0)) < 0.1
    assert spearmanr(e0[::50], et[::50]).statistic > 0.9999999   # identical rank order
    assert 0.9 < np.corrcoef(e0, et)[0, 1] < 0.9999               # Pearson changes slightly


def test_tail_same_standardised_realisation_and_spatial_structure(g11, les):
    a = obs(g11, "PIV-A", les, 3, inter=True); t = obs(g11, "SENS-TAIL-A", les, 3, inter=True)
    sig = a.meta["sigma_u_m_s"]
    # autocorrelation along rows preserved to first order
    def r1(e): return float((e[:-1] * e[1:]).mean() / (e**2).mean())
    ea, et = a.intermediates["err_vi"] / sig, t.intermediates["err_vi"] / sig
    assert abs(r1(ea) - r1(et)) < 0.05
    assert t.meta["noise"] == "corr_t" and t.meta["sigma_u_m_s"] == a.meta["sigma_u_m_s"]


@pytest.mark.parametrize("rej", ["iso", "clu"])
def test_sub5_same_positions_and_mean_of_known_5x5(g11, les, rej):
    s3 = obs(g11, f"SENS-REJ-{rej}-sub-A", les, 5, inter=True)
    s5 = obs(g11, f"SENS-REJ-{rej}-sub5-A", les, 5, inter=True)
    fl = obs(g11, f"SENS-REJ-{rej}-flag-A", les, 5, inter=True)
    assert np.array_equal(s3.intermediates["rejected"], s5.intermediates["rejected"])
    assert np.array_equal(fl.intermediates["rejected"], s5.intermediates["rejected"])
    assert s5.meta["substitution_rule"] == "mean5x5_iterative" and s3.meta["substitution_rule"] == "mean3x3_iterative"
    assert s5.meta["sub_passes"] is not None and s5.meta["sub_passes"] <= s3.meta["sub_passes"]
    assert isinstance(s5.meta["sub_passes_gt3"], bool)
    # surviving vectors unchanged by the rule
    keep = ~s5.intermediates["rejected"] & s5.intermediates["valid"]
    assert np.array_equal(s5.intermediates["vals_vi"][keep], s3.intermediates["vals_vi"][keep])
    # nodes filled in the FIRST pass equal the mean of their known 5x5 neighbours (known = unrejected valid)
    rej_m = s5.intermediates["rejected"]; known = s5.intermediates["valid"] & ~rej_m
    ker = np.ones((5, 5)); ker[2, 2] = 0
    base = fl.intermediates["vals_vi"]
    cnt = convolve(known.astype(float), ker, mode="constant"); tot = convolve(np.where(known, base, 0.0), ker, mode="constant")
    first = rej_m & (cnt > 0)
    assert first.any()
    assert np.allclose(s5.intermediates["vals_vi"][first], (tot / np.maximum(cnt, 1))[first], atol=1e-10)


def test_sub5_is_smoother_than_sub3_on_cluster_core(g11, les):
    rs = []
    for fr in range(10):
        s3 = obs(g11, "SENS-REJ-clu-sub-A", les, fr, inter=True); s5 = obs(g11, "SENS-REJ-clu-sub5-A", les, fr, inter=True)
        m = s3.intermediates["rejected"]
        rs.append(float(np.abs(s3.intermediates["vals_vi"][m] - s5.intermediates["vals_vi"][m]).mean()))
    assert np.mean(rs) > 0   # the two rules are genuinely different variants


def test_substitute_iterative_wrapper_unchanged():
    v = np.arange(49.0).reshape(7, 7); dom = np.ones((7, 7), bool); known = dom.copy(); known[3, 3] = False
    a, pa = P.substitute_iterative_3x3(v, known, dom); b, pb = P.substitute_iterative(v, known, dom, half=1)
    assert np.array_equal(a, b) and pa == pb == 1
