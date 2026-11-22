from __future__ import annotations

import numpy as np
import pytest
import torch

from combustion_recon.models import glu_assisted as GA
from combustion_recon.models import glu_reconstructor as G
from combustion_recon.models.unet import ASSISTED_CHANNEL_ORDER, ASSISTED_P_CHANNELS, build_sparse_unet_assisted_from_config

H = W = 128


@pytest.fixture(scope="module")
def geom(tmp_path_factory):
    x = -0.0369 + 0.000581 * np.arange(W)
    y = 0.0628 + 0.000758 * np.arange(H)
    p = tmp_path_factory.mktemp("g") / "geometry.npz"
    np.savez(p, x=x, y=y, x_scaled=np.linspace(-1, 1, W), y_scaled=np.linspace(-1, 1, H))
    return p, x, y


def _batch(x, y, B, K, seed=0, N=64):
    g = torch.Generator().manual_seed(seed)
    sites = torch.stack([torch.randperm(H * W, generator=g)[:N] for _ in range(B)])
    tsite = torch.stack([torch.randperm(H * W, generator=g)[:K] for _ in range(B)]) if K else torch.zeros(B, 0, dtype=torch.long)
    xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def co(s):
        return torch.stack([xt[s % W], yt[s // W]], dim=-1)

    vals = torch.randn(B, N, 2, generator=g)
    tv = torch.randn(B, K, 1, generator=g)
    p = torch.randn(B, 2, generator=g)
    xx, yy = torch.meshgrid(xt, yt, indexing="xy")
    Y = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1).unsqueeze(0).expand(B, -1, -1).contiguous()
    return co(sites), vals, sites, p, Y, co(tsite), tv, tsite


def test_unet_assisted_has_named_channels_and_runs():
    m = build_sparse_unet_assisted_from_config(init_seed=3)
    assert m.in_channels == 9 and list(m.channel_order) == list(ASSISTED_CHANNEL_ORDER)
    assert ASSISTED_CHANNEL_ORDER[-2:] == ASSISTED_P_CHANNELS
    assert m(torch.randn(2, 9, H, W)).shape == (2, 1, H, W)


def test_glu_shared_weights_equal_free_and_k0_matches_free_forward(geom):
    p, x, y = geom
    torch.manual_seed(7)
    free = G.build_glu_from_config(domain_geometry_path=p).eval()
    torch.manual_seed(7)
    asst = GA.build_assisted_glu_from_config(domain_geometry_path=p).eval()
    fs, as_ = free.state_dict(), asst.state_dict()
    assert all(torch.equal(fs[k], as_[k]) for k in fs)                       # shared weights bit-identical at init
    co, vals, sites, pn, Y, tco, tv, ts = _batch(x, y, 2, 0)
    with torch.no_grad():
        a = free(co, vals, sites, pn, Y, p_conditioning=True, decode_chunked=True, chunk_size=512)
        b = asst(co, vals, sites, pn, Y, t_coords_phys=tco, t_values_norm=tv, t_site_index=ts, p_conditioning=True, decode_chunked=True, chunk_size=512)
    assert (a - b).abs().max() < 1e-3                                         # lattice vs cdist distance only


def test_glu_uses_observed_t_and_p_off_ignores_p(geom):
    p, x, y = geom
    torch.manual_seed(1)
    m = GA.build_assisted_glu_from_config(domain_geometry_path=p).eval()
    co, vals, sites, pn, Y, tco, tv, ts = _batch(x, y, 2, 9)
    with torch.no_grad():
        base = m(co, vals, sites, pn, Y, t_coords_phys=tco, t_values_norm=tv, t_site_index=ts)
        moved = m(co, vals, sites, pn, Y, t_coords_phys=tco, t_values_norm=tv + 1.0, t_site_index=ts)
        none = m(co, vals, sites, pn, Y, t_coords_phys=tco[:, :0], t_values_norm=tv[:, :0], t_site_index=ts[:, :0])
        off1 = m(co, vals, sites, pn, Y, t_coords_phys=tco, t_values_norm=tv, t_site_index=ts, p_conditioning=False)
        off2 = m(co, vals, sites, pn + 5.0, Y, t_coords_phys=tco, t_values_norm=tv, t_site_index=ts, p_conditioning=False)
    assert (base - moved).abs().max() > 0 and (base - none).abs().max() > 0
    assert torch.equal(off1, off2)


def test_glu_neighbours_and_predictions_do_not_depend_on_batch_shape_or_mates(geom):
    p, x, y = geom
    torch.manual_seed(2)
    m = GA.build_assisted_glu_from_config(domain_geometry_path=p).eval()
    rec = {}

    def run(B, mates, tag):
        co, vals, sites, pn, Y, tco, tv, ts = _batch(x, y, 8, 4, seed=5)
        sel = lambda t: t[:1].expand(B, *t.shape[1:]).contiguous() if mates == "self" else t[:B].contiguous()  # noqa: E731
        orig = G.topk_distance_then_site_index
        sets = []

        def wrapped(d, s, k):
            out = orig(d, s, k)
            sets.append(out[0].clone())
            return out

        G.topk_distance_then_site_index = wrapped
        try:
            with torch.no_grad():
                out = m(sel(co), sel(vals), sel(sites), sel(pn), sel(Y), t_coords_phys=sel(tco), t_values_norm=sel(tv), t_site_index=sel(ts), decode_chunked=True, chunk_size=512)
        finally:
            G.topk_distance_then_site_index = orig
        rec[tag] = (out[0].clone(), torch.cat(sets, 0))

    run(8, "self", "b8")
    run(1, "self", "b1")
    run(3, "self", "b3")
    run(8, "real", "b8real")
    for t in ("b1", "b3", "b8real"):
        assert torch.equal(rec[t][1], rec["b8"][1]), f"top-k sets differ for {t}"          # no tie flips with the batch
        assert (rec[t][0] - rec["b8"][0]).abs().max() < 1e-3
