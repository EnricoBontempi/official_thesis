from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pytest

from combustion_recon.data import t_anchors as T

MANIFEST = Path(__file__).resolve().parents[1] / "configs/observations/month2_temperature_anchors_nested_v1.json"


def test_manifest_contract_nested_fixed_and_hash_checked():
    m = T.AnchorManifest.load(MANIFEST)
    assert sorted(m.orders) == [0, 1, 2] and len({tuple(o) for o in m.orders.values()}) == 3
    for r in (0, 1, 2):
        for a, b in zip(T.K_VALUES[1:-1], T.K_VALUES[2:]):
            assert set(m.sites(r, a)) <= set(m.sites(r, b))
        assert m.common_domain_mask(r).sum() == 128 * 128 - 32
        assert not m.common_domain_mask(r).ravel()[m.orders[r]].any()
    assert m.sites(0, 0).size == 0
    with pytest.raises(ValueError):
        m.sites(0, 5)


def test_train_k_is_uniform_deterministic_and_model_independent():
    seq = [T.train_K_for_update(u) for u in range(6000)]
    assert seq == [T.train_K_for_update(u) for u in range(6000)]
    c = collections.Counter(seq)
    assert set(c) == set(T.K_VALUES) and max(c.values()) - min(c.values()) < 150


def test_train_layouts_differ_by_frame_and_epoch_and_from_eval_namespace():
    a = T.train_sites(0, "s", 1, 32)
    assert not np.array_equal(a, T.train_sites(1, "s", 1, 32)) and not np.array_equal(a, T.train_sites(0, "s", 2, 32))
    assert np.array_equal(a[:4], T.train_sites(0, "s", 1, 4))                       # nested prefix
    m = T.AnchorManifest.load(MANIFEST)
    assert not any(np.array_equal(a, m.orders[r]) for r in (0, 1, 2))


def test_domain_metrics_exclude_the_right_sites():
    true = np.zeros((128, 128))
    pred = np.zeros((128, 128))
    layout = np.arange(32)
    anchors = layout[:4]
    pred.ravel()[anchors] = 100.0          # error only at the first 4 anchors
    out = T.domain_metrics(pred, true, anchors, layout)
    assert out["anchors_rmse_K"] == 100.0 and out["offanchor_rmse_K"] == 0.0 and out["common_rmse_K"] == 0.0
    assert out["full_rmse_K"] > 0 and out["n_common"] == 128 * 128 - 32
