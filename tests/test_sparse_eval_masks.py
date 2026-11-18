"""S1 nested eval mask persistence and historical 5% join."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from combustion_recon.data.month2_observations import mask_hash
from combustion_recon.data.sparse_eval_masks import (
    DEFAULT_MANIFEST_OUT,
    DEFAULT_PAYLOAD_ROOT,
    N_SITES,
    assert_nesting,
    build_nested_for_frame,
    historical_5pct_sites,
    load_nested_sites,
    sites_to_mask,
    verify_payload_against_manifest,
)
from combustion_recon.data.sparse_protocol import load_sparse_protocol

ROOT = Path(__file__).resolve().parents[1]
HAS_PAYLOAD = (DEFAULT_PAYLOAD_ROOT / "train_fit_nested_sites.npz").exists()


def test_sparse_protocol_loads_and_refuses_unknown(tmp_path: Path):
    cfg = load_sparse_protocol()
    assert cfg["observation"]["K"] == 0
    assert cfg["unet_sparse"]["channels"] == 7
    bad = tmp_path / "bad.yaml"
    bad.write_text("protocol_id: x\nscientific_training_authorized: false\nlr_tune_on_val: 1e-2\n")
    with pytest.raises(ValueError, match="unsupported keys"):
        load_sparse_protocol(bad)


def test_nested_construction_and_independence_from_T():
    a = build_nested_for_frame(
        role="train_fit", sim_id="0NH3_1.h5", frame_index=0, replica_id=0
    )
    assert_nesting(a.sites_1pct, a.sites_5pct, a.sites_20pct)
    assert a.sites_1pct.size == N_SITES[0.01]
    assert a.sites_5pct.size == N_SITES[0.05]
    assert a.sites_20pct.size == N_SITES[0.20]
    # Sorted unique
    assert np.all(a.sites_5pct[:-1] < a.sites_5pct[1:])
    # T-independence: recompute via seed path only
    _, s5, h5 = historical_5pct_sites(sim_id="0NH3_1.h5", frame_index=0, replica_id=0)
    assert h5 == a.hash_5pct
    assert np.array_equal(s5, a.sites_5pct)


@pytest.mark.skipif(not HAS_PAYLOAD, reason="sparse eval mask payload not materialized")
def test_payload_reload_and_historical_join_manifest():
    man = json.loads(DEFAULT_MANIFEST_OUT.read_text())
    assert man["historical_5pct_join"]["ok"] is True
    assert man["n_instances"] == 3309
    assert man["historical_5pct_join"]["n_matched"] == 3309
    v = verify_payload_against_manifest()
    assert v["ok"] is True
    loaded = load_nested_sites(
        role="train_fit", sim_id="0NH3_1.h5", frame_index=0, replica_id=0
    )
    assert mask_hash(sites_to_mask(loaded["sites_5pct"])) == loaded["hash_5pct"]
    assert_nesting(loaded["sites_1pct"], loaded["sites_5pct"], loaded["sites_20pct"])


@pytest.mark.skipif(not HAS_PAYLOAD, reason="sparse eval mask payload not materialized")
def test_verified_examples_5pct_match():
    obs = json.loads((ROOT / "data/manifests/month2_observations.json").read_text())
    for ex in obs["verified_examples"]:
        if abs(float(ex["site_fraction"]) - 0.05) > 1e-12:
            continue
        _, _, h5 = historical_5pct_sites(
            sim_id=ex["sim_id"],
            frame_index=int(ex["frame_index"]),
            replica_id=int(ex["replica_id"]),
        )
        assert h5 == ex["velocity_mask_hash"]
