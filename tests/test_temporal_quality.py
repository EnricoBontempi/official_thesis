"""Temporal provenance flags and association guard (post-consolidation correction)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from combustion_recon.data.shard_access import (
    KNOWN_T_LEN_MISMATCH,
    ShardStore,
    TimelineError,
    timeline_is_approximately_regular,
)
from combustion_recon.data.temporal_quality import (
    require_verified_field_time_association,
    temporal_quality_from_mapping,
)


RAW = Path("data/raw/combustion/hf_dataset/numerical")
ANOM = "40NH3_1.2.h5"
HEALTHY = "0NH3_0.85.h5"


def _raw_available() -> bool:
    return (RAW / "data-00022-of-00030.arrow").exists()


@pytest.mark.skipif(not _raw_available(), reason="raw Arrow shards not available")
def test_anomalous_shard_all_indices_association_unverified():
    store = ShardStore(raw_dir=RAW, max_cached_frames=8)
    # Protocol-relevant indices plus ends
    for fi in [0, 40, 1480, 1600, 1999, 2000]:
        fr = store.get_frame(ANOM, fi)
        assert fr.field_time_association_verified is False
        assert fr.time_validated is False
        assert fr.timeline_regular is False
        if fi < 2000:
            assert fr.has_original_time_coordinate is True
            assert np.isfinite(fr.t)
            assert fr.duplicate_of is None
        else:
            assert fr.has_original_time_coordinate is False
            assert np.isnan(fr.t)
            assert fr.duplicate_of == 1999


@pytest.mark.skipif(not _raw_available(), reason="raw Arrow shards not available")
def test_original_coordinate_not_confused_with_verified_association():
    store = ShardStore(raw_dir=RAW, max_cached_frames=4)
    fr = store.get_frame(ANOM, 100)
    assert fr.has_original_time_coordinate is True
    assert fr.field_time_association_verified is False
    q = temporal_quality_from_mapping(fr)
    assert q["has_original_time_coordinate"] is True
    assert q["field_time_association_verified"] is False
    assert q["time_validated"] is False


@pytest.mark.skipif(not _raw_available(), reason="raw Arrow shards not available")
def test_duplicate_of_and_physical_fields_unchanged():
    store = ShardStore(raw_dir=RAW, max_cached_frames=4)
    a = store.get_frame(ANOM, 1999)
    b = store.get_frame(ANOM, 2000)
    assert b.duplicate_of == 1999
    assert np.array_equal(a.temperature, b.temperature)
    assert np.array_equal(a.velocity_i, b.velocity_i)
    assert np.array_equal(a.velocity_j, b.velocity_j)
    # Healthy shard still verified
    h = store.get_frame(HEALTHY, 2000)
    assert h.field_time_association_verified is True
    assert h.time_validated is True
    assert h.duplicate_of is None


@pytest.mark.skipif(not _raw_available(), reason="raw Arrow shards not available")
def test_dataset_metadata_propagates_flags_not_model_inputs():
    from combustion_recon.data.month2_dataset import Month2ReconDataset

    ds = Month2ReconDataset(
        "seen_train_eval",
        task_mode="temperature_free",
        K=0,
        site_fraction=0.05,
        p_conditioning=False,
        mask_seed=20261116,
        replica_id=0,
        sim_ids=[ANOM],
        store=ShardStore(raw_dir=RAW, max_cached_frames=16),
        max_cached_frames=16,
    )
    # Find frame 2000 sample
    sample = None
    for i in range(len(ds)):
        if ds.get_identity(i)["frame_index"] == 2000:
            sample = ds[i]
            break
    assert sample is not None
    md = sample["metadata"]
    assert md["field_time_association_verified"] is False
    assert md["time_validated"] is False
    assert md["duplicate_of"] == 1999
    assert md["has_original_time_coordinate"] is False
    assert "field_time_association_verified" not in sample["model_inputs"]
    assert "time_validated" not in sample["model_inputs"]
    assert "duplicate_of" not in sample["model_inputs"]
    # Numerical single-frame inputs still present
    assert "mask_velocity" in sample["model_inputs"]
    assert sample["targets"]["temperature"].shape == (128, 128)


def test_guard_rejects_unverified_association_synthetic():
    meta = {
        "field_time_association_verified": False,
        "time_validated": False,
        "has_original_time_coordinate": True,
        "timeline_regular": False,
        "time_coordinate_original": 1.7,
    }
    with pytest.raises(TimelineError, match="field↔time association"):
        require_verified_field_time_association(meta, context="unit")
    ok = {
        "field_time_association_verified": True,
        "time_validated": True,
        "has_original_time_coordinate": True,
        "timeline_regular": True,
        "time_coordinate_original": 1.0,
    }
    require_verified_field_time_association(ok)


def test_timeline_regularity_helper():
    t = np.linspace(0.0, 1.0, 11)
    assert timeline_is_approximately_regular(t)
    t2 = t.copy()
    t2[5] += 0.05
    # make non-decreasing still but irregular diffs
    t_bad = np.array([0.0, 0.1, 0.2, 0.45, 0.55, 0.65])
    assert not timeline_is_approximately_regular(t_bad)


def test_known_exception_spec_identity():
    spec = KNOWN_T_LEN_MISMATCH[ANOM]
    assert spec["duplicate_frame_index"] == 2000
    assert spec["duplicate_of_index"] == 1999
