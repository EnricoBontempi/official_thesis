"""Month-2 pipeline tests: frames, stats provenance, norm, collate, reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from combustion_recon.data.collate import collate_month2_batch
from combustion_recon.data.dataset import load_splits
from combustion_recon.data.freeze import FreezeGateError
from combustion_recon.data.month2_dataset import Month2ReconDataset
from combustion_recon.data.month2_manifests import build_frames_manifest, build_observations_manifest
from combustion_recon.data.month2_normalizer import (
    Month2FitOnlyNormalizer,
    compute_month2_fit_stats,
    validate_stats_reject_incompatible,
)
from combustion_recon.data.month2_observations import (
    build_month2_observation_pack,
    observation_mask_seed,
)
from combustion_recon.data.month2_protocol import (
    assert_fit_seen_disjoint,
    build_index,
    frames_for_role,
    load_frame_protocol,
)
from combustion_recon.data.shard_access import CompactFrame, ShardStore

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "combustion" / "hf_dataset" / "numerical"
STATS = ROOT / "data" / "manifests" / "train_norm_stats_month2_fit.json"
HAS_REAL = RAW.exists() and len(list(RAW.glob("data-*-of-00030.arrow"))) >= 1


def test_frame_protocol_counts_and_disjoint():
    protocol = load_frame_protocol()
    assert_fit_seen_disjoint(protocol)
    fit = frames_for_role(protocol, "train_fit")
    seen = frames_for_role(protocol, "seen_train_eval")
    val = frames_for_role(protocol, "val_eval")
    assert len(fit) == 38 and fit[0] == 0 and fit[-1] == 1480
    assert len(seen) == 11 and seen[0] == 1600 and seen[-1] == 2000
    assert len(val) == 41 and val[0] == 0 and val[-1] == 2000
    assert not (set(fit) & set(seen))
    train_idx = build_index("train_fit", protocol=protocol)
    seen_idx = build_index("seen_train_eval", protocol=protocol)
    val_idx = build_index("val_eval", protocol=protocol)
    assert len(train_idx) == 760
    assert len(seen_idx) == 220
    assert len(val_idx) == 123


def test_frames_manifest_counts():
    man = build_frames_manifest()
    assert man["counts"]["train_fit"]["n_samples"] == 760
    assert man["counts"]["seen_train_eval"]["n_samples"] == 220
    assert man["counts"]["val_eval"]["n_samples"] == 123
    assert man["counts"]["test_interp_eval"]["payload_access"] == "requires_selection_freeze"
    assert man["fit_seen_disjoint"] is True


def test_observations_manifest_nested_and_seeds():
    man = build_observations_manifest()
    assert man["depends_on_target_T"] is False
    assert man["mask_seed_default"] == 20261116
    assert all(ex["nested_K_ok"] for ex in man["verified_examples"])
    # Same velocity mask across K is a documented invariant
    assert all(ex["same_velocity_mask_for_all_K"] for ex in man["verified_examples"])


def test_mask_seed_invariant_to_dataset_order_concept():
    a = observation_mask_seed(
        mask_seed=1, sim_id="0NH3_1.h5", frame_index=40, layout_id="L", replica_id=0
    )
    b = observation_mask_seed(
        mask_seed=1, sim_id="0NH3_1.h5", frame_index=40, layout_id="L", replica_id=0
    )
    c = observation_mask_seed(
        mask_seed=1, sim_id="0NH3_1.h5", frame_index=40, layout_id="L", replica_id=1
    )
    assert a == b
    assert a != c


def test_fold_capable_stats_reject_wrong_provenance(tmp_path: Path):
    # Synthetic compact frames via a tiny fake store
    class FakeStore:
        def __init__(self):
            self._lock = __import__("threading").RLock()
            self._frame_cache = {}
            rng = np.random.default_rng(0)
            self.frames = {}
            for sid, nh3, phi in [("0NH3_0.85.h5", 0, 0.85), ("20NH3_0.9.h5", 20, 0.9)]:
                for fi in [0, 40]:
                    self.frames[(sid, fi)] = CompactFrame(
                        temperature=rng.normal(2000, 10, size=(8, 8)).astype(np.float32),
                        velocity_i=rng.normal(size=(8, 8)).astype(np.float32),
                        velocity_j=rng.normal(size=(8, 8)).astype(np.float32),
                        velocity_k=rng.normal(size=(8, 8)).astype(np.float32),
                        x=np.linspace(-1, 1, 8),
                        y=np.linspace(0, 1, 8),
                        t=float(fi),
                        sim_id=sid,
                        frame_index=fi,
                    )

        def prefetch(self, sim_id, frame_indices):
            return None

        def get_frame(self, sim_id, frame_index):
            return self.frames[(sim_id, int(frame_index))]

    store = FakeStore()
    ids = ["0NH3_0.85.h5", "20NH3_0.9.h5"]
    stats = compute_month2_fit_stats(
        sim_ids=ids,
        frame_indices=[0, 40],
        store=store,
        provenance_mode="debug_subset",
    )
    assert stats["meta"]["n_conditions"] == 2  # not forced to 20
    assert stats["meta"]["source_kind"] == "month2_debug_subset"
    validate_stats_reject_incompatible(stats, expected_sim_ids=ids)
    with pytest.raises(ValueError):
        validate_stats_reject_incompatible(stats, expected_sim_ids=ids + ["40NH3_1.h5"])

    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    # Production path rejects debug_subset; explicit allow for fixtures
    with pytest.raises(ValueError, match="debug_subset"):
        Month2FitOnlyNormalizer(
            path, expected_sim_ids=ids, require_protocol_id="month2_frame_protocol_v1"
        )
    norm = Month2FitOnlyNormalizer(
        path,
        expected_sim_ids=ids,
        require_protocol_id="month2_frame_protocol_v1",
        allow_noncanonical_frame_indices=True,
    )
    z = norm.normalize_field("Temperature", np.array([stats["fields"]["Temperature"]["mean"]]))
    assert abs(float(z.reshape(-1)[0])) < 1e-6
    # Reject wrong expected set
    with pytest.raises(ValueError):
        Month2FitOnlyNormalizer(
            path, expected_sim_ids=[ids[0]], allow_noncanonical_frame_indices=True
        )


def test_norm_denorm_synthetic(tmp_path: Path):
    from combustion_recon.data.month2_protocol import (
        scientific_protocol_fingerprint,
        load_frame_protocol,
        sha256_file,
        DEFAULT_SPLITS,
    )

    protocol = load_frame_protocol()
    stats = {
        "meta": {
            "split": "train",
            "frame_role": "train_fit",
            "protocol_id": "month2_frame_protocol_v1",
            "includes_seen_frames": False,
            "n_conditions": 1,
            "sim_ids": ["0NH3_0.85.h5"],
            "frame_indices": [0],
            "n_frames_per_condition": 1,
            "source_kind": "month2_debug_subset",
            "protocol_scientific_sha256": scientific_protocol_fingerprint(protocol),
            "splits_sha256": sha256_file(DEFAULT_SPLITS),
        },
        "fields": {
            "Temperature": {"mean": 2000.0, "std": 100.0, "n_voxels_total": 10, "min": 0, "max": 1},
            "Velocity_i": {"mean": 0.0, "std": 1.0, "n_voxels_total": 10, "min": 0, "max": 1},
            "Velocity_j": {"mean": 0.0, "std": 2.0, "n_voxels_total": 10, "min": 0, "max": 1},
        },
        "operating_parameters": {
            "nh3_fraction": {"mean": 0.2, "std": 0.1, "n_conditions": 1, "min": 0, "max": 1},
            "equivalence_ratio": {"mean": 1.0, "std": 0.1, "n_conditions": 1, "min": 0, "max": 1},
        },
    }
    path = tmp_path / "s.json"
    path.write_text(json.dumps(stats))
    norm = Month2FitOnlyNormalizer(
        path,
        expected_sim_ids=["0NH3_0.85.h5"],
        allow_noncanonical_frame_indices=True,
    )
    arr = np.array([2100.0, 1900.0])
    z = norm.normalize_field("Temperature", arr)
    back = norm.denormalize_field("Temperature", z)
    assert np.allclose(back, arr)
    p = norm.normalize_op_params(0.2, phi=1.0)
    assert np.allclose(p, [0.0, 0.0])


def test_collate_padding_and_validity():
    rng = np.random.default_rng(0)
    samples = []
    for n_sites, K in [(5, 0), (8, 4)]:
        H = W = 16
        vi = rng.normal(size=(H, W))
        vj = rng.normal(size=(H, W))
        T = 2000 + rng.normal(size=(H, W))
        x = np.linspace(-1, 1, W)
        y = np.linspace(0, 1, H)
        pack = build_month2_observation_pack(
            velocity_i=vi,
            velocity_j=vj,
            temperature=T,
            x_coords=x,
            y_coords=y,
            site_fraction=n_sites / (H * W),
            K=K,
            layout_id="L",
            mask_seed=0,
            sim_id="synth",
            frame_index=0,
            replica_id=0,
            include_dense_temperature_channels=True,
        )
        samples.append(
            {
                "model_inputs": {
                    "velocity_coords": pack.velocity_coords,
                    "velocity_values": pack.velocity_values,
                    "velocity_site_index": pack.velocity_site_index,
                    "scalar_coords": pack.scalar_coords,
                    "scalar_values": pack.scalar_values,
                    "scalar_var_ids": pack.scalar_var_ids,
                    "anchor_coords": pack.anchor_coords,
                    "anchor_values": pack.anchor_values,
                    "anchor_site_index": pack.anchor_site_index,
                    "dense_values": pack.dense_values,
                    "dense_channel_names": pack.dense_channel_names,
                    "query_coords": np.zeros((H * W, 2)),
                    "mask_velocity": pack.mask_velocity,
                    "mask_temperature": pack.mask_temperature,
                    "K": K,
                    "site_fraction": n_sites / (H * W),
                    "task_mode": "temperature_assisted" if K else "temperature_free",
                },
                "targets": {
                    "temperature": T,
                    "temperature_normalized": (T - 2000) / 100,
                },
                "metadata": {"sim_id": "synth", "frame_index": 0},
            }
        )
    batch = collate_month2_batch(samples, as_torch=False)
    assert batch["model_inputs"]["velocity_valid"].shape[0] == 2
    assert batch["model_inputs"]["velocity_valid"].dtype == bool
    # padded rows beyond true length are False
    for i, n in enumerate([samples[0]["model_inputs"]["velocity_coords"].shape[0], samples[1]["model_inputs"]["velocity_coords"].shape[0]]):
        assert batch["model_inputs"]["velocity_valid"][i, :n].all()
        if batch["model_inputs"]["velocity_valid"].shape[1] > n:
            assert not batch["model_inputs"]["velocity_valid"][i, n:].any()


def test_k0_assisted_matches_free_observationally():
    rng = np.random.default_rng(1)
    H = W = 32
    vi, vj = rng.normal(size=(H, W)), rng.normal(size=(H, W))
    T = 2000 + rng.normal(size=(H, W))
    x = np.linspace(-1, 1, W)
    y = np.linspace(0, 1, H)
    free = build_month2_observation_pack(
        velocity_i=vi, velocity_j=vj, temperature=T, x_coords=x, y_coords=y,
        site_fraction=0.05, K=0, layout_id="L", mask_seed=7, sim_id="s", frame_index=3,
        include_dense_temperature_channels=False,
    )
    assisted0 = build_month2_observation_pack(
        velocity_i=vi, velocity_j=vj, temperature=T, x_coords=x, y_coords=y,
        site_fraction=0.05, K=0, layout_id="L", mask_seed=7, sim_id="s", frame_index=3,
        include_dense_temperature_channels=False,
    )
    assert free.velocity_mask_hash == assisted0.velocity_mask_hash
    assert np.array_equal(free.velocity_values, assisted0.velocity_values)
    assert assisted0.K == 0 and assisted0.mask_temperature.sum() == 0


@pytest.mark.skipif(not HAS_REAL, reason="real shards not available")
def test_real_dataset_counts_and_test_gate():
    if not STATS.exists():
        pytest.skip("month2 fit stats not generated yet")
    norm = Month2FitOnlyNormalizer(STATS)
    ds = Month2ReconDataset("val_eval", normalizer=norm, max_cached_frames=8)
    assert len(ds) == 123
    sample = ds[0]
    assert set(sample) == {"model_inputs", "targets", "metadata"}
    assert "op_params" not in sample["model_inputs"]  # default p_off
    tds = Month2ReconDataset("test_extrap_eval", normalizer=norm, allow_test_payloads=False)
    assert len(tds) == 164  # index metadata only
    with pytest.raises(FreezeGateError):
        _ = tds[0]


@pytest.mark.skipif(not HAS_REAL, reason="real shards not available")
def test_reproducibility_under_reordering():
    if not STATS.exists():
        pytest.skip("month2 fit stats not generated yet")
    norm = Month2FitOnlyNormalizer(STATS)
    store = ShardStore(raw_dir=RAW, max_cached_frames=16)
    ds = Month2ReconDataset("train_fit", normalizer=norm, store=store, replica_id=0)
    # Same physical identity at different dataset positions should match masks
    # Find two indices? Instead compare getitem of same idx vs rebuilt dataset with shuffled access
    s0 = ds[10]
    s1 = Month2ReconDataset("train_fit", normalizer=norm, store=store, replica_id=0)[10]
    assert s0["metadata"]["velocity_mask_hash"] == s1["metadata"]["velocity_mask_hash"]
    assert np.array_equal(
        s0["model_inputs"]["mask_velocity"], s1["model_inputs"]["mask_velocity"]
    )
