"""Step 2.2 corrective guards: role membership, stats provenance, aggregation, prefetch."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from combustion_recon.data.dataset import load_splits
from combustion_recon.data.month2_normalizer import (
    Month2FitOnlyNormalizer,
    compute_month2_fit_stats,
)
from combustion_recon.data.month2_protocol import (
    RoleMembershipError,
    build_index,
    frames_for_role,
    load_frame_protocol,
    scientific_protocol_fingerprint,
    sha256_file,
    validate_sim_ids_for_role,
)
from combustion_recon.data.shard_access import CompactFrame, ShardStore
from combustion_recon.evaluation.month2_evaluator import (
    MixedConfigurationError,
    aggregate_frame_then_condition_then_split,
    evaluate_frame_prediction,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "combustion" / "hf_dataset" / "numerical"
STATS = ROOT / "data" / "manifests" / "train_norm_stats_month2_fit.json"
HAS_REAL = RAW.exists() and len(list(RAW.glob("data-*-of-00030.arrow"))) >= 1


# ---------------------------------------------------------------------------
# 1. Role membership
# ---------------------------------------------------------------------------

def test_reject_test_sim_id_on_train_fit_before_store():
    splits = load_splits()
    test_id = splits["test_interp"][0]
    with pytest.raises(RoleMembershipError, match="incompatible"):
        validate_sim_ids_for_role("train_fit", [test_id])
    with pytest.raises(RoleMembershipError):
        build_index("train_fit", sim_ids=[test_id])
    with pytest.raises(RoleMembershipError):
        build_index("seen_train_eval", sim_ids=[test_id])


def test_reject_train_sim_id_on_val_and_duplicates():
    splits = load_splits()
    train_id = splits["train"][0]
    with pytest.raises(RoleMembershipError):
        build_index("val_eval", sim_ids=[train_id])
    with pytest.raises(RoleMembershipError, match="duplicate"):
        validate_sim_ids_for_role("train_fit", [train_id, train_id])


def test_train_subset_allowed_for_folds_without_payload():
    splits = load_splits()
    subset = splits["train"][:3]
    idx = build_index("train_fit", sim_ids=subset)
    assert len(idx) == 3 * 38
    assert {e["sim_id"] for e in idx} == set(subset)


def test_build_index_does_not_touch_store(monkeypatch):
    """Membership validation must not require shard access."""
    called = {"n": 0}

    def boom(*args, **kwargs):
        called["n"] += 1
        raise RuntimeError("store should not be called")

    monkeypatch.setattr(
        "combustion_recon.data.shard_access.ShardStore.get_frame", boom, raising=False
    )
    splits = load_splits()
    with pytest.raises(RoleMembershipError):
        build_index("train_fit", sim_ids=[splits["test_extrap"][0]])
    # Also success path has no store
    build_index("train_fit", sim_ids=splits["train"][:1])
    assert called["n"] == 0


# ---------------------------------------------------------------------------
# 2. Stats provenance
# ---------------------------------------------------------------------------

def test_existing_production_stats_load():
    norm = Month2FitOnlyNormalizer(STATS)
    assert len(norm.sim_ids) == 20
    protocol = load_frame_protocol()
    assert norm.stats["meta"]["frame_indices"] == frames_for_role(protocol, "train_fit")
    assert (
        norm.stats["meta"]["protocol_scientific_sha256"]
        == scientific_protocol_fingerprint(protocol)
    )


def test_reject_stats_with_wrong_frames(tmp_path: Path):
    stats = json.loads(STATS.read_text())
    stats["meta"]["frame_indices"] = list(frames_for_role(load_frame_protocol(), "seen_train_eval"))
    stats["meta"]["n_frames_per_condition"] = len(stats["meta"]["frame_indices"])
    path = tmp_path / "bad_frames.json"
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="frame_indices"):
        Month2FitOnlyNormalizer(path)


def test_compute_rejects_non_train_before_io():
    store = MagicMock()
    splits = load_splits()
    with pytest.raises(RoleMembershipError):
        compute_month2_fit_stats(
            sim_ids=[splits["test_interp"][0]],
            frame_indices=frames_for_role(load_frame_protocol(), "train_fit"),
            store=store,
        )
    store.prefetch.assert_not_called()
    store.get_frame.assert_not_called()


def test_compute_rejects_noncanonical_frames_in_production():
    store = MagicMock()
    splits = load_splits()
    with pytest.raises(ValueError, match="exact protocol train_fit"):
        compute_month2_fit_stats(
            sim_ids=splits["train"][:1],
            frame_indices=[0, 40],
            store=store,
            provenance_mode="production",
        )
    store.prefetch.assert_not_called()


def test_debug_subset_stats_require_explicit_allow(tmp_path: Path):
    class FakeStore:
        def __init__(self):
            self._lock = __import__("threading").RLock()
            self._frame_cache = {}
            rng = np.random.default_rng(0)
            self.frames = {}
            for sid in ["0NH3_0.85.h5", "20NH3_0.9.h5"]:
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

    ids = ["0NH3_0.85.h5", "20NH3_0.9.h5"]
    stats = compute_month2_fit_stats(
        sim_ids=ids,
        frame_indices=[0, 40],
        store=FakeStore(),
        provenance_mode="debug_subset",
    )
    assert stats["meta"]["source_kind"] == "month2_debug_subset"
    path = tmp_path / "dbg.json"
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="debug_subset"):
        Month2FitOnlyNormalizer(path, expected_sim_ids=ids)
    norm = Month2FitOnlyNormalizer(
        path, expected_sim_ids=ids, allow_noncanonical_frame_indices=True
    )
    assert abs(float(norm.normalize_field("Temperature", np.array([norm.temperature_mean])).ravel()[0])) < 1e-6


# ---------------------------------------------------------------------------
# 3. Aggregation homogeneity
# ---------------------------------------------------------------------------

def _synth_row(role, sim_id, rmse, **tags):
    pred = np.full((4, 4), 2000.0 + rmse)
    true = np.full((4, 4), 2000.0)
    split = {
        "train_fit": "train",
        "seen_train_eval": "train",
        "val_eval": "val",
    }[role]
    return evaluate_frame_prediction(
        pred,
        true,
        train_sigma_T=300.0,
        sim_id=sim_id,
        split=split,
        role=role,
        frame_index=0,
        method=tags.get("method", "A"),
        p_conditioning=tags.get("p_conditioning", False),
        K=tags.get("K", 0),
        site_fraction=tags.get("site_fraction", 0.05),
        model_seed=tags.get("model_seed", 0),
        mask_replica_id=tags.get("mask_replica_id", 0),
    )


def test_aggregation_refuses_mixed_roles_and_methods():
    rows = [
        _synth_row("train_fit", "0NH3_0.85.h5", 10.0, method="A"),
        _synth_row("seen_train_eval", "0NH3_0.85.h5", 50.0, method="A"),
    ]
    with pytest.raises(MixedConfigurationError, match="role"):
        aggregate_frame_then_condition_then_split(rows)

    rows2 = [
        _synth_row("val_eval", "0NH3_1.05.h5", 10.0, method="A"),
        _synth_row("val_eval", "0NH3_1.05.h5", 80.0, method="B"),
    ]
    with pytest.raises(MixedConfigurationError, match="method"):
        aggregate_frame_then_condition_then_split(rows2)


def test_aggregation_keeps_train_fit_and_seen_separate():
    fit_rows = [
        _synth_row("train_fit", "0NH3_0.85.h5", 10.0, method="A"),
        _synth_row("train_fit", "20NH3_0.9.h5", 12.0, method="A"),
    ]
    seen_rows = [
        _synth_row("seen_train_eval", "0NH3_0.85.h5", 40.0, method="A"),
        _synth_row("seen_train_eval", "20NH3_0.9.h5", 42.0, method="A"),
    ]
    fit_agg = aggregate_frame_then_condition_then_split(fit_rows)
    seen_agg = aggregate_frame_then_condition_then_split(seen_rows)
    fit_rmse = float(fit_agg["per_condition"]["rmse_K"].mean())
    seen_rmse = float(seen_agg["per_condition"]["rmse_K"].mean())
    assert fit_rmse < 20
    assert seen_rmse > 30
    assert fit_agg["config_fingerprint"]["role"] == "train_fit"
    assert seen_agg["config_fingerprint"]["role"] == "seen_train_eval"
    # selection not eligible for train roles
    assert (fit_agg["selection_mean_condition_rmse"]["selection_eligible"] == False).all()


# ---------------------------------------------------------------------------
# 4. Prefetch decode accounting (small, not full-dataset)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_REAL, reason="real shards unavailable")
def test_prefetch_one_decode_for_multiple_frames():
    store = ShardStore(raw_dir=RAW, max_cached_frames=64)
    sid = load_splits()["train"][0]
    frames = [0, 40, 80]
    store.clear_cache()
    store.stats["shard_decodes"] = 0
    store.prefetch(sid, frames)
    assert store.stats["shard_decodes"] == 1
    # subsequent get_frame must not decode again
    for fi in frames:
        store.get_frame(sid, fi)
    assert store.stats["shard_decodes"] == 1
    # cold get_frame path would decode per miss: demonstrate contrast on another sid
    store2 = ShardStore(raw_dir=RAW, max_cached_frames=2)
    sid2 = load_splits()["train"][1]
    store2.stats["shard_decodes"] = 0
    store2.get_frame(sid2, 0)
    store2.get_frame(sid2, 40)  # likely miss if cache small / different key
    # At least first call decoded once; second may decode again if not prefetched
    assert store2.stats["shard_decodes"] >= 1
