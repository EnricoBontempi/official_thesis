"""Cache-id contract on the dataset path used by the Opus trainer.

Attempt 1 of O-C149-on/off-s0 died in SparseUNetCacheDataset because
validate_dense_cache_manifest defaults to the historical cache id. These tests
stay on tiny manifests: they stop at the allowlist or the next cheap gate and
do not build the 2980-frame cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from combustion_recon.training.dense_cache import (
    CHANNEL_ORDER,
    HISTORICAL_CACHE_ID,
    OPUS_C149_CACHE_ID,
    DenseCacheError,
    cache_validation_kwargs,
    validate_dense_cache_manifest,
)
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset

ROOT = Path(__file__).resolve().parents[1]


def _manifest(cache_id: str, *, train_fit: int = 2980) -> dict:
    return {
        "cache_id": cache_id,
        "test_payloads": "none",
        "task_mode": "temperature_free",
        "K": 0,
        "site_fraction": 1.0,
        "channel_order": list(CHANNEL_ORDER),
        "counts": {
            "train_fit": train_fit,
            "seen_train_eval": 220,
            "val_eval": 123,
        },
    }


def _write(root: Path, manifest: dict) -> None:
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_contract_ids_are_exclusive():
    c149 = cache_validation_kwargs("C149")
    historical = cache_validation_kwargs(None)
    a38 = cache_validation_kwargs("A38")
    b19 = cache_validation_kwargs("B19")
    assert c149["allowed_cache_ids"] == [OPUS_C149_CACHE_ID]
    assert HISTORICAL_CACHE_ID not in c149["allowed_cache_ids"]
    assert c149["expected_role_counts"] == {
        "train_fit": 2980,
        "seen_train_eval": 220,
        "val_eval": 123,
    }
    assert str(c149["frame_protocol_path"]).endswith(
        "month2_frame_protocol_opus_c149.yaml"
    )
    for kwargs in (historical, a38, b19):
        assert kwargs["allowed_cache_ids"] == [HISTORICAL_CACHE_ID]
        assert OPUS_C149_CACHE_ID not in kwargs["allowed_cache_ids"]
        assert "expected_role_counts" not in kwargs
        assert "frame_protocol_path" not in kwargs
    with pytest.raises(ValueError, match="unknown pool_id"):
        cache_validation_kwargs("S7")


def test_default_dataset_path_rejects_c149_id(tmp_path: Path):
    """Reproduce the attempt-1 failure: no allowlist, historical default."""
    _write(tmp_path, _manifest(OPUS_C149_CACHE_ID))
    with pytest.raises(DenseCacheError, match="unknown cache_id") as exc:
        SparseUNetCacheDataset("train_fit", cache_root=tmp_path)
    message = str(exc.value)
    assert OPUS_C149_CACHE_ID in message
    assert HISTORICAL_CACHE_ID in message


@pytest.mark.parametrize("pool_id", ["A38", "B19"])
def test_a38_b19_dataset_path_rejects_c149_cache(tmp_path: Path, pool_id: str):
    _write(tmp_path, _manifest(OPUS_C149_CACHE_ID))
    with pytest.raises(DenseCacheError, match="unknown cache_id"):
        SparseUNetCacheDataset(
            "train_fit",
            cache_root=tmp_path,
            **cache_validation_kwargs(pool_id),
        )


def test_c149_dataset_path_rejects_historical_cache(tmp_path: Path):
    _write(tmp_path, _manifest(HISTORICAL_CACHE_ID))
    with pytest.raises(DenseCacheError, match="unknown cache_id") as exc:
        SparseUNetCacheDataset(
            "train_fit",
            cache_root=tmp_path,
            **cache_validation_kwargs("C149"),
        )
    assert HISTORICAL_CACHE_ID in str(exc.value)
    assert OPUS_C149_CACHE_ID in str(exc.value)


def test_c149_dataset_path_accepts_id_and_keeps_count_gate(tmp_path: Path):
    """Id passes; a 760-frame count is still refused for a C149 run."""
    _write(tmp_path, _manifest(OPUS_C149_CACHE_ID, train_fit=760))
    with pytest.raises(DenseCacheError, match="count mismatch") as exc:
        SparseUNetCacheDataset(
            "train_fit",
            cache_root=tmp_path,
            **cache_validation_kwargs("C149"),
        )
    assert "unknown cache_id" not in str(exc.value)


def test_c149_dataset_path_accepts_id_and_keeps_payload_gate(tmp_path: Path):
    """Matching id and counts still require the role payloads."""
    _write(tmp_path, _manifest(OPUS_C149_CACHE_ID))
    with pytest.raises(DenseCacheError, match="missing") as exc:
        SparseUNetCacheDataset(
            "train_fit",
            cache_root=tmp_path,
            **cache_validation_kwargs("C149"),
        )
    assert "unknown cache_id" not in str(exc.value)


def test_historical_default_allowlist_unchanged(tmp_path: Path):
    with pytest.raises(DenseCacheError, match="unknown cache_id") as rejected:
        validate_dense_cache_manifest(
            {"cache_id": OPUS_C149_CACHE_ID},
            cache_root=tmp_path,
        )
    assert HISTORICAL_CACHE_ID in str(rejected.value)
    with pytest.raises(DenseCacheError, match="missing geometry") as accepted_id:
        validate_dense_cache_manifest(
            {
                "cache_id": HISTORICAL_CACHE_ID,
                "test_payloads": "none",
                "task_mode": "temperature_free",
                "K": 0,
                "site_fraction": 1.0,
                "channel_order": list(CHANNEL_ORDER),
                "counts": {},
            },
            cache_root=tmp_path,
        )
    assert "unknown cache_id" not in str(accepted_id.value)


def test_trainer_passes_contract_into_every_dataset():
    source = (ROOT / "scripts/16_train_sparse_scientific.py").read_text()
    assert "cache_kwargs = cache_validation_kwargs(pool_id)" in source
    # 6 call sites: train, train-eval(ref760), val, seen, the specialist SELECTION dataset (added by 06ee61f) and the manifest validation; verified
    # in mission 02 that the sixth passes the C149 cache contract into the dataset it builds.
    assert source.count("**cache_kwargs") == 6
    on = (ROOT / "configs/experiments/opus_O-C149-on-s0.yaml").read_text()
    off = (ROOT / "configs/experiments/opus_O-C149-off-s0.yaml").read_text()
    for text, job in ((on, "29613545"), (off, "29613546")):
        assert "attempt: 2" in text
        assert job in text
        assert "official_thesis_opus_c149_attempt2_runs" in text
        assert "warm_start: false" in text
