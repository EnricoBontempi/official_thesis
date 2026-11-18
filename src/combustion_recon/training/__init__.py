"""Training utilities for Month-2 reconstructors."""

from combustion_recon.training.dense_cache import (
    DenseUNetCacheDataset,
    assert_dense_scientific_contract,
    build_dense_cache,
    collate_dense_batch,
    compute_cache_payload_hashes,
    refresh_dense_cache_manifest_metadata,
    validate_dense_cache_manifest,
    verify_cache_payload_hashes,
)
from combustion_recon.training.epoch_loop import (
    EpochDenseTrainer,
    EpochTrainConfig,
    assert_u4_candidate_contract,
    assert_u6_candidate_contract,
    select_best_val,
    should_evaluate_epoch,
)
from combustion_recon.training.run_provenance import collect_run_provenance
from combustion_recon.training.trainer import DenseUNetTrainer, TrainConfig

__all__ = [
    "DenseUNetCacheDataset",
    "assert_dense_scientific_contract",
    "build_dense_cache",
    "collate_dense_batch",
    "refresh_dense_cache_manifest_metadata",
    "validate_dense_cache_manifest",
    "DenseUNetTrainer",
    "TrainConfig",
    "EpochDenseTrainer",
    "EpochTrainConfig",
    "assert_u4_candidate_contract",
    "assert_u6_candidate_contract",
    "collect_run_provenance",
    "compute_cache_payload_hashes",
    "verify_cache_payload_hashes",
    "select_best_val",
    "should_evaluate_epoch",
]
