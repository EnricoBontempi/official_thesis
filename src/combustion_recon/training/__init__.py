"""Training utilities for Month-2 reconstructors."""

from combustion_recon.training.dense_cache import (
    DenseUNetCacheDataset,
    assert_dense_scientific_contract,
    build_dense_cache,
    collate_dense_batch,
    refresh_dense_cache_manifest_metadata,
    validate_dense_cache_manifest,
)
from combustion_recon.training.epoch_loop import (
    EpochDenseTrainer,
    EpochTrainConfig,
    select_best_val,
    should_evaluate_epoch,
)
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
    "select_best_val",
    "should_evaluate_epoch",
]
