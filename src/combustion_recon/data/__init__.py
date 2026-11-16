"""Data loading, masking, and train-only normalization."""

from combustion_recon.data.masks import ObservationTokens, build_observation_tokens
from combustion_recon.data.normalizer import TrainOnlyNormalizer

__all__ = [
    "ObservationTokens",
    "TrainOnlyNormalizer",
    "build_observation_tokens",
]

try:
    from combustion_recon.data.month2_dataset import InputIsolationProbe, Month2ReconDataset
    from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
    from combustion_recon.data.collate import collate_month2_batch

    __all__ += [
        "Month2ReconDataset",
        "Month2FitOnlyNormalizer",
        "InputIsolationProbe",
        "collate_month2_batch",
    ]
except ImportError:
    # Allow Month-1 imports even if optional Month-2 deps fail in odd environments.
    pass
