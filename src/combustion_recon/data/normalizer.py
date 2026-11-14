"""Training-only field / operating-parameter normalization.

Loads statistics from ``data/manifests/train_norm_stats.json`` produced by
``scripts/03_lock_splits_and_stats.py``. Stats must have been computed
exclusively on the locked ``train`` operating conditions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATS_PATH = ROOT / "data" / "manifests" / "train_norm_stats.json"
DEFAULT_SPLITS_PATH = ROOT / "configs" / "splits" / "splits_interp_extrap.yaml"


class TrainOnlyNormalizer:
    """Z-score (and optional min-max) normalizer fitted on the train split only."""

    def __init__(self, stats_path: str | Path | None = None):
        path = Path(stats_path) if stats_path is not None else DEFAULT_STATS_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Missing train_norm_stats at {path}. Run scripts/03_lock_splits_and_stats.py first."
            )
        with path.open() as f:
            self.stats: dict[str, Any] = json.load(f)
        self._validate_train_only()
        self.fields = self.stats["fields"]
        self.op_params = self.stats["operating_parameters"]

    def _validate_train_only(self) -> None:
        meta = self.stats.get("meta", {})
        split = meta.get("split")
        if split != "train":
            raise ValueError(
                f"train_norm_stats.json meta.split must be 'train', got {split!r}"
            )
        n = meta.get("n_conditions")
        if n is not None and int(n) != 20:
            raise ValueError(
                f"Expected 20 training conditions in stats meta, got n_conditions={n}"
            )
        sim_ids = set(meta.get("sim_ids", []))
        if sim_ids:
            import yaml

            with DEFAULT_SPLITS_PATH.open() as f:
                splits = yaml.safe_load(f)
            train_ids = set(splits["train"]["sim_ids"])
            if sim_ids != train_ids:
                raise ValueError(
                    "train_norm_stats sim_ids do not match locked train split "
                    f"(extra={sim_ids - train_ids}, missing={train_ids - sim_ids})"
                )

    def normalize_field(self, name: str, arr: np.ndarray) -> np.ndarray:
        s = self.fields[name]
        mean = float(s["mean"])
        std = float(s["std"])
        if std <= 0:
            raise ValueError(f"Non-positive std for field {name}")
        return (arr.astype(np.float64) - mean) / std

    def denormalize_field(self, name: str, arr: np.ndarray) -> np.ndarray:
        s = self.fields[name]
        return arr.astype(np.float64) * float(s["std"]) + float(s["mean"])

    def normalize_op_params(
        self, nh3_fraction: float, equivalence_ratio: float
    ) -> np.ndarray:
        """Return z-scored ``[nh3_fraction, equivalence_ratio]``."""
        out = np.empty(2, dtype=np.float64)
        for i, (name, val) in enumerate(
            [("nh3_fraction", nh3_fraction), ("equivalence_ratio", equivalence_ratio)]
        ):
            s = self.op_params[name]
            out[i] = (float(val) - float(s["mean"])) / float(s["std"])
        return out

    def denormalize_op_params(self, z: np.ndarray) -> tuple[float, float]:
        nh3 = float(z[0]) * float(self.op_params["nh3_fraction"]["std"]) + float(
            self.op_params["nh3_fraction"]["mean"]
        )
        phi = float(z[1]) * float(self.op_params["equivalence_ratio"]["std"]) + float(
            self.op_params["equivalence_ratio"]["mean"]
        )
        return nh3, phi

    @property
    def temperature_std(self) -> float:
        return float(self.fields["Temperature"]["std"])

    @property
    def temperature_mean(self) -> float:
        return float(self.fields["Temperature"]["mean"])
