"""Temperature-anchor contract of the temperature study (frozen manifest, K schedule, domains, metrics).

Nothing here reads temperature values to choose sites. Eval layouts come from the frozen manifest
(``configs/observations/month2_temperature_anchors_nested_v1.json``); TRAIN layouts are drawn per frame and
epoch in a separate namespace; the TRAIN K is a pure function of the optimizer update number.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from combustion_recon.data.masks import stable_seed

H = W = 128
K_VALUES: tuple[int, ...] = (0, 1, 4, 9, 16, 32)
EVAL_NAMESPACE = "month2_temperature_anchors_nested_v1"
TRAIN_LAYOUT_NAMESPACE = "month2_assisted_t_train_layout_v1"
TRAIN_K_NAMESPACE = "month2_assisted_k_train_v1"
BASE_SEED = 20261121


@dataclass(frozen=True)
class AnchorManifest:
    path: str
    sha256: str
    orders: dict[int, np.ndarray]      # replica -> (32,) flat site order (nested prefixes)
    seeds: dict[int, int]

    @classmethod
    def load(cls, path: str | Path) -> "AnchorManifest":
        p = Path(path)
        raw = p.read_bytes()
        d = json.loads(raw)
        if d.get("schema") != EVAL_NAMESPACE or int(d["base_seed"]) != BASE_SEED or tuple(d["K_values"]) != K_VALUES:
            raise ValueError("anchor manifest does not match the frozen contract")
        body = {k: v for k, v in d.items() if k != "manifest_content_sha256"}
        if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != d["manifest_content_sha256"]:
            raise ValueError("anchor manifest content hash mismatch")
        orders, seeds = {}, {}
        for lay in d["layouts"]:
            r = int(lay["replica_id"])
            o = np.asarray(lay["site_order_flat"], dtype=np.int64)
            if o.shape != (32,) or len(set(o.tolist())) != 32 or o.min() < 0 or o.max() >= H * W:
                raise ValueError(f"layout {r} invalid")
            if seeds.setdefault(r, int(lay["seed"])) != stable_seed(EVAL_NAMESPACE, BASE_SEED, r):
                raise ValueError(f"layout {r} seed differs from the namespace rule")
            orders[r] = o
        if sorted(orders) != [0, 1, 2]:
            raise ValueError("three layouts (replicas 0/1/2) required")
        return cls(str(p), hashlib.sha256(raw).hexdigest(), orders, seeds)

    def sites(self, replica: int, K: int) -> np.ndarray:
        if K not in K_VALUES:
            raise ValueError(f"K must be in {K_VALUES}")
        return self.orders[int(replica)][:K].copy()

    def common_domain_mask(self, replica: int) -> np.ndarray:
        m = np.ones(H * W, dtype=bool)
        m[self.orders[int(replica)]] = False
        return m.reshape(H, W)


def train_sites(epoch: int, sim_id: str, frame_index: int, K: int) -> np.ndarray:
    """Random TRAIN anchors for one frame/epoch (nested prefix of a permutation; separate namespace)."""
    if K == 0:
        return np.zeros((0,), dtype=np.int64)
    seed = stable_seed(TRAIN_LAYOUT_NAMESPACE, BASE_SEED, int(epoch), str(sim_id), int(frame_index))
    return np.random.default_rng(seed).permutation(H * W)[:K].astype(np.int64)


def train_K_for_update(update: int) -> int:
    """Uniform over K_VALUES, one draw per optimizer update; same sequence for every family, p and model seed."""
    rng = np.random.default_rng(stable_seed(TRAIN_K_NAMESPACE, BASE_SEED, int(update)))
    return int(K_VALUES[int(rng.integers(len(K_VALUES)))])


def _core(err: np.ndarray) -> dict[str, float]:
    e = err.astype(np.float64)
    if e.size == 0:
        return {"rmse_K": float("nan"), "mae_K": float("nan"), "bias_K": float("nan"), "centered_rmse_K": float("nan")}
    mean = float(e.mean())
    return {"rmse_K": float(np.sqrt(np.mean(e ** 2))), "mae_K": float(np.abs(e).mean()), "bias_K": mean,
            "centered_rmse_K": float(np.sqrt(np.mean((e - mean) ** 2)))}


def domain_metrics(pred_K: np.ndarray, true_K: np.ndarray, anchor_sites: np.ndarray, layout32_sites: np.ndarray) -> dict[str, float]:
    """Full-field, common-domain (grid minus the 32 layout sites), off-anchor(K) and anchors metrics for one frame."""
    err = (np.asarray(pred_K, dtype=np.float64) - np.asarray(true_K, dtype=np.float64)).ravel()
    full = np.ones(err.size, dtype=bool)
    common = full.copy()
    common[np.asarray(layout32_sites, dtype=np.int64)] = False
    off = full.copy()
    off[np.asarray(anchor_sites, dtype=np.int64)] = False
    anc = np.zeros(err.size, dtype=bool)
    anc[np.asarray(anchor_sites, dtype=np.int64)] = True
    out: dict[str, float] = {}
    for name, m in (("full", full), ("common", common), ("offanchor", off), ("anchors", anc)):
        for k, v in _core(err[m]).items():
            out[f"{name}_{k}"] = v
    out["n_common"] = int(common.sum())
    return out
