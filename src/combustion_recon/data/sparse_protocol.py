"""Load and validate month2_sparse_protocol.yaml (refuse unknown scientific keys)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPARSE_PROTOCOL = ROOT / "configs" / "experiments" / "month2_sparse_protocol.yaml"

ALLOWED_TOP = {
    "protocol_id",
    "roadmap",
    "status",
    "scientific_training_authorized",
    "roles",
    "observation",
    "training_masks",
    "seeds",
    "normalization",
    "cache",
    "unet_sparse",
    "glu",
    "forbidden_this_mission",
    "diagnostic_overfit",
}


def load_sparse_protocol(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_SPARSE_PROTOCOL
    with p.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("sparse protocol must be a mapping")
    unknown = sorted(set(cfg) - ALLOWED_TOP)
    if unknown:
        raise ValueError(f"sparse protocol has unsupported keys (refused): {unknown}")
    if cfg.get("scientific_training_authorized") is not False:
        raise ValueError("scientific_training_authorized must be false for S1–S2 delivery")
    if cfg.get("glu", {}).get("wrapper_authorized_this_mission") is not False:
        raise ValueError("GLU wrapper not authorized in this mission")
    roles = cfg.get("roles") or {}
    if roles.get("test_payloads") not in (None, "none", False):
        raise ValueError("test_payloads must be none")
    obs = cfg.get("observation") or {}
    if int(obs.get("K", -1)) != 0:
        raise ValueError("K must be 0")
    seeds = cfg.get("seeds") or {}
    for k in ("model_seed", "shuffle_seed", "train_mask_seed", "eval_mask_seed"):
        if k not in seeds:
            raise ValueError(f"seeds.{k} required")
    if int(seeds["train_mask_seed"]) == int(seeds["eval_mask_seed"]):
        # Not forbidden mathematically, but our contract uses distinct values.
        raise ValueError("train_mask_seed and eval_mask_seed must be distinct namespaces")
    unet = cfg.get("unet_sparse") or {}
    if int(unet.get("channels", -1)) != 7:
        raise ValueError("unet_sparse.channels must be 7")
    order = list(unet.get("channel_order") or [])
    expected = [
        "vi_obs_norm",
        "vj_obs_norm",
        "Mv",
        "x_scaled",
        "y_scaled",
        "nh3_norm",
        "phi_norm",
    ]
    if order != expected:
        raise ValueError(f"unet_sparse.channel_order must be {expected}")
    return cfg
