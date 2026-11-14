"""Anti-leakage guardrails for locked operating-condition splits and train stats."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = ROOT / "configs" / "splits" / "splits_interp_extrap.yaml"
STATS_PATH = ROOT / "data" / "manifests" / "train_norm_stats.json"
REMAIN_PATH = ROOT / "data" / "manifests" / "hf_small_metadata" / "combustion__remain_params_numerical.json"


def _load_splits():
    with SPLITS_PATH.open() as f:
        cfg = yaml.safe_load(f)
    return {
        "train": set(cfg["train"]["sim_ids"]),
        "val": set(cfg["val"]["sim_ids"]),
        "test_interp": set(cfg["test_interp"]["sim_ids"]),
        "test_extrap": set(cfg["test_extrap"]["sim_ids"]),
    }


def _params(sim_id: str) -> tuple[float, float]:
    remain = json.loads(REMAIN_PATH.read_text())
    nh3, phi = remain[sim_id]
    return float(nh3), float(phi)


def test_splits_form_exact_partition_of_30():
    splits = _load_splits()
    remain = set(json.loads(REMAIN_PATH.read_text()).keys())
    flat = set().union(*splits.values())
    assert len(flat) == 30
    assert flat == remain
    assert sum(len(v) for v in splits.values()) == 30
    assert len(splits["train"]) == 20
    assert len(splits["val"]) == 3
    assert len(splits["test_interp"]) == 3
    assert len(splits["test_extrap"]) == 4


def test_zero_overlap_across_splits():
    splits = _load_splits()
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = splits[a] & splits[b]
            assert not inter, f"Overlap between {a} and {b}: {inter}"


def test_test_interp_inside_train_ranges():
    splits = _load_splits()
    train_nh3 = [_params(s)[0] for s in splits["train"]]
    train_phi = [_params(s)[1] for s in splits["train"]]
    nh3_min, nh3_max = min(train_nh3), max(train_nh3)
    phi_min, phi_max = min(train_phi), max(train_phi)
    for sid in splits["test_interp"]:
        nh3, phi = _params(sid)
        assert nh3_min <= nh3 <= nh3_max
        assert phi_min <= phi <= phi_max


def test_test_extrap_on_or_outside_frontier():
    splits = _load_splits()
    train_phi = [_params(s)[1] for s in splits["train"]]
    phi_min, phi_max = min(train_phi), max(train_phi)
    # Expected frontier cases from the locked YAML
    lean = "20NH3_0.75.h5"
    rich0 = "0NH3_1.3.h5"
    rich20 = "20NH3_1.3.h5"
    corner = "80NH3_1.25.h5"
    assert lean in splits["test_extrap"]
    assert _params(lean)[1] < phi_min
    assert rich0 in splits["test_extrap"] and _params(rich0)[1] > phi_max
    assert rich20 in splits["test_extrap"] and _params(rich20)[1] > phi_max
    assert corner in splits["test_extrap"]
    # corner is at max NH3 and at/above rich frontier for that blend
    assert _params(corner)[0] == max(_params(s)[0] for s in splits["train"])


@pytest.mark.skipif(not STATS_PATH.exists(), reason="train_norm_stats.json not yet generated")
def test_train_norm_stats_use_only_train_conditions():
    splits = _load_splits()
    stats = json.loads(STATS_PATH.read_text())
    assert stats["meta"]["split"] == "train"
    assert int(stats["meta"]["n_conditions"]) == 20
    assert set(stats["meta"]["sim_ids"]) == splits["train"]
    # Must not include any held-out ids
    held = splits["val"] | splits["test_interp"] | splits["test_extrap"]
    assert set(stats["meta"]["sim_ids"]).isdisjoint(held)
    assert "Temperature" in stats["fields"]
    assert stats["fields"]["Temperature"]["std"] > 0
