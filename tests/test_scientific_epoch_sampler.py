"""Opus scientific-epoch presentation stream and pool nesting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from combustion_recon.training.scientific_epoch_sampler import (
    ListSampler,
    ScientificPresentationStream,
    assert_pool_nesting,
    cache_indices_for_pool,
)
from combustion_recon.training.sparse_scientific_loop import (
    assert_sparse_scientific_candidate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/processed/month2_dense_unet_cache"


def test_protocol_yaml_parses_with_s6_and_opus_keys():
    from combustion_recon.data.sparse_protocol import load_sparse_protocol

    cfg = load_sparse_protocol()
    assert "authorized_s6_offline" in cfg
    assert isinstance(cfg["authorized_s6_offline"], dict)
    assert "items" in cfg["authorized_s6_offline"]
    assert cfg["scientific_training_authorized"] is False


def test_a38_stream_matches_raw_randperm_seed_formula():
    """95 updates / 760 presentations; A38 order == one raw randperm on the seed formula.

    This is not the order produced by DataLoader(shuffle=True). That loader
    draws one integer from the same generator before randperm. See
    test_opus_s5_bridge_order.py.
    """
    pool = 760
    shuffle_seed = 0
    stream = ScientificPresentationStream(
        pool, shuffle_seed=shuffle_seed, presentations_per_epoch=760
    )
    for epoch in range(3):
        got = stream.next_scientific_epoch()
        assert len(got) == 760
        assert len(set(got)) == 760  # one full permutation
        g = torch.Generator()
        g.manual_seed(int(shuffle_seed) + 100_003 * (epoch + 1))
        expected = torch.randperm(pool, generator=g).tolist()
        assert got == expected
    # 95 updates at batch 8
    assert 760 // 8 == 95


def test_b19_two_perms_per_scientific_epoch_and_shared_mask_epoch():
    stream = ScientificPresentationStream(380, shuffle_seed=0, presentations_per_epoch=760)
    idxs = stream.next_scientific_epoch()
    assert len(idxs) == 760
    # Two full permutations of 380 → each pool index appears twice.
    from collections import Counter

    counts = Counter(idxs)
    assert set(counts.values()) == {2}
    assert len(counts) == 380
    # Stream advanced two cycles.
    assert stream.state.cycle == 2
    assert stream.state.pos == 0


def test_stream_resume_mid_cycle_continuous():
    s1 = ScientificPresentationStream(380, shuffle_seed=7, presentations_per_epoch=760)
    first = s1.next_scientific_epoch()
    mid_state = s1.state_dict()
    # Consume one more epoch continuously.
    second_cont = s1.next_scientific_epoch()

    s2 = ScientificPresentationStream(380, shuffle_seed=7, presentations_per_epoch=760)
    s2.load_state_dict(mid_state)
    second_res = s2.next_scientific_epoch()
    assert second_res == second_cont
    assert first != second_cont


def test_c149_pool_residual_across_scientific_epochs():
    """2980 pool: one permutation spans multiple scientific epochs; no dropped tails."""
    stream = ScientificPresentationStream(
        2980, shuffle_seed=1, presentations_per_epoch=760
    )
    # Consume exactly one full permutation via scientific epochs.
    first_perm: list[int] = []
    while len(first_perm) < 2980:
        first_perm.extend(stream.next_scientific_epoch())
    first_perm = first_perm[:2980]
    g = torch.Generator()
    g.manual_seed(1 + 100_003 * 1)
    expected = torch.randperm(2980, generator=g).tolist()
    assert first_perm == expected
    assert set(first_perm) == set(range(2980))
    # Residual position after crossing into next cycle is restored on resume.
    mid = stream.state_dict()
    cont = stream.next_scientific_epoch()
    resumed = ScientificPresentationStream(
        2980, shuffle_seed=1, presentations_per_epoch=760
    )
    resumed.load_state_dict(mid)
    assert resumed.next_scientific_epoch() == cont
    assert 760 // 8 == 95


def test_list_sampler_length_and_order():
    order = [3, 1, 4, 1, 5]
    samp = ListSampler(order)
    assert list(samp) == order
    assert len(samp) == 5
    loader = DataLoader(list(range(10)), batch_size=2, sampler=samp)
    flat = [int(x) for batch in loader for x in batch]
    assert flat == order


@pytest.mark.skipif(not (CACHE / "train_fit" / "frames.json").exists(), reason="cache missing")
def test_pool_nesting_and_counts_on_real_cache():
    frames = json.loads((CACHE / "train_fit" / "frames.json").read_text())
    report = assert_pool_nesting(frames)
    assert report["ok_b19_a38_on_cache"] is True
    assert len(cache_indices_for_pool(frames, "A38")) == 760
    assert len(cache_indices_for_pool(frames, "B19")) == 380
    b19 = set(cache_indices_for_pool(frames, "B19"))
    a38 = set(cache_indices_for_pool(frames, "A38"))
    assert b19 < a38


@pytest.mark.parametrize(
    "name",
    [
        "opus_O-A38-on-s1.yaml",
        "opus_O-A38-off-s1.yaml",
        "opus_O-B19-on-s0.yaml",
        "opus_O-B19-off-s0.yaml",
        "opus_O-C149-on-s0.yaml",
        "opus_O-C149-off-s0.yaml",
    ],
)
def test_opus_candidate_contracts(name: str):
    cand = yaml.safe_load((ROOT / "configs/experiments" / name).read_text())
    meta = assert_sparse_scientific_candidate_contract(cand)
    assert meta["family"] == "sparse_unet"
    assert cand["opus_submission_authorized"] is True
    assert cand["training_candidate"]["presentations_per_scientific_epoch"] == 760
    assert cand["training_candidate"]["amp"] is False
    assert cand["training_candidate"]["torch_compile"] is False
    assert cand["training_candidate"]["require_eval_mask_payload"] is True
