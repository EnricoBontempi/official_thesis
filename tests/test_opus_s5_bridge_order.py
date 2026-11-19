"""Synthetic bridge: new A38 stream versus the DataLoader path S5 executed."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from combustion_recon.training.scientific_epoch_sampler import ScientificPresentationStream
from combustion_recon.training.sparse_view import sites_from_training_mask


class _IndexDataset(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        return int(index)


def _dataloader_order(n: int, seed: int, batch_size: int = 8) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        _IndexDataset(n),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    order: list[int] = []
    for batch in loader:
        order.extend(int(v) for v in batch)
    return order


def _s5_seed(epoch: int, shuffle_seed: int = 0) -> int:
    return int(shuffle_seed) + 100_003 * (epoch + 1)


def test_a38_stream_differs_from_executed_dataloader_order():
    pool = 760
    stream = ScientificPresentationStream(
        pool, shuffle_seed=0, presentations_per_epoch=760
    )
    streamed_by_epoch = [stream.next_scientific_epoch() for _ in range(60)]
    for epoch in (0, 1, 59):
        streamed = streamed_by_epoch[epoch]
        loaded = _dataloader_order(pool, _s5_seed(epoch))
        assert set(streamed) == set(loaded) == set(range(pool))
        assert streamed != loaded
        n_pos = sum(a != b for a, b in zip(streamed, loaded))
        n_batch = sum(
            streamed[i : i + 8] != loaded[i : i + 8] for i in range(0, pool, 8)
        )
        assert n_pos >= 758
        assert n_batch == 95


def test_dataloader_order_is_randperm_after_one_generator_draw():
    n = 760
    seed = _s5_seed(0)
    loaded = _dataloader_order(n, seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    expected = torch.randperm(n, generator=generator).tolist()
    assert loaded == expected


def test_train_mask_depends_on_epoch_not_call_order():
    first = sites_from_training_mask(
        train_mask_seed=20261118,
        epoch=3,
        sim_id="0NH3_1",
        frame_index=40,
    )
    second = sites_from_training_mask(
        train_mask_seed=20261118,
        epoch=3,
        sim_id="0NH3_1",
        frame_index=40,
    )
    later = sites_from_training_mask(
        train_mask_seed=20261118,
        epoch=4,
        sim_id="0NH3_1",
        frame_index=40,
    )
    assert first[0].tolist() == second[0].tolist()
    assert first[0].tolist() != later[0].tolist()
