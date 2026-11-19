"""A38 seed-0 references stay on the Opus stream and the seed-1 contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import yaml

from combustion_recon.models.unet import build_sparse_unet_from_config
from combustion_recon.training.scientific_epoch_sampler import ScientificPresentationStream
from combustion_recon.training.sparse_view import sites_from_training_mask

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/experiments"

_IDENTITY_KEYS = {
    "experiment_id",
    "status",
    "decision",
    "notes",
    "artifacts",
    "training_candidate",
}


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text())


def _state_sha(seed: int) -> str:
    model = build_sparse_unet_from_config(init_seed=seed, base_channels=32)
    digest = hashlib.sha256()
    for key in sorted(model.state_dict()):
        tensor = model.state_dict()[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def test_seed0_yaml_changes_only_identity_seeds_and_output():
    for side in ("on", "off"):
        old = _load(f"opus_O-A38-{side}-s1.yaml")
        new = _load(f"opus_O-A38-{side}-s0.yaml")
        assert set(old) == set(new)
        for key in set(old) - _IDENTITY_KEYS:
            assert new[key] == old[key], key
        assert new["training_candidate"]["seed"] == 0
        assert new["training_candidate"]["shuffle_seed"] == 0
        assert old["training_candidate"]["seed"] == 1
        tc_old = dict(old["training_candidate"])
        tc_new = dict(new["training_candidate"])
        for field in ("seed", "shuffle_seed", "init"):
            tc_old.pop(field)
            tc_new.pop(field)
        assert tc_new == tc_old
        assert new["training_candidate"]["p_conditioning"] is (side == "on")
        assert new["training_candidate"]["pool_id"] == "A38"
        assert new["training_candidate"]["presentations_per_scientific_epoch"] == 760
        assert new["training_candidate"]["train_mask_seed"] == 20261118
        assert new["training_candidate"]["eval_mask_seed"] == 20261116
        assert new["training_candidate"]["save_best_seen_diagnostic"] is True
        assert new["family"] == "sparse_unet"
        assert new["decision"]["by"] == old["decision"]["by"]
        assert new["decision"]["date"] == old["decision"]["date"]
        art_old = dict(old["artifacts"])
        art_new = dict(new["artifacts"])
        for field in ("results_root_engaging", "run_id"):
            art_old.pop(field)
            art_new.pop(field)
        assert art_new == art_old
        assert new["artifacts"]["run_id"] == f"O-A38-{side}-s0"
        assert new["artifacts"]["results_root_engaging"].endswith("official_thesis_opus_a38s0_runs")
        assert new["artifacts"]["results_root_engaging"] != old["artifacts"]["results_root_engaging"]


def test_seed0_stream_differs_from_seed1_and_is_not_the_dataloader():
    pool = 760
    s0 = ScientificPresentationStream(pool, shuffle_seed=0, presentations_per_epoch=760)
    s1 = ScientificPresentationStream(pool, shuffle_seed=1, presentations_per_epoch=760)
    order0 = s0.next_scientific_epoch()
    order1 = s1.next_scientific_epoch()
    assert order0 != order1
    assert set(order0) == set(range(pool))
    generator = torch.Generator()
    generator.manual_seed(100_003)
    assert order0 == torch.randperm(pool, generator=generator).tolist()


def test_train_mask_is_shared_by_seed_and_depends_on_the_mask_seed():
    common = dict(train_mask_seed=20261118, epoch=0, sim_id="0NH3_1", frame_index=0)
    assert sites_from_training_mask(**common)[0].tolist() == sites_from_training_mask(**common)[0].tolist()
    other = dict(common)
    other["train_mask_seed"] = 20261119
    assert sites_from_training_mask(**common)[0].tolist() != sites_from_training_mask(**other)[0].tolist()


def test_seed0_init_is_deterministic_and_distinct_from_seed1():
    first = _state_sha(0)
    second = _state_sha(0)
    assert first == second
    assert first != _state_sha(1)
