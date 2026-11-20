"""The ten S7 configs are the Opus C149 candidate with only the matrix fields changed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from combustion_recon.training.sparse_scientific_loop import (
    assert_sparse_scientific_candidate_contract,
    config_from_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
MATRIX = json.loads((ROOT / "configs/experiments/month2_s7_c149_matrix.json").read_text())
OPUS = {
    True: yaml.safe_load((ROOT / "configs/experiments/opus_O-C149-on-s0.yaml").read_text()),
    False: yaml.safe_load((ROOT / "configs/experiments/opus_O-C149-off-s0.yaml").read_text()),
}
RUNS = MATRIX["new_runs"]


def test_matrix_is_exactly_the_approved_ten():
    ids = [r["run_id"] for r in RUNS]
    assert sorted(ids) == sorted(
        [f"S7-C149-unet-{p}-s{s}" for p in ("on", "off") for s in (1, 2)]
        + [f"S7-C149-glu-{p}-s{s}" for p in ("on", "off") for s in (0, 1, 2)]
    )
    assert [r["array_index"] for r in RUNS] == list(range(10))
    assert len(set(ids)) == 10
    assert {r["run_id"] for r in MATRIX["reused_runs"]} == {"O-C149-on-s0", "O-C149-off-s0"}
    assert {r["training_job"] for r in MATRIX["reused_runs"]} == {29645053, 29645054}
    # nothing new may collide with a reused run
    assert not ({r["run_id"] for r in RUNS} & {r["run_id"] for r in MATRIX["reused_runs"]})
    assert len({r["out_dir"] for r in RUNS}) == 10


def test_frozen_input_hashes_match_the_roadmap():
    s = MATRIX["shared_inputs"]
    assert s["expect_cache_aggregate_sha256"] == "fa1c9358deb3bda0313cd74f9ab6855af87bc969c4069a463073d4c4120f03b4"
    assert s["mask_manifest_file_sha256"] == "2e2d8ff6edabf62ad56702a155140d064c0b4014d06ee96debf52811efa5c75e"
    assert s["mask_aggregate_payload_sha256"] == "7be78f35ffea2ebc598adce8cec39dc719339aac17ecadde3137f2064891ffeb"
    assert s["mask_manifest_file_sha256"] != s["mask_aggregate_payload_sha256"]
    assert s["cache_counts"] == {"train_fit": 2980, "seen_train_eval": 220, "val_eval": 123}


@pytest.mark.parametrize("run", RUNS, ids=[r["run_id"] for r in RUNS])
def test_config_is_opus_recipe_with_only_matrix_fields_changed(run):
    cfg = yaml.safe_load((ROOT / run["config"]).read_text())
    meta = assert_sparse_scientific_candidate_contract(cfg)
    assert meta["family"] == run["family"]
    assert cfg["s7_submission_authorized"] is True and "opus_submission_authorized" not in cfg
    tc, ref = cfg["training_candidate"], OPUS[run["p_conditioning"]]["training_candidate"]
    assert tc["p_conditioning"] is run["p_conditioning"]
    assert tc["seed"] == tc["shuffle_seed"] == run["seed"]
    assert tc["family"] == run["family"] == cfg["family"]
    changed_allowed = {"family", "seed", "shuffle_seed", "init", "wall_time_cap_hours"}
    for key in set(tc) | set(ref):
        if key in changed_allowed:
            continue
        assert tc.get(key) == ref.get(key), f"scientific field {key} differs from Opus C149"
    assert tc["pool_id"] == "C149" and tc["presentations_per_scientific_epoch"] == 760
    assert tc["train_mask_seed"] == 20261118 and tc["eval_mask_seed"] == 20261116
    assert tc["batch_size"] == 8 and tc["warmup_updates"] == 950 and tc["microbatch_size"] is None
    assert tc["require_eval_mask_payload"] is True and tc["save_best_seen_diagnostic"] is True
    assert tc["early_stopping"] == ref["early_stopping"] and tc["plateau"] == ref["plateau"]
    assert tc["n_epochs"] == 1000 and tc["eval_every_epochs"] == 10
    rt = config_from_candidate(cfg, device="cpu")
    assert rt.family == run["family"] and rt.pool_id == "C149"
    assert rt.amp_enabled is False and rt.torch_compile is False and rt.allow_tf32 is False


def test_seed_pairs_share_seed_and_differ_only_in_p():
    by = {r["run_id"]: yaml.safe_load((ROOT / r["config"]).read_text()) for r in RUNS}
    for fam in ("unet", "glu"):
        for s in (1, 2) if fam == "unet" else (0, 1, 2):
            on, off = by[f"S7-C149-{fam}-on-s{s}"], by[f"S7-C149-{fam}-off-s{s}"]
            a, b = dict(on["training_candidate"]), dict(off["training_candidate"])
            assert a.pop("p_conditioning") is True and b.pop("p_conditioning") is False
            assert a == b
