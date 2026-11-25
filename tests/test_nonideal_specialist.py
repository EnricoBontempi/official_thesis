"""CPU tests of the specialist retraining design (no checkpoint, no VAL/TEST)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from combustion_recon.data.nonideal_observations import (
    CellSpec, DegradationSpec, degradation_seed_for, file_sha256, load_degradation_spec, load_velocity_scale,
)
from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.training import nonideal_specialist as sp
from combustion_recon.training.dense_cache import cache_validation_kwargs
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset, collate_sparse_batch

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/processed/month2_dense_unet_cache_c149"
STATS = ROOT / "data/manifests/train_norm_stats_month2_fit.json"
SPEC = ROOT / "configs/nonideal/degradation_spec_v1_12models.yaml"
HAS = (CACHE / "train_fit" / "vi_norm.npy").exists()
needs_cache = pytest.mark.skipif(not HAS, reason="C149 cache not available")


@pytest.fixture(scope="module")
def spec():
    return load_degradation_spec(SPEC)


@pytest.fixture(scope="module")
def scale():
    return load_velocity_scale(STATS)


def _block(regime="G1", **kw):
    b = {"regime": regime, "train_degradation_seed": sp.TRAIN_SEED, "selection_degradation_seed": sp.SELECTION_SEED,
         "clean_fraction": 0, "degradation_spec": "configs/nonideal/degradation_spec_v1_12models.yaml",
         "degradation_spec_sha256": file_sha256(SPEC)}
    b.update(kw)
    return b


GLU_TC = {"p_conditioning": True, "microbatch_size": 1}
UNET_TC = {"p_conditioning": True, "microbatch_size": None}


def test_namespace_seeds_distinct_from_eval_and_historical(spec):
    assert len({sp.TRAIN_SEED, sp.SELECTION_SEED, sp.EVAL_SEED, 20261114, 20261116, 20261118}) == 6
    with pytest.raises(sp.SpecialistContractError):
        sp.namespace_spec(spec, sp.EVAL_SEED)
    with pytest.raises(sp.SpecialistContractError):
        sp.namespace_spec(spec, 20261118)
    seen = set()
    for ns in (sp.TRAIN_SEED, sp.SELECTION_SEED, sp.EVAL_SEED):
        for fr in range(30):
            for rep in range(5):
                for op in ("noise", "g1", "g1_eq"):
                    v = degradation_seed_for(ns, op, "0NH3_1.h5", fr, rep)
                    assert v not in seen
                    seen.add(v)


@pytest.mark.parametrize("mutate,msg", [
    (lambda b, tc: b.update(regime="R0"), "regime"),
    (lambda b, tc: b.update(train_degradation_seed=20261125), "seeds"),
    (lambda b, tc: b.update(clean_fraction=0.25), "clean"),
    (lambda b, tc: tc.update(p_conditioning=False), "p_on"),
    (lambda b, tc: tc.update(microbatch_size=None), "microbatch"),
    (lambda b, tc: b.update(degradation_spec_sha256="0" * 64), "hash"),
])
def test_contract_refusals_glu(spec, mutate, msg):
    b, tc = _block(), dict(GLU_TC)
    mutate(b, tc)
    with pytest.raises(sp.SpecialistContractError, match=msg):
        sp.validate_specialist_block(b, family="adapted_glu", tc=tc, root=ROOT)


def test_contract_unet_and_ok_paths():
    assert sp.validate_specialist_block(_block("N1G1"), family="adapted_glu", tc=GLU_TC, root=ROOT)["cell"].cell_id == "N1G1"
    assert sp.validate_specialist_block(_block("G1-eq"), family="sparse_unet", tc=UNET_TC, root=ROOT)["cell"].gap == "g1_eq"
    with pytest.raises(sp.SpecialistContractError, match="microbatch"):
        sp.validate_specialist_block(_block(), family="sparse_unet", tc={**UNET_TC, "microbatch_size": 1}, root=ROOT)


def _ds(cls, role="train_fit", idx=(0, 700, 2000), **kw):
    return cls(role, cache_root=CACHE, indices=list(idx), mask_mode="training", site_fraction=0.05,
               train_mask_seed=20261118, p_conditioning=True, normalizer=Month2FitOnlyNormalizer(STATS),
               **cache_validation_kwargs("C149"), **kw)


@needs_cache
def test_identity_cell_reproduces_original_training_items(spec, scale):
    base = _ds(SparseUNetCacheDataset)
    ident = _ds(sp.SpecialistTrainDataset, cell=CellSpec("ID", None, None), spec=spec, scale=scale)
    for ds in (base, ident):
        ds.set_epoch(3)
    for i in range(len(base)):
        a, b = base[i], ident[i]
        assert torch.equal(a["model_inputs"]["input_tensor"], b["model_inputs"]["input_tensor"])
        assert np.array_equal(a["observation"].sites, b["observation"].sites)
        assert np.array_equal(a["glu_tokens"]["values_norm"], b["glu_tokens"]["values_norm"])


@needs_cache
def test_regimes_realizations_pairing_and_validity(spec, scale):
    cells = {c: spec.cells[c] for c in sp.REGIMES}
    dss = {c: _ds(sp.SpecialistTrainDataset, cell=cells[c], spec=spec, scale=scale) for c in cells}
    base = _ds(SparseUNetCacheDataset)
    for ds in (*dss.values(), base):
        ds.set_epoch(5)
    for i in range(3):
        o = base[i]["observation"]
        it = {c: dss[c][i] for c in cells}
        obs = {c: it[c]["observation"] for c in cells}
        # N1/N2 share the standardized noise; no sites lost
        d1 = obs["N1"].velocity_values_phys - o.velocity_values_phys
        d2 = obs["N2"].velocity_values_phys - o.velocity_values_phys
        assert np.allclose(d2, 3 * d1, atol=1e-9) and obs["N1"].n_sites == o.n_sites == 819
        # G1 / N1G1 share survivors; G1-eq removes the same number
        assert np.array_equal(obs["G1"].sites, obs["N1G1"].sites)
        assert obs["G1-eq"].n_sites == obs["G1"].n_sites < 819
        assert not np.array_equal(obs["G1-eq"].sites, obs["G1"].sites)
        # survivors keep their historical sites (subset), values untouched for G1
        assert set(obs["G1"].sites) <= set(o.sites)
        pos = np.searchsorted(o.sites, obs["G1"].sites)
        assert np.array_equal(obs["G1"].velocity_values_norm, o.velocity_values_norm[pos])
        # model inputs: tokens = survivors only; U-Net grid zero + Mv=0 at removed sites; p_on channels active
        for c in ("G1", "N1G1", "G1-eq"):
            x = it[c]["model_inputs"]["input_tensor"].numpy()
            rem = np.setdiff1d(o.sites, obs[c].sites)
            ys, xs = rem // 128, rem % 128
            assert not x[0][ys, xs].any() and not x[1][ys, xs].any() and not x[2][ys, xs].any()
            assert x[2].sum() == obs[c].n_sites == it[c]["glu_tokens"]["site_index"].size
            assert np.abs(x[5:7]).max() > 0  # p_on: NH3 and phi channels are not zero
            assert not np.isin(rem, it[c]["glu_tokens"]["site_index"]).any()
    # ragged GLU batch collates (token lists are not stacked)
    b = collate_sparse_batch([dss["G1"][i] for i in range(3)])
    assert len({len(t["site_index"]) for t in b["glu_tokens"]}) > 1


@needs_cache
def test_train_realizations_vary_by_epoch_and_differ_from_eval_and_selection(spec, scale):
    ds = _ds(sp.SpecialistTrainDataset, cell=spec.cells["N1G1"], spec=spec, scale=scale)
    ds.set_epoch(0); a = ds[0]["observation"]
    ds.set_epoch(1); b = ds[0]["observation"]
    ds.set_epoch(0); a2 = ds[0]["observation"]
    assert np.array_equal(a.sites, a2.sites) and np.array_equal(a.velocity_values_norm, a2.velocity_values_norm)  # deterministic
    assert not np.array_equal(a.sites, b.sites)  # new region (and new TRAIN-mask sites) every epoch
    # same sample/epoch in a different "training run": identical packages (key has no model seed or family)
    ds2 = _ds(sp.SpecialistTrainDataset, cell=spec.cells["N1G1"], spec=spec, scale=scale)
    ds2.set_epoch(0)
    assert np.array_equal(ds2[0]["observation"].velocity_values_norm, a.velocity_values_norm)
    # TRAIN-namespace noise differs from the evaluation and selection namespaces for the same key
    from combustion_recon.data.nonideal_observations import standardized_noise
    m = ds[0]["metadata"]
    z = [standardized_noise(s, m["sim_id"], int(m["frame_index"]), 0) for s in (sp.TRAIN_SEED, sp.SELECTION_SEED, sp.EVAL_SEED)]
    assert not np.array_equal(z[0], z[1]) and not np.array_equal(z[0], z[2]) and not np.array_equal(z[1], z[2])


@needs_cache
def test_no_temperature_dependence_of_inputs(spec, scale):
    ds = _ds(sp.SpecialistTrainDataset, cell=spec.cells["N1G1"], spec=spec, scale=scale)
    ds.set_epoch(2)
    before = [ds[i]["model_inputs"]["input_tensor"].clone() for i in range(3)]
    ds.t_K = np.zeros_like(np.asarray(ds.t_K[:1]).repeat(len(ds.vi), axis=0)) + 123.0
    ds.t_norm = np.zeros_like(ds.t_K)
    after = [ds[i]["model_inputs"]["input_tensor"] for i in range(3)]
    assert all(torch.equal(a, b) for a, b in zip(before, after))


# ----------------------------------------------------------------------------- microbatch / accumulation

def _tiny_trainer(mb):
    from combustion_recon.models.glu_reconstructor import build_glu_from_config, make_synthetic_query_grid
    from combustion_recon.training.sparse_scientific_loop import SparseScientificTrainer, config_from_candidate
    cand = yaml.safe_load((ROOT / "configs/experiments/s7/S7-C149-glu-on-s0.yaml").read_text())
    cfg = config_from_candidate(cand, device="cpu", overrides={"microbatch_size": mb, "presentations_per_scientific_epoch": 760})
    x = np.linspace(-2, 2, 8)
    torch.manual_seed(0)
    model = build_glu_from_config(domain_x=x, domain_y=x, height=8, width=8)
    q = make_synthetic_query_grid(x, x)
    return SparseScientificTrainer(model, cfg=cfg, normalizer=None, n_train_samples=2980, query_coords_phys=q), model


def _batch(n_tokens, seed=0):
    rng = np.random.default_rng(seed)
    toks = []
    for n in n_tokens:
        toks.append({"coords_phys": rng.normal(size=(n, 2)) * 0.5, "values_norm": rng.normal(size=(n, 2)),
                     "site_index": np.sort(rng.choice(64, size=n, replace=False)).astype(np.int64)})
    return {"glu_tokens": toks, "target_T_norm": torch.tensor(rng.normal(size=(len(n_tokens), 1, 8, 8)), dtype=torch.float32),
            "p_norm_tensor": torch.tensor(rng.normal(size=(len(n_tokens), 2)), dtype=torch.float32),
            "metadata": [{} for _ in n_tokens]}


def test_microbatch_one_equals_full_batch_for_homogeneous_glu_batches():
    t_full, m_full = _tiny_trainer(None)
    t_mb, m_mb = _tiny_trainer(1)
    m_mb.load_state_dict(m_full.state_dict())
    for t in (t_full, t_mb):
        t._set_lr(1e-3)
        t.train_mode()
    b = _batch([40] * 8)
    la, ga = t_full._one_train_step(copy.deepcopy(b))
    lb, gb = t_mb._one_train_step(copy.deepcopy(b))
    assert abs(la - lb) < 1e-6 * max(1.0, abs(la))                       # loss: mean over the batch of 8
    assert abs(ga - gb) < 1e-5 * max(1.0, ga)                            # gradient norm before clipping
    for (k, p), (_, q) in zip(m_full.named_parameters(), m_mb.named_parameters()):
        assert torch.allclose(p, q, atol=1e-6, rtol=1e-5), k            # one optimizer step: same update
    # Adam's first step is almost sign-like, so also compare the (post-clip) gradients themselves
    worst = max(float((p.grad - q.grad).abs().max() / (p.grad.abs().max() + 1e-12))
                for p, q in zip(m_full.parameters(), m_mb.parameters()) if p.grad is not None)
    assert worst < 1e-4, worst
    assert t_full.state.update == t_mb.state.update == 1               # one optimizer step per effective batch
    assert t_full.state.samples_seen == t_mb.state.samples_seen == 8


def test_ragged_batch_needs_microbatch_one_and_clips_once():
    ragged = _batch([30, 31, 40, 45, 50, 33, 36, 39])
    t_mb, m_mb = _tiny_trainer(1)
    t_mb._set_lr(1e-3); t_mb.train_mode()
    out = t_mb._one_train_step(copy.deepcopy(ragged))
    assert not isinstance(out, str) and np.isfinite(out[0]) and t_mb.state.update == 1 and t_mb.state.samples_seen == 8
    t_full, _ = _tiny_trainer(None)
    t_full._set_lr(1e-3); t_full.train_mode()
    with pytest.raises(Exception):  # the unmodified batch-8 forward cannot stack ragged token sets
        t_full._one_train_step(copy.deepcopy(ragged))


def test_microbatch_mean_not_biased_by_token_count():
    """Equal weighting per sample: split-by-sample loss equals the batch mean even with different N per sample."""
    t, m = _tiny_trainer(1)
    t.train_mode()
    b = _batch([30, 50, 40, 36])
    losses = []
    with torch.no_grad():
        for i in range(4):
            sub = {**b, "glu_tokens": [b["glu_tokens"][i]], "target_T_norm": b["target_T_norm"][i:i + 1], "p_norm_tensor": b["p_norm_tensor"][i:i + 1]}
            pred = t._forward_norm(sub, decode_chunked=False)
            losses.append(float(torch.mean((pred - sub["target_T_norm"]) ** 2)))
    t._set_lr(0.0)
    loss_v, _ = t._one_train_step(copy.deepcopy(b))
    assert abs(loss_v - float(np.mean(losses))) < 1e-6
