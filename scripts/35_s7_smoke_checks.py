#!/usr/bin/env python3
"""S7 smoke checks on the real C149 cache and persisted masks (non-scientific).

Run on a compute node after the smoke trainings. It does not train for science and
never selects anything. It checks, for the family of the given smoke run dirs:

  * dataset/normalizer/model open on C149 with its id and counts;
  * input isolation: hidden vi/vj mutations leave the input and the prediction
    bit-identical; observed-value mutations change them (positive control);
  * p_on responds to p, p_off ignores p (input channels / GLU p are zero);
  * evaluator path at 1/5/20% x replicas 0/1/2 with site counts 164/819/3277,
    finite outputs, peak GPU memory and time;
  * one real optimizer step at batch 8 with peak memory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.models.glu_reconstructor import (  # noqa: E402
    build_glu_from_config,
    make_synthetic_query_grid,
)
from combustion_recon.models.unet import build_sparse_unet_from_config  # noqa: E402
from combustion_recon.training.dense_cache import cache_validation_kwargs  # noqa: E402
from combustion_recon.training.scientific_epoch_sampler import cache_indices_for_pool  # noqa: E402
from combustion_recon.training.sparse_dataset import (  # noqa: E402
    SparseUNetCacheDataset,
    collate_sparse_batch,
)
from combustion_recon.training.sparse_scientific_loop import (  # noqa: E402
    SparseScientificTrainer,
    config_from_candidate,
)

EXPECTED_SITES = {0.01: 164, 0.05: 819, 0.20: 3277}


def _spread(n: int, k: int) -> list[int]:
    return sorted({int(round(i * (n - 1) / max(1, k - 1))) for i in range(min(k, n))})


class _MutatedField:
    """Array view returning a copy with chosen sites replaced (for isolation checks)."""

    def __init__(self, base, resolve, hidden: bool, scale: float = 1000.0):
        self.base, self.resolve, self.hidden, self.scale = base, resolve, hidden, scale
        self.shape = base.shape

    def __getitem__(self, idx):
        arr = np.array(self.base[idx], dtype=np.float64)
        mask = self.resolve(int(idx))
        sel = ~mask if self.hidden else mask
        rng = np.random.default_rng(1234 + int(idx))
        arr[sel] = arr[sel] + self.scale * rng.standard_normal(int(sel.sum()))
        return arr


def _make_ds(role, cache_root, mask_root, cfg, normalizer, kwargs, *, p_on, coverage=0.05,
             mode="evaluation", replica=0, indices=None):
    return SparseUNetCacheDataset(
        role, cache_root=cache_root, indices=indices, mask_mode=mode,
        site_fraction=coverage, train_mask_seed=int(cfg.train_mask_seed),
        eval_mask_seed=int(cfg.eval_mask_seed), layout_id=str(cfg.layout_id), replica_id=replica,
        p_conditioning=p_on, normalizer=normalizer, eval_mask_payload_root=mask_root,
        require_eval_mask_payload=True, **kwargs,
    )


def _trainer(family, cfg, normalizer, cache_root, ds, ckpt, device):
    if family == "sparse_unet":
        model = build_sparse_unet_from_config(init_seed=int(cfg.seed))
        q = None
    else:
        model = build_glu_from_config(domain_geometry_path=Path(cache_root) / "geometry.npz")
        q = make_synthetic_query_grid(ds.x, ds.y)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    tr = SparseScientificTrainer(model, cfg=cfg, normalizer=normalizer, n_train_samples=len(ds),
                                 query_coords_phys=q)
    return tr


def _predict(tr, ds, i_list, family):
    batch = collate_sparse_batch([ds[i] for i in i_list])
    if family == "adapted_glu":
        batch["p_norm_tensor"] = torch.tensor(
            [[float(ds.p_norm[int(m["cache_index"]), 0]), float(ds.p_norm[int(m["cache_index"]), 1])]
             for m in batch["metadata"]], dtype=torch.float32)
    tr.eval_mode()
    with torch.no_grad():
        return batch, tr._forward_norm(batch, decode_chunked=True).detach().cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-on", type=Path, required=True)
    ap.add_argument("--run-off", type=Path, required=True)
    ap.add_argument("--config-on", type=Path, required=True)
    ap.add_argument("--config-off", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--mask-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda"
    assert torch.cuda.is_available()
    report: dict = {"non_scientific_smoke": True, "gpu": torch.cuda.get_device_name(0), "checks": {}}
    normalizer = Month2FitOnlyNormalizer(ROOT / "data/manifests/train_norm_stats_month2_fit.json")
    kwargs = cache_validation_kwargs("C149")
    results = {}
    for tag, run_dir, cfg_path, p_on in (
        ("p_on", args.run_on, args.config_on, True),
        ("p_off", args.run_off, args.config_off, False),
    ):
        cand = yaml.safe_load(cfg_path.read_text())
        cfg = config_from_candidate(cand, device=dev)
        family = cfg.family
        assert cfg.p_conditioning is p_on
        ck = run_dir / "checkpoints" / "last.pt"
        rep: dict = {"family": family, "checkpoint": str(ck)}
        val_all = _make_ds("val_eval", args.cache_root, args.mask_root, cfg, normalizer, kwargs, p_on=p_on)
        rep["c149_cache_opened"] = {"cache_id": val_all.manifest["cache_id"], "counts": val_all.manifest["counts"]}
        sel = _spread(len(val_all), args.frames)
        val = _make_ds("val_eval", args.cache_root, args.mask_root, cfg, normalizer, kwargs, p_on=p_on, indices=sel)
        tr = _trainer(family, cfg, normalizer, args.cache_root, val, ck, dev)

        # ---- input isolation (hidden mutation) + positive control (observed mutation)
        i0 = [0, 1]
        base_batch, base_pred = _predict(tr, val, i0, family)
        def resolve(idx):
            meta = dict(val.frames[idx])
            sites, _, _ = val._resolve_sites(idx, meta)
            m = np.zeros(val.vi.shape[1:], dtype=bool)
            m.ravel()[sites] = True
            return m
        iso = {}
        for name, hidden in (("hidden_mutation", True), ("observed_mutation", False)):
            mut = _make_ds("val_eval", args.cache_root, args.mask_root, cfg, normalizer, kwargs,
                           p_on=p_on, indices=sel)
            mut.vi = _MutatedField(val.vi, resolve, hidden)
            mut.vj = _MutatedField(val.vj, resolve, hidden)
            mb, mp = _predict(tr, mut, i0, family)
            same_in = bool(torch.equal(mb["input_tensor"], base_batch["input_tensor"]))
            same_tok = all(
                np.array_equal(a["values_norm"], b["values_norm"]) and np.array_equal(a["site_index"], b["site_index"])
                for a, b in zip(mb["glu_tokens"], base_batch["glu_tokens"]))
            same_pred = bool(torch.equal(mp, base_pred))
            iso[name] = {"input_identical": same_in, "tokens_identical": same_tok, "prediction_identical": same_pred}
        assert iso["hidden_mutation"] == {"input_identical": True, "tokens_identical": True, "prediction_identical": True}, iso
        ob = iso["observed_mutation"]
        assert not ob["tokens_identical"] and not ob["prediction_identical"], iso
        rep["isolation"] = iso

        # ---- p behaviour: change p_norm and compare
        class _PMut:
            def __init__(self, base): self.base = base; self.shape = base.shape
            def __getitem__(self, idx): return np.asarray(self.base[idx], dtype=np.float64) + 3.0
        pm = _make_ds("val_eval", args.cache_root, args.mask_root, cfg, normalizer, kwargs, p_on=p_on, indices=sel)
        pm.p_norm = _PMut(val.p_norm)
        pb, pp = _predict(tr, pm, i0, family)
        p_changed_input = not torch.equal(pb["input_tensor"], base_batch["input_tensor"])
        p_changed_pred = not torch.equal(pp, base_pred)
        if p_on:
            assert p_changed_pred, "p_on prediction must respond to p"
        else:
            assert not p_changed_pred and not p_changed_input, "p_off must ignore p"
            if family == "sparse_unet":
                assert bool(torch.all(base_batch["input_tensor"][:, -2:] == 0))
        rep["p_behaviour"] = {"input_changes_with_p": p_changed_input, "prediction_changes_with_p": p_changed_pred}

        # ---- evaluator path 1/5/20 x replicas, memory, time
        cov_rep = {}
        for cov in (0.01, 0.05, 0.20):
            for rid in (0, 1, 2):
                ds = _make_ds("val_eval", args.cache_root, args.mask_root, cfg, normalizer, kwargs,
                              p_on=p_on, coverage=cov, replica=rid, indices=sel)
                n_sites = ds[0]["observation"].n_sites
                assert n_sites == EXPECTED_SITES[cov], (cov, n_sites)
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
                out = tr.evaluate_dataset(ds, role="val_eval", replica_id=rid, method=family,
                                          variant=tag, model_seed=int(cfg.seed))
                torch.cuda.synchronize()
                rmse = out["mean_condition_rmse_K"]
                assert rmse is not None and np.isfinite(rmse)
                cov_rep[f"{cov}_r{rid}"] = {
                    "n_sites": int(n_sites), "mean_condition_rmse_K": float(rmse),
                    "frames": len(ds), "seconds": round(time.perf_counter() - t0, 2),
                    "peak_gpu_mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                }
        rep["evaluator_coverages"] = cov_rep
        # latency batch 1 at 20% (worst case)
        ds20 = _make_ds("val_eval", args.cache_root, args.mask_root, cfg, normalizer, kwargs, p_on=p_on,
                        coverage=0.20, indices=sel[:1])
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
        tr.evaluate_dataset(ds20, role="val_eval", replica_id=0, method=family, variant=tag,
                            model_seed=int(cfg.seed), batch_size=1)
        torch.cuda.synchronize()
        rep["latency_batch1_20pct"] = {"seconds": round(time.perf_counter() - t0, 2),
                                       "peak_gpu_mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3)}

        # ---- one real training step, batch 8, 5% training masks
        trn = _make_ds("train_fit", args.cache_root, args.mask_root, cfg, normalizer, kwargs, p_on=p_on,
                       mode="training", indices=cache_indices_for_pool(
                           json.loads((args.cache_root / "train_fit" / "frames.json").read_text()), "C149")[:64])
        trn.set_epoch(0)
        batch = collate_sparse_batch([trn[i] for i in range(8)])
        if family == "adapted_glu":
            batch["p_norm_tensor"] = torch.tensor([[float(trn.p_norm[int(m["cache_index"]), 0]),
                                                    float(trn.p_norm[int(m["cache_index"]), 1])] for m in batch["metadata"]])
        tr.train_mode(); tr.state.update = 100
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
        step = tr._one_train_step(batch)
        torch.cuda.synchronize()
        assert not isinstance(step, str), step
        rep["train_step_batch8_5pct"] = {"loss": step[0], "grad_norm_pre_clip": step[1],
                                         "seconds": round(time.perf_counter() - t0, 2),
                                         "peak_gpu_mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
        results[tag] = rep
        del tr; torch.cuda.empty_cache()
    report["checks"] = results
    a = json.loads((args.run_on / "init_manifest.json").read_text())["init_state_dict_sha256"]
    b = json.loads((args.run_off / "init_manifest.json").read_text())["init_state_dict_sha256"]
    report["init_hash_on_equals_off"] = a == b
    assert a == b, "p_on/p_off must share the initialization within a family"
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
