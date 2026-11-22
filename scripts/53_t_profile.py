#!/usr/bin/env python3
"""Profile the assisted path on the real cache (non-scientific): step time per K, eval throughput per frame, peak memory.
Estimates seconds per epoch and per evaluation for one family; the run costs are extrapolated from these measurements."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.data.t_anchors import K_VALUES, AnchorManifest  # noqa: E402
from combustion_recon.models.glu_assisted import build_assisted_glu_from_config  # noqa: E402
from combustion_recon.models.glu_reconstructor import make_synthetic_query_grid  # noqa: E402
from combustion_recon.models.unet import build_sparse_unet_assisted_from_config  # noqa: E402
from combustion_recon.training.assisted_dataset import AssistedDataset  # noqa: E402
from combustion_recon.training.assisted_trainer import AssistedTrainer  # noqa: E402
from combustion_recon.training.dense_cache import cache_validation_kwargs  # noqa: E402
from combustion_recon.training.scientific_epoch_sampler import cache_indices_for_pool  # noqa: E402
from combustion_recon.training.sparse_dataset import collate_sparse_batch  # noqa: E402
from combustion_recon.training.sparse_scientific_loop import SparseScientificConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("unet", "glu"), required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--mask-root", type=Path, required=True)
    ap.add_argument("--stats", type=Path, default=Path("/home/bontempi/projects/official_thesis_s5/data/manifests/train_norm_stats_month2_fit.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    fam = "sparse_unet" if args.family == "unet" else "adapted_glu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    anchors = AnchorManifest.load(ROOT / "configs/observations/month2_temperature_anchors_nested_v1.json")
    norm = Month2FitOnlyNormalizer(args.stats)
    kw = cache_validation_kwargs("C149")
    ref760 = cache_indices_for_pool(json.loads((args.cache_root / "train_fit" / "frames.json").read_text()), "A38")
    tr = AssistedDataset("train_fit", cache_root=args.cache_root, indices=ref760[:64], mask_mode="training", site_fraction=0.05, train_mask_seed=20261118,
                         p_conditioning=True, normalizer=norm, eval_mask_payload_root=args.mask_root, require_eval_mask_payload=True, anchors=anchors, **kw)
    cfg = SparseScientificConfig(family=fam, p_conditioning=True, seed=1, device="cuda", batch_size=8, decode_chunked_eval=True)
    torch.manual_seed(1)
    model = build_sparse_unet_assisted_from_config(init_seed=1) if fam == "sparse_unet" else build_assisted_glu_from_config(domain_geometry_path=args.cache_root / "geometry.npz")
    model.to("cuda")
    queries = None if fam == "sparse_unet" else make_synthetic_query_grid(tr.x, tr.y)
    trainer = AssistedTrainer(model, cfg=cfg, normalizer=norm, n_train_samples=64, query_coords_phys=queries, anchors=anchors)
    trainer._set_lr(1e-3)
    out = {"family": fam, "n_params": sum(p.numel() for p in model.parameters()), "step_s_by_K": {}, "data_s_by_K": {}, "eval_frame_s_by_K": {}}
    torch.cuda.reset_peak_memory_stats()
    for K in K_VALUES:
        steps, data = [], []
        for it in range(6):
            t0 = time.perf_counter()
            batch = collate_sparse_batch([tr[(8 * (it % 8) + j, K)] for j in range(8)])
            if fam == "adapted_glu":
                batch["p_norm_tensor"] = torch.tensor([[float(tr.p_norm[int(m["cache_index"]), 0]), float(tr.p_norm[int(m["cache_index"]), 1])] for m in batch["metadata"]])
            t1 = time.perf_counter()
            r = trainer._one_train_step(batch)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            if it >= 2:
                data.append(t1 - t0)
                steps.append(t2 - t1)
        out["step_s_by_K"][K] = float(np.mean(steps))
        out["data_s_by_K"][K] = float(np.mean(data))
    out["train_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    trainer.eval_mode()
    tr.mask_mode = "evaluation"
    tr.replica_id = 0
    for K in (0, 32):
        tr.eval_K = K
        t0 = time.perf_counter()
        n = 0
        with torch.no_grad():
            for s in range(0, 32, 8):
                b = collate_sparse_batch([tr[s + j] for j in range(8)])
                if fam == "adapted_glu":
                    b["p_norm_tensor"] = torch.tensor([[float(tr.p_norm[int(m["cache_index"]), 0]), float(tr.p_norm[int(m["cache_index"]), 1])] for m in b["metadata"]])
                trainer._forward_norm(b, decode_chunked=True)
                n += 8
        torch.cuda.synchronize()
        out["eval_frame_s_by_K"][K] = (time.perf_counter() - t0) / n
    mean_step = float(np.mean(list(out["step_s_by_K"].values())) + np.mean(list(out["data_s_by_K"].values())))
    ef = float(np.mean(list(out["eval_frame_s_by_K"].values())))
    # evaluation every 10 epochs: VAL 123 x 6K x 3 replicas, TRAIN_FIT 760 x 3 diag K, SEEN 220 x 3 diag K (+ per-frame metric cost ~ 3 ms)
    n_eval_frames = 123 * 6 * 3 + 760 * 3 + 220 * 3
    out["epoch_s_estimate"] = 95 * mean_step
    out["eval_s_estimate"] = n_eval_frames * (ef + 0.003)
    out["eval_frames_per_eval"] = n_eval_frames
    out["peak_mem_GB_all"] = torch.cuda.max_memory_allocated() / 1e9
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
