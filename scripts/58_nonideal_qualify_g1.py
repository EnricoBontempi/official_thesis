#!/usr/bin/env python3
"""CPU qualification of the G1 generator on TRAIN geometry and historical TRAIN masks.

Reads only: geometry.npz, train_fit nested sites (site indices), the spec. No velocity, no T,
no VAL/TEST, no model. Writes a small JSON summary.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.nonideal_observations import (  # noqa: E402
    GLU_TOP_K, file_sha256, g1_eq_removed, g1_remove, load_degradation_spec,
)


def q(a, ps=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, default=ROOT / "configs/nonideal/degradation_spec_v1.yaml")
    ap.add_argument("--geometry", type=Path, default=ROOT / "data/processed/month2_dense_unet_cache_c149/geometry.npz")
    ap.add_argument("--sites", type=Path, default=ROOT / "data/processed/month2_sparse_eval_masks/train_fit_nested_sites.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "results/nonideal/block1/g1_train_qualification.json")
    a = ap.parse_args()
    spec = load_degradation_spec(a.spec)
    g = np.load(a.geometry)
    x, y = g["x"].astype(np.float64), g["y"].astype(np.float64)
    d = np.load(a.sites, allow_pickle=True)
    sims, frames, reps, s5 = d["sim_id"], d["frame_index"], d["replica_id"], d["sites_5pct"]
    n = len(sims)
    rows = []
    cover = np.zeros((128, 128), dtype=np.int64)
    first_attempt_rejected = 0
    for i in range(n):
        sim, fr, rp = str(sims[i]), int(frames[i]), int(reps[i])
        sites = np.sort(s5[i].astype(np.int64))
        reg, rem = g1_remove(spec.g1, sites, spec_seed=spec.degradation_seed, sim_id=sim,
                             frame_index=fr, replica_id=rp, x_m=x, y_m=y)
        first_attempt_rejected += int(reg.attempt > 0)
        eq = g1_eq_removed(sites, int(rem.sum()), spec_seed=spec.degradation_seed, sim_id=sim,
                           frame_index=fr, replica_id=rp)
        assert eq.sum() == rem.sum()
        cover += reg.mask
        rows.append((int(sites.size), int(rem.sum()), reg.n_cells, reg.n_discs, reg.attempt,
                     reg.area_fraction_effective, reg.area_mm2, int(reg.mask.ravel()[sites].sum())))
    r = np.array(rows, dtype=np.float64)
    n_orig, n_rem, cells, discs, att, af, amm2 = (r[:, k] for k in range(7))
    surv = n_orig - n_rem
    freq = cover / n
    border = np.zeros((128, 128), bool); border[:8] = border[-8:] = True; border[:, :8] = border[:, -8:] = True
    out = {
        "spec_sha256": file_sha256(a.spec), "n_packets": n, "n_frames": int(len(set(zip(sims.tolist(), frames.tolist())))),
        "n_original_sites_unique": sorted(set(n_orig.astype(int).tolist())),
        "requested_area_fraction": spec.g1.area_fraction,
        "effective_area_fraction": q(af),
        "effective_area_mm2": q(amm2),
        "region_cells": q(cells), "n_discs": q(discs),
        "attempts_used_max": int(att.max()), "packets_needing_retry": int(first_attempt_rejected),
        "removed_sites": q(n_rem), "surviving_sites": q(surv),
        "removed_site_fraction": q(n_rem / n_orig),
        "min_surviving_sites": int(surv.min()), "min_required_by_code": 1,
        "glu_top_k": GLU_TOP_K, "packets_with_fewer_than_top_k_survivors": int((surv < GLU_TOP_K).sum()),
        "packets_with_zero_removed": int((n_rem == 0).sum()),
        "per_cell_coverage_frequency": {
            "mean": float(freq.mean()), "min": float(freq.min()), "max": float(freq.max()),
            "mean_outer_8_cell_border": float(freq[border].mean()), "mean_interior": float(freq[~border].mean()),
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
