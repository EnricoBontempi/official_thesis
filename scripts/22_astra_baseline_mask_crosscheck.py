#!/usr/bin/env python3
"""Indirect cross-check of the velocity masks consumed by the baselines (no masks are stored in baseline output).

Compares per (role, replica, sim, frame) n_velocity_sites and n_t9_on_velocity_sites between the baseline
GappyPOD_p_off rows and the neural rows (whose mask_hash was verified at runtime against the persisted
mask). Equality is consistent with, but does NOT prove, identical site sets.
"""
import argparse, json
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--astra-root", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); R = Path(a.astra_root); res = {}
for role in ("seen_train_eval", "val_eval"):
    k = ["replica_id", "sim_id", "frame_index", "cache_index"]
    res[role] = {}
    n = pd.read_csv(R / "s5_sparse_unet_pon_seed0" / f"per_frame_{role}.csv.gz"); n = n[n.treatment == "free"]
    b = pd.read_csv(R / "baselines" / f"per_frame_{role}.csv.gz")
    for m in ("GappyPOD_p_off", "GappyPOD_p_on"):
        g = b[(b.method == m) & (b.treatment == "free")]
        j = n.merge(g, on=k, suffixes=("_n", "_b"))
        res[role][m] = {"n_neural": len(n), "n_baseline": len(g), "n_matched_keys": len(j),
                        "n_velocity_sites_equal": bool((j.n_velocity_sites_n == j.n_velocity_sites_b).all()),
                        "n_t9_on_velocity_sites_equal": bool((j.n_t9_on_velocity_sites_n == j.n_t9_on_velocity_sites_b).all()),
                        "n_velocity_sites_unique": sorted(set(g.n_velocity_sites))}
Path(a.out).write_text(json.dumps(res, indent=2) + "\n"); print(json.dumps(res, indent=2))
